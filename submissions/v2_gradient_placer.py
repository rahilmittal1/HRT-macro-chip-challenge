"""
V2 Gradient Placer — PyTorch Adam Optimizer with Multi-Objective Loss

Architecture:
  Loss = 1.0 * WL + λ_den * DensityPenalty + λ_cong * CongestionPenalty + λ_ovlp * OverlapPenalty

  - WL: Differentiable HPWL via log-sum-exp over real net connectivity
  - DensityPenalty: Grid-based area overlap (matches TILOS top-10% metric)
  - CongestionPenalty: TILOS evaluator heatmap feedback every 100 steps
  - OverlapPenalty: Pairwise hard-macro overlap area

  Fixed macros (incl. soft macros) contribute to density/congestion but have NO gradients.
  Only movable hard macros are optimized.

  Dynamic weighting: λ_den starts high (spread early) → decays (let WL tighten);
                     λ_cong ramps up as TILOS feedback becomes available.

Hyperparams tunable via env vars (for Optuna meta_tuner.py):
  PLACER_LR, PLACER_NUM_STEPS, PLACER_LAMBDA_DEN_START, PLACER_LAMBDA_DEN_END,
  PLACER_LAMBDA_CONG_START, PLACER_LAMBDA_CONG_END

Usage:
    uv run evaluate submissions/v2_gradient_placer.py -b ibm01
    uv run evaluate submissions/v2_gradient_placer.py --all
"""

import math
import os
import torch
import torch.nn.functional as F
from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark_from_dir, load_benchmark
from macro_place.objective import compute_proxy_cost


# ═══════════════════════════════════════════════════════════════════════════════
# Placer class
# ═══════════════════════════════════════════════════════════════════════════════

class V2GradientPlacer:

    def __init__(self):
        self.lr = float(os.environ.get("PLACER_LR", "0.05"))
        self.num_steps = int(os.environ.get("PLACER_NUM_STEPS", "500"))
        self.lam_den_start = float(os.environ.get("PLACER_LAMBDA_DEN_START", "2.0"))
        self.lam_den_end = float(os.environ.get("PLACER_LAMBDA_DEN_END", "0.3"))
        self.lam_cong_start = float(os.environ.get("PLACER_LAMBDA_CONG_START", "0.2"))
        self.lam_cong_end = float(os.environ.get("PLACER_LAMBDA_CONG_END", "1.0"))

    # ── Main entry ────────────────────────────────────────────────────────

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        plc = self._load_plc(benchmark)
        sizes = benchmark.macro_sizes                        # [N, 2]
        cw, ch = benchmark.canvas_width, benchmark.canvas_height
        num_hard = benchmark.num_hard_macros
        num_macros = benchmark.num_macros
        movable_mask = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()

        half_w = sizes[:, 0] / 2                             # [N]
        half_h = sizes[:, 1] / 2
        grid_rows, grid_cols = benchmark.grid_rows, benchmark.grid_cols
        gw, gh = cw / grid_cols, ch / grid_rows
        cell_area = gw * gh

        # ── Extract net connectivity for HPWL ─────────────────────────────
        net_data = self._extract_nets(benchmark, plc)

        # ── Grid cell boundary tensors (for vectorised density) ───────────
        cell_x_lo = torch.arange(grid_cols, dtype=torch.float32) * gw        # [C]
        cell_x_hi = cell_x_lo + gw
        cell_y_lo = torch.arange(grid_rows, dtype=torch.float32) * gh        # [R]
        cell_y_hi = cell_y_lo + gh

        # ── Optimisable positions ─────────────────────────────────────────
        positions = benchmark.macro_positions.clone().requires_grad_(True)
        fixed_pos = benchmark.macro_positions.clone()
        optimizer = torch.optim.Adam([positions], lr=self.lr)

        # ── Congestion heatmap (updated from TILOS every 100 steps) ───────
        cong_map = torch.zeros(grid_rows, grid_cols)

        # ── Training loop ─────────────────────────────────────────────────
        best_cost = float("inf")
        best_pos = positions.detach().clone()

        for step in range(self.num_steps):
            optimizer.zero_grad()

            # Differentiable clamp to canvas
            pos = torch.stack([
                torch.clamp(positions[:, 0], min=half_w, max=cw - half_w),
                torch.clamp(positions[:, 1], min=half_h, max=ch - half_h),
            ], dim=1)                                        # [N, 2]

            # ── Losses ────────────────────────────────────────────────────
            # Wirelength (HPWL via log-sum-exp)
            gamma = 5.0 + 15.0 * (step / self.num_steps)    # sharpen over time
            wl_loss = self._hpwl_loss(pos, net_data, cw, ch, gamma)

            # Density (grid overlap, top-10% average, matches TILOS)
            den_loss = self._density_loss(
                pos, sizes, num_macros,
                cell_x_lo, cell_x_hi, cell_y_lo, cell_y_hi, cell_area,
                grid_rows, grid_cols,
            )

            # Overlap (pairwise hard-macro)
            ovlp_loss = self._overlap_loss(pos, sizes, num_hard)

            # Congestion (from TILOS heatmap, differentiable via grid_sample)
            cong_loss = self._congestion_loss(
                pos, cong_map, movable_mask,
                grid_rows, grid_cols, gw, gh,
            )

            # ── Dynamic weights ───────────────────────────────────────────
            frac = step / self.num_steps
            lam_den = self.lam_den_start + (self.lam_den_end - self.lam_den_start) * frac
            lam_cong = self.lam_cong_start + (self.lam_cong_end - self.lam_cong_start) * frac
            lam_ovlp = 10.0

            loss = wl_loss + lam_den * den_loss + lam_cong * cong_loss + lam_ovlp * ovlp_loss

            loss.backward()

            # Zero grad for non-movable macros (fixed hard + all soft)
            with torch.no_grad():
                if positions.grad is not None:
                    positions.grad[~movable_mask] = 0.0

            optimizer.step()

            # Hard reset non-movable positions (belt and suspenders)
            with torch.no_grad():
                positions[~movable_mask] = fixed_pos[~movable_mask]

            # ── TILOS congestion feedback every 100 steps ─────────────────
            if plc is not None and (step + 1) % 100 == 0:
                with torch.no_grad():
                    snap = positions.detach().clone()
                    snap[:, 0] = torch.clamp(snap[:, 0], min=half_w, max=cw - half_w)
                    snap[:, 1] = torch.clamp(snap[:, 1], min=half_h, max=ch - half_h)
                    costs = compute_proxy_cost(snap, benchmark, plc)
                    # Read congestion grids
                    v_cong = torch.tensor(plc.V_routing_cong, dtype=torch.float32)
                    h_cong = torch.tensor(plc.H_routing_cong, dtype=torch.float32)
                    cong_map = torch.max(v_cong, h_cong).reshape(grid_rows, grid_cols)

                    # Track best
                    if costs["overlap_count"] == 0 and costs["proxy_cost"] < best_cost:
                        best_cost = costs["proxy_cost"]

        # ── Final clamped positions ───────────────────────────────────────
        final = positions.detach().clone()
        final[:, 0] = torch.clamp(final[:, 0], min=half_w, max=cw - half_w)
        final[:, 1] = torch.clamp(final[:, 1], min=half_h, max=ch - half_h)

        # ── Spiral legalization ───────────────────────────────────────────
        final = self._legalize(final, benchmark)

        return final

    # ═══════════════════════════════════════════════════════════════════════
    # Loss functions
    # ═══════════════════════════════════════════════════════════════════════

    def _hpwl_loss(self, pos, net_data, cw, ch, gamma):
        """Differentiable HPWL via log-sum-exp over real net connectivity."""
        if net_data is None:
            return torch.tensor(0.0)

        padded_idx = net_data["padded_macro_idx"]     # [num_nets, max_pins]
        padded_off = net_data["padded_offsets"]        # [num_nets, max_pins, 2]
        padded_port = net_data["padded_is_port"]       # [num_nets, max_pins]
        padded_mask = net_data["padded_mask"]           # [num_nets, max_pins]
        num_nets = padded_idx.shape[0]

        if num_nets == 0:
            return torch.tensor(0.0)

        # Pin positions from macro centers + offsets
        macro_pos = pos[padded_idx.clamp(min=0)]       # [num_nets, max_pins, 2]
        pin_pos = macro_pos + padded_off
        # For port pins, offsets are absolute positions
        pin_pos = torch.where(padded_port.unsqueeze(2), padded_off, pin_pos)

        pin_x = pin_pos[:, :, 0]                       # [num_nets, max_pins]
        pin_y = pin_pos[:, :, 1]

        # Mask invalid pins with extreme values
        BIG = max(cw, ch) * 10
        pin_x_hi = torch.where(padded_mask, pin_x, torch.tensor(-BIG))
        pin_x_lo = torch.where(padded_mask, pin_x, torch.tensor(BIG))
        pin_y_hi = torch.where(padded_mask, pin_y, torch.tensor(-BIG))
        pin_y_lo = torch.where(padded_mask, pin_y, torch.tensor(BIG))

        # Normalize to [0, 1] to prevent overflow in exp
        pin_x_hi_n = pin_x_hi / cw
        pin_x_lo_n = pin_x_lo / cw
        pin_y_hi_n = pin_y_hi / ch
        pin_y_lo_n = pin_y_lo / ch

        # Smooth max/min
        max_x = torch.logsumexp(gamma * pin_x_hi_n, dim=1) / gamma * cw
        min_x = -torch.logsumexp(-gamma * pin_x_lo_n, dim=1) / gamma * cw
        max_y = torch.logsumexp(gamma * pin_y_hi_n, dim=1) / gamma * ch
        min_y = -torch.logsumexp(-gamma * pin_y_lo_n, dim=1) / gamma * ch

        hpwl = (max_x - min_x) + (max_y - min_y)      # [num_nets]
        total = hpwl.sum()

        # Normalise to ≈ [0, 1] range
        diag = math.sqrt(cw**2 + ch**2)
        return total / (num_nets * diag + 1e-8)

    def _density_loss(self, pos, sizes, num_macros,
                      cx_lo, cx_hi, cy_lo, cy_hi, cell_area,
                      grid_rows, grid_cols):
        """Differentiable density matching TILOS top-10% metric.

        Includes ALL macros (hard + soft).  Fixed macros contribute density
        but their positions are constant — no gradient flows through them.
        """
        # Macro bounding boxes
        x_lo = pos[:num_macros, 0] - sizes[:num_macros, 0] / 2   # [N]
        x_hi = pos[:num_macros, 0] + sizes[:num_macros, 0] / 2
        y_lo = pos[:num_macros, 1] - sizes[:num_macros, 1] / 2
        y_hi = pos[:num_macros, 1] + sizes[:num_macros, 1] / 2

        # Overlap area per macro per cell (vectorised)
        # ox[n, c] = clamp(min(x_hi[n], cx_hi[c]) - max(x_lo[n], cx_lo[c]), min=0)
        ox = torch.clamp(
            torch.min(x_hi.unsqueeze(1), cx_hi.unsqueeze(0))
            - torch.max(x_lo.unsqueeze(1), cx_lo.unsqueeze(0)),
            min=0,
        )  # [N, C]

        oy = torch.clamp(
            torch.min(y_hi.unsqueeze(1), cy_hi.unsqueeze(0))
            - torch.max(y_lo.unsqueeze(1), cy_lo.unsqueeze(0)),
            min=0,
        )  # [N, R]

        # Full overlap: [N, R, C] = oy[:, :, None] * ox[:, None, :]
        overlap = oy.unsqueeze(2) * ox.unsqueeze(1)        # [N, R, C]
        density = overlap.sum(0) / cell_area                # [R, C]

        # Top-10% average (matches TILOS get_density_cost)
        flat = density.flatten()
        k = max(1, int(flat.numel() * 0.10))
        top_vals, _ = torch.topk(flat, k)
        return 0.5 * top_vals.mean()

    def _overlap_loss(self, pos, sizes, num_hard):
        """Differentiable pairwise hard-macro overlap area."""
        if num_hard <= 1:
            return torch.tensor(0.0)

        hx = pos[:num_hard, 0]                             # [H]
        hy = pos[:num_hard, 1]
        hw = sizes[:num_hard, 0]
        hh = sizes[:num_hard, 1]

        dx = torch.abs(hx.unsqueeze(1) - hx.unsqueeze(0))  # [H, H]
        dy = torch.abs(hy.unsqueeze(1) - hy.unsqueeze(0))
        min_sep_x = (hw.unsqueeze(1) + hw.unsqueeze(0)) / 2
        min_sep_y = (hh.unsqueeze(1) + hh.unsqueeze(0)) / 2

        ovlp_x = torch.clamp(min_sep_x - dx, min=0)
        ovlp_y = torch.clamp(min_sep_y - dy, min=0)
        ovlp = ovlp_x * ovlp_y                             # [H, H]

        # Zero diagonal, take upper triangle
        mask = torch.triu(torch.ones(num_hard, num_hard, dtype=torch.bool), diagonal=1)
        total = ovlp[mask].sum()

        # Normalise by canvas area
        canvas_area = sizes[:num_hard, 0].sum() * sizes[:num_hard, 1].mean()
        return total / (canvas_area + 1e-8)

    def _congestion_loss(self, pos, cong_map, movable_mask,
                         grid_rows, grid_cols, gw, gh):
        """Congestion penalty using TILOS heatmap (updated every 100 steps).

        Uses F.grid_sample for differentiable bilinear interpolation:
        macros in congested cells get pushed toward lower-congestion regions.
        """
        if cong_map.sum() == 0:
            return torch.tensor(0.0)

        movable_idx = torch.where(movable_mask)[0]
        if len(movable_idx) == 0:
            return torch.tensor(0.0)

        mov_pos = pos[movable_idx]                          # [M, 2]

        # Normalise coordinates to [-1, 1] for grid_sample
        norm_x = (mov_pos[:, 0] / (grid_cols * gw)) * 2 - 1
        norm_y = (mov_pos[:, 1] / (grid_rows * gh)) * 2 - 1

        # grid_sample expects [B, C, H, W] input and [B, N, 1, 2] grid
        inp = cong_map.unsqueeze(0).unsqueeze(0)            # [1, 1, R, C]
        grid = torch.stack([norm_x, norm_y], dim=1)         # [M, 2]
        grid = grid.unsqueeze(0).unsqueeze(2)               # [1, M, 1, 2]

        sampled = F.grid_sample(
            inp, grid, mode="bilinear", padding_mode="border", align_corners=False,
        )                                                   # [1, 1, M, 1]

        return sampled.squeeze().mean()

    # ═══════════════════════════════════════════════════════════════════════
    # Net extraction
    # ═══════════════════════════════════════════════════════════════════════

    def _extract_nets(self, benchmark, plc):
        """Extract net connectivity as padded tensors for vectorised HPWL."""
        if plc is None:
            return None

        # plc module index → benchmark tensor index
        plc_to_bench = {}
        for bi, pi in enumerate(benchmark.hard_macro_indices):
            plc_to_bench[pi] = bi
        for bi, pi in enumerate(benchmark.soft_macro_indices):
            plc_to_bench[pi] = benchmark.num_hard_macros + bi

        port_set = set(getattr(plc, "port_indices", []))

        MAX_PINS = 50  # cap net size for memory
        nets = []       # list of [(is_port, bench_idx, ox, oy), ...]

        for driver_name in plc.nets:
            pins = []
            sink_names = plc.nets[driver_name]

            # -- driver pin --
            drv_plc = plc.mod_name_to_indices[driver_name]
            drv_mod = plc.modules_w_pins[drv_plc]
            drv_parent = plc.get_ref_node_id(drv_plc)

            if drv_parent in plc_to_bench:
                bi = plc_to_bench[drv_parent]
                ox = getattr(drv_mod, "x_offset", 0.0)
                oy = getattr(drv_mod, "y_offset", 0.0)
                pins.append((False, bi, ox, oy))
            elif drv_parent in port_set:
                x, y = plc.modules_w_pins[drv_parent].get_pos()
                pins.append((True, 0, x, y))

            # -- sink pins --
            for sname in sink_names:
                s_plc = plc.mod_name_to_indices[sname]
                s_mod = plc.modules_w_pins[s_plc]
                s_parent = plc.get_ref_node_id(s_plc)
                if s_parent in plc_to_bench:
                    bi = plc_to_bench[s_parent]
                    ox = getattr(s_mod, "x_offset", 0.0)
                    oy = getattr(s_mod, "y_offset", 0.0)
                    pins.append((False, bi, ox, oy))
                elif s_parent in port_set:
                    x, y = plc.modules_w_pins[s_parent].get_pos()
                    pins.append((True, 0, x, y))

            if 2 <= len(pins) <= MAX_PINS:
                nets.append(pins)

        if not nets:
            return None

        num_nets = len(nets)
        max_pins = max(len(n) for n in nets)

        padded_idx = torch.zeros(num_nets, max_pins, dtype=torch.long)
        padded_off = torch.zeros(num_nets, max_pins, 2)
        padded_port = torch.zeros(num_nets, max_pins, dtype=torch.bool)
        padded_mask = torch.zeros(num_nets, max_pins, dtype=torch.bool)

        for n, net in enumerate(nets):
            for k, (is_port, bi, ox, oy) in enumerate(net):
                padded_idx[n, k] = bi
                padded_off[n, k, 0] = ox
                padded_off[n, k, 1] = oy
                padded_port[n, k] = is_port
                padded_mask[n, k] = True

        return {
            "padded_macro_idx": padded_idx,
            "padded_offsets": padded_off,
            "padded_is_port": padded_port,
            "padded_mask": padded_mask,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # Spiral legalization
    # ═══════════════════════════════════════════════════════════════════════

    def _legalize(self, placement, benchmark):
        """Spiral-search legalization: largest macro first, nearest clear spot."""
        sizes = benchmark.macro_sizes
        cw, ch = benchmark.canvas_width, benchmark.canvas_height
        movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        movable_idx = torch.where(movable)[0].tolist()
        movable_idx.sort(key=lambda i: -(sizes[i, 0].item() * sizes[i, 1].item()))

        gap = 0.01
        placed = []  # (idx, x, y, w, h)

        for idx in movable_idx:
            x, y = placement[idx, 0].item(), placement[idx, 1].item()
            w, h = sizes[idx, 0].item(), sizes[idx, 1].item()
            hw, hh = w / 2, h / 2
            x = max(hw, min(cw - hw, x))
            y = max(hh, min(ch - hh, y))

            if not self._overlaps(x, y, w, h, placed, gap):
                placement[idx, 0] = x
                placement[idx, 1] = y
                placed.append((idx, x, y, w, h))
                continue

            found = False
            for rs in range(1, 400):
                r = rs * 0.2
                na = max(8, rs * 4)
                for ai in range(na):
                    a = 2 * math.pi * ai / na
                    nx = x + r * math.cos(a)
                    ny = y + r * math.sin(a)
                    nx = max(hw, min(cw - hw, nx))
                    ny = max(hh, min(ch - hh, ny))
                    if not self._overlaps(nx, ny, w, h, placed, gap):
                        placement[idx, 0] = nx
                        placement[idx, 1] = ny
                        placed.append((idx, nx, ny, w, h))
                        found = True
                        break
                if found:
                    break
            if not found:
                placed.append((idx, x, y, w, h))

        return placement

    @staticmethod
    def _overlaps(x, y, w, h, placed, gap):
        for _, px, py, pw, ph in placed:
            if abs(x - px) < (w + pw) / 2 + gap and abs(y - py) < (h + ph) / 2 + gap:
                return True
        return False

    # ═══════════════════════════════════════════════════════════════════════
    # PlacementCost loader
    # ═══════════════════════════════════════════════════════════════════════

    def _load_plc(self, benchmark):
        testcase_root = "external/MacroPlacement/Testcases/ICCAD04"
        bench_dir = os.path.join(testcase_root, benchmark.name)
        ng45 = {
            "ariane133": "external/MacroPlacement/Flows/NanGate45/ariane133/netlist/output_CT_Grouping",
            "ariane136": "external/MacroPlacement/Flows/NanGate45/ariane136/netlist/output_CT_Grouping",
            "mempool_tile": "external/MacroPlacement/Flows/NanGate45/mempool_tile/netlist/output_CT_Grouping",
            "nvdla": "external/MacroPlacement/Flows/NanGate45/nvdla/netlist/output_CT_Grouping",
        }
        if os.path.exists(bench_dir):
            _, plc = load_benchmark_from_dir(bench_dir)
            return plc
        if benchmark.name in ng45:
            d = ng45[benchmark.name]
            _, plc = load_benchmark(f"{d}/netlist.pb.txt", f"{d}/initial.plc")
            return plc
        return None
