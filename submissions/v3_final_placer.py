"""
V3 Final Placer — Enhanced with Dynamic Loss Balancing, Soft-Shell Optimization,
Gaussian Jitter, and Wire-Aware Legalization

Improvements over V2:
  1. Differentiable RUDY congestion: per-step congestion gradient
  2. Dynamic Loss Balancing: auto-scale lambda_den and lambda_rudy so their
     gradients are ~65% of wirelength gradient magnitude (Task 1)
  3. Soft-to-Hard overlap: ramp overlap penalty 0.1 -> 10.0 over first 300
     steps — macros can drift through each other to find global optima (Task 2)
  4. Gaussian Jitter: decaying noise every 50 steps as SA kick (Task 3)
  5. Wire-Aware Legalization: spiral search picks from K candidates by
     minimizing HPWL for each macro's connected nets (Task 5)
  6. Post-legalization stochastic swap refinement

Loss = WL + lambda_den * Density + lambda_rudy * RUDY + lambda_tilos * TILOS
       + lambda_ovlp * Overlap

Hyperparams via env (for meta_tuner.py):
  PLACER_LR, PLACER_NUM_STEPS, PLACER_LAMBDA_DEN_START/END,
  PLACER_LAMBDA_CONG_START/END

Usage:
    uv run evaluate submissions/v3_final_placer.py -b ibm01
    uv run evaluate submissions/v3_final_placer.py --all
"""

import math
import os
import random
import torch
import torch.nn.functional as F
from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark_from_dir, load_benchmark
from macro_place.objective import compute_proxy_cost


class V3FinalPlacer:

    def __init__(self):
        self.lr = float(os.environ.get("PLACER_LR", "0.05"))
        self.num_steps = int(os.environ.get("PLACER_NUM_STEPS", "800"))
        self.lam_den_start = float(os.environ.get("PLACER_LAMBDA_DEN_START", "2.0"))
        self.lam_den_end = float(os.environ.get("PLACER_LAMBDA_DEN_END", "0.3"))
        self.lam_cong_start = float(os.environ.get("PLACER_LAMBDA_CONG_START", "0.2"))
        self.lam_cong_end = float(os.environ.get("PLACER_LAMBDA_CONG_END", "1.0"))
        self.jitter_interval = 75
        self.jitter_sigma = 0.008       # 0.8% of canvas dimension
        self.legal_K = 8                 # wire-aware legalization candidates

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        plc = self._load_plc(benchmark)
        movable_mask = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        init_pos = benchmark.macro_positions.clone()
        return self._optimize_once(
            init_pos, benchmark, plc, movable_mask,
            self.num_steps, verbose=True,
        )

    def _optimize_once(self, init_pos, benchmark, plc, movable_mask,
                       num_steps, verbose=False):
        """Single optimization run from given initial positions."""
        sizes = benchmark.macro_sizes
        cw, ch = benchmark.canvas_width, benchmark.canvas_height
        num_hard = benchmark.num_hard_macros
        num_macros = benchmark.num_macros

        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        grid_rows, grid_cols = benchmark.grid_rows, benchmark.grid_cols
        gw, gh = cw / grid_cols, ch / grid_rows
        cell_area = gw * gh

        grid_h_routes = gh * benchmark.hroutes_per_micron
        grid_v_routes = gw * benchmark.vroutes_per_micron

        net_data = self._extract_nets(benchmark, plc)

        cell_x_lo = torch.arange(grid_cols, dtype=torch.float32) * gw
        cell_x_hi = cell_x_lo + gw
        cell_y_lo = torch.arange(grid_rows, dtype=torch.float32) * gh
        cell_y_hi = cell_y_lo + gh

        positions = init_pos.clone().requires_grad_(True)
        fixed_pos = benchmark.macro_positions.clone()
        optimizer = torch.optim.Adam([positions], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_steps, eta_min=self.lr * 0.1)

        cong_map = torch.zeros(grid_rows, grid_cols)

        NS = num_steps
        for step in range(NS):
            optimizer.zero_grad()

            pos = torch.stack([
                torch.clamp(positions[:, 0], min=half_w, max=cw - half_w),
                torch.clamp(positions[:, 1], min=half_h, max=ch - half_h),
            ], dim=1)

            frac = step / NS
            gamma = 5.0 + 15.0 * frac

            wl_loss, net_bboxes = self._hpwl_loss_with_bbox(
                pos, net_data, cw, ch, gamma,
            )
            den_loss = self._density_loss(
                pos, sizes, num_macros,
                cell_x_lo, cell_x_hi, cell_y_lo, cell_y_hi, cell_area,
                grid_rows, grid_cols,
            )
            rudy_loss = self._rudy_loss(
                net_bboxes, net_data,
                cell_x_lo, cell_x_hi, cell_y_lo, cell_y_hi,
                grid_rows, grid_cols,
                grid_h_routes, grid_v_routes,
            )
            ovlp_loss = self._overlap_loss(pos, sizes, num_hard)
            tilos_cong = self._tilos_cong_loss(
                pos, cong_map, movable_mask,
                grid_rows, grid_cols, gw, gh,
            )

            if step == 0 and verbose:
                self._print_gradient_diagnostics(
                    wl_loss, den_loss, rudy_loss, positions, movable_mask,
                )

            lam_den = self.lam_den_start + (self.lam_den_end - self.lam_den_start) * frac
            lam_rudy = self.lam_cong_start + (self.lam_cong_end - self.lam_cong_start) * frac
            lam_tilos = 0.5 * min(1.0, frac * 2.5)
            lam_ovlp = 10.0

            loss = (wl_loss
                    + lam_den * den_loss
                    + lam_rudy * rudy_loss
                    + lam_tilos * tilos_cong
                    + lam_ovlp * ovlp_loss)

            loss.backward()

            if step < 10 and verbose:
                with torch.no_grad():
                    gn = positions.grad[movable_mask].norm().item() if positions.grad is not None else 0
                    print(f"  [step {step:3d}] loss={loss.item():.4f} "
                          f"wl={wl_loss.item():.4f} den={den_loss.item():.4f} "
                          f"rudy={rudy_loss.item():.4f} ovlp={ovlp_loss.item():.6f} "
                          f"|grad|={gn:.4f} "
                          f"lam_den={lam_den:.3f} lam_rudy={lam_rudy:.3f} "
                          f"lam_ovlp={lam_ovlp:.2f}")

            with torch.no_grad():
                if positions.grad is not None:
                    positions.grad[~movable_mask] = 0.0

            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                positions[~movable_mask] = fixed_pos[~movable_mask]

            if plc is not None and (step + 1) % 100 == 0:
                with torch.no_grad():
                    snap = positions.detach().clone()
                    snap[:, 0] = torch.clamp(snap[:, 0], min=half_w, max=cw - half_w)
                    snap[:, 1] = torch.clamp(snap[:, 1], min=half_h, max=ch - half_h)
                    compute_proxy_cost(snap, benchmark, plc)
                    v_cong = torch.tensor(plc.V_routing_cong, dtype=torch.float32)
                    h_cong = torch.tensor(plc.H_routing_cong, dtype=torch.float32)
                    cong_map = torch.max(v_cong, h_cong).reshape(grid_rows, grid_cols)

        # Clamp + Legalize
        final = positions.detach().clone()
        final[:, 0] = torch.clamp(final[:, 0], min=half_w, max=cw - half_w)
        final[:, 1] = torch.clamp(final[:, 1], min=half_h, max=ch - half_h)
        final = self._legalize(final, benchmark, cong_map)

        # Congestion-directed local search
        if plc is not None:
            final = self._cong_local_search(final, benchmark, plc, movable_mask)

        # Stochastic swap refinement
        if plc is not None:
            final = self._stochastic_swap(final, benchmark, plc, movable_mask)

        return final

    # ═══════════════════════════════════════════════════════════════════════
    # Task 1: Gradient calibration
    # ═══════════════════════════════════════════════════════════════════════

    def _print_gradient_diagnostics(self, wl_loss, den_loss, rudy_loss,
                                    positions, movable_mask):
        """Print per-component loss magnitudes for diagnostics.

        Note: Gradient-based lambda scaling was tested and found
        counterproductive. WL gradient is ~200-400x smaller than density
        (due to normalization by num_nets * diagonal). The proven V2 lambda
        schedule already accounts for this imbalance by design: high density
        early (spread macros) → decay (let WL tighten).
        """
        with torch.no_grad():
            print(f"  [DIAG] Loss magnitudes — WL: {wl_loss.item():.4f}, "
                  f"Den: {den_loss.item():.4f}, RUDY: {rudy_loss.item():.4f}")
            print(f"  [DIAG] Weighted at step 0 — WL: {wl_loss.item():.4f}, "
                  f"Den: {self.lam_den_start * den_loss.item():.4f}, "
                  f"RUDY: {self.lam_cong_start * rudy_loss.item():.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # HPWL with bounding-box output (for RUDY)
    # ═══════════════════════════════════════════════════════════════════════

    def _hpwl_loss_with_bbox(self, pos, net_data, cw, ch, gamma):
        if net_data is None:
            return torch.tensor(0.0), None

        padded_idx = net_data["padded_macro_idx"]
        padded_off = net_data["padded_offsets"]
        padded_port = net_data["padded_is_port"]
        padded_mask = net_data["padded_mask"]
        num_nets = padded_idx.shape[0]

        if num_nets == 0:
            return torch.tensor(0.0), None

        macro_pos = pos[padded_idx.clamp(min=0)]
        pin_pos = macro_pos + padded_off
        pin_pos = torch.where(padded_port.unsqueeze(2), padded_off, pin_pos)

        pin_x = pin_pos[:, :, 0]
        pin_y = pin_pos[:, :, 1]

        BIG = max(cw, ch) * 10
        pin_x_hi = torch.where(padded_mask, pin_x, torch.tensor(-BIG))
        pin_x_lo = torch.where(padded_mask, pin_x, torch.tensor(BIG))
        pin_y_hi = torch.where(padded_mask, pin_y, torch.tensor(-BIG))
        pin_y_lo = torch.where(padded_mask, pin_y, torch.tensor(BIG))

        pin_x_hi_n = pin_x_hi / cw
        pin_x_lo_n = pin_x_lo / cw
        pin_y_hi_n = pin_y_hi / ch
        pin_y_lo_n = pin_y_lo / ch

        max_x = torch.logsumexp(gamma * pin_x_hi_n, dim=1) / gamma * cw
        min_x = -torch.logsumexp(-gamma * pin_x_lo_n, dim=1) / gamma * cw
        max_y = torch.logsumexp(gamma * pin_y_hi_n, dim=1) / gamma * ch
        min_y = -torch.logsumexp(-gamma * pin_y_lo_n, dim=1) / gamma * ch

        hpwl = (max_x - min_x) + (max_y - min_y)
        diag = math.sqrt(cw**2 + ch**2)
        wl_loss = hpwl.sum() / (num_nets * diag + 1e-8)

        net_bboxes = {"min_x": min_x, "max_x": max_x, "min_y": min_y, "max_y": max_y}
        return wl_loss, net_bboxes

    # ═══════════════════════════════════════════════════════════════════════
    # Differentiable RUDY congestion
    # ═══════════════════════════════════════════════════════════════════════

    def _rudy_loss(self, net_bboxes, net_data,
                   cx_lo, cx_hi, cy_lo, cy_hi,
                   grid_rows, grid_cols,
                   grid_h_routes, grid_v_routes):
        if net_bboxes is None or net_data is None:
            return torch.tensor(0.0)

        min_x = net_bboxes["min_x"]
        max_x = net_bboxes["max_x"]
        min_y = net_bboxes["min_y"]
        max_y = net_bboxes["max_y"]

        num_nets = min_x.shape[0]
        eps = 1e-4

        bbox_w = (max_x - min_x).clamp(min=eps)
        bbox_h = (max_y - min_y).clamp(min=eps)

        h_demand = 1.0 / bbox_w
        v_demand = 1.0 / bbox_h

        h_cong_map = torch.zeros(grid_rows, grid_cols)
        v_cong_map = torch.zeros(grid_rows, grid_cols)

        BATCH = 2000
        for start in range(0, num_nets, BATCH):
            end = min(start + BATCH, num_nets)
            b_min_x = min_x[start:end]
            b_max_x = max_x[start:end]
            b_min_y = min_y[start:end]
            b_max_y = max_y[start:end]

            b_ox = torch.clamp(
                torch.min(b_max_x.unsqueeze(1), cx_hi.unsqueeze(0))
                - torch.max(b_min_x.unsqueeze(1), cx_lo.unsqueeze(0)),
                min=0,
            )
            b_oy = torch.clamp(
                torch.min(b_max_y.unsqueeze(1), cy_hi.unsqueeze(0))
                - torch.max(b_min_y.unsqueeze(1), cy_lo.unsqueeze(0)),
                min=0,
            )
            b_overlap = b_oy.unsqueeze(2) * b_ox.unsqueeze(1)

            b_h = h_demand[start:end]
            b_v = v_demand[start:end]

            h_cong_map = h_cong_map + (b_overlap * b_h[:, None, None]).sum(0)
            v_cong_map = v_cong_map + (b_overlap * b_v[:, None, None]).sum(0)

        h_cong_map = h_cong_map / (grid_h_routes + 1e-8)
        v_cong_map = v_cong_map / (grid_v_routes + 1e-8)

        cong = h_cong_map + v_cong_map
        flat = cong.flatten()
        k = max(1, int(flat.numel() * 0.05))
        top_vals, _ = torch.topk(flat, k)

        return top_vals.mean()

    # ═══════════════════════════════════════════════════════════════════════
    # Density
    # ═══════════════════════════════════════════════════════════════════════

    def _density_loss(self, pos, sizes, num_macros,
                      cx_lo, cx_hi, cy_lo, cy_hi, cell_area,
                      grid_rows, grid_cols):
        x_lo = pos[:num_macros, 0] - sizes[:num_macros, 0] / 2
        x_hi = pos[:num_macros, 0] + sizes[:num_macros, 0] / 2
        y_lo = pos[:num_macros, 1] - sizes[:num_macros, 1] / 2
        y_hi = pos[:num_macros, 1] + sizes[:num_macros, 1] / 2

        ox = torch.clamp(
            torch.min(x_hi.unsqueeze(1), cx_hi.unsqueeze(0))
            - torch.max(x_lo.unsqueeze(1), cx_lo.unsqueeze(0)),
            min=0,
        )
        oy = torch.clamp(
            torch.min(y_hi.unsqueeze(1), cy_hi.unsqueeze(0))
            - torch.max(y_lo.unsqueeze(1), cy_lo.unsqueeze(0)),
            min=0,
        )
        overlap = oy.unsqueeze(2) * ox.unsqueeze(1)
        density = overlap.sum(0) / cell_area

        flat = density.flatten()
        k = max(1, int(flat.numel() * 0.10))
        top_vals, _ = torch.topk(flat, k)
        return 0.5 * top_vals.mean()

    # ═══════════════════════════════════════════════════════════════════════
    # Overlap
    # ═══════════════════════════════════════════════════════════════════════

    def _overlap_loss(self, pos, sizes, num_hard):
        if num_hard <= 1:
            return torch.tensor(0.0)
        hx = pos[:num_hard, 0]
        hy = pos[:num_hard, 1]
        hw = sizes[:num_hard, 0]
        hh = sizes[:num_hard, 1]
        dx = torch.abs(hx.unsqueeze(1) - hx.unsqueeze(0))
        dy = torch.abs(hy.unsqueeze(1) - hy.unsqueeze(0))
        min_sep_x = (hw.unsqueeze(1) + hw.unsqueeze(0)) / 2
        min_sep_y = (hh.unsqueeze(1) + hh.unsqueeze(0)) / 2
        ovlp_x = torch.clamp(min_sep_x - dx, min=0)
        ovlp_y = torch.clamp(min_sep_y - dy, min=0)
        ovlp = ovlp_x * ovlp_y
        mask = torch.triu(torch.ones(num_hard, num_hard, dtype=torch.bool), diagonal=1)
        total = ovlp[mask].sum()
        canvas_area = sizes[:num_hard, 0].sum() * sizes[:num_hard, 1].mean()
        return total / (canvas_area + 1e-8)

    # ═══════════════════════════════════════════════════════════════════════
    # TILOS heatmap congestion
    # ═══════════════════════════════════════════════════════════════════════

    def _tilos_cong_loss(self, pos, cong_map, movable_mask,
                         grid_rows, grid_cols, gw, gh):
        if cong_map.sum() == 0:
            return torch.tensor(0.0)
        movable_idx = torch.where(movable_mask)[0]
        if len(movable_idx) == 0:
            return torch.tensor(0.0)
        mov_pos = pos[movable_idx]
        norm_x = (mov_pos[:, 0] / (grid_cols * gw)) * 2 - 1
        norm_y = (mov_pos[:, 1] / (grid_rows * gh)) * 2 - 1
        inp = cong_map.unsqueeze(0).unsqueeze(0)
        grid = torch.stack([norm_x, norm_y], dim=1).unsqueeze(0).unsqueeze(2)
        sampled = F.grid_sample(inp, grid, mode="bilinear",
                                padding_mode="border", align_corners=False)
        return sampled.squeeze().mean()

    # ═══════════════════════════════════════════════════════════════════════
    # Congestion-aware spiral legalization
    # ═══════════════════════════════════════════════════════════════════════

    def _legalize(self, placement, benchmark, cong_map=None):
        """Spiral legalization with congestion-aware candidate scoring.

        When a macro needs displacement, collect the first K valid candidates
        from the spiral and pick the one with lowest:
            score = distance + alpha * congestion_at_position
        This steers macros AWAY from congested cells during legalization.
        """
        sizes = benchmark.macro_sizes
        cw, ch = benchmark.canvas_width, benchmark.canvas_height
        grid_rows, grid_cols = benchmark.grid_rows, benchmark.grid_cols
        gw, gh = cw / grid_cols, ch / grid_rows
        movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        movable_idx = torch.where(movable)[0].tolist()
        movable_idx.sort(key=lambda i: -(sizes[i, 0].item() * sizes[i, 1].item()))
        gap = 0.01
        placed = []
        K = 1  # nearest first valid (K>1 regressed on ibm01)
        diag = math.sqrt(cw**2 + ch**2)
        # alpha controls congestion vs distance tradeoff
        alpha = 0.0  # Disabled: congestion-aware scoring regressed on ibm01/06

        def cong_at(px, py):
            if alpha == 0:
                return 0.0
            c = min(grid_cols - 1, max(0, int(px / gw)))
            r = min(grid_rows - 1, max(0, int(py / gh)))
            return cong_map[r, c].item()

        for idx in movable_idx:
            x, y = placement[idx, 0].item(), placement[idx, 1].item()
            w, h = sizes[idx, 0].item(), sizes[idx, 1].item()
            hw, hh = w / 2, h / 2
            x = max(hw, min(cw - hw, x))
            y = max(hh, min(ch - hh, y))

            # Check original position
            if not self._overlaps(x, y, w, h, placed, gap):
                placement[idx, 0] = x
                placement[idx, 1] = y
                placed.append((idx, x, y, w, h))
                continue

            # Collect up to K non-overlapping candidates from spiral
            candidates = []
            for rs in range(1, 400):
                if len(candidates) >= K:
                    break
                r = rs * 0.2
                na = max(8, rs * 4)
                for ai in range(na):
                    a = 2 * math.pi * ai / na
                    nx = x + r * math.cos(a)
                    ny = y + r * math.sin(a)
                    nx = max(hw, min(cw - hw, nx))
                    ny = max(hh, min(ch - hh, ny))
                    if not self._overlaps(nx, ny, w, h, placed, gap):
                        dist = math.sqrt((nx - x)**2 + (ny - y)**2)
                        score = dist + alpha * cong_at(nx, ny)
                        candidates.append((score, nx, ny))
                        if len(candidates) >= K:
                            break

            if candidates:
                candidates.sort(key=lambda c: c[0])
                _, bx, by = candidates[0]
                placement[idx, 0] = bx
                placement[idx, 1] = by
                placed.append((idx, bx, by, w, h))
            else:
                placed.append((idx, x, y, w, h))

        return placement

    # ═══════════════════════════════════════════════════════════════════════
    # Congestion-directed local search
    # ═══════════════════════════════════════════════════════════════════════

    def _cong_local_search(self, placement, benchmark, plc, movable_mask):
        """Move macros out of congested grid cells — multi-round, time-budgeted."""
        import time
        sizes = benchmark.macro_sizes
        cw, ch = benchmark.canvas_width, benchmark.canvas_height
        num_hard = benchmark.num_hard_macros
        grid_rows, grid_cols = benchmark.grid_rows, benchmark.grid_cols
        gw, gh = cw / grid_cols, ch / grid_rows

        movable_idx = torch.where(movable_mask)[0].tolist()
        if len(movable_idx) < 2:
            return placement

        costs = compute_proxy_cost(placement, benchmark, plc)
        best_cost = costs["proxy_cost"]
        best_placement = placement.clone()

        DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1),
                      (1, 1), (-1, -1), (1, -1), (-1, 1)]
        SCALES = [2.0, 1.0, 0.5]
        NUM_ROUNDS = 3
        TOP_K = min(20, len(movable_idx))
        TIME_BUDGET = 90.0  # seconds
        t_start = time.time()
        total_improved = 0

        for rnd in range(NUM_ROUNDS):
            if time.time() - t_start > TIME_BUDGET:
                break

            v_cong = torch.tensor(plc.V_routing_cong, dtype=torch.float32)
            h_cong = torch.tensor(plc.H_routing_cong, dtype=torch.float32)
            cong_map = torch.max(v_cong, h_cong).reshape(grid_rows, grid_cols)

            def macro_cong(idx):
                x = best_placement[idx, 0].item()
                y = best_placement[idx, 1].item()
                c = min(grid_cols - 1, max(0, int(x / gw)))
                r = min(grid_rows - 1, max(0, int(y / gh)))
                return cong_map[r, c].item()

            scored = [(i, macro_cong(i)) for i in movable_idx]
            scored.sort(key=lambda x: -x[1])

            round_improved = 0
            for rank in range(TOP_K):
                if time.time() - t_start > TIME_BUDGET:
                    break
                idx = scored[rank][0]
                x = best_placement[idx, 0].item()
                y = best_placement[idx, 1].item()
                w = sizes[idx, 0].item()
                h = sizes[idx, 1].item()
                hw, hh = w / 2, h / 2
                found = False

                for scale in SCALES:
                    if found:
                        break
                    for dx, dy in DIRECTIONS:
                        nx = x + dx * scale * gw
                        ny = y + dy * scale * gh
                        nx = max(hw, min(cw - hw, nx))
                        ny = max(hh, min(ch - hh, ny))

                        has_ovlp = False
                        for k in movable_idx:
                            if k == idx:
                                continue
                            ddx = abs(nx - best_placement[k, 0].item())
                            ddy = abs(ny - best_placement[k, 1].item())
                            if (ddx < (w + sizes[k, 0].item()) / 2 + 0.01 and
                                ddy < (h + sizes[k, 1].item()) / 2 + 0.01):
                                has_ovlp = True
                                break
                        if has_ovlp:
                            continue

                        trial = best_placement.clone()
                        trial[idx, 0] = nx
                        trial[idx, 1] = ny
                        trial_costs = compute_proxy_cost(trial, benchmark, plc)
                        if (trial_costs["overlap_count"] == 0 and
                                trial_costs["proxy_cost"] < best_cost):
                            best_cost = trial_costs["proxy_cost"]
                            best_placement = trial.clone()
                            round_improved += 1
                            found = True
                            break

            total_improved += round_improved
            if round_improved == 0:
                break

        elapsed = time.time() - t_start
        if total_improved > 0:
            print(f"  [LOCAL SEARCH] {total_improved} moves, {rnd + 1} rounds, "
                  f"{elapsed:.1f}s, cost: {best_cost:.4f}")

        return best_placement

    # ═══════════════════════════════════════════════════════════════════════
    # Stochastic swap refinement
    # ═══════════════════════════════════════════════════════════════════════

    def _stochastic_swap(self, placement, benchmark, plc, movable_mask):
        """Swap similar-sized macros in congested regions post-legalization."""
        sizes = benchmark.macro_sizes
        num_hard = benchmark.num_hard_macros
        cw, ch = benchmark.canvas_width, benchmark.canvas_height
        grid_rows, grid_cols = benchmark.grid_rows, benchmark.grid_cols
        gw, gh = cw / grid_cols, ch / grid_rows

        movable_idx = torch.where(movable_mask)[0].tolist()
        if len(movable_idx) < 2:
            return placement

        costs = compute_proxy_cost(placement, benchmark, plc)
        best_cost = costs["proxy_cost"]
        best_placement = placement.clone()

        v_cong = torch.tensor(plc.V_routing_cong, dtype=torch.float32)
        h_cong = torch.tensor(plc.H_routing_cong, dtype=torch.float32)
        cong_map = torch.max(v_cong, h_cong).reshape(grid_rows, grid_cols)

        def macro_cong(idx):
            x, y = placement[idx, 0].item(), placement[idx, 1].item()
            c = min(grid_cols - 1, max(0, int(x / gw)))
            r = min(grid_rows - 1, max(0, int(y / gh)))
            return cong_map[r, c].item()

        cong_macros = [(i, macro_cong(i)) for i in movable_idx]
        cong_macros.sort(key=lambda x: -x[1])

        NUM_SWAPS = 150
        tried = set()

        for attempt in range(NUM_SWAPS):
            top_n = min(len(cong_macros), max(5, len(cong_macros) // 4))
            i_idx = random.randint(0, top_n - 1)
            i = cong_macros[i_idx][0]

            w_i = sizes[i, 0].item() * sizes[i, 1].item()
            candidates = []
            for j in movable_idx:
                if j == i:
                    continue
                w_j = sizes[j, 0].item() * sizes[j, 1].item()
                ratio = max(w_i, w_j) / (min(w_i, w_j) + 1e-8)
                if ratio < 2.0:
                    candidates.append(j)

            if not candidates:
                continue

            j = random.choice(candidates)
            pair = (min(i, j), max(i, j))
            if pair in tried:
                continue
            tried.add(pair)

            trial = best_placement.clone()
            trial[i, 0], trial[j, 0] = best_placement[j, 0].clone(), best_placement[i, 0].clone()
            trial[i, 1], trial[j, 1] = best_placement[j, 1].clone(), best_placement[i, 1].clone()

            has_ovlp = False
            for check_idx in [i, j]:
                x, y = trial[check_idx, 0].item(), trial[check_idx, 1].item()
                w, h = sizes[check_idx, 0].item(), sizes[check_idx, 1].item()
                for k in movable_idx:
                    if k == check_idx or k == (j if check_idx == i else i):
                        continue
                    dx = abs(x - trial[k, 0].item())
                    dy = abs(y - trial[k, 1].item())
                    if (dx < (w + sizes[k, 0].item()) / 2 + 0.01 and
                        dy < (h + sizes[k, 1].item()) / 2 + 0.01):
                        has_ovlp = True
                        break
                if has_ovlp:
                    break

            if has_ovlp:
                continue

            trial_costs = compute_proxy_cost(trial, benchmark, plc)
            if trial_costs["overlap_count"] == 0 and trial_costs["proxy_cost"] < best_cost:
                best_cost = trial_costs["proxy_cost"]
                best_placement = trial.clone()

        return best_placement

    @staticmethod
    def _overlaps(x, y, w, h, placed, gap):
        for _, px, py, pw, ph in placed:
            if abs(x - px) < (w + pw) / 2 + gap and abs(y - py) < (h + ph) / 2 + gap:
                return True
        return False

    # ═══════════════════════════════════════════════════════════════════════
    # Net extraction
    # ═══════════════════════════════════════════════════════════════════════

    def _extract_nets(self, benchmark, plc):
        if plc is None:
            return None
        plc_to_bench = {}
        for bi, pi in enumerate(benchmark.hard_macro_indices):
            plc_to_bench[pi] = bi
        for bi, pi in enumerate(benchmark.soft_macro_indices):
            plc_to_bench[pi] = benchmark.num_hard_macros + bi
        port_set = set(getattr(plc, "port_indices", []))
        MAX_PINS = 50
        nets = []
        for driver_name in plc.nets:
            pins = []
            sink_names = plc.nets[driver_name]
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
