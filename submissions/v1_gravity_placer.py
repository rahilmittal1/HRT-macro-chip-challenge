"""
V1 Gravity Placer — Center-of-Gravity Heuristic with Legalization

Algorithm:
1. Build macro adjacency matrix from PlacementCost
2. Start from initial positions
3. Iteratively move each movable hard macro toward the weighted center
   of its connected neighbors (force-directed / gravity pull)
4. Legalize by resolving overlaps greedily

Usage:
    uv run evaluate submissions/v1_gravity_placer.py -b ibm01
    uv run evaluate submissions/v1_gravity_placer.py --all
"""

import math
import torch
from macro_place.benchmark import Benchmark
from macro_place.loader import load_benchmark_from_dir
import os


class V1GravityPlacer:

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        placement = benchmark.macro_positions.clone()
        sizes = benchmark.macro_sizes
        cw, ch = benchmark.canvas_width, benchmark.canvas_height
        movable = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
        movable_idx = torch.where(movable)[0]
        num_hard = benchmark.num_hard_macros
        num_soft = benchmark.num_soft_macros
        num_total = num_hard + num_soft

        # --- Build adjacency by re-loading plc ---
        adj_matrix = self._get_adjacency(benchmark)

        # --- Force-directed iterations ---
        for iteration in range(80):
            step = 0.35 * (1.0 - iteration / 80)
            for i in movable_idx.tolist():
                # Weighted center of gravity of all connected macros
                weights = adj_matrix[i]
                w_sum = weights.sum().item()
                if w_sum == 0:
                    continue

                cx = (weights * placement[:num_total, 0]).sum().item() / w_sum
                cy = (weights * placement[:num_total, 1]).sum().item() / w_sum

                # Move toward center of gravity
                new_x = placement[i, 0].item() + step * (cx - placement[i, 0].item())
                new_y = placement[i, 1].item() + step * (cy - placement[i, 1].item())

                # Clamp to canvas
                hw, hh = sizes[i, 0].item() / 2, sizes[i, 1].item() / 2
                new_x = max(hw, min(cw - hw, new_x))
                new_y = max(hh, min(ch - hh, new_y))

                placement[i, 0] = new_x
                placement[i, 1] = new_y

        # --- Legalization: resolve overlaps ---
        placement = self._legalize(placement, sizes, movable_idx, cw, ch)

        return placement

    def _get_adjacency(self, benchmark):
        """Load plc and extract macro adjacency matrix."""
        # Find benchmark directory
        testcase_root = "external/MacroPlacement/Testcases/ICCAD04"
        bench_dir = os.path.join(testcase_root, benchmark.name)

        if not os.path.exists(bench_dir):
            # NG45 or other — return empty adjacency
            n = benchmark.num_macros
            return torch.zeros(n, n)

        _, plc = load_benchmark_from_dir(bench_dir)

        # get_macro_adjacency returns flat list of size (hard+soft)^2
        n = benchmark.num_hard_macros + benchmark.num_soft_macros
        flat_adj = plc.get_macro_adjacency()
        adj = torch.tensor(flat_adj, dtype=torch.float32).reshape(n, n)

        return adj

    def _legalize(self, placement, sizes, movable_idx, cw, ch):
        """Greedy overlap removal: sort by area descending, shift collisions."""
        gap = 0.01
        hard_idx = movable_idx.tolist()
        hard_idx.sort(key=lambda i: -(sizes[i, 0].item() * sizes[i, 1].item()))

        placed = []

        for idx in hard_idx:
            x = placement[idx, 0].item()
            y = placement[idx, 1].item()
            w = sizes[idx, 0].item()
            h = sizes[idx, 1].item()

            x, y = self._find_legal_pos(x, y, w, h, placed, cw, ch, gap)

            placement[idx, 0] = x
            placement[idx, 1] = y
            placed.append((idx, x, y, w, h))

        return placement

    def _find_legal_pos(self, x, y, w, h, placed, cw, ch, gap):
        hw, hh = w / 2, h / 2
        x = max(hw, min(cw - hw, x))
        y = max(hh, min(ch - hh, y))

        if not self._has_overlap(x, y, w, h, placed, gap):
            return x, y

        # Spiral search around target position
        for radius_step in range(1, 300):
            r = radius_step * 0.2
            num_angles = max(8, radius_step * 4)
            for angle_step in range(num_angles):
                angle = 2 * math.pi * angle_step / num_angles
                nx = x + r * math.cos(angle)
                ny = y + r * math.sin(angle)
                nx = max(hw, min(cw - hw, nx))
                ny = max(hh, min(ch - hh, ny))

                if not self._has_overlap(nx, ny, w, h, placed, gap):
                    return nx, ny

        return x, y  # fallback

    def _has_overlap(self, x, y, w, h, placed, gap):
        for _, px, py, pw, ph in placed:
            dx = abs(x - px)
            dy = abs(y - py)
            if dx < (w + pw) / 2 + gap and dy < (h + ph) / 2 + gap:
                return True
        return False
