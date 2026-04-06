"""
DREAMPlace-based macro placer for IBM ICCAD04 benchmarks.

Converts the competition's pb.txt benchmark to Bookshelf format,
runs DREAMPlace global placement (CPU, no GPU), and maps the output
macro positions back to the competition's tensor format.

Usage:
    uv run evaluate submissions/src/dreamplace_placer.py -b ibm01
    uv run evaluate submissions/src/dreamplace_placer.py --all
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from collections import defaultdict

import torch

from macro_place._plc import PlacementCost
from macro_place.benchmark import Benchmark

# Paths
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DREAMPLACE_DIR = os.path.join(_THIS_DIR, "DREAMPlace")
TESTCASE_ROOT = "external/MacroPlacement/Testcases/ICCAD04"

# Scale factor: microns → integer database units (1 µm = 1000 units).
# DREAMPlace's Bookshelf SCL parser requires integer coordinates.
_SCALE = 1000

# Row height used when writing fake .scl rows (microns).
# Doesn't affect global placement quality since legalization is disabled.
_ROW_HEIGHT = 0.19
_SITE_WIDTH = 0.19


def _s(v: float) -> int:
    """Scale a micron value to integer database units."""
    return int(round(v * _SCALE))


# ---------------------------------------------------------------------------
# Bookshelf helpers
# ---------------------------------------------------------------------------

def _pin_to_macro_name(pin_name: str, plc: PlacementCost) -> str:
    """
    Map a pin/port name to its parent macro/port name.

    In pb.txt format, macro pins are named  <macro>/<pin_suffix>.
    Ports are standalone nodes.
    """
    idx = plc.mod_name_to_indices.get(pin_name)
    if idx is None:
        return pin_name  # unknown – use as-is
    node = plc.modules_w_pins[idx]
    t = node.get_type()
    if t == "PORT":
        return pin_name
    # MACRO_PIN and SOFT_MACRO_PIN share the parent/child naming convention
    return pin_name.rsplit("/", 1)[0]


def _build_macro_nets(plc: PlacementCost):
    """
    Return a list of nets, each net being a set of macro/port names.
    Deduplicates self-loops and single-node nets.
    """
    nets = []
    all_macro_names = set()
    for idx in plc.hard_macro_indices + plc.soft_macro_indices + plc.port_indices:
        all_macro_names.add(plc.modules_w_pins[idx].get_name())

    for driver_pin, sink_pins in plc.nets.items():
        driver_macro = _pin_to_macro_name(driver_pin, plc)
        nodes = {driver_macro}
        for sp in sink_pins:
            nodes.add(_pin_to_macro_name(sp, plc))
        # Keep only nodes that exist as physical nodes
        nodes = nodes & all_macro_names
        if len(nodes) >= 2:
            nets.append(list(nodes))
    return nets


def _write_bookshelf(tmpdir: str, benchmark: Benchmark, plc: PlacementCost):
    """Write .aux / .nodes / .pl / .nets / .scl Bookshelf files to tmpdir."""
    canvas_w, canvas_h = plc.get_canvas_width_height()

    # ── collect physical nodes ────────────────────────────────────────────
    # Order: hard macros, soft macros, ports  (all movable then fixed)
    # Bookshelf convention: movable nodes first, then terminals.
    movable_nodes = []   # (name, ll_x, ll_y, w, h)
    terminal_nodes = []  # (name, ll_x, ll_y, w, h)

    for idx in plc.hard_macro_indices:
        node = plc.modules_w_pins[idx]
        cx, cy = node.get_pos()
        w, h = node.get_width(), node.get_height()
        entry = (node.get_name(), cx - w / 2, cy - h / 2, w, h)
        if node.get_fix_flag():
            terminal_nodes.append(entry)
        else:
            movable_nodes.append(entry)

    for idx in plc.soft_macro_indices:
        node = plc.modules_w_pins[idx]
        cx, cy = node.get_pos()
        w, h = node.get_width(), node.get_height()
        entry = (node.get_name(), cx - w / 2, cy - h / 2, w, h)
        if node.get_fix_flag():
            terminal_nodes.append(entry)
        else:
            movable_nodes.append(entry)

    for idx in plc.port_indices:
        node = plc.modules_w_pins[idx]
        cx, cy = node.get_pos()
        # Ports are point-like; give them a tiny size
        w, h = _SITE_WIDTH, _ROW_HEIGHT
        terminal_nodes.append((node.get_name(), cx, cy, w, h))

    all_nodes = movable_nodes + terminal_nodes
    num_movable = len(movable_nodes)
    num_terminals = len(terminal_nodes)

    # ── .nodes ────────────────────────────────────────────────────────────
    nodes_file = os.path.join(tmpdir, "design.nodes")
    with open(nodes_file, "w") as f:
        f.write("UCLA nodes 1.0\n\n")
        f.write(f"NumNodes : {len(all_nodes)}\n")
        f.write(f"NumTerminals : {num_terminals}\n\n")
        for i, (name, x, y, w, h) in enumerate(all_nodes):
            if i >= num_movable:
                f.write(f"\t{name}\t{_s(w)}\t{_s(h)}\tterminal\n")
            else:
                f.write(f"\t{name}\t{_s(w)}\t{_s(h)}\n")

    # ── .pl ───────────────────────────────────────────────────────────────
    pl_file = os.path.join(tmpdir, "design.pl")
    with open(pl_file, "w") as f:
        f.write("UCLA pl 1.0\n\n")
        for i, (name, x, y, w, h) in enumerate(all_nodes):
            suffix = " : N /FIXED" if i >= num_movable else " : N"
            f.write(f"{name}\t{_s(x)}\t{_s(y)}{suffix}\n")

    # ── .nets ─────────────────────────────────────────────────────────────
    macro_nets = _build_macro_nets(plc)
    nets_file = os.path.join(tmpdir, "design.nets")
    with open(nets_file, "w") as f:
        f.write("UCLA nets 1.0\n\n")
        total_pins = sum(len(n) for n in macro_nets)
        f.write(f"NumNets : {len(macro_nets)}\n")
        f.write(f"NumPins : {total_pins}\n\n")
        for i, net_nodes in enumerate(macro_nets):
            f.write(f"NetDegree : {len(net_nodes)}  n{i}\n")
            for j, node_name in enumerate(net_nodes):
                direction = "O" if j == 0 else "I"
                f.write(f"\t{node_name}\t{direction}\t: 0 0\n")

    # ── .scl ─────────────────────────────────────────────────────────────
    scl_file = os.path.join(tmpdir, "design.scl")
    row_h = _s(_ROW_HEIGHT)
    site_w = _s(_SITE_WIDTH)
    num_rows = max(1, int(canvas_h / _ROW_HEIGHT))
    num_sites = max(1, int(canvas_w / _SITE_WIDTH))
    with open(scl_file, "w") as f:
        f.write("UCLA scl 1.0\n\n")
        f.write(f"NumRows : {num_rows}\n\n")
        for i in range(num_rows):
            y_coord = i * row_h
            f.write("CoreRow Horizontal\n")
            f.write(f"  Coordinate    : {y_coord}\n")
            f.write(f"  Height        : {row_h}\n")
            f.write(f"  Sitewidth     : {site_w}\n")
            f.write(f"  Sitespacing   : {site_w}\n")
            f.write(f"  Siteorient    : 1\n")
            f.write(f"  Sitesymmetry  : 1\n")
            f.write(f"  SubrowOrigin  : 0  NumSites : {num_sites}\n")
            f.write("End\n\n")

    # ── .aux ─────────────────────────────────────────────────────────────
    aux_file = os.path.join(tmpdir, "design.aux")
    with open(aux_file, "w") as f:
        f.write(
            "RowBasedPlacement : "
            "design.nodes design.nets design.pl design.scl\n"
        )

    return movable_nodes


def _check_gpu_available() -> bool:
    """Return True if DREAMPlace was compiled with CUDA and a GPU is present."""
    configure_py = os.path.join(DREAMPLACE_DIR, "build", "dreamplace", "configure.py")
    try:
        ns: dict = {}
        with open(configure_py) as f:
            exec(f.read(), ns)
        cuda_compiled = ns.get("compile_configurations", {}).get("CUDA_FOUND") == "TRUE"
    except Exception:
        cuda_compiled = False
    if not cuda_compiled:
        return False
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _build_dreamplace_config(tmpdir: str, benchmark: Benchmark, plc: PlacementCost) -> dict:
    """Build DREAMPlace JSON config for global placement."""
    canvas_w, canvas_h = plc.get_canvas_width_height()
    canvas_area = canvas_w * canvas_h

    # Estimate total movable area from benchmark tensors
    movable_mask = benchmark.get_movable_mask()
    movable_area = (
        benchmark.macro_sizes[movable_mask, 0] * benchmark.macro_sizes[movable_mask, 1]
    ).sum().item()
    # Conservative density target to give legalization room
    target_density = min(0.80, movable_area / canvas_area * 1.05)

    result_dir = os.path.join(tmpdir, "results")
    os.makedirs(result_dir, exist_ok=True)

    use_gpu = _check_gpu_available()
    num_threads = min(16, os.cpu_count() or 1)

    # Two-stage global placement + built-in legalization tuned for macro-only designs
    return {
        "aux_input": os.path.join(tmpdir, "design.aux"),
        "gpu": 1 if use_gpu else 0,
        "num_bins_x": 64,
        "num_bins_y": 64,
        "global_place_stages": [
            {
                # Stage 1: Coarse (8×8) to globally separate macros
                "num_bins_x": 8,
                "num_bins_y": 8,
                "iteration": 1000,
                "learning_rate": 0.01,
                "wirelength": "weighted_average",
                "optimizer": "nesterov",
                "Llambda_density_weight_iteration": 100,
                "Lsub_iteration": 2,
            },
            {
                # Stage 2: Fine (64×64) for detailed placement
                "num_bins_x": 64,
                "num_bins_y": 64,
                "iteration": 1500,
                "learning_rate": 0.005,
                "wirelength": "weighted_average",
                "optimizer": "nesterov",
                "Llambda_density_weight_iteration": 1,
                "Lsub_iteration": 1,
            }
        ],
        "target_density": target_density,
        "density_weight": 5e-4,
        "gamma": 100.0,
        "random_seed": 1000,
        "scale_factor": 1.0,
        "ignore_net_degree": 100,
        "enable_fillers": 0,
        "gp_noise_ratio": 0.0,
        "global_place_flag": 1,
        "legalize_flag": 1,  # Skip std-cell legalizer (greedy/abacus); we legalize macros ourselves
        "detailed_place_flag": 0,
        "stop_overflow": 0.03,
        "macro_place_flag": 0,
        "dtype": "float32",
        "plot_flag": 0,
        "random_center_init_flag": 0,  # Start from initial placement
        "sort_nets_by_degree": 0,
        "num_threads": num_threads,
        "result_dir": result_dir,
        # ── Legalization tuning for macro-only designs ──
        "max_displace": 1.0,  # Max displacement during legalization (microns)
        "batch_size": 128,    # Legalization batch size
        "num_threads_legalize": num_threads,
    }


def _read_output_placement(
    tmpdir: str,
    movable_nodes: list,
    benchmark: Benchmark,
    plc: PlacementCost,
) -> torch.Tensor:
    """
    Parse DREAMPlace's output .gp.pl file and return a [num_macros, 2] tensor
    of center positions matching the benchmark's macro ordering.
    """
    # Find the output .pl file
    result_dir = os.path.join(tmpdir, "results", "design")
    pl_files = [f for f in os.listdir(result_dir) if f.endswith(".gp.pl")]
    if not pl_files:
        raise RuntimeError(f"No .gp.pl output found in {result_dir}")
    pl_path = os.path.join(result_dir, pl_files[0])

    # Parse positions: name → (ll_x, ll_y)
    name_to_pos = {}
    with open(pl_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("UCLA") or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                name = parts[0]
                try:
                    ll_x, ll_y = float(parts[1]), float(parts[2])
                    name_to_pos[name] = (ll_x, ll_y)
                except ValueError:
                    pass

    # Build name → (width, height) from movable_nodes
    name_to_size = {name: (w, h) for name, _, _, w, h in movable_nodes}
    # Also add terminal nodes from plc for completeness
    for idx in plc.hard_macro_indices + plc.soft_macro_indices + plc.port_indices:
        node = plc.modules_w_pins[idx]
        n = node.get_name()
        if n not in name_to_size:
            name_to_size[n] = (node.get_width(), node.get_height())

    # Build updated placement tensor (start from initial positions)
    placement = benchmark.macro_positions.clone()

    # Map DREAMPlace output back to benchmark macro ordering
    hard_indices = benchmark.hard_macro_indices   # plc idx list, len = num_hard_macros
    soft_indices = benchmark.soft_macro_indices   # plc idx list, len = num_soft_macros

    for tensor_idx, plc_idx in enumerate(hard_indices + soft_indices):
        node = plc.modules_w_pins[plc_idx]
        name = node.get_name()
        if name in name_to_pos and not node.get_fix_flag():
            # DREAMPlace outputs in scaled integer units; convert back to microns
            ll_x = name_to_pos[name][0] / _SCALE
            ll_y = name_to_pos[name][1] / _SCALE
            w, h = name_to_size.get(name, (node.get_width(), node.get_height()))
            placement[tensor_idx, 0] = ll_x + w / 2
            placement[tensor_idx, 1] = ll_y + h / 2

    return placement

def _jacobi_legalize(
    hp: "np.ndarray",
    hs: "np.ndarray",
    hf: "np.ndarray",
    canvas_w: float,
    canvas_h: float,
    max_passes: int = 1000000,
) -> "np.ndarray":
    """
    Jacobi-style iterative push-apart fallback for hard macros.

    Accumulates all pairwise push forces simultaneously (avoiding
    Gauss-Seidel oscillation), then applies them.  When the algorithm
    stalls (no progress for 20 passes), adds a small random perturbation
    to break symmetry.
    """
    import numpy as np

    N = len(hp)
    EPS = 1e-3
    stall_count = 0
    prev_overlaps = -1
    rng = np.random.default_rng(42)

    for pass_idx in range(max_passes):
        cx = hp[:, 0]
        cy = hp[:, 1]
        wx = hs[:, 0]
        wy = hs[:, 1]

        dx = cx[:, None] - cx[None, :]
        dy = cy[:, None] - cy[None, :]
        gx = (wx[:, None] + wx[None, :]) / 2
        gy = (wy[:, None] + wy[None, :]) / 2

        ov_x = gx - np.abs(dx)
        ov_y = gy - np.abs(dy)
        overlap = (ov_x > 0) & (ov_y > 0)
        np.fill_diagonal(overlap, False)

        total = int(overlap.sum()) // 2
        if total == 0:
            print(f"[DEBUG] Legalization converged in {pass_idx + 1} pass(es)")
            return hp

        # Stall detection → random perturbation
        if total == prev_overlaps:
            stall_count += 1
        else:
            stall_count = 0
        prev_overlaps = total

        if stall_count >= 20:
            stall_count = 0
            ii_ov, _ = np.where(overlap)
            perturb_idx = np.unique(ii_ov)
            noise_scale = 0.5 * np.max(hs)  # up to half the largest macro dimension
            hp[perturb_idx, 0] += rng.uniform(-noise_scale, noise_scale, len(perturb_idx))
            hp[perturb_idx, 1] += rng.uniform(-noise_scale, noise_scale, len(perturb_idx))
            for i in perturb_idx:
                if not hf[i]:
                    hp[i, 0] = float(np.clip(hp[i, 0], wx[i] / 2, canvas_w - wx[i] / 2))
                    hp[i, 1] = float(np.clip(hp[i, 1], wy[i] / 2, canvas_h - wy[i] / 2))
            continue

        delta_x = np.zeros(N)
        delta_y = np.zeros(N)
        ii, jj = np.where(overlap & (np.arange(N)[:, None] < np.arange(N)[None, :]))
        for i, j in zip(ii, jj):
            ox = ov_x[i, j] + EPS
            oy = ov_y[i, j] + EPS
            if ox <= oy:
                sign = np.sign(dx[i, j]) if dx[i, j] != 0 else 1.0
                if not hf[i]: delta_x[i] += sign * ox / 2
                if not hf[j]: delta_x[j] -= sign * ox / 2
            else:
                sign = np.sign(dy[i, j]) if dy[i, j] != 0 else 1.0
                if not hf[i]: delta_y[i] += sign * oy / 2
                if not hf[j]: delta_y[j] -= sign * oy / 2

        for i in range(N):
            if hf[i]:
                continue
            hp[i, 0] = float(np.clip(hp[i, 0] + delta_x[i], wx[i] / 2, canvas_w - wx[i] / 2))
            hp[i, 1] = float(np.clip(hp[i, 1] + delta_y[i], wy[i] / 2, canvas_h - wy[i] / 2))

    cx = hp[:, 0]; cy = hp[:, 1]
    dx2 = np.abs(cx[:, None] - cx[None, :])
    dy2 = np.abs(cy[:, None] - cy[None, :])
    remaining = int(((dx2 < (hs[:, 0:1] + hs[:, 0]) / 2) & (dy2 < (hs[:, 1:2] + hs[:, 1]) / 2)).sum() - N) // 2
    print(f"[DEBUG] Jacobi fallback did NOT converge; {remaining} overlaps remain after {max_passes} passes")
    return hp


def _count_hard_overlaps(hp, hs) -> int:
    """Count number of overlapping hard macro pairs (numpy, O(N²))."""
    import numpy as np
    cx = hp[:, 0]; cy = hp[:, 1]
    wx = hs[:, 0]; wy = hs[:, 1]
    dx = np.abs(cx[:, None] - cx[None, :])
    dy = np.abs(cy[:, None] - cy[None, :])
    half_sx = (wx[:, None] + wx[None, :]) / 2
    half_sy = (wy[:, None] + wy[None, :]) / 2
    overlap = (dx < half_sx) & (dy < half_sy)
    np.fill_diagonal(overlap, False)
    return int(overlap.sum()) // 2


def _sanitize_positions(
    hp: "np.ndarray",
    hs: "np.ndarray",
    hf: "np.ndarray",
    canvas_w: float,
    canvas_h: float,
) -> None:
    """In-place: replace NaN/inf positions and clip to valid canvas range."""
    import numpy as np
    for i in range(len(hp)):
        if hf[i]:
            continue
        if not np.isfinite(hp[i, 0]):
            hp[i, 0] = canvas_w / 2
        if not np.isfinite(hp[i, 1]):
            hp[i, 1] = canvas_h / 2
        hp[i, 0] = float(np.clip(hp[i, 0], hs[i, 0] / 2, canvas_w - hs[i, 0] / 2))
        hp[i, 1] = float(np.clip(hp[i, 1], hs[i, 1] / 2, canvas_h - hs[i, 1] / 2))


def _legalize_macros(
    placement: torch.Tensor,
    sizes: torch.Tensor,
    movable_mask: torch.Tensor,
    canvas_w: float,
    canvas_h: float,
    num_hard: int,
) -> torch.Tensor:
    """
    Legalize hard macros only (soft macros are allowed to overlap).

    Strategy:
    1. Sanitize positions (fix NaN/inf, clip to canvas).
    2. Check if DREAMPlace's built-in legalization (legalize_flag: 1) already
       converged to 0 overlaps.
    3. If overlaps remain, fall back to Jacobi push-apart.
    """
    import numpy as np

    pos   = placement.detach().float().numpy().copy()
    sz    = sizes.detach().float().numpy()
    fixed = (~movable_mask).numpy()

    hp = pos[:num_hard].copy()
    hs = sz[:num_hard]
    hf = fixed[:num_hard]

    # ── 0. Sanitize positions ────────────────────────────────────────────
    _sanitize_positions(hp, hs, hf, canvas_w, canvas_h)

    initial_overlaps = _count_hard_overlaps(hp, hs)
    print(f"[DEBUG] Hard overlaps after DREAMPlace global placement: {initial_overlaps}")

    # ── 1. Check if DREAMPlace's built-in legalization succeeded ─────────
    if initial_overlaps == 0:
        print(f"[DEBUG] DREAMPlace's built-in legalization converged to 0 overlaps")
        return torch.from_numpy(pos).to(placement.dtype)

    # ── 2. Fallback: Jacobi for remaining overlaps ─────────────────────
    print(f"[DEBUG] {initial_overlaps} overlaps remain; applying Jacobi fallback...")
    pos[:num_hard] = _jacobi_legalize(hp, hs, hf, canvas_w, canvas_h)
    return torch.from_numpy(pos).to(placement.dtype)


# ---------------------------------------------------------------------------
# Placer class
# ---------------------------------------------------------------------------

class DREAMPlacePlacer:
    """
    Wraps DREAMPlace as a competition-compatible placer.

    Converts each benchmark to Bookshelf format, runs DREAMPlace global
    placement (CPU, single-threaded, no legalization), and returns macro
    center positions as a [num_macros, 2] tensor.
    """

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        # ── load PlacementCost for net connectivity ───────────────────────
        testcase_dir = os.path.join(TESTCASE_ROOT, benchmark.name)
        netlist_file = os.path.join(testcase_dir, "netlist.pb.txt")
        plc_file = os.path.join(testcase_dir, "initial.plc")

        if not os.path.exists(netlist_file):
            raise FileNotFoundError(
                f"Netlist not found: {netlist_file}\n"
                "Run: git submodule update --init external/MacroPlacement"
            )

        plc = PlacementCost(netlist_file)
        if os.path.exists(plc_file):
            plc.restore_placement(plc_file, ifInital=True, ifReadComment=True)

        canvas_w_dbg, canvas_h_dbg = plc.get_canvas_width_height()
        print(f"[DEBUG] benchmark={benchmark.name}  canvas={canvas_w_dbg:.4f}x{canvas_h_dbg:.4f}")
        print(f"[DEBUG] hard_macros={benchmark.num_hard_macros}  soft_macros={benchmark.num_soft_macros}  fixed={benchmark.macro_fixed.sum().item()}")
        movable_area = (benchmark.macro_sizes[~benchmark.macro_fixed, 0] * benchmark.macro_sizes[~benchmark.macro_fixed, 1]).sum().item()
        print(f"[DEBUG] movable_area={movable_area:.2f}  canvas_area={canvas_w_dbg * canvas_h_dbg:.2f}  density={movable_area / (canvas_w_dbg * canvas_h_dbg):.4f}")

        with tempfile.TemporaryDirectory(prefix=f"dp_{benchmark.name}_") as tmpdir:
            # 1. Write Bookshelf files
            movable_nodes = _write_bookshelf(tmpdir, benchmark, plc)

            # 2. Write JSON config
            config = _build_dreamplace_config(tmpdir, benchmark, plc)
            config_file = os.path.join(tmpdir, "config.json")
            with open(config_file, "w") as f:
                json.dump(config, f, indent=2)

            # 3. Run DREAMPlace
            placer_script = os.path.join(DREAMPLACE_DIR, "dreamplace", "Placer.py")
            env = os.environ.copy()
            # build/ must come before DREAMPLACE_DIR so `import dreamplace`
            # resolves to build/dreamplace/ (which has configure.py + .so files).
            build_dir = os.path.join(DREAMPLACE_DIR, "build")
            env["PYTHONPATH"] = (
                build_dir
                + os.pathsep
                + DREAMPLACE_DIR
                + os.pathsep
                + env.get("PYTHONPATH", "")
            )

            # DREAMPlace was compiled against the system Python/PyTorch
            # (/usr/bin/python3.10 + ~/.local/lib/.../torch), not the venv.
            # Using sys.executable here would cause an ABI mismatch.
            system_python = "/usr/bin/python3.10"
            result = subprocess.run(
                [system_python, placer_script, config_file],
                capture_output=True,
                text=True,
                env=env,
            )
            if result.returncode != 0:
                print("[DREAMPlace STDERR]", result.stderr[-3000:])
                raise RuntimeError(
                    f"DREAMPlace failed with exit code {result.returncode}"
                )

            # 4. Parse output, legalize overlaps, return placement
            placement = _read_output_placement(tmpdir, movable_nodes, benchmark, plc)
            canvas_w, canvas_h = plc.get_canvas_width_height()

            # Debug: count hard-hard overlaps before legalization
            nh = benchmark.num_hard_macros
            pre_overlaps = 0
            for ii in range(nh):
                for jj in range(ii + 1, nh):
                    xi, yi = placement[ii, 0].item(), placement[ii, 1].item()
                    xj, yj = placement[jj, 0].item(), placement[jj, 1].item()
                    wi, hi = benchmark.macro_sizes[ii, 0].item(), benchmark.macro_sizes[ii, 1].item()
                    wj, hj = benchmark.macro_sizes[jj, 0].item(), benchmark.macro_sizes[jj, 1].item()
                    if (abs(xi - xj) < (wi + wj) / 2) and (abs(yi - yj) < (hi + hj) / 2):
                        pre_overlaps += 1
            print(f"[DEBUG] Hard-hard overlaps from DREAMPlace output (before legalization): {pre_overlaps}")
            print(f"[DEBUG] canvas={canvas_w:.3f}x{canvas_h:.3f}  num_hard={nh}  num_soft={benchmark.num_soft_macros}")
            print(f"[DEBUG] config target_density={config['target_density']:.4f}  stop_overflow={config['stop_overflow']}  gpu={config['gpu']}")

            return _legalize_macros(
                placement,
                benchmark.macro_sizes,
                benchmark.get_movable_mask(),
                canvas_w,
                canvas_h,
                benchmark.num_hard_macros,
            )
