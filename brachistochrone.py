"""Discrete brachistochrone: minimum-time descent on a smooth, frictionless track.

Physics
-------
On a frictionless *constrained* path (a bead on a wire), energy conservation
fixes the speed from the altitude alone::

    v(y) = sqrt(v0**2 + 2*g*(h - y))

regardless of the route taken to get there.  That is what makes the search
tractable: speed never has to be carried as a search variable, only position
(and, optionally, heading).

Along a straight chord the tangential acceleration a = g*sin(phi) is constant,
so the traversal time is *exact*, not an approximation::

    v2**2 = v1**2 + 2*a*L   and   v2 = v1 + a*t   =>   t = 2*L / (v1 + v2)

A polyline built from grid chords is therefore timed exactly.  The only
approximation is that the optimal curve is restricted to grid nodes, and since
every grid polyline is itself an admissible path, the computed time is always an
*upper* bound on the true continuous brachistochrone time.

Search structure
----------------
The state is ``(ix, iy, ir)`` -- horizontal cell, altitude cell, and the 10-degree
bin of the velocity vector's angle below horizontal on arrival.  At the start
node ``ir`` is the launch angle; everywhere else it is the arrival heading.  The
heading axis is only load-bearing when ``max_turn_deg`` is set (it forbids
kinked paths); with no turn limit the answer is identical for every bin, which
is a useful self-check.

Edge weights are strictly positive, so Dijkstra is exact on this graph.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np

__all__ = [
    "Config",
    "Solution",
    "solve",
    "free_fall_time",
    "max_horizontal_extent",
    "analytic_brachistochrone",
]


# --------------------------------------------------------------------------
# Analytic reference results
# --------------------------------------------------------------------------


def free_fall_time(drop: float, g: float = 9.81, v0: float = 0.0) -> float:
    """Time for a vertical free fall through ``drop`` metres from speed ``v0``."""
    if drop < 0:
        raise ValueError("drop must be non-negative")
    return (math.sqrt(v0 * v0 + 2.0 * g * drop) - v0) / g


def _cycloid_descent_time(theta: float, drop: float, g: float) -> float:
    """Time along the cycloid that falls ``drop`` metres by parameter ``theta``."""
    a = drop / (1.0 - math.cos(theta))
    return math.sqrt(a / g) * theta


def max_horizontal_extent(
    drop: float, g: float = 9.81, time_budget: float | None = None
) -> tuple[float, str]:
    """Largest horizontal offset worth putting on the grid.

    Returns ``(x_max, binding_constraint)``.

    The bound is geometric.  The brachistochrone from ``(0, h)`` to ``(X, h-drop)``
    is the cycloid ``x = a(t - sin t)``, ``y_drop = a(1 - cos t)``.  Its lowest
    point is at ``t = pi``, where the drop is ``2a`` and the offset is ``a*pi``.
    Setting ``2a = drop`` gives::

        x_max = pi * drop / 2

    For any endpoint farther out than that, the fastest path must dip *below* the
    target altitude and climb back up -- i.e. it leaves the altitude band the
    grid covers, so widening the grid past this point buys nothing but cells.

    If ``time_budget`` is given and is tighter than the ``t = pi`` traversal time
    (``pi*sqrt(drop/(2g))``), the budget binds instead and the bound is found by
    bisection on theta (descent time is monotonic in theta over ``(0, pi]``).
    """
    if drop <= 0:
        raise ValueError("drop must be positive")

    theta = math.pi
    binding = "geometry"

    if time_budget is not None:
        t_at_pi = _cycloid_descent_time(math.pi, drop, g)
        if time_budget < t_at_pi:
            t_floor = free_fall_time(drop, g)  # theta -> 0 limit: the vertical drop
            if time_budget <= t_floor:
                return 0.0, "time (nothing beyond a vertical drop is reachable)"
            lo, hi = 1e-9, math.pi
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if _cycloid_descent_time(mid, drop, g) < time_budget:
                    lo = mid
                else:
                    hi = mid
            theta = 0.5 * (lo + hi)
            binding = "time budget"

    a = drop / (1.0 - math.cos(theta))
    return a * (theta - math.sin(theta)), binding


def analytic_brachistochrone(
    x_end: float, drop: float, g: float = 9.81
) -> tuple[float, float]:
    """Exact continuous brachistochrone from ``(0, 0)`` to ``(x_end, -drop)``.

    Returns ``(time, arc_length)``.  Valid for a start at rest.  ``x_end`` may
    exceed ``pi*drop/2``, in which case the cycloid dips below the endpoint.
    """
    if drop <= 0:
        raise ValueError("drop must be positive")
    if x_end < 0:
        raise ValueError("x_end must be non-negative")
    if x_end == 0:
        return free_fall_time(drop, g), drop

    # Solve (theta - sin theta) / (1 - cos theta) = x_end / drop for theta in (0, 2pi).
    target = x_end / drop
    lo, hi = 1e-9, 2.0 * math.pi - 1e-9
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        ratio = (mid - math.sin(mid)) / (1.0 - math.cos(mid))
        if ratio < target:
            lo = mid
        else:
            hi = mid
    theta = 0.5 * (lo + hi)
    a = drop / (1.0 - math.cos(theta))
    time = math.sqrt(a / g) * theta
    # Cycloid arc length: integral of a*sqrt(2(1-cos t)) dt = 4a(1 - cos(theta/2)).
    arc = 4.0 * a * (1.0 - math.cos(0.5 * theta))
    return time, arc


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Config:
    """Simulation parameters.

    Attributes
    ----------
    h, y_end:
        Start and end altitude in metres.
    g:
        Gravitational acceleration.
    dt:
        Output sampling interval.  Used to resample the winning path onto a
        uniform time grid -- *not* to build the search graph (see module notes
        and ``neighbourhood_cells``).
    v0:
        Initial speed magnitude.  With the default 0, a purely horizontal first
        move is impossible and is correctly reported as unreachable.
    x_step, y_step:
        Requested grid spacing.  Both are snapped so the axes span their ranges
        exactly; the realised values are on the ``Solution``.
    x_max:
        Horizontal extent.  ``None`` derives it from :func:`max_horizontal_extent`.
    time_budget:
        Total time allowed, used only when deriving ``x_max``.  ``None`` derives
        it as ``2 * ceil(sqrt(2*drop/g))``.
    angle_step_deg, angle_max_deg:
        Heading bins, measured below horizontal.  0 deg is horizontal, 90 deg is
        straight down.
    neighbourhood_cells:
        Chords span up to this many cells in each axis, which sets the angular
        resolution of the path.  Only direction-primitive offsets are kept
        (gcd == 1); collinear multiples are exactly equivalent under the chord
        time formula, so dropping them is lossless.
    max_turn_deg:
        Optional limit on the heading change at a node.  ``None`` leaves the
        path free to kink, which is what reproduces the true brachistochrone.
    launch_angles_deg:
        Headings seeded at the start node.  ``None`` seeds all of them, giving
        the globally fastest path.
    """

    h: float = 100.0
    y_end: float = 0.0
    g: float = 9.81
    dt: float = 0.1
    v0: float = 0.0
    x_step: float = 5.0
    y_step: float = 5.0
    x_max: float | None = None
    time_budget: float | None = None
    angle_step_deg: float = 10.0
    angle_max_deg: float = 90.0
    neighbourhood_cells: int = 5
    max_turn_deg: float | None = None
    launch_angles_deg: Sequence[float] | None = None

    def __post_init__(self) -> None:
        if self.h <= self.y_end:
            raise ValueError("h must be above y_end")
        if self.g <= 0:
            raise ValueError("g must be positive")
        if self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.v0 < 0:
            raise ValueError("v0 must be non-negative")
        if self.x_step <= 0 or self.y_step <= 0:
            raise ValueError("grid steps must be positive")
        if self.angle_step_deg <= 0 or self.angle_max_deg <= 0:
            raise ValueError("angle step and maximum must be positive")
        if self.neighbourhood_cells < 1:
            raise ValueError("neighbourhood_cells must be at least 1")

    @property
    def drop(self) -> float:
        return self.h - self.y_end

    @property
    def default_time_budget(self) -> float:
        return 2.0 * math.ceil(free_fall_time(self.drop, self.g))


# --------------------------------------------------------------------------
# Solution
# --------------------------------------------------------------------------


@dataclass
class Solution:
    """Result of a solve: the state-space arrays plus path helpers."""

    config: Config
    x: np.ndarray  # (nx,) horizontal coordinates, ascending
    y: np.ndarray  # (ny,) altitudes, descending from h
    r_deg: np.ndarray  # (nr,) heading bin centres, below horizontal
    speed: np.ndarray  # (ny,) speed at each altitude, from energy conservation
    time: np.ndarray  # (nx, ny, nr) minimum arrival time, inf where unreachable
    length: np.ndarray  # (nx, ny, nr) arc length of that minimum-time path
    prev: np.ndarray  # (nx, ny, nr, 3) predecessor state index, -1 at start/unreached
    x_max: float
    x_max_binding: str

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.time.shape

    def best_at(self, ix: int, iy: int) -> tuple[float, int]:
        """Fastest arrival at a cell over all headings: ``(time, ir)``."""
        column = self.time[ix, iy]
        ir = int(np.argmin(column))
        return float(column[ir]), ir

    def target_index(self, x_target: float) -> int:
        """Grid column nearest to ``x_target``."""
        return int(np.argmin(np.abs(self.x - x_target)))

    def path_nodes(self, ix: int, iy: int, ir: int | None = None) -> list[tuple[float, float]]:
        """Reconstruct the minimum-time polyline to a cell, start first."""
        if ir is None:
            _, ir = self.best_at(ix, iy)
        if not math.isfinite(self.time[ix, iy, ir]):
            raise ValueError(f"state ({ix}, {iy}, {ir}) is unreachable")
        nodes: list[tuple[float, float]] = []
        j, i, r = ix, iy, ir
        while j >= 0:
            nodes.append((float(self.x[j]), float(self.y[i])))
            j, i, r = (int(v) for v in self.prev[j, i, r])
        nodes.reverse()
        return nodes

    def sample_path(
        self, nodes: Sequence[tuple[float, float]], dt: float | None = None
    ) -> np.ndarray:
        """Resample a polyline onto a uniform time grid.

        Returns a structured array with fields ``t, x, y, v, angle_deg`` holding
        the ball's position and velocity every ``dt`` seconds -- the original
        "all positions at time intervals dt" output.  Within each chord the
        motion is uniformly accelerated, so ``s(tau) = v1*tau + a*tau**2/2`` is
        exact.
        """
        cfg = self.config
        step = cfg.dt if dt is None else dt
        if step <= 0:
            raise ValueError("dt must be positive")
        if len(nodes) < 2:
            raise ValueError("need at least two nodes to sample")

        segments = []  # (t_start, v1, accel, length, p1, p2)
        t_cursor = 0.0
        for (x1, y1), (x2, y2) in zip(nodes, nodes[1:]):
            length = math.hypot(x2 - x1, y1 - y2)
            v1 = self._speed_at_altitude(y1)
            v2 = self._speed_at_altitude(y2)
            if v1 + v2 <= 0.0:
                raise ValueError("segment is untraversable from rest")
            duration = 2.0 * length / (v1 + v2)
            accel = (v2 * v2 - v1 * v1) / (2.0 * length) if length > 0 else 0.0
            segments.append((t_cursor, v1, accel, length, (x1, y1), (x2, y2)))
            t_cursor += duration
        total = t_cursor

        dtype = np.dtype(
            [("t", "f8"), ("x", "f8"), ("y", "f8"), ("v", "f8"), ("angle_deg", "f8")]
        )
        times = [k * step for k in range(int(math.floor(total / step)) + 1)]
        if not times or times[-1] < total - 1e-12:
            times.append(total)  # always land the exact endpoint
        out = np.empty(len(times), dtype=dtype)

        seg_idx = 0
        for k, t in enumerate(times):
            while seg_idx + 1 < len(segments) and t >= segments[seg_idx + 1][0]:
                seg_idx += 1
            t0, v1, accel, length, (x1, y1), (x2, y2) = segments[seg_idx]
            tau = t - t0
            s = v1 * tau + 0.5 * accel * tau * tau
            frac = 0.0 if length == 0 else min(max(s / length, 0.0), 1.0)
            out[k] = (
                t,
                x1 + frac * (x2 - x1),
                y1 + frac * (y2 - y1),
                v1 + accel * tau,
                math.degrees(math.atan2(y1 - y2, x2 - x1)),
            )
        return out

    def as_structured_array(self) -> np.ndarray:
        """The full ``(nx, ny, nr)`` state space as one structured array.

        Fields: ``x, y, r_deg`` (the cell's coordinates), ``t_min`` (fastest
        arrival), ``path_len`` (arc length of that path), ``v`` (speed there),
        ``dx, dy`` (the step that led in), ``v_in`` (speed entering that step),
        ``angle_in_deg`` (heading of that step) and ``reachable``.
        """
        nx, ny, nr = self.shape
        dtype = np.dtype(
            [
                ("x", "f8"),
                ("y", "f8"),
                ("r_deg", "f8"),
                ("t_min", "f8"),
                ("path_len", "f8"),
                ("v", "f8"),
                ("dx", "f8"),
                ("dy", "f8"),
                ("v_in", "f8"),
                ("angle_in_deg", "f8"),
                ("reachable", "?"),
            ]
        )
        arr = np.zeros((nx, ny, nr), dtype=dtype)
        arr["x"] = self.x[:, None, None]
        arr["y"] = self.y[None, :, None]
        arr["r_deg"] = self.r_deg[None, None, :]
        arr["t_min"] = self.time
        arr["path_len"] = self.length
        arr["v"] = self.speed[None, :, None]
        arr["reachable"] = np.isfinite(self.time)

        pj, pi = self.prev[..., 0], self.prev[..., 1]
        has_prev = pj >= 0
        px = np.where(has_prev, self.x[np.clip(pj, 0, None)], np.nan)
        py = np.where(has_prev, self.y[np.clip(pi, 0, None)], np.nan)
        arr["dx"] = np.where(has_prev, arr["x"] - px, np.nan)
        arr["dy"] = np.where(has_prev, arr["y"] - py, np.nan)
        arr["v_in"] = np.where(has_prev, self.speed[np.clip(pi, 0, None)], np.nan)
        arr["angle_in_deg"] = np.degrees(np.arctan2(-arr["dy"], arr["dx"]))
        return arr

    def _speed_at_altitude(self, y: float) -> float:
        cfg = self.config
        return math.sqrt(cfg.v0 * cfg.v0 + 2.0 * cfg.g * (cfg.h - y))


# --------------------------------------------------------------------------
# Solver
# --------------------------------------------------------------------------


def _primitive_moves(max_cells: int) -> Iterator[tuple[int, int]]:
    """Forward-and-down cell offsets with no collinear duplicates."""
    for dj in range(max_cells + 1):
        for di in range(max_cells + 1):
            if dj == 0 and di == 0:
                continue
            if math.gcd(dj, di) != 1:
                continue
            yield dj, di


def _bin_index(angle_deg: float, step: float, n_bins: int) -> int:
    # floor(x + 0.5) rather than round(): round() is banker's, which would send
    # a 45 deg chord to the 40 deg bin.
    return min(max(int(math.floor(angle_deg / step + 0.5)), 0), n_bins - 1)


def solve(config: Config | None = None, **overrides) -> Solution:
    """Build the state space and run Dijkstra over it."""
    cfg = config or Config(**overrides)
    if config is not None and overrides:
        raise TypeError("pass either a Config or keyword overrides, not both")

    drop = cfg.drop
    budget = cfg.time_budget if cfg.time_budget is not None else cfg.default_time_budget
    if cfg.x_max is not None:
        x_max, binding = cfg.x_max, "caller-supplied"
    else:
        x_max, binding = max_horizontal_extent(drop, cfg.g, budget)

    # Snap both axes so they span their range exactly.
    ny = max(2, int(round(drop / cfg.y_step)) + 1)
    nx = max(2, int(round(x_max / cfg.x_step)) + 1)
    y = cfg.h - np.linspace(0.0, drop, ny)
    x = np.linspace(0.0, x_max, nx)
    x_step = x_max / (nx - 1)
    y_step = drop / (ny - 1)

    nr = int(round(cfg.angle_max_deg / cfg.angle_step_deg)) + 1
    r_deg = np.arange(nr, dtype=float) * cfg.angle_step_deg

    speed = np.sqrt(cfg.v0 * cfg.v0 + 2.0 * cfg.g * (cfg.h - y))

    # Precompute the move set: geometry is identical at every node.
    moves = []
    for dj, di in _primitive_moves(cfg.neighbourhood_cells):
        seg_x, seg_y = dj * x_step, di * y_step
        length = math.hypot(seg_x, seg_y)
        angle = math.degrees(math.atan2(seg_y, seg_x))
        if angle > cfg.angle_max_deg + 1e-9:
            continue
        moves.append((dj, di, length, _bin_index(angle, cfg.angle_step_deg, nr)))

    shape = (nx, ny, nr)
    time = np.full(shape, np.inf)
    length = np.full(shape, np.inf)
    prev = np.full(shape + (3,), -1, dtype=np.int32)
    settled = np.zeros(shape, dtype=bool)

    if cfg.launch_angles_deg is None:
        seeds = range(nr)
    else:
        seeds = {_bin_index(a, cfg.angle_step_deg, nr) for a in cfg.launch_angles_deg}

    heap: list[tuple[float, int, int, int]] = []
    for ir in seeds:
        time[0, 0, ir] = 0.0
        length[0, 0, ir] = 0.0
        heapq.heappush(heap, (0.0, 0, 0, ir))

    while heap:
        t, j, i, ir = heapq.heappop(heap)
        if settled[j, i, ir]:
            continue
        settled[j, i, ir] = True
        v_from = speed[i]
        len_here = length[j, i, ir]
        heading = r_deg[ir]

        for dj, di, seg_len, ir_next in moves:
            nj, ni = j + dj, i + di
            if nj >= nx or ni >= ny:
                continue
            if settled[nj, ni, ir_next]:
                continue
            if cfg.max_turn_deg is not None:
                if abs(r_deg[ir_next] - heading) > cfg.max_turn_deg + 1e-9:
                    continue
            v_sum = v_from + speed[ni]
            if v_sum <= 0.0:
                # Both ends at rest: a horizontal chord off the start point.
                continue
            t_next = t + 2.0 * seg_len / v_sum
            if t_next < time[nj, ni, ir_next]:
                time[nj, ni, ir_next] = t_next
                length[nj, ni, ir_next] = len_here + seg_len
                prev[nj, ni, ir_next] = (j, i, ir)
                heapq.heappush(heap, (t_next, nj, ni, ir_next))

    return Solution(
        config=cfg,
        x=x,
        y=y,
        r_deg=r_deg,
        speed=speed,
        time=time,
        length=length,
        prev=prev,
        x_max=x_max,
        x_max_binding=binding,
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--height", type=float, default=100.0, help="start altitude (m)")
    p.add_argument("--end-altitude", type=float, default=0.0, help="end altitude (m)")
    p.add_argument("--gravity", type=float, default=9.81, help="g (m/s^2)")
    p.add_argument("--dt", type=float, default=0.1, help="output sampling interval (s)")
    p.add_argument("--v0", type=float, default=0.0, help="initial speed (m/s)")
    p.add_argument("--x-step", type=float, default=5.0, help="horizontal grid step (m)")
    p.add_argument("--y-step", type=float, default=5.0, help="vertical grid step (m)")
    p.add_argument("--x-max", type=float, default=None, help="override horizontal extent (m)")
    p.add_argument("--angle-step", type=float, default=10.0, help="heading bin width (deg)")
    p.add_argument("--neighbourhood", type=int, default=5, help="chord span in cells")
    p.add_argument("--max-turn", type=float, default=None, help="heading change limit (deg)")
    p.add_argument("--target", type=float, default=None, help="endpoint offset to report (m)")
    p.add_argument("--csv", type=str, default=None, help="write the sampled path here")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = Config(
        h=args.height,
        y_end=args.end_altitude,
        g=args.gravity,
        dt=args.dt,
        v0=args.v0,
        x_step=args.x_step,
        y_step=args.y_step,
        x_max=args.x_max,
        angle_step_deg=args.angle_step,
        neighbourhood_cells=args.neighbourhood,
        max_turn_deg=args.max_turn,
    )
    sol = solve(cfg)
    nx, ny, nr = sol.shape

    print(f"drop {cfg.drop:g} m, g {cfg.g:g} m/s^2, v0 {cfg.v0:g} m/s")
    print(f"free-fall time      {free_fall_time(cfg.drop, cfg.g, cfg.v0):.4f} s")
    print(f"time budget         {cfg.default_time_budget:g} s")
    print(f"horizontal extent   {sol.x_max:.3f} m  (limited by {sol.x_max_binding})")
    print(f"grid                {nx} x {ny} x {nr} = {nx*ny*nr} states")
    print(f"reachable           {int(np.isfinite(sol.time).sum())} states")
    print()

    targets = [args.target] if args.target is not None else [0.0, 25.0, 50.0, 100.0, sol.x_max]
    print(f"{'x_end (m)':>10} {'discrete (s)':>13} {'analytic (s)':>13} {'error':>8} {'len (m)':>9}")
    for xt in targets:
        ix = sol.target_index(xt)
        t_num, ir = sol.best_at(ix, ny - 1)
        if not math.isfinite(t_num):
            print(f"{sol.x[ix]:10.2f} {'unreachable':>13}")
            continue
        t_ref, _ = analytic_brachistochrone(float(sol.x[ix]), cfg.drop, cfg.g)
        err = (t_num - t_ref) / t_ref
        print(
            f"{sol.x[ix]:10.2f} {t_num:13.4f} {t_ref:13.4f} "
            f"{err:7.2%} {sol.length[ix, ny-1, ir]:9.2f}"
        )

    if args.csv is not None:
        ix = sol.target_index(args.target if args.target is not None else sol.x_max)
        samples = sol.sample_path(sol.path_nodes(ix, ny - 1))
        with open(args.csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(samples.dtype.names)
            for row in samples:
                writer.writerow([f"{v:.6f}" for v in row])
        print(f"\nwrote {len(samples)} samples to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
