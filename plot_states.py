"""Interactive 3D view of the brachistochrone state space.

Every reachable ``(ix, iy, ir)`` state is drawn as a point, coloured by the arc
length of the minimum-time path that reaches it -- black/navy at the start,
running through yellow, green and red to purple at the farthest states.  The
optimal path to a chosen endpoint is overlaid on top, plus a dashed shadow of
its physical ``(x, altitude)`` shape on the back wall.

Plot axes
---------
The array axes are ``(x, y, r)``, but the *plot* puts altitude on the vertical
so the picture reads physically::

    plot X  <- x        horizontal position (m)
    plot Y  <- r        heading below horizontal (deg)
    plot Z  <- y        altitude (m), increasing upwards

Because ``r`` is binned in 10 degree steps, the cloud forms ten discrete planes.
That is the state space itself, not a rendering artefact: two paths arriving at
the same point with different headings are different states.

Rotation
--------
Drag with the mouse for azimuth and elevation.  The three sliders give explicit
control of all three angles including roll, which the mouse cannot reach.
"""

from __future__ import annotations

import argparse
import math
from typing import Sequence

import numpy as np

from brachistochrone import Config, solve

# Anchor stops requested for the colour ramp.  The two unnamed intermediates
# (deep blue, orange) only keep saturation up: interpolating navy straight to
# yellow in RGB passes through olive grey, which reads as a dead zone.
COLOUR_STOPS = [
    (0.00, "#000008"),  # black
    (0.10, "#141d63"),  # navy
    (0.22, "#2f6fb0"),  # (blue, transitional)
    (0.38, "#f0e34a"),  # yellow
    (0.56, "#1e9c4a"),  # green
    (0.72, "#e07a1f"),  # (orange, transitional)
    (0.86, "#cc2b28"),  # red
    (1.00, "#6a1b9a"),  # purple
]


def make_colormap(name: str = "descent"):
    """Build the navy -> yellow -> green -> red -> purple ramp."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, COLOUR_STOPS)


def plot(
    sol,
    target_x: float | None = None,
    stride: int = 1,
    point_size: float = 9.0,
    alpha: float = 0.55,
    elev: float = 22.0,
    azim: float = -58.0,
    roll: float = 0.0,
    margin: float = 0.05,
    interactive: bool = True,
):
    """Render the state space.  Returns ``(fig, ax)``."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    nx, ny, nr = sol.shape
    iyt = sol.iy_target
    cfg_y_end = sol.config.y_end
    cmap = make_colormap()

    # --- point cloud -----------------------------------------------------
    reachable = np.isfinite(sol.length)
    if stride > 1:
        mask = np.zeros_like(reachable)
        mask[::stride, ::stride, :] = True
        reachable &= mask
    ix, iy, ir = np.nonzero(reachable)
    px, py, pz = sol.x[ix], sol.r_deg[ir], sol.y[iy]
    pc = sol.length[ix, iy, ir]

    fig = plt.figure(figsize=(12.0, 8.5))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.02, right=0.90, bottom=0.16, top=0.97)

    scat = ax.scatter(
        px,
        py,
        pz,
        c=pc,
        cmap=cmap,
        vmin=0.0,
        vmax=float(pc.max()),
        s=point_size,
        alpha=alpha,
        linewidths=0.0,
        depthshade=False,  # depth dimming would fight the colour encoding
    )

    # --- optimal path overlay -------------------------------------------
    if target_x is None:
        target_x = float(sol.x[-1])
    jt = sol.target_index(target_x)
    t_best, ir_best = sol.best_at(jt, iyt)
    if not math.isfinite(t_best):
        raise SystemExit(f"no path reaches x = {sol.x[jt]:.2f} m")

    states = sol.path_states(jt, iyt, ir_best)
    lx = np.array([sol.x[j] for j, _, _ in states])
    ly = np.array([sol.r_deg[r] for _, _, r in states])
    lz = np.array([sol.y[i] for _, i, _ in states])

    # A ball starting at rest has no heading, so the start state's r bin is an
    # arbitrary seed.  Drawing it literally throws a spurious segment across the
    # whole r axis; snap it to the first real heading instead.
    if sol.config.v0 == 0.0 and len(ly) > 1:
        ly[0] = ly[1]

    # Dark underlay first, bright core on top: readable against every colour
    # in the ramp, including the yellow band.
    ax.plot(lx, ly, lz, color="black", linewidth=6.5, solid_capstyle="round", zorder=5)
    ax.plot(
        lx,
        ly,
        lz,
        color="white",
        linewidth=3.0,
        solid_capstyle="round",
        zorder=6,
        label="optimal path (min time)",
    )
    ax.scatter(lx, ly, lz, s=34, facecolor="white", edgecolor="black", linewidths=1.0, zorder=7)

    # Shadow on the r = 0 wall: the physical trajectory shape.
    ax.plot(
        lx,
        np.zeros_like(ly),
        lz,
        color="0.25",
        linewidth=1.6,
        linestyle="--",
        zorder=4,
        label="physical (x, altitude) trace",
    )

    # --- framing ---------------------------------------------------------
    ax.set_xlabel("x  —  horizontal position (m)", labelpad=12)
    ax.set_ylabel("r  —  heading below horizontal (deg)", labelpad=12)
    ax.set_zlabel("y  —  altitude (m)", labelpad=10)
    # Pad every axis past the data range.  The optimal path runs exactly along
    # y = y_end, x = x_max and r = 0, so limits set to the bare data range clip
    # those runs to half a line width and they read as missing.
    def _padded(lo: float, hi: float) -> tuple[float, float]:
        pad = margin * (hi - lo)
        return lo - pad, hi + pad

    ax.set_xlim(*_padded(0.0, float(sol.x[-1])))
    ax.set_ylim(*_padded(float(sol.r_deg[0]), float(sol.r_deg[-1])))
    ax.set_zlim(*_padded(float(sol.y[-1]), float(sol.y[0])))

    # With headroom below the floor, mark where the target altitude actually is.
    x0, x1 = 0.0, float(sol.x[-1])
    r0, r1 = float(sol.r_deg[0]), float(sol.r_deg[-1])
    ax.plot(
        [x0, x1, x1, x0, x0],
        [r0, r0, r1, r1, r0],
        [float(sol.y[iyt])] * 5,
        color="0.45",
        linewidth=1.0,
        linestyle="--",
        alpha=0.8,
        zorder=3,
        label=f"target altitude ({cfg_y_end:g} m)",
    )
    ax.set_box_aspect((1.6, 1.0, 1.1))
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.pane.set_facecolor("#ececef")  # light: near-black points must read
        pane.pane.set_alpha(1.0)
    ax.grid(True, color="#c9c9cf", linewidth=0.5)
    ax.view_init(elev=elev, azim=azim, roll=roll)

    cbar = fig.colorbar(scat, ax=ax, pad=0.10, shrink=0.68)
    cbar.set_label("path length travelled from (0, h)  (m)")
    cbar.solids.set_alpha(1.0)

    ax.set_title(
        f"Brachistochrone state space  —  {int(np.isfinite(sol.length).sum())} reachable states\n"
        f"optimal path to x = {sol.x[jt]:.1f} m:  t = {t_best:.4f} s,  "
        f"length = {sol.length[jt, iyt, ir_best]:.1f} m",
        fontsize=11,
        pad=4,
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)

    # --- rotation sliders ------------------------------------------------
    if interactive:
        s_elev = Slider(fig.add_axes([0.13, 0.085, 0.62, 0.022]), "elev (X)", -90.0, 90.0, valinit=elev)
        s_azim = Slider(fig.add_axes([0.13, 0.050, 0.62, 0.022]), "azim (Z)", -180.0, 180.0, valinit=azim)
        s_roll = Slider(fig.add_axes([0.13, 0.015, 0.62, 0.022]), "roll (Y)", -180.0, 180.0, valinit=roll)

        def update(_):
            ax.view_init(elev=s_elev.val, azim=s_azim.val, roll=s_roll.val)
            fig.canvas.draw_idle()

        for s in (s_elev, s_azim, s_roll):
            s.on_changed(update)
        fig._rotation_sliders = (s_elev, s_azim, s_roll)  # keep them alive

    return fig, ax


def plot_curve(sol, target_x: float | None = None, show_samples: bool = True):
    """Plot the descent curve assembled from its individual chord segments.

    Upper panel: the physical ``(x, altitude)`` track.  Each chord is drawn as
    its own coloured segment so the piecewise construction is visible, with the
    continuous cycloid overlaid for reference.  Lower panel: speed against time,
    which must lie exactly on ``sqrt(2*g*(h-y))`` whatever the route.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    from brachistochrone import analytic_brachistochrone, cycloid_curve

    cfg = sol.config
    iyt = sol.iy_target
    if target_x is None:
        target_x = float(sol.x[-1])
    jt = sol.target_index(target_x)
    t_total, ir = sol.best_at(jt, iyt)
    if not math.isfinite(t_total):
        raise SystemExit(f"no path reaches x = {sol.x[jt]:.2f} m")

    seg = sol.path_segments(jt, iyt, ir)
    x_end = float(sol.x[jt])

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(11.0, 9.0), height_ratios=(2.4, 1.0), constrained_layout=True
    )

    # --- chord segments, coloured by the speed they are traversed at ---------
    pts = np.column_stack([seg["x1"], seg["y1"]])
    pts = np.vstack([pts, [[seg["x2"][-1], seg["y2"][-1]]]])
    lines = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(lines, cmap=make_colormap(), linewidths=4.0, zorder=3)
    lc.set_array(0.5 * (seg["v1"] + seg["v2"]))
    ax.add_collection(lc)
    cbar = fig.colorbar(lc, ax=ax, pad=0.02)
    cbar.set_label("segment speed (m/s)")

    # Chord joints: these are the grid nodes Dijkstra actually settled on.
    ax.plot(
        pts[:, 0], pts[:, 1], linestyle="none", marker="o", markersize=5,
        markerfacecolor="white", markeredgecolor="black", markeredgewidth=1.0,
        zorder=4, label=f"chord joints ({len(seg)} segments)",
    )

    # --- continuous reference ------------------------------------------------
    cx, cdrop = cycloid_curve(x_end, cfg.drop)
    t_ref, arc_ref = analytic_brachistochrone(x_end, cfg.drop, cfg.g)
    ax.plot(cx, cfg.h - cdrop, color="0.35", linewidth=1.6, linestyle="--",
            zorder=2, label=f"continuous cycloid  ({t_ref:.4f} s)")
    ax.plot([0.0, x_end], [cfg.h, cfg.y_end], color="0.6", linewidth=1.2,
            linestyle=":", zorder=1, label="straight chord (slower)")

    # --- ball positions every dt --------------------------------------------
    if show_samples:
        s = sol.sample_path(sol.path_nodes(jt, iyt, ir))
        ax.plot(s["x"], s["y"], linestyle="none", marker=".", markersize=7,
                color="black", alpha=0.55, zorder=5,
                label=f"ball every dt = {cfg.dt:g} s  ({len(s)} samples)")

    ax.set_xlabel("x  —  horizontal position (m)")
    ax.set_ylabel("altitude (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="0.9")
    ax.set_title(
        f"Minimum-time descent  (0, {cfg.h:g})  ->  ({x_end:.2f}, {cfg.y_end:g})\n"
        f"discrete {t_total:.4f} s over {seg['length'].sum():.2f} m   |   "
        f"cycloid {t_ref:.4f} s over {arc_ref:.2f} m   |   "
        f"gap {(t_total - t_ref) / t_ref:+.2%}",
        fontsize=11,
    )
    ax.legend(loc="lower left", fontsize=9, framealpha=0.9)

    # --- speed profile -------------------------------------------------------
    t_nodes = np.concatenate([seg["t_start"], [seg["t_end"][-1]]])
    v_nodes = np.concatenate([seg["v1"], [seg["v2"][-1]]])
    ax2.plot(t_nodes, v_nodes, color="#1f77b4", linewidth=2.0, label="speed at chord joints")
    ax2.axhline(math.sqrt(cfg.v0**2 + 2 * cfg.g * cfg.drop), color="0.5",
                linestyle="--", linewidth=1.2, label="sqrt(v0² + 2g·drop)")
    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("speed (m/s)")
    ax2.grid(True, color="0.9")
    ax2.legend(loc="lower right", fontsize=9)

    return fig, (ax, ax2)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--height", type=float, default=100.0, help="start altitude (m)")
    p.add_argument("--end-altitude", type=float, default=0.0, help="end altitude (m)")
    p.add_argument("--gravity", type=float, default=9.81, help="g (m/s^2)")
    p.add_argument("--v0", type=float, default=0.0, help="initial speed (m/s)")
    p.add_argument("--x-step", type=float, default=5.0, help="horizontal grid step (m)")
    p.add_argument("--y-step", type=float, default=5.0, help="vertical grid step (m)")
    p.add_argument("--x-max", type=float, default=None, help="override horizontal extent (m)")
    p.add_argument("--theta-ratio", type=float, default=None,
                   help="endpoint as theta/pi of the cycloid; >1 dips below the target")
    p.add_argument("--depth-below", type=float, default=None,
                   help="grid headroom below the target altitude (m); enables climbing")
    p.add_argument("--neighbourhood", type=int, default=5, help="chord span in cells")
    p.add_argument("--max-turn", type=float, default=None, help="heading change limit (deg)")
    p.add_argument("--target", type=float, default=None, help="endpoint for the overlaid path (m)")
    p.add_argument("--stride", type=int, default=1, help="thin the cloud by this factor")
    p.add_argument("--point-size", type=float, default=9.0, help="scatter marker size")
    p.add_argument("--alpha", type=float, default=0.55, help="scatter opacity")
    p.add_argument("--elev", type=float, default=22.0, help="initial elevation (deg)")
    p.add_argument("--azim", type=float, default=-58.0, help="initial azimuth (deg)")
    p.add_argument("--roll", type=float, default=0.0, help="initial roll (deg)")
    p.add_argument("--curve", action="store_true",
                   help="plot the descent curve from its chord segments instead of the state space")
    p.add_argument("--segments-csv", type=str, default=None,
                   help="write the per-segment breakdown here")
    p.add_argument("--save", type=str, default=None, help="write the figure here instead of showing")
    p.add_argument("--dpi", type=int, default=140, help="resolution for --save")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.save is not None:
        import matplotlib

        matplotlib.use("Agg")  # must precede the pyplot import
    import matplotlib.pyplot as plt

    sol = solve(
        Config(
            h=args.height,
            y_end=args.end_altitude,
            g=args.gravity,
            v0=args.v0,
            x_step=args.x_step,
            y_step=args.y_step,
            x_max=args.x_max,
            theta_ratio=args.theta_ratio,
            depth_below=args.depth_below,
            neighbourhood_cells=args.neighbourhood,
            max_turn_deg=args.max_turn,
        )
    )
    if args.segments_csv is not None:
        import csv

        jt = sol.target_index(args.target if args.target is not None else sol.x_max)
        seg = sol.path_segments(jt, sol.shape[1] - 1)
        with open(args.segments_csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(seg.dtype.names)
            for row in seg:
                writer.writerow([f"{v:.6f}" for v in row])
        print(f"wrote {len(seg)} segments to {args.segments_csv}")

    if args.curve:
        fig, _ = plot_curve(sol, target_x=args.target)
    else:
        fig, _ = plot(
            sol,
            target_x=args.target,
            stride=args.stride,
            point_size=args.point_size,
            alpha=args.alpha,
            elev=args.elev,
            azim=args.azim,
            roll=args.roll,
            interactive=args.save is None,
        )

    if args.save is not None:
        fig.savefig(args.save, dpi=args.dpi, bbox_inches="tight")
        print(f"wrote {args.save}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
