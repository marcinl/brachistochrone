"""Regenerate the variable-gravity comparison figures used in the README.

    python gravity_examples.py

Writes two PNGs into docs/images/:

- gravity-fields.png : four fields (uniform, rising, falling, oscillating) on
  the default 100 m -> 0 problem, showing how the field's *shape* moves where
  the descent happens.
- gravity-dip.png    : three fields on a longer 200 m -> 100 m problem with
  headroom below the target, so the paths dip past the endpoint and climb back.

Each curve is a single ``solve`` call; the figures just overlay them.  The
per-field command-line equivalents are listed in the README.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")  # write files, no display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from brachistochrone import Config, solve  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parent / "docs" / "images"


def _descent(ax, sol, colour, label):
    """Overlay one solution's physical (x, altitude) path on ``ax``."""
    ixt = sol.shape[0] - 1
    t, _ = sol.best_at(ixt, sol.iy_target)
    seg = sol.path_segments(ixt, sol.iy_target)
    xs = np.append(seg["x1"], seg["x2"][-1])
    ys = np.append(seg["y1"], seg["y2"][-1])
    ax.plot(xs, ys, color=colour, lw=2.4, label=label.format(
        t=t, v=seg["v2"][-1], dip=sol.config.y_end - ys.min(),
        L=seg["length"].sum()))
    return t, seg


def fields_figure() -> pathlib.Path:
    """Uniform / rising / falling / oscillating on the default problem."""
    cases = [
        ("uniform  g = 9.81",        "const:9.81",           "#1f77b4"),
        ("g rises   (poly:9.81,0.05)",  "poly:9.81,0.05",    "#ff7f0e"),
        ("g falls   (poly:9.81,-0.04)", "poly:9.81,-0.04",   "#2ca02c"),
        ("g ripples (sin:3,0.05,0,9.81)", "sin:3,0.05,0,9.81", "#d62728"),
    ]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5),
                                   height_ratios=(2.1, 1.0), constrained_layout=True)
    for name, spec, c in cases:
        sol = solve(Config(gravity_spec=spec, x_step=5.0, y_step=5.0))
        _descent(ax1, sol, c, name + "    t = {t:.4f} s")
        ax2.plot(sol.x, sol.field.g, color=c, lw=2)
    ax1.set_xlabel("x (m)"); ax1.set_ylabel("altitude (m)"); ax1.set_aspect("equal")
    ax1.grid(True, color="0.92"); ax1.legend(fontsize=9)
    ax1.set_title("Fastest descent 100 m -> 0 under different gravity fields g(x)")
    ax2.set_xlabel("x (m)"); ax2.set_ylabel("g(x)  (m/s^2)")
    ax2.grid(True, color="0.92"); ax2.set_title("the fields themselves", fontsize=10)
    out = OUT / "gravity-fields.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def dip_figure() -> pathlib.Path:
    """Longer problem with headroom below the target, so paths dip and climb."""
    H, YE, XMAX, XS, YS, DEPTH, TOL = 200.0, 100.0, 250.0, 8.0, 8.0, 30.0, 3.0
    a = (9.81 - 1.0) / XMAX**2
    cases = [
        ("uniform  g = 9.81",        "const:9.81",            "#1f77b4"),
        (f"g = 9.81 - {a:.2e}*x^2",  f"poly:9.81,0,{-a:.6g}", "#d62728"),
        ("g = 9.81*e^(-0.01*x^1.1)", "exp:9.81,1.1,0.01,0",   "#2ca02c"),
    ]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 9.5),
                                   height_ratios=(2.5, 1.0), constrained_layout=True)
    floor = YE
    for name, spec, c in cases:
        sol = solve(Config(h=H, y_end=YE, x_max=XMAX, x_step=XS, y_step=YS,
                           depth_below=DEPTH, gravity_spec=spec, energy_tol=TOL))
        _descent(ax1, sol, c,
                 name + "    t = {t:.3f} s,  dip = {dip:.1f} m,  v_end = {v:.1f} m/s")
        ax2.plot(sol.x, sol.field.g, color=c, lw=2.2)
        floor = float(sol.y[-1])
    ax1.axhline(YE, color="0.4", ls="--", lw=1.3, zorder=1)
    ax1.text(XMAX, YE + 2, "target altitude 100 m", ha="right", va="bottom",
             fontsize=9, color="0.35")
    ax1.plot([0], [H], marker="o", ms=8, mfc="white", mec="black", zorder=5)
    ax1.plot([XMAX], [YE], marker="s", ms=8, mfc="white", mec="black", zorder=5)
    ax1.set_xlabel("x (m)"); ax1.set_ylabel("altitude (m)"); ax1.set_aspect("equal")
    ax1.grid(True, color="0.92"); ax1.legend(fontsize=9, loc="upper right")
    ax1.set_xlim(-6, XMAX + 6); ax1.set_ylim(floor - 3, H + 5)
    ax1.set_title("Fastest descent (0, 200) -> (250, 100)  —  "
                  "paths dip below the target and climb back")
    ax2.set_ylim(0, 11); ax2.set_xlabel("x (m)"); ax2.set_ylabel("g(x)  (m/s^2)")
    ax2.grid(True, color="0.92"); ax2.set_title("the gravity fields", fontsize=10)
    out = OUT / "gravity-dip.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (fields_figure(), dip_figure()):
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
