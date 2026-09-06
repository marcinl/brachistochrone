"""Gravity fields that vary with horizontal position, g = g(x).

Why this is not a small change
------------------------------
With uniform gravity the force field is conservative, so a potential exists and
the speed at a point follows from its altitude alone::

    v(y) = sqrt(v0**2 + 2*g*(h - y))          # independent of the route

Make ``g`` depend on ``x`` and that stops being true.  The field is
``F = (0, -g(x))`` per unit mass, whose 2D curl is::

    dF_y/dx - dF_x/dy = -dg/dx

which is non-zero wherever ``g`` varies.  A non-conservative field has no
potential, so the work done between two points depends on the path taken.
Concretely, under ``g(x) = 1 + x`` three routes from ``(0, 10)`` to ``(10, 0)``
arrive with speeds 4.47, 10.95 and 14.83 -- the same endpoints, a 3.3x spread.

The consequence for the solver is structural: arrival speed becomes part of the
*state*, not a function of it, so the search can no longer merge every path that
reaches a cell.  See ``brachistochrone.solve`` for how that is handled.

Work along a chord
------------------
Gravity acts downwards with magnitude ``g(x)``, so the work per unit mass over a
straight chord from ``(x1, y1)`` to ``(x2, y2)`` is::

    dKE = integral of g(x) * (-dy)

On a straight chord ``dy/dx`` is constant, which pulls it out of the integral::

    run  = x2 - x1
    drop = y1 - y2
    dKE  = (drop / run) * integral_{x1}^{x2} g(x) dx        (run > 0)
    dKE  = g(x1) * drop                                     (run == 0)

So a field only has to supply ``g(x)`` and its running integral.  Both are
served from a cumulative trapezoid table sampled on the solver's own x grid, so
an analytic field and a JSON field are treated identically and give comparable
answers.

Spec strings
------------
``parse_spec`` accepts ``"family:p1,p2,..."`` -- one shell-safe token::

    const:9.81                 g = 9.81
    poly:9.81,0.02,-1e-4       g = 9.81 + 0.02*x - 1e-4*x**2      (ascending powers)
    power:0.5,2,9.81           g = 9.81 + 0.5 * x**2
    sin:2,0.05,0,9.81          g = 9.81 + 2*sin(0.05*x + 0)
    cos:2,0.05,0,9.81          g = 9.81 + 2*cos(0.05*x + 0)
    exp:5,2,0.001,9.81         g = 9.81 + 5*exp(-0.001 * x**2)

Trailing parameters may be omitted and fall back to the documented defaults.
Note ``poly`` takes coefficients in *ascending* powers, constant first, so the
offset is always the first number.
"""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "GravityField",
    "uniform_field",
    "sampled_field",
    "parse_spec",
    "load_json",
    "write_json",
    "FAMILIES",
]


@dataclass(frozen=True)
class GravityField:
    """``g`` sampled on an ascending x grid, with its cumulative integral.

    ``x`` and ``g`` must be the same length.  ``cum[k]`` holds
    ``integral_{x[0]}^{x[k]} g dx`` by the trapezoid rule, which makes
    :meth:`integral` exact for the piecewise-linear field the samples define.
    """

    x: np.ndarray
    g: np.ndarray
    cum: np.ndarray
    label: str
    uniform: bool

    def __post_init__(self) -> None:
        if self.x.ndim != 1 or self.x.size < 2:
            raise ValueError("gravity field needs at least two sample points")
        if self.x.shape != self.g.shape:
            raise ValueError("x and g must have the same length")
        if not np.all(np.diff(self.x) > 0):
            raise ValueError("gravity field x samples must be strictly ascending")

    # -- queries ---------------------------------------------------------
    def at(self, x: float | np.ndarray) -> np.ndarray:
        """Local ``g`` at one or more positions (linear interpolation)."""
        return np.interp(x, self.x, self.g)

    def integral(self, x1: float, x2: float) -> float:
        """``integral_{x1}^{x2} g dx``, signed."""
        return float(np.interp(x2, self.x, self.cum) - np.interp(x1, self.x, self.cum))

    def chord_work(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """Work per unit mass done by gravity along a straight chord.

        Positive means the ball gains kinetic energy.
        """
        run, drop = x2 - x1, y1 - y2
        if run == 0.0:
            return float(self.at(x1)) * drop
        return (drop / run) * self.integral(x1, x2)

    @property
    def mean(self) -> float:
        """Length-weighted mean of ``g`` across the sampled range."""
        span = float(self.x[-1] - self.x[0])
        if span <= 0:
            return float(self.g[0])
        return float(self.cum[-1] / span)

    def resampled(self, x_new: np.ndarray) -> "GravityField":
        """This field re-sampled onto another grid, integrals recomputed."""
        return sampled_field(x_new, self.at(x_new), label=self.label)

    def describe(self) -> str:
        if self.uniform:
            return f"{self.label} (uniform, g = {self.g[0]:g})"
        return (
            f"{self.label} (variable, g in [{self.g.min():g}, {self.g.max():g}], "
            f"mean {self.mean:g}, {self.g.size} samples)"
        )


def _cumulative(x: np.ndarray, g: np.ndarray) -> np.ndarray:
    steps = np.diff(x) * 0.5 * (g[:-1] + g[1:])  # trapezoid
    return np.concatenate([[0.0], np.cumsum(steps)])


def sampled_field(
    x: Sequence[float] | np.ndarray,
    g: Sequence[float] | np.ndarray,
    label: str = "sampled",
) -> GravityField:
    """Build a field from explicit samples."""
    xa = np.asarray(x, dtype=float)
    ga = np.asarray(g, dtype=float)
    is_uniform = bool(ga.size and np.ptp(ga) <= 1e-12)
    return GravityField(xa, ga, _cumulative(xa, ga), label, is_uniform)


def uniform_field(g: float, x_max: float, n: int = 2) -> GravityField:
    """A constant field, so the uniform and variable code paths agree exactly."""
    xa = np.linspace(0.0, max(x_max, 1e-9), max(n, 2))
    return sampled_field(xa, np.full(xa.shape, float(g)), label=f"const:{g:g}")


# --------------------------------------------------------------------------
# Analytic families
# --------------------------------------------------------------------------


def _const(params):
    (g0,) = _take(params, 1, "const", ("g",))
    return (lambda x: np.full_like(np.asarray(x, dtype=float), g0)), f"const:{g0:g}"


def _poly(params):
    if not params:
        raise ValueError("poly needs at least one coefficient")
    coeffs = list(params)
    # Ascending powers: c0 + c1*x + c2*x**2 + ...  Constant first, so the
    # offset that keeps g positive is always the leading number.
    def fn(x):
        xa = np.asarray(x, dtype=float)
        out = np.zeros_like(xa)
        for k, c in enumerate(coeffs):
            out = out + c * xa**k
        return out

    terms = " + ".join(f"{c:g}*x^{k}" if k else f"{c:g}" for k, c in enumerate(coeffs))
    return fn, f"poly({terms})"


def _power(params):
    A, n, c = _take(params, 3, "power", ("A", "n", "offset"), defaults=(1.0, 1.0, 0.0))
    return (lambda x: c + A * np.asarray(x, dtype=float) ** n), f"power({c:g}+{A:g}*x^{n:g})"


def _sin(params):
    A, k, phi, c = _take(params, 4, "sin", ("A", "k", "phi", "offset"),
                         defaults=(1.0, 1.0, 0.0, 0.0))
    return ((lambda x: c + A * np.sin(k * np.asarray(x, dtype=float) + phi)),
            f"sin({c:g}+{A:g}*sin({k:g}x+{phi:g}))")


def _cos(params):
    A, k, phi, c = _take(params, 4, "cos", ("A", "k", "phi", "offset"),
                         defaults=(1.0, 1.0, 0.0, 0.0))
    return ((lambda x: c + A * np.cos(k * np.asarray(x, dtype=float) + phi)),
            f"cos({c:g}+{A:g}*cos({k:g}x+{phi:g}))")


def _exp(params):
    A, n, k, c = _take(params, 4, "exp", ("A", "n", "k", "offset"),
                       defaults=(1.0, 1.0, 1.0, 0.0))
    return ((lambda x: c + A * np.exp(-k * np.asarray(x, dtype=float) ** n)),
            f"exp({c:g}+{A:g}*e^(-{k:g}x^{n:g}))")


FAMILIES: dict[str, Callable] = {
    "const": _const,
    "poly": _poly,
    "power": _power,
    "sin": _sin,
    "cos": _cos,
    "exp": _exp,
}

#: Human-readable parameter order, for CLI help and error messages.
FAMILY_HELP = {
    "const": "const:g",
    "poly": "poly:c0,c1,c2,...        g = c0 + c1*x + c2*x^2 + ...  (ascending)",
    "power": "power:A,n,offset         g = offset + A*x^n",
    "sin": "sin:A,k,phi,offset       g = offset + A*sin(k*x + phi)",
    "cos": "cos:A,k,phi,offset       g = offset + A*cos(k*x + phi)",
    "exp": "exp:A,n,k,offset         g = offset + A*exp(-k*x^n)",
}


def _take(params, count, family, names, defaults=None):
    if len(params) > count:
        raise ValueError(
            f"{family} takes at most {count} parameters ({', '.join(names)}), "
            f"got {len(params)}"
        )
    if defaults is None:
        if len(params) != count:
            raise ValueError(
                f"{family} needs {count} parameters ({', '.join(names)}), got {len(params)}"
            )
        return tuple(params)
    filled = list(params) + list(defaults[len(params):])
    return tuple(filled)


def parse_spec(spec: str) -> tuple[Callable, str]:
    """Turn ``"family:p1,p2"`` into ``(callable, label)``.

    The callable maps an x array to g values.  Raises ``ValueError`` with a
    listing of the families on anything unrecognised.
    """
    text = spec.strip()
    if not text:
        raise ValueError("empty gravity spec")
    family, _, rest = text.partition(":")
    family = family.strip().lower()
    if family not in FAMILIES:
        raise ValueError(
            f"unknown gravity family {family!r}; choose from "
            + ", ".join(sorted(FAMILIES))
        )
    params: list[float] = []
    for chunk in (c.strip() for c in rest.split(",")):
        if not chunk:
            continue
        try:
            params.append(float(chunk))
        except ValueError:
            raise ValueError(f"{family}: {chunk!r} is not a number") from None
    return FAMILIES[family](params)


def field_from_spec(spec: str, x: np.ndarray) -> GravityField:
    """Evaluate a spec string on grid ``x``."""
    fn, label = parse_spec(spec)
    return sampled_field(x, np.asarray(fn(x), dtype=float), label=label)


# --------------------------------------------------------------------------
# JSON interchange
# --------------------------------------------------------------------------


def write_json(path, x_max: float, x_step: float, g: np.ndarray, label: str) -> dict:
    """Write a gravity table.  Returns the payload for inspection."""
    payload = {
        "description": "gravity magnitude g(x) sampled on the solver x grid",
        "generated_by": label,
        "x_max": float(x_max),
        "x_step": float(x_step),
        "count": int(np.size(g)),
        "g": [float(v) for v in np.asarray(g).ravel()],
    }
    pathlib.Path(path).write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def load_json(path, x: np.ndarray) -> GravityField:
    """Load a gravity table and bind it to grid ``x``.

    The stored sequence must have exactly one value per grid column.  A
    mismatch is the most likely mistake with hand-made files, so the error
    names both counts and the x_max/x_step that would produce the right one.
    """
    raw = json.loads(pathlib.Path(path).read_text())
    if isinstance(raw, list):  # bare array is accepted
        values = raw
        meta = {}
    elif isinstance(raw, dict):
        if "g" not in raw:
            raise ValueError(f"{path}: JSON object has no 'g' key")
        values = raw["g"]
        meta = raw
    else:
        raise ValueError(f"{path}: expected a JSON object or array")

    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{path}: 'g' must be a flat array of numbers")
    if arr.size != x.size:
        step = (x[-1] - x[0]) / (x.size - 1) if x.size > 1 else 0.0
        raise ValueError(
            f"{path}: gravity table has {arr.size} values but the grid needs "
            f"{x.size} (x_max={x[-1]:g}, x_step={step:g}). "
            f"Regenerate with: make_gravity_json.py --x-max {x[-1]:g} "
            f"--x-step {step:g}"
        )
    if meta:
        # A loose check: it catches a table built for a different problem, but
        # must tolerate a rounded x_max being typed by hand (157.08 for the
        # exact pi*50 = 157.0796...).  The length check above is the strict one.
        for key, actual in (("x_max", x[-1]),):
            if key in meta and not math.isclose(float(meta[key]), float(actual), rel_tol=1e-3):
                raise ValueError(
                    f"{path}: file declares {key}={float(meta[key]):.6g} but the grid "
                    f"has {key}={float(actual):.6g}"
                )
    label = str(meta.get("generated_by", pathlib.Path(path).name)) if meta else str(path)
    return sampled_field(x, arr, label=label)


def check_positive(field: GravityField) -> None:
    """Reject fields that would stall or reverse the ball.

    A non-positive ``g`` anywhere on the grid means gravity does no work (or
    negative work) there, which the search cannot use and which almost always
    means a sinusoid or polynomial was given without an offset.
    """
    if np.all(field.g > 0):
        return
    bad = np.flatnonzero(field.g <= 0)
    where = ", ".join(f"x={field.x[k]:g} -> g={field.g[k]:g}" for k in bad[:3])
    more = "" if bad.size <= 3 else f" (and {bad.size - 3} more)"
    raise ValueError(
        f"gravity must be positive everywhere on the grid; {where}{more}. "
        "Add an offset large enough to lift the whole curve above zero."
    )
