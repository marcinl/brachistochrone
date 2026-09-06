"""Tests for variable gravity fields and the energy-augmented search.

The anchor test is :class:`TestVariableMatchesUniform`: run the variable-gravity
machinery on a field that happens to be constant and it must reproduce the
uniform solver exactly.  Same physics, different algorithm, so any disagreement
is a bug in the new code rather than a modelling choice.
"""

import json
import math
import pathlib
import tempfile
import unittest

import numpy as np

import brachistochrone as B
from brachistochrone import Config, solve
from gravity import (
    GravityField,
    _cumulative,
    check_positive,
    field_from_spec,
    load_json,
    parse_spec,
    sampled_field,
    uniform_field,
    write_json,
)
from make_gravity_json import grid_for

COARSE = dict(x_step=10.0, y_step=10.0)  # keep the label search quick


def constant_but_variable(g0: float, x: np.ndarray) -> GravityField:
    """A constant field flagged non-uniform, to force the label search."""
    g = np.full(x.shape, float(g0))
    return GravityField(x, g, _cumulative(x, g), f"forced:{g0:g}", False)


class TestSpecs(unittest.TestCase):
    def test_each_family_parses(self):
        x = np.linspace(0.0, 100.0, 21)
        for spec in ("const:9.81", "poly:9.81,0.02,-1e-4", "power:0.5,2,9.81",
                     "sin:2,0.05,0,9.81", "cos:2,0.05,0,9.81", "exp:5,2,1e-3,9.81"):
            with self.subTest(spec=spec):
                field = field_from_spec(spec, x)
                self.assertEqual(field.g.shape, x.shape)
                self.assertTrue(np.all(np.isfinite(field.g)))

    def test_poly_is_ascending_powers(self):
        """Constant first, so the offset that keeps g positive leads."""
        fn, _ = parse_spec("poly:3,2,1")          # 3 + 2x + x^2
        np.testing.assert_allclose(fn(np.array([0.0, 1.0, 2.0])), [3.0, 6.0, 11.0])

    def test_const_is_flagged_uniform(self):
        x = np.linspace(0.0, 50.0, 11)
        self.assertTrue(field_from_spec("const:9.81", x).uniform)
        self.assertFalse(field_from_spec("poly:9.81,0.1", x).uniform)

    def test_trailing_params_default(self):
        x = np.linspace(0.0, 10.0, 11)
        # sin:A only -> k=1, phi=0, offset=0
        np.testing.assert_allclose(field_from_spec("sin:2", x).g, 2 * np.sin(x))

    def test_unknown_family_lists_the_options(self):
        with self.assertRaises(ValueError) as ctx:
            parse_spec("wobble:1")
        self.assertIn("poly", str(ctx.exception))

    def test_non_numeric_parameter(self):
        with self.assertRaises(ValueError):
            parse_spec("sin:a,b")

    def test_too_many_parameters(self):
        with self.assertRaises(ValueError):
            parse_spec("sin:1,2,3,4,5")


class TestFieldMaths(unittest.TestCase):
    def test_integral_is_exact_for_a_constant_field(self):
        f = uniform_field(9.81, 100.0)
        self.assertAlmostEqual(f.integral(0.0, 100.0), 981.0, places=9)
        self.assertAlmostEqual(f.integral(20.0, 50.0), 9.81 * 30, places=9)

    def test_integral_is_exact_for_a_linear_field(self):
        """Trapezoid is exact on a piecewise-linear field."""
        x = np.linspace(0.0, 10.0, 11)
        f = sampled_field(x, 1.0 + x)
        self.assertAlmostEqual(f.integral(0.0, 10.0), 10.0 + 50.0, places=9)

    def test_chord_work_matches_uniform_formula(self):
        f = uniform_field(9.81, 100.0)
        # Descending 10 m over any run gains g*drop under uniform gravity.
        for run in (1.0, 20.0, 100.0):
            self.assertAlmostEqual(f.chord_work(0.0, 10.0, run, 0.0), 9.81 * 10, places=9)

    def test_vertical_chord_uses_local_g(self):
        x = np.linspace(0.0, 10.0, 11)
        f = sampled_field(x, 1.0 + x)
        self.assertAlmostEqual(f.chord_work(4.0, 10.0, 4.0, 0.0), 5.0 * 10.0, places=9)

    def test_climbing_costs_energy(self):
        f = uniform_field(9.81, 100.0)
        self.assertLess(f.chord_work(0.0, 0.0, 10.0, 5.0), 0.0)

    def test_field_is_non_conservative(self):
        """The fact that forces the whole redesign: route changes arrival energy."""
        x = np.linspace(0.0, 10.0, 1001)
        f = sampled_field(x, 1.0 + x)

        def energy(nodes):
            return sum(f.chord_work(a[0], a[1], b[0], b[1])
                       for a, b in zip(nodes, nodes[1:]))

        down_across = energy([(0, 10), (0, 0), (10, 0)])
        across_down = energy([(0, 10), (10, 10), (10, 0)])
        self.assertAlmostEqual(down_across, 10.0, places=6)
        self.assertAlmostEqual(across_down, 110.0, places=6)
        self.assertGreater(abs(across_down - down_across), 1.0)

    def test_uniform_field_is_conservative(self):
        f = uniform_field(9.81, 20.0)

        def energy(nodes):
            return sum(f.chord_work(a[0], a[1], b[0], b[1])
                       for a, b in zip(nodes, nodes[1:]))

        self.assertAlmostEqual(energy([(0, 10), (0, 0), (10, 0)]),
                               energy([(0, 10), (10, 10), (10, 0)]), places=9)

    def test_mean_is_length_weighted(self):
        x = np.linspace(0.0, 10.0, 11)
        self.assertAlmostEqual(sampled_field(x, 1.0 + x).mean, 6.0, places=9)


class TestPositivity(unittest.TestCase):
    def test_rejects_a_sinusoid_without_an_offset(self):
        x = np.linspace(0.0, 100.0, 21)
        with self.assertRaises(ValueError) as ctx:
            check_positive(field_from_spec("sin:2,0.05,0,0", x))
        self.assertIn("offset", str(ctx.exception))

    def test_accepts_a_lifted_sinusoid(self):
        x = np.linspace(0.0, 100.0, 21)
        check_positive(field_from_spec("sin:2,0.05,0,9.81", x))  # must not raise

    def test_solver_rejects_non_positive_gravity(self):
        with self.assertRaises(ValueError):
            solve(Config(gravity_spec="poly:9.81,-1.0", **COARSE))


class TestJson(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.dir.name) / "g.json"

    def tearDown(self):
        self.dir.cleanup()

    def test_round_trip(self):
        x = grid_for(100.0, 5.0)
        field = field_from_spec("poly:9.81,0.02", x)
        write_json(self.path, 100.0, 5.0, field.g, field.label)
        back = load_json(self.path, x)
        np.testing.assert_allclose(back.g, field.g)

    def test_size_mismatch_names_both_counts(self):
        x = grid_for(100.0, 5.0)
        write_json(self.path, 100.0, 5.0, np.full(x.size, 9.81), "const")
        with self.assertRaises(ValueError) as ctx:
            load_json(self.path, grid_for(100.0, 2.0))
        message = str(ctx.exception)
        self.assertIn(str(x.size), message)
        self.assertIn("make_gravity_json.py", message)

    def test_bare_array_is_accepted(self):
        x = grid_for(50.0, 10.0)
        self.path.write_text(json.dumps([9.81] * x.size))
        self.assertTrue(load_json(self.path, x).uniform)

    def test_missing_g_key(self):
        self.path.write_text(json.dumps({"x_max": 100.0}))
        with self.assertRaises(ValueError):
            load_json(self.path, grid_for(100.0, 5.0))

    def test_rounded_x_max_is_tolerated(self):
        """157.08 typed by hand must match the exact pi*50 grid."""
        x = grid_for(math.pi * 50, 5.0)
        write_json(self.path, 157.08, 5.0, np.full(x.size, 9.81), "const")
        load_json(self.path, x)  # must not raise

    def test_grid_for_matches_the_solver_grid(self):
        for x_max, x_step in ((100.0, 5.0), (157.08, 5.0), (250.0, 2.5)):
            with self.subTest(x_max=x_max):
                sol = solve(Config(x_max=x_max, x_step=x_step, y_step=10.0))
                np.testing.assert_allclose(grid_for(x_max, x_step), sol.x)


class TestVariableMatchesUniform(unittest.TestCase):
    """A constant field run through the label search must match plain Dijkstra."""

    def setUp(self):
        self.real = B._build_field

    def tearDown(self):
        B._build_field = self.real

    def _forced(self, **kw):
        B._build_field = lambda cfg, x, x_max: constant_but_variable(cfg.g, x)
        try:
            return solve(Config(**kw))
        finally:
            B._build_field = self.real

    def test_times_agree_to_machine_precision(self):
        uni = solve(Config(**COARSE))
        var = self._forced(**COARSE)
        for xt in (0.0, 25.0, 50.0, uni.x_max):
            with self.subTest(x=xt):
                ix = uni.target_index(xt)
                tu, _ = uni.best_at(ix, uni.iy_target)
                tv, _ = var.best_at(ix, var.iy_target)
                self.assertAlmostEqual(tu, tv, places=12)

    def test_paths_are_identical(self):
        uni = solve(Config(**COARSE))
        var = self._forced(**COARSE)
        self.assertEqual(uni.path_nodes(uni.shape[0] - 1, uni.iy_target),
                         var.path_nodes(var.shape[0] - 1, var.iy_target))

    def test_label_bookkeeping_is_populated(self):
        var = self._forced(**COARSE)
        self.assertIsNotNone(var.label_of)
        self.assertIsNotNone(var.label_link)
        self.assertIsNotNone(var.arrival_speed)


class TestVariableSolver(unittest.TestCase):
    def setUp(self):
        self.sol = solve(Config(gravity_spec="poly:9.81,0.05", **COARSE))
        self.ixt = self.sol.shape[0] - 1
        self.seg = self.sol.path_segments(self.ixt, self.sol.iy_target)

    def test_uses_the_label_search(self):
        self.assertFalse(self.sol.field.uniform)
        self.assertIsNotNone(self.sol.label_of)

    def test_speeds_match_accumulated_work(self):
        """Speed is path-dependent now, so it must follow the chords taken."""
        energy = 0.0
        for row in self.seg:
            energy += self.sol.field.chord_work(row["x1"], row["y1"], row["x2"], row["y2"])
            self.assertAlmostEqual(math.sqrt(2 * energy), row["v2"], places=9)

    def test_durations_sum_to_the_reported_minimum(self):
        t, _ = self.sol.best_at(self.ixt, self.sol.iy_target)
        self.assertAlmostEqual(self.seg["dt_seg"].sum(), t, places=9)

    def test_path_is_contiguous(self):
        np.testing.assert_allclose(self.seg["x2"][:-1], self.seg["x1"][1:])
        np.testing.assert_allclose(self.seg["y2"][:-1], self.seg["y1"][1:])

    def test_speed_is_no_longer_an_altitude_column(self):
        self.assertTrue(np.all(np.isnan(self.sol.speed)))

    def test_stronger_gravity_downstream_is_faster(self):
        """More g where the ball spends its speed must beat less."""
        rising = solve(Config(gravity_spec="poly:9.81,0.05", **COARSE))
        falling = solve(Config(gravity_spec="poly:9.81,-0.04", **COARSE))
        t_rise, _ = rising.best_at(rising.shape[0] - 1, rising.iy_target)
        t_fall, _ = falling.best_at(falling.shape[0] - 1, falling.iy_target)
        self.assertLess(t_rise, t_fall)

    def test_energy_tolerance_does_not_change_the_answer_much(self):
        loose = solve(Config(gravity_spec="poly:9.81,0.05", energy_tol=0.1, **COARSE))
        t_exact, _ = self.sol.best_at(self.ixt, self.sol.iy_target)
        t_loose, _ = loose.best_at(loose.shape[0] - 1, loose.iy_target)
        self.assertGreaterEqual(t_loose, t_exact - 1e-12)
        self.assertLess((t_loose - t_exact) / t_exact, 1e-3)

    def test_structured_array_reports_searched_speeds(self):
        arr = self.sol.as_structured_array()
        finite = np.isfinite(arr["t_min"])
        np.testing.assert_allclose(arr["v"][finite], self.sol.arrival_speed[finite])


class TestConfigCombinations(unittest.TestCase):
    def test_spec_and_json_are_mutually_exclusive(self):
        with self.assertRaises(TypeError):
            Config(gravity_spec="const:9.81", gravity_json="g.json")

    def test_negative_energy_tol_rejected(self):
        with self.assertRaises(ValueError):
            Config(energy_tol=-1.0)

    def test_has_variable_gravity_flag(self):
        self.assertFalse(Config().has_variable_gravity)
        self.assertTrue(Config(gravity_spec="poly:9,1").has_variable_gravity)

    def test_const_spec_takes_the_uniform_fast_path(self):
        sol = solve(Config(gravity_spec="const:9.81", **COARSE))
        self.assertTrue(sol.field.uniform)
        self.assertIsNone(sol.label_of)
        plain, _ = solve(Config(**COARSE)).best_at(0, 10)
        self.assertAlmostEqual(sol.best_at(0, sol.iy_target)[0], plain, places=12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
