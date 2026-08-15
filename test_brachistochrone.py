"""Tests for the discrete brachistochrone solver.

The two anchors are analytic: a vertical drop must reproduce sqrt(2h/g) exactly,
and every grid path must be an upper bound on the continuous cycloid time that
tightens as the grid refines.
"""

import math
import unittest

import numpy as np

from brachistochrone import (
    Config,
    _primitive_moves,
    analytic_brachistochrone,
    cycloid_curve,
    endpoint_for_theta_ratio,
    free_fall_time,
    max_horizontal_extent,
    solve,
)


class TestAnalytics(unittest.TestCase):
    def test_free_fall_from_rest(self):
        self.assertAlmostEqual(free_fall_time(100.0, 9.81), math.sqrt(2 * 100 / 9.81), places=12)

    def test_free_fall_with_initial_speed(self):
        # v0 = 10 m/s downward through 100 m: (sqrt(v0^2 + 2gh) - v0) / g
        expected = (math.sqrt(100 + 2 * 9.81 * 100) - 10.0) / 9.81
        self.assertAlmostEqual(free_fall_time(100.0, 9.81, v0=10.0), expected, places=12)

    def test_extent_is_half_cycloid_arch(self):
        """Default budget is slack, so geometry binds at x_max = pi*drop/2."""
        x_max, binding = max_horizontal_extent(100.0, 9.81, time_budget=10.0)
        self.assertAlmostEqual(x_max, math.pi * 100.0 / 2.0, places=9)
        self.assertEqual(binding, "geometry")

    def test_extent_falls_back_to_time_budget(self):
        t_at_pi = math.pi * math.sqrt(100.0 / (2 * 9.81))
        x_max, binding = max_horizontal_extent(100.0, 9.81, time_budget=t_at_pi * 0.8)
        self.assertEqual(binding, "time budget")
        self.assertLess(x_max, math.pi * 100.0 / 2.0)
        self.assertGreater(x_max, 0.0)

    def test_extent_zero_when_budget_below_free_fall(self):
        x_max, binding = max_horizontal_extent(100.0, 9.81, time_budget=1.0)
        self.assertEqual(x_max, 0.0)
        self.assertIn("time", binding)

    def test_analytic_endpoint_matches_free_fall_at_zero_offset(self):
        t, arc = analytic_brachistochrone(0.0, 100.0, 9.81)
        self.assertAlmostEqual(t, math.sqrt(2 * 100 / 9.81), places=12)
        self.assertAlmostEqual(arc, 100.0, places=12)

    def test_analytic_cycloid_at_the_arch_apex(self):
        """At x_end = pi*drop/2 the cycloid is a half arch: a = drop/2, theta = pi."""
        drop = 100.0
        t, arc = analytic_brachistochrone(math.pi * drop / 2, drop, 9.81)
        self.assertAlmostEqual(t, math.pi * math.sqrt((drop / 2) / 9.81), places=6)
        self.assertAlmostEqual(arc, 4.0 * (drop / 2) * (1 - math.cos(math.pi / 2)), places=6)


class TestVerticalDrop(unittest.TestCase):
    """The straight-down column is a chain of exact chords, so it must be exact."""

    def setUp(self):
        self.cfg = Config(h=100.0, y_end=0.0, x_step=5.0, y_step=5.0)
        self.sol = solve(self.cfg)

    def test_matches_closed_form(self):
        t, _ = self.sol.best_at(0, self.sol.shape[1] - 1)
        self.assertAlmostEqual(t, math.sqrt(2 * 100 / 9.81), places=9)

    def test_path_length_equals_the_drop(self):
        _, ir = self.sol.best_at(0, self.sol.shape[1] - 1)
        self.assertAlmostEqual(self.sol.length[0, -1, ir], 100.0, places=9)

    def test_horizontal_move_from_rest_is_unreachable(self):
        """v0 = 0 at the top: a horizontal chord takes infinite time, not a crash."""
        self.assertTrue(np.all(np.isinf(self.sol.time[1, 0, :])))

    def test_start_cell_is_the_only_zero_time_state(self):
        zeros = np.argwhere(self.sol.time == 0.0)
        self.assertTrue(np.all(zeros[:, 0] == 0))
        self.assertTrue(np.all(zeros[:, 1] == 0))


class TestAgainstContinuousOptimum(unittest.TestCase):
    def _time_to(self, x_end, **kw):
        cfg = Config(h=100.0, y_end=0.0, **kw)
        sol = solve(cfg)
        ix = sol.target_index(x_end)
        t, _ = sol.best_at(ix, sol.shape[1] - 1)
        return t, float(sol.x[ix])

    def test_never_beats_the_continuous_optimum(self):
        """Every grid polyline is an admissible path, so it can only be slower."""
        for x_end in (25.0, 50.0, 100.0, 150.0):
            with self.subTest(x_end=x_end):
                t, x_actual = self._time_to(x_end, x_step=5.0, y_step=5.0)
                t_ref, _ = analytic_brachistochrone(x_actual, 100.0, 9.81)
                self.assertGreaterEqual(t, t_ref - 1e-9)

    def test_beats_the_straight_chord(self):
        """The whole point: the cycloid is longer than the chord but faster."""
        x_end = 100.0
        t, x_actual = self._time_to(x_end, x_step=5.0, y_step=5.0)
        chord_len = math.hypot(x_actual, 100.0)
        v_end = math.sqrt(2 * 9.81 * 100.0)
        t_chord = 2.0 * chord_len / v_end
        self.assertLess(t, t_chord)

    def test_refining_the_grid_tightens_the_bound(self):
        """Compare at the far edge -- the one column every grid shares exactly."""

        def edge_error(**kw):
            sol = solve(Config(h=100.0, y_end=0.0, **kw))
            nx, ny, _ = sol.shape
            t, _ = sol.best_at(nx - 1, ny - 1)
            t_ref, _ = analytic_brachistochrone(float(sol.x[-1]), 100.0, 9.81)
            return (t - t_ref) / t_ref

        coarse = edge_error(x_step=10.0, y_step=10.0, neighbourhood_cells=4)
        fine = edge_error(x_step=2.5, y_step=2.5, neighbourhood_cells=6)
        self.assertLessEqual(fine, coarse)
        self.assertLess(fine, 0.01)


class TestStateSpace(unittest.TestCase):
    def setUp(self):
        self.sol = solve(Config(h=100.0, x_step=10.0, y_step=10.0))

    def test_grid_dimensions(self):
        nx, ny, nr = self.sol.shape
        self.assertEqual(ny, 11)  # 100 m in 10 m steps, inclusive
        self.assertEqual(nr, 10)  # 0..90 deg in 10 deg steps
        self.assertEqual(nx, int(round(math.pi * 100 / 2 / 10.0)) + 1)

    def test_launch_angle_is_inert_without_a_turn_limit(self):
        """No curvature constraint => the launch heading cannot matter."""
        nx, ny, _ = self.sol.shape
        times = []
        for angle in (0.0, 30.0, 60.0, 90.0):
            sol = solve(
                Config(h=100.0, x_step=10.0, y_step=10.0, launch_angles_deg=(angle,))
            )
            t, _ = sol.best_at(nx - 1, ny - 1)
            times.append(t)
        self.assertAlmostEqual(min(times), max(times), places=9)

    def test_arrival_heading_does_matter(self):
        """The r axis is real state: reaching a cell steeply differs from shallowly."""
        ny = self.sol.shape[1]
        finite = [t for t in self.sol.time[-1, ny - 1] if math.isfinite(t)]
        self.assertGreater(len(finite), 1)
        self.assertGreater(max(finite) - min(finite), 1e-6)

    def test_turn_limit_costs_time(self):
        free, _ = self.sol.best_at(self.sol.shape[0] - 1, self.sol.shape[1] - 1)
        limited = solve(Config(h=100.0, x_step=10.0, y_step=10.0, max_turn_deg=10.0))
        constrained, _ = limited.best_at(limited.shape[0] - 1, limited.shape[1] - 1)
        self.assertGreaterEqual(constrained, free - 1e-9)

    def test_speed_depends_only_on_altitude(self):
        expected = np.sqrt(2 * 9.81 * (100.0 - self.sol.y))
        np.testing.assert_allclose(self.sol.speed, expected)

    def test_structured_array_fields(self):
        arr = self.sol.as_structured_array()
        self.assertEqual(arr.shape, self.sol.shape)
        np.testing.assert_allclose(arr["t_min"], self.sol.time)
        # Incoming steps go forward and down everywhere they exist.
        seen = np.isfinite(arr["dx"])
        self.assertTrue(np.all(arr["dx"][seen] >= -1e-9))
        self.assertTrue(np.all(arr["dy"][seen] <= 1e-9))
        angles = arr["angle_in_deg"][seen]
        self.assertTrue(np.all((angles >= -1e-9) & (angles <= 90.0 + 1e-9)))


class TestSampling(unittest.TestCase):
    def setUp(self):
        self.sol = solve(Config(h=100.0, x_step=5.0, y_step=5.0))
        self.ix = self.sol.target_index(100.0)
        self.nodes = self.sol.path_nodes(self.ix, self.sol.shape[1] - 1)

    def test_path_endpoints(self):
        self.assertAlmostEqual(self.nodes[0][0], 0.0)
        self.assertAlmostEqual(self.nodes[0][1], 100.0)
        self.assertAlmostEqual(self.nodes[-1][0], float(self.sol.x[self.ix]))
        self.assertAlmostEqual(self.nodes[-1][1], 0.0)

    def test_path_states_agree_with_path_nodes(self):
        states = self.sol.path_states(self.ix, self.sol.shape[1] - 1)
        self.assertEqual(len(states), len(self.nodes))
        for (j, i, _), (x, y) in zip(states, self.nodes):
            self.assertAlmostEqual(float(self.sol.x[j]), x)
            self.assertAlmostEqual(float(self.sol.y[i]), y)

    def test_path_states_start_at_the_origin_cell(self):
        states = self.sol.path_states(self.ix, self.sol.shape[1] - 1)
        self.assertEqual(states[0][:2], (0, 0))
        self.assertEqual(states[-1][:2], (self.ix, self.sol.shape[1] - 1))

    def test_path_state_headings_match_the_step_taken(self):
        """The r index of each state is the bin of the chord that arrived there."""
        cfg = self.sol.config
        states = self.sol.path_states(self.ix, self.sol.shape[1] - 1)
        for (j0, i0, _), (j1, i1, r1) in zip(states, states[1:]):
            dx = float(self.sol.x[j1] - self.sol.x[j0])
            dy = float(self.sol.y[i0] - self.sol.y[i1])
            angle = math.degrees(math.atan2(dy, dx))
            self.assertAlmostEqual(angle, float(self.sol.r_deg[r1]), delta=cfg.angle_step_deg / 2)

    def test_path_states_rejects_unreachable(self):
        with self.assertRaises(ValueError):
            self.sol.path_states(1, 0, 0)  # horizontal first move from rest

    def test_samples_are_uniform_in_time_and_end_exactly(self):
        s = self.sol.sample_path(self.nodes, dt=0.1)
        self.assertAlmostEqual(s["t"][0], 0.0)
        np.testing.assert_allclose(np.diff(s["t"][:-1]), 0.1, atol=1e-9)
        t_total, _ = self.sol.best_at(self.ix, self.sol.shape[1] - 1)
        self.assertAlmostEqual(s["t"][-1], t_total, places=9)

    def test_sampled_motion_is_physical(self):
        s = self.sol.sample_path(self.nodes, dt=0.05)
        self.assertTrue(np.all(np.diff(s["y"]) <= 1e-9))  # never climbs
        self.assertTrue(np.all(np.diff(s["x"]) >= -1e-9))  # never reverses
        # Speed is set by altitude alone, whatever the route.
        np.testing.assert_allclose(s["v"], np.sqrt(2 * 9.81 * (100.0 - s["y"])), atol=1e-6)

    def test_final_speed_is_the_free_fall_speed(self):
        s = self.sol.sample_path(self.nodes)
        self.assertAlmostEqual(s["v"][-1], math.sqrt(2 * 9.81 * 100.0), places=6)


class TestSegments(unittest.TestCase):
    """The per-chord breakdown must reconstruct the DP result exactly."""

    def setUp(self):
        self.sol = solve(Config(h=100.0, x_step=5.0, y_step=5.0))
        self.ny = self.sol.shape[1]
        self.ix = self.sol.shape[0] - 1
        self.seg = self.sol.path_segments(self.ix, self.ny - 1)

    def test_durations_sum_to_the_reported_minimum(self):
        t_total, _ = self.sol.best_at(self.ix, self.ny - 1)
        self.assertAlmostEqual(self.seg["dt_seg"].sum(), t_total, places=9)

    def test_lengths_sum_to_the_reported_path_length(self):
        _, ir = self.sol.best_at(self.ix, self.ny - 1)
        self.assertAlmostEqual(
            self.seg["length"].sum(), self.sol.length[self.ix, self.ny - 1, ir], places=9
        )

    def test_segments_are_contiguous(self):
        np.testing.assert_allclose(self.seg["x2"][:-1], self.seg["x1"][1:])
        np.testing.assert_allclose(self.seg["y2"][:-1], self.seg["y1"][1:])

    def test_running_totals_are_consistent(self):
        np.testing.assert_allclose(
            self.seg["t_end"] - self.seg["t_start"], self.seg["dt_seg"], atol=1e-12
        )
        np.testing.assert_allclose(self.seg["s_end"], np.cumsum(self.seg["length"]))

    def test_exit_speed_matches_next_entry_speed(self):
        np.testing.assert_allclose(self.seg["v2"][:-1], self.seg["v1"][1:], atol=1e-12)

    def test_speeds_follow_energy_conservation(self):
        np.testing.assert_allclose(
            self.seg["v1"], np.sqrt(2 * 9.81 * (100.0 - self.seg["y1"])), atol=1e-9
        )

    def test_chord_angles_shallow_out(self):
        """A cycloid from rest starts vertical and flattens; never the reverse."""
        self.assertGreater(self.seg["angle_deg"][0], 80.0)
        self.assertLess(self.seg["angle_deg"][-1], 10.0)
        self.assertTrue(np.all(np.diff(self.seg["angle_deg"]) <= 1e-9))


class TestCycloidCurve(unittest.TestCase):
    def test_endpoints(self):
        x, drop = cycloid_curve(157.08, 100.0, n=200)
        self.assertAlmostEqual(x[0], 0.0, places=9)
        self.assertAlmostEqual(drop[0], 0.0, places=9)
        self.assertAlmostEqual(x[-1], 157.08, places=6)
        self.assertAlmostEqual(drop[-1], 100.0, places=6)

    def test_monotone_within_the_half_arch(self):
        x, drop = cycloid_curve(math.pi * 50.0, 100.0, n=200)
        self.assertTrue(np.all(np.diff(x) >= -1e-12))
        self.assertTrue(np.all(np.diff(drop) >= -1e-12))

    def test_degenerate_vertical_case(self):
        x, drop = cycloid_curve(0.0, 100.0, n=50)
        np.testing.assert_allclose(x, 0.0)
        self.assertAlmostEqual(drop[-1], 100.0)


class TestThetaRatio(unittest.TestCase):
    """Endpoints past theta = pi must dip below the target and climb back."""

    def test_ratio_one_is_the_threshold(self):
        x_end, dip = endpoint_for_theta_ratio(1.0, 100.0)
        self.assertAlmostEqual(x_end, math.pi * 50.0, places=9)
        self.assertEqual(dip, 0.0)

    def test_dip_grows_past_the_threshold(self):
        dips = [endpoint_for_theta_ratio(r, 100.0)[1] for r in (1.05, 1.117, 1.216, 1.29)]
        self.assertTrue(all(d > 0 for d in dips))
        self.assertTrue(all(b > a for a, b in zip(dips, dips[1:])))

    def test_no_dip_below_the_threshold(self):
        for r in (0.3, 0.521, 0.862, 0.999):
            self.assertEqual(endpoint_for_theta_ratio(r, 100.0)[1], 0.0)

    def test_matches_the_plotted_endpoints(self):
        """The four curves from the comparison figure."""
        for ratio, expected in ((1.0, 157.08), (1.117, 200.11), (1.216, 250.10), (1.29, 300.25)):
            self.assertAlmostEqual(endpoint_for_theta_ratio(ratio, 100.0)[0], expected, places=1)

    def test_rejects_out_of_range(self):
        for bad in (0.0, -0.5, 2.0, 2.5):
            with self.assertRaises(ValueError):
                endpoint_for_theta_ratio(bad, 100.0)
        with self.assertRaises(ValueError):
            Config(theta_ratio=2.5)

    def test_rejects_theta_ratio_with_x_max(self):
        with self.assertRaises(TypeError):
            Config(theta_ratio=1.2, x_max=200.0)


class TestClimbing(unittest.TestCase):
    def setUp(self):
        self.sol = solve(Config(theta_ratio=1.29, x_step=5.0, y_step=5.0))
        self.seg = self.sol.path_segments(self.sol.shape[0] - 1, self.sol.iy_target)

    def test_target_row_is_not_the_last_row(self):
        self.assertLess(self.sol.iy_target, self.sol.shape[1] - 1)
        self.assertAlmostEqual(float(self.sol.y[self.sol.iy_target]), 0.0, places=9)

    def test_grid_has_headroom_below_the_target(self):
        self.assertGreater(self.sol.depth_below, 0.0)
        self.assertLess(float(self.sol.y[-1]), 0.0)

    def test_path_dips_below_the_target(self):
        lowest = min(self.seg["y1"].min(), self.seg["y2"].min())
        self.assertLess(lowest, -1e-9)

    def test_path_climbs_back_to_the_target(self):
        self.assertGreater(int((self.seg["dy"] > 1e-9).sum()), 0)
        self.assertAlmostEqual(float(self.seg["y2"][-1]), 0.0, places=9)

    def test_lowest_point_precedes_the_endpoint(self):
        """The question that motivated this: the minimum is not at the end."""
        i_low = int(np.argmin(self.seg["y1"]))
        self.assertLess(self.seg["x1"][i_low], self.seg["x2"][-1])

    def test_climbing_segments_have_negative_angles(self):
        climbs = self.seg[self.seg["dy"] > 1e-9]
        self.assertTrue(np.all(climbs["angle_deg"] < 0.0))

    def test_heading_axis_extends_negative(self):
        self.assertLess(float(self.sol.r_deg[0]), 0.0)
        self.assertAlmostEqual(float(self.sol.r_deg[0]), -90.0)
        self.assertAlmostEqual(float(self.sol.r_deg[-1]), 90.0)

    def test_speeds_still_follow_energy_conservation_below_zero(self):
        np.testing.assert_allclose(
            self.seg["v1"], np.sqrt(2 * 9.81 * (100.0 - self.seg["y1"])), atol=1e-9
        )

    def test_beats_the_monotone_solution(self):
        """Dipping must pay, or the extra states were pointless."""
        x_end, _ = endpoint_for_theta_ratio(1.29, 100.0)
        flat = solve(Config(x_max=x_end, x_step=5.0, y_step=5.0))
        t_flat, _ = flat.best_at(flat.shape[0] - 1, flat.iy_target)
        t_dip, _ = self.sol.best_at(self.sol.shape[0] - 1, self.sol.iy_target)
        self.assertLess(t_dip, t_flat)

    def test_refining_tightens_the_bound(self):
        from brachistochrone import analytic_brachistochrone

        x_end, _ = endpoint_for_theta_ratio(1.29, 100.0)
        t_ref, _ = analytic_brachistochrone(x_end, 100.0, 9.81)

        def err(step, nb):
            s = solve(Config(theta_ratio=1.29, x_step=step, y_step=step, neighbourhood_cells=nb))
            t, _ = s.best_at(s.shape[0] - 1, s.iy_target)
            return (t - t_ref) / t_ref

        self.assertLess(err(2.5, 6), err(5.0, 5))
        self.assertGreater(err(5.0, 5), 0.0)  # still an upper bound


class TestDefaultsUnchanged(unittest.TestCase):
    """The climbing machinery must stay dormant unless asked for."""

    def setUp(self):
        self.sol = solve(Config())

    def test_no_headroom_by_default(self):
        self.assertEqual(self.sol.depth_below, 0.0)
        self.assertEqual(self.sol.iy_target, self.sol.shape[1] - 1)

    def test_heading_axis_stays_non_negative(self):
        self.assertEqual(self.sol.shape[2], 10)
        self.assertAlmostEqual(float(self.sol.r_deg[0]), 0.0)

    def test_no_upward_moves_generated(self):
        self.assertEqual(sum(1 for _, di in _primitive_moves(5, allow_up=False) if di < 0), 0)
        self.assertGreater(sum(1 for _, di in _primitive_moves(5, allow_up=True) if di < 0), 0)

    def test_vertical_climbs_are_never_emitted(self):
        self.assertEqual(
            sum(1 for dj, di in _primitive_moves(5, allow_up=True) if dj == 0 and di < 0), 0
        )


class TestConfigValidation(unittest.TestCase):
    def test_rejects_inverted_altitudes(self):
        with self.assertRaises(ValueError):
            Config(h=0.0, y_end=100.0)

    def test_rejects_non_positive_dt(self):
        with self.assertRaises(ValueError):
            Config(dt=0.0)

    def test_configurable_end_altitude_shortens_the_drop(self):
        sol = solve(Config(h=100.0, y_end=40.0, x_step=5.0, y_step=5.0))
        self.assertAlmostEqual(float(sol.y[-1]), 40.0)
        t, _ = sol.best_at(0, sol.shape[1] - 1)
        self.assertAlmostEqual(t, math.sqrt(2 * 60.0 / 9.81), places=9)

    def test_initial_speed_enables_horizontal_launch(self):
        sol = solve(Config(h=100.0, v0=5.0, x_step=5.0, y_step=5.0))
        self.assertTrue(math.isfinite(sol.time[1, 0, 0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
