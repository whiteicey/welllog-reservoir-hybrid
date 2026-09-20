import unittest
import numpy as np
from src.decision import utility_action, por_prediction
from src.metrics import official_target_score


class DecisionTests(unittest.TestCase):
    def test_optimum_at_all_piecewise_linear_breakpoints(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            y = np.sort(rng.uniform(.1, 30, 12))
            p = rng.dirichlet(np.ones(len(y)), size=6)
            width = .08 * (y + 1e-5)
            actions = np.r_[y - width, y, y + width]
            all_utilities = np.maximum(0., 1. - abs(actions[:, None] - y) / width) @ p.T
            chosen = utility_action(p, y, chunk_size=2)
            actual = (p * np.maximum(0., 1. - abs(chosen[:, None] - y) / width)).sum(axis=1)
            np.testing.assert_allclose(actual, all_utilities.max(axis=0), atol=1e-12)

    def test_beats_argmax_on_multimodal_example(self):
        p = np.array([[.4, .3, .3]])
        y = np.array([5., 10., 10.1])
        self.assertGreater(utility_action(p, y)[0], 9.)
        self.assertEqual(y[p.argmax()], 5.)

    def test_kernel_matches_official_metric(self):
        y = np.array([.1, 1., 10., 10.5])
        p = np.array([[.1, .2, .3, .4]])
        pred = utility_action(p, y)[0]
        replicated = np.repeat(y, [1, 2, 3, 4])
        score = official_target_score(replicated, np.full(10, pred), 'POR')
        expected = (p[0] * np.maximum(0., 1. - abs(pred-y)/(.08*(y+1e-5)))).sum()
        self.assertAlmostEqual(score, expected)

    def test_joint_mass_and_legacy_gate(self):
        p = np.array([[.4, .3, .3], [.4, .3, .3]])
        centers = np.log1p([5., 10., 10.1])
        cfg = {'threshold': .5, 'scale': 1., 'offset': 0.}
        np.testing.assert_allclose(por_prediction(p, [1., 0.], centers, cfg, mode='joint'), [.1, 10.])
        np.testing.assert_allclose(por_prediction(p, [.5, .49], centers, cfg, mode='argmax'), [.1, 5.])

    def test_invalid_inputs_fail(self):
        for p in ([[np.nan]], [[-1.]], [[0.]]):
            with self.assertRaises(ValueError):
                utility_action(p, [1.])
        with self.assertRaises(ValueError):
            por_prediction([[.2]], [.5], [1.], {}, mode='joint')

    def test_empty_batch_and_duplicate_support(self):
        self.assertEqual(utility_action(np.empty((0, 2)), [1., 2.]).shape, (0,))
        np.testing.assert_allclose(utility_action([[.3, .7]], [2., 2.]), [2.])


if __name__ == '__main__':
    unittest.main()
