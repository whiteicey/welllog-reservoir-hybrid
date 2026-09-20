import unittest
import numpy as np
from train import make_por_bins
from validate_por import summarize


class ValidationTests(unittest.TestCase):
    def test_quantile_ties_have_finite_centers(self):
        nondefault, classes, centers = make_por_bins(np.r_[.1, np.full(100, 1.), np.full(100, 2.), 3.])
        self.assertFalse(nondefault[0])
        self.assertTrue(np.isfinite(centers).all())
        self.assertTrue(np.all((classes >= 0) & (classes < len(centers))))

    def test_degenerate_labels_rejected(self):
        with self.assertRaises(ValueError):
            make_por_bins(np.full(20, 1.))

    def test_macro_micro_and_paired_delta(self):
        y = np.ones(4)
        pred = np.array([1., 1., 1., 2.])
        wells = np.array(['long', 'long', 'long', 'short'])
        r = summarize(y, {'argmax': pred, 'utility': y.copy()}, wells, np.array([0, 0, 0, 1]))
        self.assertEqual(r['argmax']['micro'], .75)
        self.assertEqual(r['argmax']['macro'], .5)
        self.assertEqual(r['utility']['total_delta_if_other_targets_unchanged'], 7.5)
        self.assertEqual(r['utility']['well_wins'], 1)
