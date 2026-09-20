import unittest
import numpy as np
import pandas as pd
from src.data import INPUT_COLUMNS
from src.features import FeatureBuilder


class FeatureTests(unittest.TestCase):
    def test_saved_preprocessing_and_no_label_dependency(self):
        n = 12
        f = pd.DataFrame({c: np.full(n, 1.) for c in INPUT_COLUMNS})
        f['DEPTH'] = np.arange(n) + 500.
        f['AC'] = 200.
        f['DEN'] = 2.4
        f['GR'] = np.nan
        fit = FeatureBuilder().fit([f])
        before = fit.transform(f)
        restored = FeatureBuilder()
        restored.medians_ = dict(fit.medians_)
        restored.base_feature_names_ = fit.base_feature_names_
        restored.enhanced_feature_names_ = fit.enhanced_feature_names_
        f['POR'] = np.arange(n)
        after = restored.transform(f)
        np.testing.assert_array_equal(before.base, after.base)
        np.testing.assert_array_equal(before.enhanced, after.enhanced)
        self.assertTrue(np.isfinite(after.enhanced).all())
