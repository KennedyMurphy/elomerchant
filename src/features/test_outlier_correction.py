import unittest
import pandas as pd
import numpy as np
import src.features.outlier_correction as oc

class TestFlagNormalOutliers(unittest.TestCase):

    def setUp(self):
        self.series = pd.Series(np.random.normal(0, 0.1, 1000))

        self.series.loc[500] = -5
        self.series.loc[42] = 5
    
    def test_flag(self):
        outliers = oc.flag_normal_outliers(self.series, 5)

        self.assertTrue(outliers.loc[500])
        self.assertTrue(outliers.loc[42])

class TestCorrectNormalOutliers(unittest.TestCase):

    def setUp(self):
        self.series = pd.Series(np.random.normal(0, 0.1, 1000))

        self.series.loc[500] = -5
        self.series.loc[42] = 5

    def test_correction(self):
        m, sd = self.series.agg(['mean', 'std'])

        corrected = oc.correct_normal_outliers(self.series.copy(), 5)

        self.assertEqual(m - 5 * sd, corrected.loc[500])
        self.assertEqual(m + 5 * sd, corrected.loc[42])

        self.assertEqual(len(corrected), len(self.series))