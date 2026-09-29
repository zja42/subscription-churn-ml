import unittest

import numpy as np
import pandas as pd

import train


class ChurnProjectTests(unittest.TestCase):
    def test_data_contract(self):
        X, y, customer_ids = train.load_data()
        self.assertEqual(len(X), 7043)
        self.assertEqual(len(customer_ids), 7043)
        self.assertEqual(int(y.sum()), 1869)
        self.assertEqual(set(X.columns), set(train.NUMERIC + train.CATEGORICAL))

    def test_value_threshold_is_on_grid(self):
        y = pd.Series([0, 0, 1, 1])
        probability = np.array([0.1, 0.4, 0.6, 0.9])
        threshold, table = train.choose_value_threshold(y, probability)
        self.assertTrue(0.05 <= threshold <= 0.80)
        self.assertEqual(len(table), 151)


if __name__ == "__main__":
    unittest.main()
