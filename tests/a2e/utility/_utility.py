import unittest
import numpy as np
from a2e.utility import grid_run, build_samples


class TestUtility(unittest.TestCase):

    def test_grid_run_2d(self):
        param_grid = {
            'a': ['a1', 'a2', 'a3'],
            'b': ['b1', 'b2'],
        }
        expected_param_list = [
            {'a': 'a1', 'b': 'b1'},
            {'a': 'a1', 'b': 'b2'},
            {'a': 'a2', 'b': 'b1'},
            {'a': 'a2', 'b': 'b2'},
            {'a': 'a3', 'b': 'b1'},
            {'a': 'a3', 'b': 'b2'},
        ]
        actual_param_list = []

        grid_run(param_grid, lambda x: actual_param_list.append(x))

        self.assertListEqual(actual_param_list, expected_param_list)

    def test_grid_run_3d(self):
        param_grid = {
            'a': ['a1', 'a2', 'a3'],
            'b': ['b1', 'b2'],
            'c': ['c1', 'c2'],
        }
        expected_param_list = [
            {'a': 'a1', 'b': 'b1', 'c': 'c1'},
            {'a': 'a1', 'b': 'b1', 'c': 'c2'},
            {'a': 'a1', 'b': 'b2', 'c': 'c1'},
            {'a': 'a1', 'b': 'b2', 'c': 'c2'},
            {'a': 'a2', 'b': 'b1', 'c': 'c1'},
            {'a': 'a2', 'b': 'b1', 'c': 'c2'},
            {'a': 'a2', 'b': 'b2', 'c': 'c1'},
            {'a': 'a2', 'b': 'b2', 'c': 'c2'},
            {'a': 'a3', 'b': 'b1', 'c': 'c1'},
            {'a': 'a3', 'b': 'b1', 'c': 'c2'},
            {'a': 'a3', 'b': 'b2', 'c': 'c1'},
            {'a': 'a3', 'b': 'b2', 'c': 'c2'},
        ]
        actual_param_list = []

        grid_run(param_grid, lambda x: actual_param_list.append(x))

        self.assertListEqual(actual_param_list, expected_param_list)

    def test_build_samples(self):
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        expected_samples_2 = [[1, 2], [3, 4], [5, 6], [7, 8]]
        expected_samples_3 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        actual_data_2 = build_samples(data, 2).tolist()
        actual_data_3 = build_samples(data, 3).tolist()

        self.assertListEqual(actual_data_2, expected_samples_2)
        self.assertListEqual(actual_data_3, expected_samples_3)


if __name__ == '__main__':
    unittest.main()
