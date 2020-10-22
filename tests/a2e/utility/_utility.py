import unittest
from a2e.utility import grid_run


class TestUtility(unittest.TestCase):

    def test_grid_run(self):
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

        self.assertEquals(actual_param_list, expected_param_list)


if __name__ == '__main__':
    unittest.main()
