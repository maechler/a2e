import unittest
from functools import partial
from a2e.model.keras import create_deep_feed_forward_autoencoder


class TestUtility(unittest.TestCase):

    def test_create_deep_feed_forward_autoencoder_shallow(self):
        model = create_deep_feed_forward_autoencoder(100, number_of_hidden_layers=1, compression_per_layer=0.5)

        self.assertEqual(len(model.layers), 3)
        self.assertEqual(model.layers[0].input_shape[0][1], 100)
        self.assertEqual(model.layers[1].output_shape[1], 50)
        self.assertEqual(model.layers[2].output_shape[1], 100)

    def test_create_deep_feed_forward_autoencoder_deep(self):
        model = create_deep_feed_forward_autoencoder(120, number_of_hidden_layers=3, compression_per_layer=0.5)

        self.assertEqual(len(model.layers), 5)
        self.assertEqual(model.layers[0].input_shape[0][1], 120)
        self.assertEqual(model.layers[1].output_shape[1], 60)
        self.assertEqual(model.layers[2].output_shape[1], 30)
        self.assertEqual(model.layers[3].output_shape[1], 60)
        self.assertEqual(model.layers[4].output_shape[1], 120)

    def test_create_deep_feed_forward_autoencoder_invalid(self):
        self.assertRaises(ValueError, partial(create_deep_feed_forward_autoencoder, 100, number_of_hidden_layers=2))
        self.assertRaises(ValueError, partial(create_deep_feed_forward_autoencoder, 100, number_of_hidden_layers=40))


if __name__ == '__main__':
    unittest.main()
