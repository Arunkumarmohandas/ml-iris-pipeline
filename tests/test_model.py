import unittest
from model import train_model


class TestModel(unittest.TestCase):

    def test_model_training(self):
        print("Running test_model_training")
        model = train_model()
        self.assertIsNotNone(model)


if __name__ == "__main__":
    unittest.main()
