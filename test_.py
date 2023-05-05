import unittest
import os
from utils import *
import Config as c


class TestChurnPrediction(unittest.TestCase):

    def setUp(self):
        df = PreprocessData(c.PATH).create_dataset()
        self.model = Classifier(df, c.save_path)

        match self._testMethodName:
            case 'test_create_dataset' :
                self.test_message = " >>>>> Dataset created correctly"

            case 'test_train':
                self.test_message = " >>>>> model training and load both correct"

            case 'test_save_model':
                self.test_message = " >>>>> model saved correctly"

            case 'test_test':
                self.test_message = " >>>>> model accuracy greater than 75%"

            case 'test_model_visualization':
                self.test_message = " >>>>> Results plot correctly"

            case _:
                self.test_message ="Invalid test case"

    def test_create_dataset(self):
        df = PreprocessData(c.PATH).create_dataset()
        # Check dataset shape
        self.assertEqual(df.shape, (7043, 42))
        # Check if all columns are numeric of boolean
        expected = [np.dtype('int64'), np.dtype('bool'), np.dtype('float64'), np.dtype('int32')]
        unique_types = list(set(df.dtypes))
        self.assertEqual(sorted(unique_types), sorted(expected))

        # Check if there are no NaN values
        self.assertFalse(df.isna().any().any())

        # Check if there are no empty values
        self.assertFalse(df.isin(['', ' ']).any().any())

    def test_train(self):
        self.model.train()
        loaded_model = self.model.load_model()
        self.assertTrue(isinstance(loaded_model, LogisticRegression))

    def test_save_model(self):
        self.model.train()
        self.model.save_model()
        self.assertTrue(os.path.exists(c.save_path))

    def test_accuracy(self):
        """
        Test the model on the test set and check if accuracy is greater than 0.75
        """
        self.model.train()
        accuracy = self.model.test()
        self.assertTrue(accuracy > 0.75, f"Test accuracy ({accuracy}) is not greater than 75%")

    def test_model_visualization(self):
        # # Check if the resulats plot correctly
        pass

    def tearDown(self):
        print(self.test_message)



if __name__ == '__main__':
    unittest.main()

