import unittest
from utils import *
import Config as c


class TestChurnPrediction(unittest.TestCase):

    def setUp(self):
        match self._testMethodName:
            case 'test_create_dataset' :
                self.test_message = "1- Dataset created correctly"

            case 'test_model_training':
                self.test_message = "3- model training correctly"

            case 'test_model_evaluation':
                self.test_message = "4- model is evaluating correctly"

            case 'test_model_visualization':
                self.test_message = "5- resulats plot correctly"

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


    def test_model_training(self):
        # Check if the model is training correctly
        pass

    def test_model_evaluation(self):
        # Check if the model is evaluating correctly
        pass

    def test_model_visualization(self):
        # # Check if the resulats plot correctly
        pass

    def tearDown(self):
        print(self.test_message)



if __name__ == '__main__':
    unittest.main()

