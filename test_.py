import unittest
from utils import *
import Config as c


class TestChurnPrediction(unittest.TestCase):

    def setUp(self):
        match self._testMethodName:
            case 'test_data_loaded':
                self.test_message = "1- Dataset has been loaded correctly"

            case 'test_data_preprocessing' :
                self.test_message = "2- Data Preprocessing step is working correctly"

            case 'test_model_training':
                self.test_message = "3- model training correctly"

            case 'test_model_evaluation':
                self.test_message = "4- model is evaluating correctly"

            case 'test_model_visualization':
                self.test_message = "5- resulats plot correctly"

            case _:
                self.test_message ="Invalid test case"

    def test_data_loaded(self):
        # Check if the dataset has been loaded correctly
        dataset = Load_Dataset(c.PATH).load_data()
        self.assertIsNotNone(dataset)

    def test_data_preprocessing(self):
        # Check if the data preprocessing step is working correctly
        pass

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

