from utils import *
import Config as c


def predict_churn():
    """
    Predict whether a customer will churn or not based on their demographic and contract information.
    """
    # Load and preprocess the data
    df = PreprocessData(c.PATH).create_dataset()

    # Train the model
    model = Classifier(df, c.save_path)
    model.train()
    model.save_model()
    # Test the model
    model.test()

    model.plot_performance()


if __name__ == '__main__':
    predict_churn()
