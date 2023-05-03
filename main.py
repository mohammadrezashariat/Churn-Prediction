from utils import *
import Config as c


def predict_churn():
    """
    Predict whether a customer will churn or not based on their demographic and contract information.
    """
    # Load the dataset
    dataset = Load_Dataset(c.PATH).load_data()

    # Preprocess the data
    X_train, Y_train, X_test, Y_test = Preprocess_Data(dataset).prepro()

    # Train the model
    model = Train_Model(X_train, Y_train).train()

    # get accuracy
    accuracy = Evaluate_Model(model, X_test, Y_test).eval

    Plot_Results().plot()


if __name__ == '__main__':
    predict_churn()