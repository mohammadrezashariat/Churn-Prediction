from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import Config as c


class Classifier:
    """
    class to create machine learning model, train, save and test it.

    Attributes:
        df(pandas.Dataframe): csv file as pandas dataframe
        test_size(int): precent for split train and test set
        save_path(str): path for saving model

    Methods:
        __split_dataset(): Split Dataset into train and test
        train(): train model
        save_model(): to save model after training
        test(): to evaluate on test set
    """
    def __init__(self, df, save_path, test_size=0.2):
        self.df = df
        self.test_size = test_size
        self.save_path = save_path
        self.model = LogisticRegression(random_state=42, C=200, max_iter=1000)
        self.__split_dataset()

    def __split_dataset(self):
        """
        Split the preprocessed dataset into train and test sets, and set the corresponding attributes.
        """
        # Print message to indicate start of function
        print("Split Dataset into train and test")

        # Split the dataset into train and test sets using train_test_split function from scikit-learn
        # with the specified test_size and random_state.
        # stratify=self.df['Churn'] makes sure that the proportion of churners is same in both train and test sets.
        train_df, test_df = train_test_split(self.df, test_size=c.TEST_SIZE, random_state=42, stratify=self.df['Churn'])

        # Reset the index of the train and test sets.
        train_df.reset_index(drop=True, inplace=True)
        test_df.reset_index(drop=True, inplace=True)

        # Set the X_train, Y_train, X_test, Y_test attributes of the class.
        # X_train and X_test contain all columns except 'Churn', while Y_train and Y_test only contain 'Churn'.
        self.X_train = train_df.drop(columns=['Churn'])
        self.Y_train = train_df['Churn']
        self.X_test = test_df.drop(columns=['Churn'])
        self.Y_test = test_df['Churn']

        # Delete the original preprocessed dataframe and train_df and test_df to save memory.
        del self.df, train_df, test_df

    def train(self):
        """
        Train the Logistic Regression model and compute accuracy on train set.
        """
        print("train model")
        # Fit the model on the train data
        self.model.fit(self.X_train, self.Y_train)

        # Make predictions on the train set
        y_pred_train = self.model.predict(self.X_train)

        # Compute the accuracy on the train set
        acc_train = accuracy_score(self.Y_train, y_pred_train)
        print("+++  Accuracy on train set:", acc_train)

    def save_model(self):
        """
        Save the trained model to a file.
        """
        print(f"Saving model in {self.save_path}")
        joblib.dump(self.model, self.save_path)

    def load_model(self):
        """
        Load a trained model from the specified file path.

        Returns:
            loaded_model (object): The trained model loaded from the file.
        """
        loaded_model = joblib.load(self.save_path)
        return loaded_model

    def test(self):
        """
        Evaluate the accuracy of the trained model on the test dataset.

        Returns:
        --------
        accuracy : float
            The accuracy score on the test dataset.
        """

        # Predict the labels of the test dataset using the trained model
        y_predict = self.model.predict(self.X_test)

        # Compute the accuracy of the predicted labels against the true labels
        accuracy = accuracy_score(self.Y_test, y_predict)

        # Print the accuracy score on the test dataset
        print(f"Accuracy on test set: {accuracy}")

        return accuracy
