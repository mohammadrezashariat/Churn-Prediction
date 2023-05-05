import pandas as pd
import numpy as np


class PreprocessData:
    """
    class to preprocess data before feed to machine learning model

    Attributes:
        path (str): path of csv file

    Methods:
        __load_data(): read csv and return the data
        __data_cleaning(): + fill NaN values with mean
                           + drop unnecessary columns
                           + replace meaningful values with useful values
        __feature_engineering(): Convert discrete values to bins of values
        __feature_scaling(): Calculates log(1 + x) for dataset
        __encoding_categorical_features(): turn every column values to categories
    """
    def __init__(self, path):
        """
        Initializes a new Preprocess_Data object.

        Args:
            path (str): path of csv file
        """
        self.path = path

    def __load_dataframe(self):
        """
        read csv file as pandas dataframe
        """
        print("Reading csv file")
        self.df = pd.read_csv(self.path)

    def __data_cleaning(self):
        """
        Cleans the data by dropping unnecessary columns, converting data types,
        replacing values, and filling null values with the mean of the column.
        """
        # Drop the 'customerID' column
        print("Cleaning data")
        self.df = self.df.drop(columns=['customerID'])

        # Convert the 'TotalCharges' column to numeric data type, and coerce any errors
        self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')

        # Replace the values 'No internet service' and 'No phone service' with 'No'
        self.df.replace(['No internet service', 'No phone service'], 'No', inplace=True)

        # Fill null values with the mean of the column
        for col in self.df.columns:
            if self.df[col].isna().sum() > 0:
                self.df[col].fillna(self.df[col].mean(), inplace=True)

    def __feature_engineering(self):
        """
        Features Engineered:
        - tenure_range: the range of the customer's tenure in
        years (0-1, 1-2, 2-3, 3-4, 4-5, more than 5).

        Returns:
            None
        """
        print("Feature Engineering")
        # Create conditions for each tenure range
        condition = [((self.df.tenure >= 0) & (self.df.tenure <= 12)),
                     ((self.df.tenure > 12) & (self.df.tenure <= 24)),
                     ((self.df.tenure > 24) & (self.df.tenure <= 36)),
                     ((self.df.tenure > 36) & (self.df.tenure <= 48)),
                     ((self.df.tenure > 48) & (self.df.tenure <= 60)),
                     (self.df.tenure > 60)]

        # Assign each tenure range a corresponding value
        # choice = ['0-1year','1-2years', '2-3years', '3-4years','4-5years','more than 5 years']
        choice = [0, 1, 2, 3, 4, 5]

        # Create a new column 'tenure_range' in the dataframe based on the conditions and choices
        self.df['tenure_range'] = np.select(condition, choice)

    def __feature_scaling(self):
        """
        Apply feature scaling to the 'MonthlyCharges' and 'TotalCharges' columns of the dataset
        by taking the natural logarithm of each value plus 1 to prevent negative values.
        """
        print("Feature Scaling")
        self.df['MonthlyCharges'] = np.log1p(self.df['MonthlyCharges'])
        self.df['TotalCharges'] = np.log1p(self.df['TotalCharges'])

    def __encoding_categorical_features(self):
        """
        Encoding the categorical features of the dataset.
        This method performs one-hot encoding on the categorical features of the dataset and drops the original
        categorical columns. It also converts the target variable `Churn` into binary format.
        """
        print("Encoding Features")
        # List of categorical columns
        cat_cols = ['gender', 'InternetService', 'PaymentMethod', 'Partner', 'Dependents', 'PhoneService',
                    'PaperlessBilling', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling']

        # Perform one-hot encoding on categorical columns
        self.df = pd.concat([self.df, pd.get_dummies(self.df[cat_cols])], axis='columns')

        # Drop the original categorical columns
        self.df = self.df.drop(columns=cat_cols)

        # Convert the target variable `Churn` into binary format
        self.df['Churn'] = np.where(self.df['Churn'] == 'Yes', 1, 0)

    def create_dataset(self):
        """
        Read and preprocess the customer churn dataset by
        performing data cleaning,
        encoding categorical features,
        feature engineering, and
        feature scaling.

        Returns:
            pd.DataFrame: Preprocessed dataset with encoded categorical features, feature engineered columns,
                          and scaled numerical columns.
        """
        self.__load_dataframe()
        self.__data_cleaning()
        self.__encoding_categorical_features()
        self.__feature_engineering()
        self.__feature_scaling()
        print("-- Dataset is ready!")

        return self.df
