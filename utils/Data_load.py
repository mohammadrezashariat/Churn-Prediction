import pandas as pd


class Load_Dataset():
    """
    A class to load dataset from .csv

    Attributes:
        path (str): path of csv file

    Methods:
        load_data(): read csv and return the data
    """
    def __init__(self, path):
        """
        Initializes a new Load_Dataset object.

        Args:
            path (str): path of csv file
        """
        self.path = path

    def load_data(self):
        """
        read csv and return the data

        Returns:
           pandas.DataFrame: multiple columns of the data
        """
        return pd.read_csv(self.path)