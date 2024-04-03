import numpy as np
from sklearn.model_selection import train_test_split
from math import sqrt
from decision_tree import Decision_Tree  # Importing the Decision Tree class
import random


class Random_Forest:
    """
    Class implementing a Random Forest model.
    """

    def __init__(self, min_sample=1, n_estimators=3, max_depth=5, split_size=0.2):
        """
        Initialize the Random Forest model with specified parameters.

        Args:
        - min_sample (int): Minimum number of samples required to split a node. Default is 1.
        - n_estimators (int): Number of decision trees in the forest. Default is 3.
        - max_depth (int): Maximum depth of the decision trees. Default is 5.
        - split_size (float): Size of the test split when splitting the data. Default is 0.2.
        """
        self.id_estimators = None  # List to store the IDs of the individual decision trees
        self.n_estimators = n_estimators  # Number of decision trees in the forest
        self.max_depth = max_depth  # Maximum depth of each decision tree
        self.min_sample = min_sample  # Minimum number of samples required to split a node
        self.max_features = None  # Maximum number of features to consider for splitting
        self.data = None  # Training data
        self.target_data = None  # Target variable
        self.columns = None  # List of column names in the training data
        self.split_size = split_size  # Size of the test split when splitting the data

    def fit_model(self, X_tr, y_tr):
        """
        Fit the Random Forest model to the training data.

        Args:
        - X_tr (DataFrame): Features of the training data.
        - y_tr (DataFrame): Target variable of the training data.
        """
        self.data, self.target_data = X_tr, y_tr
        self.columns = self.data.columns
        self.max_features = int(sqrt(len(self.data.columns)))

        self.id_estimators = self.process_estimator()

    def process_estimator(self):
        """
        Process individual decision tree estimators for the Random Forest model.

        Returns:
        - id_list (list): List containing IDs of the individual decision tree estimators.
        """
        id_list = []
        for i in range(self.n_estimators):
            columns_selected = self.pick_random()  # Randomly select features
            X_tr = self.data[columns_selected]
            X_tr_sample, X_tst_sample, y_tr_sample, y_tst_sample = \
                train_test_split(X_tr, self.target_data, test_size=self.split_size, random_state=i)
            estimator = Decision_Tree(min_sample=self.min_sample, max_depht=self.max_depth)

            estimator.fit_model(X_tr_sample, y_tr_sample)

            id_list.append(estimator.DT_id)

        return id_list

    def pick_random(self):
        """
        Randomly select a subset of features for each decision tree estimator.

        Returns:
        - columns_selected (list): List of column names representing the selected features.
        """
        index_list = [i for i in range(len(self.columns))]
        columns_selected = []
        for i in range(self.max_features):
            col_index = random.choice(index_list)
            index_list.remove(col_index)

            columns_selected.append(self.columns[col_index])
        return columns_selected

    def make_predict(self, X_tst):
        """
        Make predictions using the Random Forest model.

        Args:
        - X_tst (DataFrame): Features of the test data.

        Returns:
        - predict_list (list): List of predicted values.
        """
        predict_list = []
        predict_list_total_estimators = np.array(self.prediction_process(X_tst))
        predict_list_reformed = self.reformed_list(predict_list_total_estimators)

        for i in predict_list_reformed:
            unique, counts = np.unique(i, return_counts=True)
            unique_counts_dict = {unique[j]: counts[j] for j in range(len(unique))}
            predict_sample = [j for j in unique_counts_dict
                              if unique_counts_dict[j] == max(unique_counts_dict.values())][0]
            predict_list.append(predict_sample)
        return predict_list

    def prediction_process(self, X_tst):
        """
        Process predictions from individual decision tree estimators.

        Args:
        - X_tst (DataFrame): Features of the test data.

        Returns:
        - predict_list (list): List of prediction lists from individual decision tree estimators.
        """
        predict_list = []
        for estimator in self.id_estimators:
            estimator_predict_list = estimator.make_predict(X_tst)

            predict_list.append(estimator_predict_list)

        return predict_list

    @staticmethod
    def reformed_list(basic_list):
        """
        Reform a basic list into a list of lists.

        Args:
        - basic_list (list): Basic list to be reformed.

        Returns:
        - list_reformed (list): Reformatted list.
        """
        list_reformed = []
        for sample_index in range(basic_list.shape[1]):
            prediction_of_each_sample = []
            for estimator in basic_list:
                prediction_of_each_sample.append(estimator[sample_index])
            list_reformed.append(prediction_of_each_sample)

        return list_reformed

    def score(self, y_pre, y_true):
        """
        Calculate the accuracy score of the Random Forest model.

        Args:
        - y_pre (list): Predicted values.
        - y_true (DataFrame): True values of the target variable.

        Returns:
        - score (float): Accuracy score of the model.
        """
        zipped_data = zip(y_pre, y_true[self.target_data.columns[0]].values)

        correct_predict, wrong_predict = 0, 0
        for i in zipped_data:

            if int(i[0]) == int(i[1]):
                correct_predict += 1
            else:
                wrong_predict += 1
        score = correct_predict / (correct_predict + wrong_predict)
        return score
