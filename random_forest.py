import numpy as np
from sklearn.model_selection import train_test_split
from math import sqrt
from decision_tree import Decision_Tree
import random


class Random_Forest:

    def __init__(self, min_sample=1, n_estimators=3, max_depth=5):
        self.id_estimators = None
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_sample = min_sample
        self.max_features = None
        self.data = None  # data
        self.target_data = None  # target feature
        self.columns = None

    def fit_model(self, X_tr, y_tr):
        self.data, self.target_data = X_tr, y_tr
        self.columns = self.data.columns
        self.max_features = int(sqrt(len(self.data.columns)))

        self.id_estimators = self.process_estimator()

    def process_estimator(self):
        id_list = []
        for i in range(self.n_estimators):
            columns_selected = self.pick_random()
            X_tr = self.data[columns_selected]
            # X_tr_sample, X_tst_sample, y_tr_sample, y_tst_sample = train_test_split(X_tr, self.target_data,
            #                                                                         test_size=0.0, random_state=i)
            estimator = Decision_Tree(min_sample=self.min_sample, max_depht=self.max_depth)

            estimator.fit_model(X_tr, self.target_data)

            id_list.append(estimator.DT_id)

        return id_list

    def pick_random(self):

        index_list = [i for i in range(len(self.columns))]
        columns_selected = []
        for i in range(self.max_features):
            col_index = random.choice(index_list)
            index_list.remove(col_index)

            columns_selected.append(self.columns[col_index])
        return columns_selected

    def make_predict(self, X_tst):
        predict_list = []
        predict_list_total_estimators = self.prediction_process(X_tst)
        predict_list_reformed = np.array(predict_list_total_estimators).reshape(-1, self.n_estimators)

        for i in predict_list_reformed:
            unique, counts = np.unique(i, return_counts=True)
            predict_sample = np.argmax(counts)
            predict_list.append(predict_sample)
        # print(predict_list)
        return predict_list

    def prediction_process(self, X_tst):
        predict_list = []
        for estimator in self.id_estimators:
            estimator_predict_list = estimator.make_predict(X_tst)
            predict_list.append(estimator_predict_list)

        return predict_list

    def score(self, y_pre, y_true):
        # Zip the predicted and true labels together
        zipped_data = zip(y_pre, y_true[self.target_data.columns[0]].values)
        correct_predict, wrong_predict = 0, 0
        # Iterate through the zipped data
        for i in zipped_data:
            # Check if the predicted and true labels match
            if int(i[0]) == int(i[1]):
                correct_predict += 1
            else:
                wrong_predict += 1
        # Calculate and return the score
        score = correct_predict / (correct_predict + wrong_predict)
        return score
