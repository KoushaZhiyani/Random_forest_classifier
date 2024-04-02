from sklearn.model_selection import train_test_split
from math import sqrt
import ctypes
from decision_tree import Decision_Tree
import random

class Random_Forest:

    def __init__(self, min_sample=1, n_estimators=1, max_depth=5):
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
        # input()

        self.id_estimators = self.process_estimator()
        print(self.id_estimators)

    def process_estimator(self):
        id_list = []

        columns_selected = self.pick_random()
        X_tr = self.data[columns_selected]
        X_tr_sample, X_tst_sample, y_tr_sample, y_tst_sample = train_test_split(X_tr, self.target_data,
                                                                                    test_size=0.2, random_state=42)
        estimator = Decision_Tree(min_sample=self.min_sample, max_depht=self.max_depth)

        estimator.fit_model(X_tr_sample, y_tr_sample)

        id_list.append(id(estimator))

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
        # Iterate through test data rows
        for i in X_tst.iloc:
            for j in self.id_estimators:
                q = self.return_obj_from_id(j)
                print(q.data)
                input()

    @staticmethod
    def return_obj_from_id(id2):
        return ctypes.cast(id2, ctypes.py_object).value

