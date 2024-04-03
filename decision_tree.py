import pandas as pd
import warnings

warnings.filterwarnings("ignore")


class Decision_Tree:
    def __init__(self, min_sample=None, max_depht=None, depth=0):
        self.data = None  # data
        self.target_data = None  # target feature

        self.columns = None  # columns name
        self.classes = []
        self.depth = depth  # current depth
        self.leaf = False  # leaf check

        self.gini_list = []
        self.minimum_gini = []  # minimum gini in Node

        self.left_link = None
        self.right_link = None

        self.max_depht = max_depht  # maximum tree depth
        self.min_sample = min_sample  # minimum sample in Node

        self.DT_id = self
    # process feature to find minimum gini

    def fit_model(self, X_tr, y_tr):        
        self.data = X_tr
        self.target_data = y_tr
        self.columns = self.data.columns
        self.find_classes()
        if self.max_depht is not None:
            if (self.depth > self.max_depht - 1 and self.max_depht != -1) or self.is_leaf() == 0:
                self.leaf = True
                return
        else:
            if self.is_leaf() == 0:
                self.leaf = True
                return

        self.gini_list = []
        for i in self.data.columns:
            self.subset_size_check(i)
        if len(self.gini_list) == 0:
            self.leaf = True
            return

        self.find_minimum_gini()

        self.split_data()

    # Get the size of two Nodes
    def find_classes(self):
        for i in self.target_data.iloc:
            # print(i)
            # input()
            if i[0] not in self.classes:
                self.classes.append(i[0])
        self.classes.sort()

    def subset_size_check(self, col):

        for i in range(round(min(self.data[col])) - 1, round(max(self.data[col]) + 2)):

            first = []
            second = []
            for j in self.data.iloc:

                if j[col] <= i:
                    first.append(self.target_data.loc[j.name][0])
                else:
                    second.append(self.target_data.loc[j.name][0])
            if len(second) > int(self.min_sample) and len(first) > int(self.min_sample):
                self.gini_score(first, second, i, col)

    def gini_score(self, lst_pos, lst_neg, base, col):
        total = 0

        for i in lst_pos, lst_neg:
            label_each_col = {label: len([j for j in i if j == label]) for label in self.classes}
            total_label = sum(label_each_col.values())

            gini_score_i = (total_label / len(self.data)) * \
                           (1 - sum((label / total_label) ** 2 for label in label_each_col.values()))

            total += gini_score_i
        self.gini_list.append((base, col, total))

    def split_data(self):

        first = []
        second = []
        for i in self.data.iloc():

            if float(i[self.minimum_gini[1]]) <= float(self.minimum_gini[0]):
                first.append(i)
            else:
                second.append(i)

        first = pd.DataFrame(first, columns=self.columns)
        second = pd.DataFrame(second, columns=self.columns)

        target_first = self.target_data.drop(second.index, axis=0)
        target_second = self.target_data.drop(first.index, axis=0)

        left_node = Decision_Tree(min_sample=self.min_sample, max_depht=self.max_depht, depth=self.depth + 1)
        right_node = Decision_Tree(min_sample=self.min_sample, max_depht=self.max_depht, depth=self.depth + 1)
        self.left_link = left_node
        self.right_link = right_node
        left_node.fit_model(first, target_first)
        right_node.fit_model(second, target_second)

    def is_leaf(self):
        # Calculate the Gini score for the current node
        label_each_col = {label: len([j[0] for j in self.target_data.values if j == label]) for label in self.classes}
        total_label = sum(label_each_col.values())

        gini_score_i = (total_label / len(self.data)) * \
                       (1 - sum((label / total_label) ** 2 for label in label_each_col.values()))

        return gini_score_i  # Return the calculated Gini score

    def find_minimum_gini(self):
        # Define column names for the DataFrame
        columns = ["base", "col_name", "score"]
        # Create a DataFrame from the gini_list
        self.gini_list = pd.DataFrame(self.gini_list, columns=columns)
        # Find the index of the minimum Gini score
        minimum_gini_index = self.gini_list["score"].idxmin(axis=0)
        # Get the minimum Gini values
        self.minimum_gini = self.gini_list.loc[minimum_gini_index]

    def make_predict(self, X_tst):
        predict_list = []
        # Iterate through test data rows
        for i in X_tst.iloc:
            q = self
            # Traverse the decision tree until a leaf node is reached
            while not q.leaf:
                # Get the column and base value for splitting
                columns = q.minimum_gini[1]
                base_label = q.minimum_gini[0]
                # Traverse left or right based on the split condition
                if float(i[columns]) <= float(base_label):
                    q = q.left_link
                else:
                    q = q.right_link
            # Count the occurrences of each class label in the leaf node
            label_each_col = {label: len([j[0] for j in q.target_data.values if j == label]) for label in
                              self.classes}
            # Predict the class with the maximum occurrences
            predict = list(filter(lambda key: label_each_col[key] == max(label_each_col.values()), label_each_col))[0]
            predict_list.append(predict)
        return predict_list
