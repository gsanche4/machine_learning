"""
Script: project_1_sanchezfeliz.py
Description: Random Forest
Author: Genaro Sanchez Feliz
Date: 2025-06-13
"""

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import math
import random


###############################################################################
# Helper functions
###############################################################################
def read_iris_data(file_name):
    """
    Takes in iris.data file path string,
    reads in the data and renames the columns,
    and returns the df
    """
    df = pd.read_csv(file_name, header=None)
    df = df.rename(columns={0: 'sepal_length',
                            1: 'sepal_width',
                            2: 'petal_length',
                            3: 'petal_width',
                            4: 'class'})
    return df


def test_model(df, target, model):

    # Track the correct and total number of predictions
    correct = 0
    total = 0

    # Loops through each point, and predicts the y
    # then checks it against the actual y.
    for index, row in df.iterrows():
        query = row.drop(target).to_dict()
        actual = row[target]
        predicted = model.predict(query)
        if predicted == actual:
            correct += 1
        total += 1

    # Return the accuracy
    return correct / total


class DecisionTree():
    def __init__(self, max_depth=5):
        self.tree = None
        self.max_depth = max_depth

    def fit(self, df, target):
        self.default_class = df[target].mode()[0]
        self.tree = self._build_tree(df, target)

    def predict(self, query):
        return self._predict(query, self.tree)

    def _calculate_entropy(self, column):
        """Takes in a list, column of a dataframe, or array
        and returns the entropy."""
        # Get array of elements and counts
        elements, counts = np.unique(column, return_counts=True)

        # Initializing entropy
        entropy = 0

        # Loop through each element
        for i in range(len(elements)):

            # Calculate the probability of that element occuring
            prob = counts[i] / np.sum(counts)

            # Update entropy vairable
            entropy -= prob * math.log2(prob)

        return entropy

    def _calculate_info_gain(self, df, attribute, target):
        # Entropy for target column
        total_entropy = self._calculate_entropy(df[target])

        # Get all unique values, and the count of occurances
        vals, counts = np.unique(df[attribute], return_counts=True)

        # Initialize weighted entropy
        weighted_entropy = 0

        # For every unique value in that column
        for i in range(len(vals)):

            # Get a subset df where that unique value exists
            subset = df[df[attribute] == vals[i]]

            # Get the entropy of the subset
            subset_entropy = self._calculate_entropy(subset[target])

            # Multiply the entropy by its weight, and then add it to the
            # weighted entropy
            weighted_entropy += (counts[i] / np.sum(counts)) * subset_entropy

        # Return the difference between the total entropy
        # and the weighetd entropy
        return total_entropy - weighted_entropy

    def _build_tree(self, df, target, depth=0):

        # Get all the unique classes
        unique_classes = np.unique(df[target])

        # If there is only one class, then just return that class
        if len(unique_classes) == 1:
            return unique_classes[0]

        # Base Case, if there are no features in the data frame
        # besides the target column, then just returns the mode.
        if len(df.columns) == 1:
            return df[target].mode()[0]

        # Stops the model from having too many branches and overfitting
        if depth >= self.max_depth:
            return df[target].mode()[0]

        # Calculate info gain for all attributes
        info_gain = {}
        for attribute in df.columns:
            # Excluding the target column
            if attribute != target:
                # Calculate the information gain for that attribute
                info_gain[attribute] = self._calculate_info_gain(df, attribute,
                                                                 target)

        # Get the attribute with the most information gain
        best_attribute = max(info_gain, key=info_gain.get)
        tree = {best_attribute: {}}

        for val in np.unique(df[best_attribute]):
            subset = df[df[best_attribute] == val]
            subset = subset.drop(columns=[best_attribute])

            # Recursively build subtree until all the unique values
            # in the target class are the same, then it will return
            # that value.
            subtree = self._build_tree(subset, target, depth + 1)
            tree[best_attribute][val] = subtree

        return tree

    def _predict(self, query, tree):
        for key in query.keys():
            if key in tree:
                value = query[key]
                branches = tree[key]

                if value not in branches:
                    return self.default_class

                result = branches[value]

                if isinstance(result, dict):
                    return self._predict(query, result)
                else:
                    return result

        # If no class, return the default
        return self.default_class


class RandomForest():
    def __init__(self, num_trees=10):
        self.num_trees = num_trees
        self.forest = []

    def fit(self, df, target):
        self._build_forest(df, cls_target=target)

    def predict(self, query):
        predictions = []

        for tree, default_class in self.forest:
            pred = self._predict_tree(query, tree, default_class=default_class)
            predictions.append(pred)

        # Find the aggregate or the most common
        return max(set(predictions), key=predictions.count)

    def _calculate_entropy(self, column):
        """Takes in a list, column of a dataframe, or array
        and returns the entropy."""
        # Get array of elements and counts
        elements, counts = np.unique(column, return_counts=True)

        # Initializing entropy
        entropy = 0

        # Loop through each element
        for i in range(len(elements)):

            # Calculate the probability of that element occuring
            prob = counts[i] / np.sum(counts)

            # Update entropy vairable
            entropy -= prob * math.log2(prob)

        return entropy

    def _calculate_info_gain(self, df, attribute, target):
        # Entropy for target column
        total_entropy = self._calculate_entropy(df[target])

        # Get all unique values, and the count of occurances
        vals, counts = np.unique(df[attribute], return_counts=True)

        # Initialize weighted entropy
        weighted_entropy = 0

        # For every unique value in that column
        for i in range(len(vals)):

            # Get a subset df where that unique value exists
            subset = df[df[attribute] == vals[i]]

            # Get the entropy of the subset
            subset_entropy = self._calculate_entropy(subset[target])

            # Multiply the entropy by its weight, and then add it to the
            # weighted entropy
            weighted_entropy += (counts[i] / np.sum(counts)) * subset_entropy

        # Return the difference between the total entropy
        # and the weighetd entropy
        return total_entropy - weighted_entropy

    def _build_tree(self, df, target, depth=0, max_depth=5):

        default_class = df[target].mode()[0]

        # Get all the unique classes
        unique_classes = np.unique(df[target])

        # If there is only one class, then just return that class
        if len(unique_classes) == 1:
            return unique_classes[0], default_class

        # Base Case, if there are no features in the data frame
        # besides the target column, then just returns the mode.
        if len(df.columns) == 1:
            return df[target].mode()[0], default_class

        # Stops the model from having too many branches and overfitting
        if depth >= max_depth:
            return df[target].mode()[0], default_class

        # Get the random features
        n_features = int(math.sqrt(len(df.columns)))
        feature_cols = [col for col in df.columns if col != target]
        features = random.sample(feature_cols, n_features)

        # Calculate info gain for all attributes
        info_gain = {}
        for attribute in features:
            info_gain[attribute] = self._calculate_info_gain(df, attribute,
                                                             target)

        # Get the attribute with the most information gain
        best_attribute = max(info_gain, key=info_gain.get)
        tree = {best_attribute: {}}

        for val in np.unique(df[best_attribute]):
            subset = df[df[best_attribute] == val]
            subset = subset.drop(columns=[best_attribute])

            # Recursively build subtree until all the unique values
            # in the target class are the same, then it will return
            # that value.
            subtree, _ = self._build_tree(subset, target, depth + 1)
            tree[best_attribute][val] = subtree

        return (tree, default_class)

    def _predict_tree(self, query, tree, default_class):
        for key in query.keys():
            if key in tree:
                value = query[key]
                branches = tree[key]

                if value not in branches:
                    return default_class

                result = branches[value]

                if isinstance(result, dict):
                    return self._predict_tree(query, result, default_class)
                else:
                    return result

        # If no class, return the default
        return default_class

    def _get_bootstrap(self, df, sample_size=None):

        # Get the length of the data frame
        n = len(df)

        if sample_size is None:
            sample_size = int(2 * (n / 3))

        # Gets a list of random integers with replacement
        rand_ints = np.random.choice(n, size=sample_size, replace=True)

        # Returns all the rows that correlate to those random integers
        return df.iloc[rand_ints].reset_index(drop=True)

    def _build_forest(self, df, cls_target):

        for _ in range(self.num_trees):
            # Get the bootstrap df
            bootstrap_df = self._get_bootstrap(df)

            # Create a new decision tree
            tree = self._build_tree(bootstrap_df, cls_target)

            # Append the tree to forest
            self.forest.append(tree)


###############################################################################
# Main
###############################################################################
def main():

    # Read in iris data
    file_name = 'random_tree/iris.data'
    iris_csv = read_iris_data(file_name)

    # Create Decision Tree Model
    iris_dt = DecisionTree()

    # Split the data into 70% training, 30% testing
    train_df, test_df = train_test_split(iris_csv, test_size=0.30,
                                         random_state=42)

    # Fit the Decision Tree Model
    iris_dt.fit(train_df, target='class')

    # Test the Decision Tree
    dt_accuracy = round(test_model(test_df, 'class', iris_dt), 4) * 100
    print(f"Decision Tree Accuracy: {dt_accuracy}")

    # Create the Random Foreset
    iris_rand_f = RandomForest()

    # Fit the Random Forest
    iris_rand_f.fit(train_df, 'class')

    # Iris predict
    rand_f_accuracy = round(test_model(test_df, 'class', iris_rand_f), 4) * 100
    print(f"Random Forest Accuracy: {rand_f_accuracy}")


###############################################################################
# Entry point
###############################################################################

if __name__ == '__main__':
    main()


"""The accuracy for the random forrest is better than the decision tree alone.
"""
