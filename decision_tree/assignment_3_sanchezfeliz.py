"""
Genaro Sanchez Feliz
gsanche4@ramapo.edu
CMPS-320-50
Assignment 3 - Decision Tree
Jun 4th, 2025
"""

import pandas as pd
import numpy as np
import math
from sklearn.model_selection import train_test_split


# Create Decision Tree Class
class DecisionTree():
    def __init__(self):
        self.tree = None

    def fit(self, df, target):
        self.tree = self._build_tree(df, target)
        self.default_class = df[target].mode()[0]

    def predict(self, query):
        return self._predict(query, self.tree)

    def _build_tree(self, df, target):
        # Base Case, if all the values in your target class
        # are the same, then returns that same value.
        unique_classes = np.unique(df[target])
        if len(unique_classes) == 1:
            return unique_classes[0]

        # Base Case, if there are no features in the data frame
        # besides the target column, then just returns the mode.
        if len(df.columns) == 1:
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
            subtree = self._build_tree(subset, target)
            tree[best_attribute][val] = subtree

        return tree

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

    def _predict(self, query, tree):
        for key in query.keys():
            if key in tree:
                value = query[key]
                result = tree[key].get(value)

                if isinstance(result, dict):
                    return self._predict(query, result)
                else:
                    return result
        return None


def main():
    # Import Data
    cancer_csv = pd.read_csv('decision_tree/breast-cancer.data',
                             header=None,
                             names=['class', 'age', 'menopause',
                                    'tumor-size', 'inv-nodes',
                                    'node-caps', 'deg-malig',
                                    'breast', 'breast-quad', 'irradit'],
                             usecols=['class', 'age', 'tumor-size',
                                      'node-caps', 'deg-malig'])

    # Split the data into 70% training, 30% testing
    train_df, test_df = train_test_split(cancer_csv,
                                         test_size=0.30,
                                         random_state=0)

    # Create the decision tree
    cancer_tree = DecisionTree()

    # Fit the decision tree for the target 'class'
    cancer_tree.fit(train_df, target='class')

    # Predict Values in the test data frame
    correct = 0
    total = 0

    for index, row in test_df.iterrows():
        query = row.drop('class').to_dict()
        actual = row['class']
        predicted = cancer_tree.predict(query)
        if predicted == actual:
            correct += 1
        total += 1

    # Calculate accuracy
    accuracy = correct / total
    print(f"Decision Tree Algorithm Accuracy: {accuracy:.2f}")


if __name__ == "__main__":
    main()
