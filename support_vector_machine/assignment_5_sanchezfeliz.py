"""
Script: assignment_5_sanchezfeliz.py
Description: Linear Support Vector Machine
Author: Genaro Sanchez Feliz
Date: 2025-06-10
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


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


class SVM:
    def __init__(self, features, label):

        # User input
        self.features = features
        self.label = label

        # Computed
        self.classes = list(label.unique())
        self.num_classes = len(self.classes)
        self.num_features = len(self.features.columns)
        self.ovr_sets = self.make_ovr_dataset()

        # Set hyperperparamters
        self.weights = np.zeros((self.num_classes, self.num_features))
        self.biases = np.zeros((self.num_classes))
        self.learning_rate = 0.001
        self.C = 1
        self.epochs = 1000
        self.cls_num_to_name = {i: cls for i, cls in enumerate(self.classes)}
        self.cls_name_to_num = {cls: i for i, cls in enumerate(self.classes)}

    def make_ovr_dataset(self):

        # Initialize one-vs-rest dataset
        ovr_sets = {}

        # Loop through each class
        for cls in self.classes:
            # Initialize class in dictionary
            ovr_sets[cls] = []
            # Loop through each point in the data
            for i in range(len(self.features)):
                # Get the x vector
                x = self.features.iloc[i].values
                # Get the label for that x vector
                lbl = self.label.iloc[i]
                # Evaluate y
                y = 1 if lbl == cls else -1
                ovr_sets[cls].append({'x': x, 'y': y})

        return ovr_sets

    def train(self):

        # Loop through each class
        for i in range(self.num_classes):
            cls = self.classes[i]
            # Initialize the class weight and bias
            cls_w = np.zeros(self.num_features)
            cls_b = 0

            # Loop through the data
            for epoch in range(self.epochs):
                # Loop through each point in the data
                for j in range(len(self.ovr_sets[cls])):
                    # Get my x vector and y binary value
                    x_i = self.ovr_sets[cls][j]['x']
                    y_i = self.ovr_sets[cls][j]['y']
                    # Get condition value
                    condition = y_i * (np.dot(cls_w, x_i) + cls_b)

                    if condition >= 1:
                        grad_w = cls_w
                        grad_b = 0
                    else:
                        grad_w = cls_w - self.C * y_i * x_i
                        grad_b = -self.C * y_i

                    # Update the class's w and b
                    cls_w -= self.learning_rate * grad_w
                    cls_b -= self.learning_rate * grad_b

            # Update the class's w and b in model
            self.weights[i] = cls_w
            self.biases[i] = cls_b

    def predict(self, X):
        """Takes in a numpy array X, and dot multiplies the
        weights (3 classes, 4 features) to the X data
        (n rows, 4 features). To do so X needs to be transposed
        so that it will take the form of (4 features, n rows).
        Finally you add the bias to the results to get the scores.
        Then you get the max of those scores.
        """
        scores = np.dot(self.weights, X.T).T + self.biases
        return np.argmax(scores, axis=1)

    def score(self, X, y):
        """Get the predictions, since the predictions are numbers,
        the function converts back to class labels, then checks if the
        predicted labels match their actual labels, and returns the mean
        of all these predictions"""
        preds = self.predict(X.to_numpy())
        pred_labels = [self.cls_num_to_name[i] for i in preds]
        return np.mean(pred_labels == y.to_numpy())


###############################################################################
# Main
###############################################################################

def main():

    # Read in iris data
    file_name = 'support_vector_machine/iris.data'
    iris_df = read_iris_data(file_name)

    # Split Data into X (independent) and Ys (dependent)
    X = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y = iris_df['class']

    # Keep track of the scores
    scores = []

    # Score the model 10 times
    for i in range(0, 20):
        # Split data into 70% training, 30% testing
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.3,
                                                            train_size=0.7)

        # Create the model
        iris_svm = SVM(X_train, y_train)

        # Train the model
        iris_svm.train()

        # Score and record the score of the model
        scores.append(iris_svm.score(X_test, y_test))

    # Print out the average and max score
    print("Average Score:", round(np.average(scores), 2))
    print("Max Score:", round(max(scores), 2))


###############################################################################
# Entry point
###############################################################################

if __name__ == '__main__':
    main()


"""
Using this simple vector machine,  the average score is around 66-70%
and the max score is around 84-90%, so in some cases the model does really well
however in other cases it does not do well. Of course, this is just a linear
classification model, for further improvements in the future we could try a
non-linear classification model as well. """
