"""
Genaro Sanchez Feliz
gsanche4@ramapo.edu
CMPS-320-50
Assignment 2 - Logistic Regression
May 30th, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# --- LogReg Class --- #
class LogReg:
    def __init__(self, X, y, learning_rate=0.01,
                 max_iterations=100000, max_error=0.125):
        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.max_error = max_error
        self.m = len(X)

        # Initialize the weights and tracking
        self.w1 = 0
        self.w0 = 0
        self.error = 1.0
        self.error_values = {'iteration': [], 'value': []}

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def predict(self, X_new):
        z = (self.w1 * X_new) + self.w0
        y_hat = self.sigmoid(z)
        return (y_hat > 0.5).astype(int)

    def calc_error(self, predic_vals, y):
        m = len(predic_vals)
        val_sum = np.sum(np.abs(np.array(predic_vals) - y))
        return val_sum / m

    def train_model(self):
        iteration = 0
        self.error = 1.0

        while self.error > self.max_error and iteration < self.max_iterations:
            predictions = []

            for x_val, y_val in zip(self.X, self.y):
                z = (self.w1 * x_val) + self.w0
                y_hat = self.sigmoid(z)

                # Update the weights
                error = y_hat - y_val
                self.w1 -= self.learning_rate * error * x_val
                self.w0 -= self.learning_rate * error

            # Predict using current model
            predictions = self.predict(self.X)

            # Update the error
            self.error = self.calc_error(predictions, self.y)

            # Record the error values for plotting purposes
            self.error_values['iteration'].append(iteration)
            self.error_values['value'].append(self.error)

            # Increment iterations
            iteration += 1

    def predict_stats(self, X_new, y):
        predictions = self.predict(X_new)
        error = self.calc_error(predictions, y) * 100
        accuracy = 100 - error

        print(f"Error:         {error:.4f}%")
        print(f"Weight (w1):   {self.w1:.4f}")
        print(f"Bias (w0):    {self.w0:.4f}")
        print(f"Accuracy:      {accuracy:.4f}%")


def main():
    # Read in data into dataframe and rename column headers, only using:
    # column index 1: Diagnosis (1 Malignent or 0 Benign)
    # column index 2: Radius (mean distance from center to the perimeter)
    df = pd.read_csv('logistic_regression/wdbc.data',
                     header=None, usecols=[1, 2])
    df = df.rename(columns={1: 'diagnosis', 2: 'radius'})
    df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

    # Seperate my independent and dependent variables
    X = df['radius']
    y = df['diagnosis']

    # Split the data into 70% training, 30% testing
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.30,
                                                        random_state=0)

    # Create a new logistic regression model
    cancer = LogReg(X_train, y_train)

    # Train the logistic regression model
    cancer.train_model()

    # Predict using the test values
    cancer.predict_stats(X_test, y_test)

    # Plotting the results
    plt.scatter(X_test, y_test, color='blue', label='Actual')
    plt.scatter(X_test, cancer.predict(X_test),
                color='red', marker='x', label='Predicted')
    plt.yticks([0, 1])
    plt.title('Malignent Tumors Classified by Radius')
    plt.xlabel('Radius')
    plt.ylabel('Is Malignent')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()

"""
The data shows that we can get around as little as 10.5% accurate,
however tumors are complicated and radius alone is not enough to
have a 5% or less error value. In the future we could use more dimensions
in order to achieve a much less error value and much higher accuracy.
These findings are supported by the Figure 1 error chart, which shows that
after around a 12% error, the model starts becoming overfit.
"""
