"""
Genaro Sanchez Feliz
gsanche4@ramapo.edu
CMPS-320-50
Assignment 1
May 30th, 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Step 1: Read the dataset from AirQualityUCI.csv
file_path = 'week_1/AirQualityUCI.csv'
df = pd.read_csv(file_path, sep=';', usecols=['CO(GT)', 'T'])

# Drop all empty rows
df = df.dropna(subset='T')

# Convert all ',' to '.' and data type to float
df['CO(GT)'] = df['CO(GT)'].str.replace(',', '.').astype(float)
df['T'] = df['T'].str.replace(',', '.').astype(float)

# Make all -200 values an NaN then drop it
df.loc[df['CO(GT)'] == -200, 'CO(GT)'] = np.nan
df.loc[df['T'] == -200, 'T'] = np.nan
df = df.dropna()

# Step 2: Apply Linear Regression from Scratch
# Assign independent and dependent values to variables
X = df['CO(GT)']
y = df['T']

# Split into 70% for training and 30% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=0)

# Find the means of the X_train and y_train
X_train_mean = np.mean(X_train)
y_train_mean = np.mean(y_train)

# Calculate the coefficients
numerator = 0
denominator = 0

for x_val, y_val in zip(X_train, y_train):
    numerator += (x_val - X_train_mean) * (y_val - y_train_mean)
    denominator += (x_val - X_train_mean) ** 2

m = numerator / denominator
c = y_train_mean - (m * X_train_mean)

# Linear regression equation
print(f'Linear regression equation: y = {m:.4f}(x) + {c:.4f}')

# Find the menas of X_test and y_test
X_test_mean = np.mean(X_test)
y_test_mean = np.mean(y_test)

# Calculate R²
R2 = sum((m * X_test + c - y_test_mean) ** 2) / sum((y_test - y_test_mean) ** 2)
print(f'R²: {R2:.4f}')


# Plot the data and the regression line
plt.scatter(X, y, color='blue', label='Data points', alpha=0.5)
plt.plot(X, c + m * X, color='red',
         label=f'Regression Line (y = {m:.2f}x + {c:.2f})')

plt.title('Relationship between CO(GT) and Temperature', weight='bold')
plt.xlabel('CO(GT)')
plt.ylabel('Temperature')
plt.legend()
plt.show()

"""
Outcome: This data shows little to no correlation between CO(GT) and
Temperature. Furthermore, the low R^2 of about 0.0004 shows that the
linear regression does not capture or explain most data points. Finally,
in the scatter plot graph we can see that there is no direction or
pattern that data points are taking. There may be a slight pattern in
that temperatures are getting closer to the linear regression line the higher
the CO(GT) value is. This gives the data points an almost blurry horizontal
triangle apperance.
"""
