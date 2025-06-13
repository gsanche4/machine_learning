"""
Genaro Sanchez Feliz
gsanche4@ramapo.edu
CMPS-320-50
Assignment 5 - K-Means Clustering
Jun 9th, 2025
"""

# Import Libraries
import pandas as pd
import numpy as np
import random
import math
from statistics import mode


class k_clustering:

    def __init__(self, data, num_clusters, numeric_cols, categorical_cols):

        # User input attributes
        self.data = data
        self.k = num_clusters
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.numeric_col_idx = [data.columns.get_loc(col) for col in numeric_cols]
        self.categorical_col_idx = [data.columns.get_loc(col) for col in categorical_cols]

        # Get tuples of each row of mixed data
        self.data_points = [tuple(row) for row in data.values]

        # Get the number of rows
        self.N = len(self.data_points)

        # Randomly select starting centroids
        self.mu = random.sample(self.data_points, self.k)

        # Initialize attributes
        self.distance = np.zeros((self.k, self.N))
        self.cluster = [[] for _ in range(self.k)]
        self.temp_cluster = [[] for _ in range(self.k)]

    def calc_centroid(self):

        # Look through (k) clusters
        for j in range(0, self.k):

            # Skip empty clusters
            if len(self.cluster[j]) == 0:
                continue

            # Initialize the new centroids
            new_centroid = []

            # Look through each numerical column
            for col in self.numeric_col_idx:

                # Get the values for those numerical columns
                vals = [point[col] for point in self.cluster[j]]

                # Append the mean of those values to the centroid list
                new_centroid.append(np.mean(vals))

            # Loop through each categorical column
            for col in self.categorical_col_idx:

                # Get the values for those categorical columns
                vals = [point[col] for point in self.cluster[j]]

                # Append the mode for those values to the centroid list
                new_centroid.append(mode(vals))

            # Mu is now the new tuple of centroids
            self.mu[j] = tuple(new_centroid)

    def calc_distance(self, x, centroid):
        # --- Numeric Calculation ------- #
        # Get the index values for the numerical columns
        numeric_indices = [self.data.columns.get_loc(col) for col in self.numeric_cols]

        # Get the index values for the categorical columns
        categorical_indices = [self.data.columns.get_loc(col) for col in self.categorical_cols]

        # Initialize numeric distance
        num_dist = 0

        # Loop through each numeric column
        for col in numeric_indices:

            # Get the sum of squares for x minus the centroid of that col
            num_dist += (x[col] - centroid[col]) ** 2

        # Get the square root of all the sum of squares for all columns
        num_dist = math.sqrt(num_dist)

        # --- Categorical Calculation --- #
        # Initialize categorical distance
        cat_dist = 0

        # Loop through each categorical column
        for col in categorical_indices:

            # If each value in the column is not equal to 
            # the centroid then add 1, this gives equal distance
            # since one category isnt greater than another one,
            # for example spanish isnt greater than english, they are
            # both 1 away.
            if x[col] != centroid[col]:
                cat_dist += 1

        return num_dist + cat_dist

    def my_kmeans(self):

        # Loop through each row
        for j in range(0, self.k):

            # Initialize cluster/row for later use
            self.cluster[j] = []

            # Loop through all data
            for i in range(0, self.N):

                # Calcualte the distance of that point to the centroid
                self.distance[j][i] = self.calc_distance(self.data_points[i], self.mu[j])

        # Loop through all data points
        for i in range(0, self.N):

            # Keeps tracks of smallest distance to a centroid, and the centroid
            min_dist = float('inf')
            closest_cluster = 0

            # Check all clusters
            for j in range(0, self.k):

                # Check to see if the distance is the smallest
                if self.distance[j][i] < min_dist:

                    # Update the new smallest distance, and closest cluster
                    min_dist = self.distance[j][i]
                    closest_cluster = j

            # After all clusters are checked, assign to closest cluster
            self.cluster[closest_cluster].append(self.data_points[i])

        # Sort clusters
        for j in range(self.k):
            self.cluster[j].sort()

        # Sort the data points in the clusteres
        for j in range(0, self.k):
            self.cluster[j] = tuple(sorted(self.cluster[j]))

        # Check if all clusters equal all temp clusters
        if self.cluster == self.temp_cluster:
            # Return 1, Passed
            return 1
        else:
            # Copy all the clusters to the temp clusters
            self.temp_cluster = self.cluster.copy()

            # Return 0, Failed
            return 0


def main():

    # Define data file path
    file_path = 'k_means_clustering/flags/flag.data'

    # Read in data
    df = pd.read_csv(file_path,
                     header=None,
                     usecols=[0, 3, 4, 5, 6],
                     names=['name', 'area', 'population',
                            'language', 'religion'])

    # Normalize the Area and Population data as they are vastly different
    for col in ['area', 'population']:
        min_val = df[col].min()
        max_val = df[col].max()
        range_val = max_val - min_val
        df[col] = (df[col] - min_val) / range_val

    # State the number of clusteres
    k = 4

    # Split numerical and categorical columns
    num_cols = ['area', 'population']
    cat_cols = ['language', 'religion']

    # Create the k_cluster_model
    kmeans = k_clustering(df[num_cols + cat_cols], k,
                          numeric_cols=num_cols,
                          categorical_cols=cat_cols)

    # Fit the model
    while True:
        kmeans.calc_centroid()
        if kmeans.my_kmeans():
            break

    name_map = dict(zip([tuple(row) for row in df[['area', 'population', 'language', 'religion']].values],
                        df['name']))

    # Create a mapping from each data point to all matching country names
    point_to_names = {}

    for i in range(len(df)):
        # Locate each country by its values
        row = df.iloc[i]
        key = (row['area'], row['population'], row['language'], row['religion'])
        
        if key in point_to_names:
            if row['name'] not in point_to_names[key]: 
                point_to_names[key].append(row['name'])
        else:
            point_to_names[key] = [row['name']]
    
    print("\nFinal Clusters:\n")
    seen = set()

    for i, cluster in enumerate(kmeans.cluster):
        print(f"Cluster {i + 1}:")
        for point in cluster:
            if point in point_to_names:
                for name in point_to_names[point]:
                    if name not in seen:
                        print("  -", name)
                        seen.add(name)
        print()
main()