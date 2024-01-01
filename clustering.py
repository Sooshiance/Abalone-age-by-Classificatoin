import numpy as np
import pandas as pd
import random

# Load the data
# data = pd.read_csv("https://gist.githubusercontent.com/Sooshiance/bdc297dc976039bf88dd99d2c0c5823b/raw/abalone.csv")

column_names = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]

data = pd.read_csv("./DataSet.csv", names=column_names)

# Split the data into features and labels
X = data.drop("Rings", axis=1)
y = data["Rings"]

# Encode the categorical feature (Sex) as numeric values
X["Sex"] = X["Sex"].map({"M": 0, "F": 1, "I": 2})

# Define the number of clusters
k = 4

# Initialize the cluster centroids randomly
centroids = X.sample(k)

# Define the distance function
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Define the clustering algorithm
def clustering(X, centroids, k):
    # Initialize an empty list to store the cluster assignments
    clusters = []

    # Loop through all the data points
    for i in range(len(X)):
        # Initialize an empty list to store the distances to each centroid
        distances = []

    # Loop through all the centroids
    for j in range(k):
        # Calculate the distance between the current data point and the current centroid
        distance = euclidean_distance(X.iloc[i], centroids.iloc[j])

    # Append the distance to the list
    distances.append(distance)

    # Find the index of the closest centroid
    cluster = np.argmin(distances)

    # Append the cluster assignment to the list
    clusters.append(cluster)

    # Return the list of cluster assignments
    return clusters

# Define the function to update the centroids
def update_centroids(X, clusters, k):
    # Initialize an empty dataframe to store the new centroids
    new_centroids = pd.DataFrame()

    # Loop through all the clusters
    for i in range(k):
        try:
            # Find the data points that belong to the current cluster
            cluster_data = X[clusters == i]
        except:
            continue

    # Calculate the mean of the cluster data
    cluster_mean = cluster_data.mean()

    # Append the cluster mean to the dataframe
    new_centroids = new_centroids.append(cluster_mean, ignore_index=True)

    # Return the new centroids
    return new_centroids

# Define the function to check the convergence
def is_converged(centroids, new_centroids):
    # Calculate the difference between the old and new centroids
    diff = centroids - new_centroids

    # Check if the difference is zero
    return diff.sum().sum() == 0

# Define the maximum number of iterations
max_iter = 100

# Initialize a variable to store the number of iterations
iter = 0

# Initialize a variable to store the convergence status
converged = False

# Repeat until convergence or maximum iterations
while not converged and iter < max_iter:
    # Assign the data points to the closest cluster
    clusters = clustering(X, centroids, k)

    # Save the current centroids
    old_centroids = centroids.copy()

    # Update the centroids based on the cluster assignments
    centroids = update_centroids(X, clusters, k)

    # Check the convergence
    converged = is_converged(old_centroids, centroids)

    # Increment the number of iterations
    iter += 1

# Print the final centroids
print(f"The final centroids are:\n{centroids}")

# Print the number of iterations
print(f"The algorithm converged in {iter} iterations")
