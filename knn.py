import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Load the data
# data = pd.read_csv("https://gist.github.com/Sooshiance/bdc297dc976039bf88dd99d2c0c5823b")

column_names = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]

data = pd.read_csv("./DataSet.csv", names=column_names)

# Split the data into features and labels
X = data.drop("Rings", axis=1, inplace=True)
y = data["Rings"]

# Encode the categorical feature (Sex) as numeric values
X["Sex"] = X["Sex"].map({"M": 0, "F": 1, "I": 2})

# Define the distance function
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

# Define the KNN algorithm
def knn(X, y, x_test, k):
    # Initialize an empty list to store the distances
    distances = []

    # Loop through all the training examples
    for i in range(len(X)):
        # Calculate the distance between the test point and the current example
        distance = euclidean_distance(x_test, X.iloc[i])
        """
        We can use a lambda function as well
        distance = lambda x_test : 
        """

    # Append the distance and the label to the list
    distances.append((distance, y.iloc[i]))

    # Sort the list by the distance
    distances.sort(key=lambda x: x[0])

    # Get the k nearest neighbors
    neighbors = distances[:k]

    # Get the labels of the neighbors
    labels = [x[1] for x in neighbors]

    # Return the mean of the labels
    return np.mean(labels)

# Define the value of k
k = 5

# Make predictions on the test set
y_pred = []
for x_test in X:
    y_pred.append(knn(X, y, x_test, k))

# Calculate the mean squared error
mse = mean_squared_error(y, y_pred)

# Print the mean squared error
print(f"The mean squared error of the KNN algorithm is {mse:.2f}")
