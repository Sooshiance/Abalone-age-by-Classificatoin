import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score


# TODO : Load the dataset

column_names = ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]
# url = "https://gist.github.com/Sooshiance/bdc297dc976039bf88dd99d2c0c5823b"
df = pd.read_csv('./DataSet.csv', header=None, names=column_names)


# TODO : Explore the dataset

df.describe()
df.info()
df.hist(figsize=(12,10))
plt.show()


# TODO : Preprocess the dataset

X = df.drop("Rings", axis=1)
y = df["Rings"]
X = pd.get_dummies(X, prefix="Sex")
scaler = StandardScaler()
X = scaler.fit_transform(X)


# TODO : Split the dataset

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# TODO : Create a kNN model

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)


# TODO : Evaluate the model

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)


# TODO : Tune the hyperparameters

params = {"n_neighbors": range(1, 21), "weights": ["uniform", "distance"], "metric": ["euclidean", "manhattan", "minkowski"]}
grid = GridSearchCV(knn, params, cv=5, scoring="neg_mean_squared_error")
grid.fit(X_train, y_train)
print("Best parameters:", grid.best_params_)
print("Best score:", grid.best_score_)


# TODO : Compare the results

knn_best = grid.best_estimator_
y_pred_best = knn_best.predict(X_test)
mse_best = mean_squared_error(y_test, y_pred_best)
rmse_best = np.sqrt(mse_best)
r2_best = r2_score(y_test, y_pred_best)
print("MSE best:", mse_best)
print("RMSE best:", rmse_best)
print("R2 best:", r2_best)


# TODO : Plot the actual vs predicted values

plt.scatter(y_test, y_pred, label="Original")
plt.scatter(y_test, y_pred_best, label="Tuned")
plt.xlabel("Actual Rings")
plt.ylabel("Predicted Rings")
plt.legend()
plt.show()
