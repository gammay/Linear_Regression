
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Rows in data we want to use for prediction
X_rows=['BMI']
y_rows=['PFat']

# Train

# Read train data
data = pd.read_csv('Predict_FAT_TRAIN.csv')
X_train = data[X_rows]
y_train = data[y_rows]

# for i in range(len(X_train)):
#     print(i, X_train.iloc[i].values[0], y_train.iloc[i].values[0])

# Linear regression modeler
from sklearn import linear_model
lreg = linear_model.LinearRegression() 

# lreg.set_params(positive=True)

# Train and create model 
lreg.fit(X_train, y_train) 

# Test

# Read test data
data_test = pd.read_csv('Predict_FAT_TEST.csv')
X_test = data_test[X_rows]
y_test = data_test[y_rows]

# Prediction on test data
y_test_prediction = lreg.predict(X_test)

# print("BMI\t\t\t", "FAT TRUTH\t", "FAT PREDICT")
# for i in range(len(X_test)):
#     print(X_test.iloc[i].values[0],
#         "\t\t", y_test.iloc[i].values[0],
#         "\t\t", y_test_prediction[i][0])

# for i in zip(X_test, y_test):
#     print(i)

# print(X_test.merge(y_test))
# print(pd.merge(X_test, y_test, left_index=True, right_index=True))
# print(pd.merge(X_test, pd.DataFrame(y_test_prediction), left_index=True, right_index=True))
# print(pd.merge(y_test, pd.DataFrame(y_test_prediction), left_index=True, right_index=True))

# Print x and ys side by side
ys = pd.merge(y_test, pd.DataFrame(y_test_prediction), left_index=True, right_index=True)
xys = pd.merge(X_test, ys, left_index=True, right_index=True)
xys.columns = ['X', 'y_truth', 'y_predict']
print(xys)

# for v in zip(X_test, y_test):
#     print(v)

# exit(0)

# Performance and loss functions
# After train, we ask how good is this model? To assess performance, we have a few methods and metrics.
# Regression line is he line that approximates the data. In case of linear regression, regression line would obviously be linear. We can visually see this.

# Performance / errors

# Plot
# plt.xlabel("BMI")
# plt.plot(X_test, y_test, label="Truth")
# plt.plot(X_test, y_test_prediction, label="Predicted")
#
# # Indicate difference between truth and prediction (error)
# for i in range(len(X_test)):
#     x1 = X_test.iloc[i]; y1 = X_test.iloc[i]
#     x2 = y_test.iloc[i]; y2 = y_test_prediction[i]
#
#     plt.plot([x1, y1], [x2, y2], "gray", linewidth=2, linestyle=':', marker='.')
#
# plt.legend()
# plt.show()

# # Residual plot
# for i in range(len(X_test)):
#     plt.scatter(X_test, y_test - y_test_prediction, color="gray")
#
# plt.title("Residual plot")
# plt.show()

# for i in range(len(y_test_prediction)):
#     y_test_prediction[i] = (y_test.iloc[i]).values[0] * 2
#     print(y_test.iloc[i].values[0], y_test_prediction[i])

import sklearn
mae = sklearn.metrics.mean_absolute_error(y_test, y_test_prediction)
print("MAE:", mae)

r2 = sklearn.metrics.r2_score(y_test, y_test_prediction)
print('R2 score:', r2)

mse = sklearn.metrics.mean_squared_error(y_test, y_test_prediction)
print("MSE:", mse)

rmse = np.sqrt(mse)
print("RMSE:", rmse)

# for i in range(len(y_test)):
#     print(i, y_test.iloc[i].values[0], (y_test_prediction[i])[0])
#     plt.plot(i, y_test.iloc[i].values[0])
    # plt.plot(i, y_test_prediction[i])[0])
    # plt.plot(i, (y_test.iloc[i]).values(), (y_test_prediction.iloc[i]).values())
    # plt.plot(X_test, y_test_prediction, label="Predicted")

# plt.plot(range(len(y_test), y_test))
# # plt.plot(X_test, y_test_prediction)
# plt.show()

# plt.plot([1,2,3], [10,30,11])
# plt.plot(2, 30)
# plt.plot(3, 20)
# plt.plot(4, 30)
# plt.plot(5, 50)
# for i in range(len(X_test)):
#     plt.scatter(X_test, y_test - y_test_prediction, color="gray")
#     # plt.scatter(i, i+10, color="gray")

# plt.plot(range(len(X_test)), y_test)
# plt.plot(range(len(X_test)), y_test_prediction)

# for i in range(len(X_test)):
#     print(i)
#     y_test_i = y_test.iloc[i].values[0]
#     print(y_test_i)
#     y_test_prediction_i = (y_test_prediction[i])[0]
#     print("y_test_prediction_i:", y_test_prediction_i)
#     mse_i = sklearn.metrics.mean_squared_error([y_test_i], [y_test_prediction_i])
#     print(mse_i)
#
#     plt.show()
#     plt.scatter(i, mse_i)
#
# plt.show()
#
# for i in range(len(X_test)):
#     print(i)
#     # mse = sklearn.metrics.mean_squared_error(y_test.iloc[i].values[0], y_test_prediction[i])
#     y_test_i = y_test.iloc[i].values[0]
#     y_test_prediction_i = (y_test_prediction[i])[0]
#     mse = sklearn.metrics.mean_squared_error(y_test_i, y_test_prediction_i)
#     print(y_test_i, y_test_prediction_i, mse)
# # plt.title("Residual plot")
# # plt.show()

print('Coefficients:', lreg.coef_)
print('Intercept:', lreg.intercept_)

print(lreg.get_params())

print(lreg.predict([[77]]))
