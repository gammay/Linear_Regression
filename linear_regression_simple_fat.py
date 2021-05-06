import pandas as pd

# Rows in data we want to use for prediction
X_rows = ['BMI']
y_rows = ['PFat']

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Train

# Read train data
data = pd.read_csv('Predict_FAT_TRAIN.csv')
X_train = data[X_rows]
y_train = data[y_rows]

# Linear regression modeler
from sklearn import linear_model

lreg = linear_model.LinearRegression()

# Train and create model
lreg.fit(X_train, y_train)

# Test

# Read test data
data_test = pd.read_csv('Predict_FAT_TEST.csv')
X_test = data_test[X_rows]
y_test = data_test[y_rows]

# Prediction on test data
y_test_prediction = lreg.predict(X_test)

# Show predictions in readable form
ys = pd.merge(y_test, pd.DataFrame(y_test_prediction), left_index=True, right_index=True)
xys = pd.merge(X_test, ys, left_index=True, right_index=True)
xys.columns = ['X', 'y_truth', 'y_predict']
# xys.option_context('display.max_rows', None, 'display.max_columns', None)
# print(xys)
print(xys.to_string())

# Performance / errors

# Variance plot
plt.xlabel("BMI")
plt.plot(X_test, y_test, label="Truth")
plt.plot(X_test, y_test_prediction, label="Predicted")

# Indicate difference between truth and prediction (error)
for i in range(len(X_test)):
    x1 = (X_test.iloc[i])[0]; x2 = (X_test.iloc[i])[0]
    y1 = y_test.iloc[i][0]; y2 = (y_test_prediction[i])[0]
    variance = abs(y1 - y2)
    if variance >=25: plt.text(x1, (y1+y2)/2, round(variance, 2))
    else: pass
    plt.plot([x1, x2], [y1, y2], "gray", linewidth=2, linestyle=':', marker='.')

plt.legend()
plt.show()

# Residual plot
residuals = []
for i in range(len(X_test)):
    residual = (y_test.iloc[i])[0] - y_test_prediction[i]
    plt.scatter((X_test.iloc[i])[0], residual, color="gray")
    residuals.append(residual)

plt.title("Residual plot")
plt.show()


# Residuals distribution
residuals.sort()
residual_mean = np.mean(residuals)
residual_std = np.std(residuals)
pdf = stats.norm.pdf(residuals, residual_mean, residual_std)
plt.plot(residuals, pdf)

plt.title("Residual distribution")
plt.show()


import sklearn

mae = sklearn.metrics.mean_absolute_error(y_test, y_test_prediction)
print("MAE:", mae)

r2 = sklearn.metrics.r2_score(y_test, y_test_prediction)
print('R2 score:', r2)

mse = sklearn.metrics.mean_squared_error(y_test, y_test_prediction)
print("MSE:", mse)

rmse = sklearn.metrics.mean_squared_error(y_test, y_test_prediction, squared=False)
print("RMSE:", rmse)

print('Coefficients:', lreg.coef_)
print('Intercept:', lreg.intercept_)

print(lreg.get_params())
