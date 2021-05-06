import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Rows in data we want to use for prediction
x_rows=['BMI']
y_rows=['PFat']

# Read data
train_data = pd.read_csv('Predict_FAT.csv')
train_x = train_data[x_rows]
train_y = train_data[y_rows]

for x in range(len(train_data)):
    print(x, train_data.iloc[x])

# exit(0)
# Plot data
plt.scatter(train_x, train_y)
plt.plot(train_x, train_y, 'o')

# Add trend line to data

# Numpy array to simple 1D array
xx = train_x.values.flatten()
yy = train_y.values.flatten()

# Trendline
z = np.polyfit(xx, yy, 1)
p = np.poly1d(z)
plt.plot(xx, p(xx))

# Add label to graph
plt.xlabel("BMI")
plt.ylabel("Body fat")

# Display
plt.show()
