import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

data = pd.read_csv('PRACTICALS/wine-quality.csv')

X = data.iloc[:,:-1]
Y = data.iloc[:,-1]

Xtr, Xts, ytr, yts = train_test_split(X, Y, test_size=0.2)

model = LinearRegression()
model.fit(Xtr, ytr)

ypr = model.predict(Xts)

mae = mean_absolute_error(yts, ypr)
mse = mean_squared_error(yts, ypr)

print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)

plt.figure(figsize=(10, 6))

plt.scatter(yts, ypr, color='blue', alpha=0.5)
plt.plot([min(yts), max(yts)], [min(yts), max(yts)], color='red', linestyle='--')

plt.title('Actual vs. Predicted Wine Quality')
plt.xlabel('Actual Quality')
plt.ylabel('Predicted Quality')
plt.grid(True)
plt.show()