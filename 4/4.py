from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from pointgen import generate_point_set, function
from numpy import loadtxt
import numpy as np
import matplotlib.pyplot as plt


min_x, max_x, point_cnt = 0, 10, 20000

# generate and load our function's points
generate_point_set(point_cnt, min_x, max_x)
dataset = loadtxt('points.csv', delimiter=',', skiprows=1)

# points are generated randomly, so a slice = random split
train = dataset[0:16000, :]
test = dataset[16000:20000, :]

# generate NN
model = Sequential()
model.add(Dense(400, input_dim=1, activation='relu'))
model.add(Dense(400, activation='linear'))
model.add(Dense(400, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

model.fit(train[:, 0], train[:, 1], epochs=120, batch_size=100)

_, model_MSE = model.evaluate(test[:, 0], test[:, 1])
print('MSE: ' +str(model_MSE))

# plot results + ground truth
plot_range = np.linspace(0, 10, 100)
predicted = model.predict(plot_range)
ground_truth = function(plot_range)
plt.plot(plot_range, predicted, '-r')
plt.plot(plot_range, ground_truth, '--b', alpha=0.5)
plt.show()