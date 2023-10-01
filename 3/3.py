from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from pointgen import generate_point_set, function
from numpy import loadtxt
import numpy as np
import matplotlib.pyplot as plt

min_xy, max_xy, point_cnt = 0, 10, 20000

# generate and load our function's points
generate_point_set(point_cnt, min_xy, max_xy, max_xy)
dataset = loadtxt('points.csv', delimiter=',', skiprows=1)

# points are generated randomly, so a slice = random split
train = dataset[0:16000, :]
test = dataset[16000:20000, :]

# generate NN
model = Sequential()
model.add(Dense(250, input_dim=2, activation='relu'))
model.add(Dense(400, activation='relu'))
model.add(Dense(400, activation='linear'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

model.fit(train[:, 0:2], train[:, 2], epochs=100, batch_size=100)

_, model_MSE = model.evaluate(test[:, 0:2], test[:, 2])
print('MSE: ' +str(model_MSE))

# plot results
plot_range = np.random.uniform(min_xy, max_xy, (100, 2))

predicted = model.predict(plot_range)
predicted= predicted.reshape(100)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_trisurf(plot_range[:, 0], plot_range[:, 1], predicted, alpha=0.6)

plt.show()
