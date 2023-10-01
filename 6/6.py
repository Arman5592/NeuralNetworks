import matplotlib.pyplot as plt
import numpy as np
from skimage.util import random_noise
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from random import randint

def noisify(image, intensity):
    
    if intensity > 0:
        image_copy = random_noise(image, mode='gaussian', clip=True)
    if intensity > 1:
        image_copy = random_noise(image_copy, mode='gaussian', clip=True)
        image_copy = random_noise(image_copy, mode='salt', clip=True)
    if intensity > 2:
        image_copy = random_noise(image_copy, mode='gaussian', clip=True)
        image_copy = random_noise(image_copy, mode='salt', clip=True)
    if intensity > 3:
        image_copy = random_noise(image_copy, mode='gaussian', clip=True)
        image_copy = random_noise(image_copy, mode='salt', clip=True)
        image_copy = random_noise(image_copy, mode='gaussian', clip=True)
    
    return image_copy

def add_image_to_plot(axs, row, img_n, img_dn, img_r):
    axs[row, 0].imshow(img_n)
    axs[row, 1].imshow(img_dn)
    axs[row, 2].imshow(img_r)
    return axs


# load dataset from keras
(input_train, output_train), (input_test, output_test) = mnist.load_data()

# extrack 10k images and generate noised samples
images = input_train[0:12000]/255

noise_intensity = 4

images_noised = np.array(list(map(lambda x: noisify(x, noise_intensity), images))).reshape(-1,28,28)

input_train = images_noised[0:10000]
output_train = images[0:10000]
input_test = images_noised[10000:12000]
output_test = images[10000:12000]

# create and train model
model = Sequential()
model.add(Dense(300, input_dim = 28 * 28, activation= 'relu'))
model.add(Dense(700, activation = 'relu'))
model.add(Dense(28 * 28, activation = 'relu'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])

model.fit(input_train.reshape(-1, 28*28), output_train.reshape(-1, 28*28), batch_size=32, epochs=10)

_, model_MSE = model.evaluate(input_test.reshape(-1, 28*28), output_test.reshape(-1, 28*28))
print('MSE: ' +str(model_MSE))

# plot results
fig = plt.figure(dpi=100)
axs = fig.subplots(3, 3)
i1, i2, i3 = randint(10000, 12000), randint(10000, 12000), randint(10000, 12000)
add_image_to_plot(axs, 0, images_noised[i1], 
                  model.predict(images_noised[i1].reshape(-1, 28*28)).reshape(28, 28),
                  images[i1])
add_image_to_plot(axs, 1, images_noised[i2], 
                  model.predict(images_noised[i2].reshape(-1, 28*28)).reshape(28, 28),
                  images[i2])
add_image_to_plot(axs, 2, images_noised[i3], 
                  model.predict(images_noised[i3].reshape(-1, 28*28)).reshape(28, 28),
                  images[i3])

axs[0, 0].set_title("Noised")
axs[0, 1].set_title("Denoised")
axs[0, 2].set_title("Original")

plt.show()