from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# load dataset from keras
(input_train, output_train), (input_test, output_test) = fashion_mnist.load_data()

# reshape data into relu-friendly numbers
input_train = input_train.reshape((input_train.shape[0], 28*28)).astype('float32')/255
input_test = input_test.reshape((input_test.shape[0], 28*28)).astype('float32')/255
output_train = to_categorical(output_train, 10)
output_test = to_categorical(output_test, 10)

# create model
model = Sequential()
model.add(Dense(50, input_dim = 28 * 28, activation= 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(input_train, output_train, batch_size=200, epochs=15, validation_split=0.1)

# save on disk
model.save('p2_mnist')

# evaluate model
score = model.evaluate(input_test, output_test, verbose=0)

print("Test Loss: ", score[0])
print("Test Accuracy: ", score[1]*100, '%')
