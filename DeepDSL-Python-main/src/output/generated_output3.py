import tensorflow as tf
import numpy as np


class NewNetwork:
	def __init__(self):
		self.model = tf.keras.Sequential()
		self.model.add(tf.keras.layers.Dense(units=128, activation='Relu', input_shape=(32,32,)))
		self.model.add(tf.keras.layers.Dense(units=2, activation='Sigmoid'))

	def compile_model(self):
		self.model.compile(optimizer='rmsprop',
			loss='SparseCategoricalCrossentropy',
			metrics = ['accuracy'])

	def train_model(self, x_train, y_train, x_val, y_val):
		self.model.fit(
			x_train, y_train,
			validation_data=(x_val, y_val),
			epochs=20,
			batch_size=64
		)

if __name__ == "__main__":
	network = NewNetwork()
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
	network.compile_model()
	network.train_model(x_train, y_train, x_test, y_test)
