import tensorflow as tf
import numpy as np


class NewNetwork:
	def __init__(self):
		self.model = tf.keras.Sequential()
		self.model.add(tf.keras.layers.Dense(units=256, activation='Tanh', input_shape=(50,)))
		self.model.add(tf.keras.layers.Dense(units=128, activation='Relu'))
		self.model.add(tf.keras.layers.Dense(units=5, activation='Softmax'))

	def compile_model(self):
		self.model.compile(optimizer='adam',
			loss='SparseCategoricalCrossentropy',
			metrics = ['accuracy'])

	def train_model(self, x_train, y_train, x_val, y_val):
		self.model.fit(
			x_train, y_train,
			validation_data=(x_val, y_val),
			epochs=25,
			batch_size=16
		)

	def generate_dataset(self):
		(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
		return x_train, x_train, y_train, y_test 

	def visualize_model(self, x, y):
		print(f"visualizing model on a {x} * {y} grid.")

if __name__ == "__main__":
	network = NewNetwork()
	x_train, x_test, y_train, y_test = network.generate_dataset()
	network.compile_model()
	network.train_model(x_train, y_train, x_test, y_test)
