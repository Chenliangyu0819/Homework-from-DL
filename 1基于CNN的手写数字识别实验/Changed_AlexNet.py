"""
作者：  cly
日期：  2022年06月21日
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt


def AlexNet_inference(in_shape):
	model = keras.Sequential(name='Changed_AlexNet')

	# model.add(layers.Conv2D(96,(11,11),strides=(4,4),input_shape=(in_shape[1],in_shape[2],in_shape[3]),
	# padding='same',activation='relu',kernel_initializer='uniform')) # 原始AlexNet的第一层

	# 注意padding='same'的意思是：使得输出尺寸和输入尺寸保持一致!
	# 当p=1，计算p的公式为：n+2p-f+1=n，最后得p=(f-1)/2;第一层中p为2，基本思路就是输出=输入/stride=28/2=14.
	model.add(layers.Conv2D(96, (11, 11), strides=(2, 2), input_shape=(in_shape[1], in_shape[2], in_shape[3]),
							padding='same', activation='relu', kernel_initializer='uniform'))
	model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))  # kernel_size(6.5)向下取整，所以是6.
	model.add(
		layers.Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
	model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))  # kernel_size(2.5)向下取整，所以是2.
	model.add(
		layers.Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
	model.add(
		layers.Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
	model.add(
		layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
	# 因为padding = same，上述三个卷积层的维度均不变

	model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
	model.add(layers.Flatten())  # 矩阵展成向量
	model.add(layers.Dense(2048, activation='relu'))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(2048, activation='relu'))
	model.add(layers.Dropout(0.5))
	model.add(layers.Dense(10, activation='softmax'))
	model.compile(optimizer=keras.optimizers.Adam(),
				  loss='sparse_categorical_crossentropy',  # 不能直接用函数，否则在与测试加载模型不成功！
				  metrics=['accuracy'])
	model.summary()
	return model


mnist = tf.keras.datasets.mnist
MODEL_DIR = "models/"

(x_train, y_train), (x_test, y_test) = mnist.load_data()  # 括号里加路径可以缓存数据集到本地
# train_images, test_images = train_images / 255.0, test_images / 255.0  # 归一化

x_train = x_train.reshape((-1, 28, 28, 1))  # 展成（28，28，1）的数据
x_test = x_test.reshape((-1, 28, 28, 1))
print(x_train.shape[1], x_train.shape[2], x_train.shape[3])
x_shape = x_train.shape

AlexNet_model = AlexNet_inference(x_shape)
epochs = 20

while True:
	# if __name__ == '__main__':
	history = AlexNet_model.fit(x_train, y_train, batch_size=64, epochs=epochs,
								validation_split=0.1  # 验证集从训练集中取10%得到
								)

	res = AlexNet_model.evaluate(x_test, y_test)

	if res[1] > 0.6:
		plt.plot(history.history['accuracy'])
		plt.plot(history.history['val_accuracy'])
		plt.plot(history.history['loss'])
		plt.plot(history.history['val_loss'])
		plt.legend(['accuracy', 'val_accuracy', 'loss', 'val_loss'], loc='upper left')
		plt.show()
		break
	else:
		epochs += epochs

print('模型在测试集上的评估结果为：{}'.format(res))
model_save_dir = MODEL_DIR + 'AlexNet_model_' + str(epochs) + '.h5'
AlexNet_model.save(model_save_dir)
