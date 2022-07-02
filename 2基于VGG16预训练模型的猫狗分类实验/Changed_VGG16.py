"""
作者：  cly
日期：  2022年06月24日
代码调整PEP8：Ctrl+Alt+L
一个epoch需要时间：5min
学习数据增强技术；
flow_from_directory准备数据时需按类别建立子文件夹；
学习使用预训练模型：迁移学习
"""

from keras import layers
from keras import models
import matplotlib.pyplot as plt
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
import os
from keras.applications import VGG16

os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error

train_dir = r'D:\学习资料\研一下学期课程\深度学习\课程实验作业\2实验二\test1\cats_and_dogs_small\train'
validation_dir = r'D:\学习资料\研一下学期课程\深度学习\课程实验作业\2实验二\test1\cats_and_dogs_small\validation'
test_dir = r'D:\学习资料\研一下学期课程\深度学习\课程实验作业\2实验二\test1\cats_and_dogs_small\test'

# 模型框架建立
conv_base = VGG16(weights='imagenet',  # 加载在ImageNet上预训练的权值
				  include_top=False,  # 是否包括全连接分类器，显然在ImageNet中有上千分类，在我们这里是不需要的
				  input_shape=(150, 150, 3))
conv_base.trainable = False  # 冻结卷积基、使之不可训练

model = models.Sequential()
model.add(conv_base)  # 使用VGG16模型
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
model.add(layers.Dense(1, activation='sigmoid'))

# 自定义框架，训练太慢
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
# model.add(layers.MaxPool2D((2, 2)))
#
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.MaxPool2D((2, 2)))
#
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPool2D((2, 2)))
#
# model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# model.add(layers.MaxPool2D((2, 2)))
#
# model.add(layers.Flatten())
# model.add(layers.Dense(512, activation='relu'))
# model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=1e-4), metrics=['acc'])

# 数据增强技术：对送入模型中每个epoch的数据进行变换，这样每个epoch的数据都不一样，从而变相地增加数据
# 数据总量依旧不变
train_datagen = ImageDataGenerator(
	rescale=1. / 255,  # 归一化
	rotation_range=40,  # 旋转角度
	width_shift_range=0.2,  # 宽度平移
	height_shift_range=0.2,  # 高度平移
	shear_range=0.2,  # 修剪
	zoom_range=0.2,  # 缩放
	horizontal_flip=True,  # 水平翻转
	fill_mode='nearest')  # 像素填充、添加新像素

test_datagen = ImageDataGenerator(rescale=1. / 255)  # 只进行归一化，不进行数据增强，用于生成验证集和测试集

# 准备数据
# 使用flow_from_directory生成数据时，需要在train、validation、test文件夹中按类别建立子文件夹，如cats类、dogs类
train_generator = train_datagen.flow_from_directory(  # 以文件夹为路径生成经数据增强、归一化后的数据
	directory=train_dir,
	target_size=(150, 150),  # 图像尺寸
	batch_size=64,
	class_mode='binary'  # 二值标签
)

validation_generator = test_datagen.flow_from_directory(
	directory=validation_dir,
	target_size=(150, 150),
	batch_size=32,
	class_mode='binary')

test_generator = test_datagen.flow_from_directory(
	test_dir,
	target_size=(150, 150),
	batch_size=32,
	class_mode='binary'
)

# 模型训练
history = model.fit_generator(
	train_generator,
	steps_per_epoch=int(4000 / 64),  # steps_per_epoch=len(x_train)/batch_size,这里batch_size指train_generator里的
	# steps_per_epoch * batch_size 要小于训练样本数4000才不会出错
	epochs=30,  # 将epochs设为100可能更好，但是耗时太长
	validation_data=validation_generator,
	validation_steps=int(500 / 32),  # 这里的batch_size是32
)

# 模型保存
model.save('cats_and_dogs_small_1.h5')

# 训练过程展示
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)
# 准确率曲线
plt.figure()  # 建立画布，即做一张图
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.savefig('acc.png')

# LOSS曲线
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('loss.png')
# plt.show()

# 模型测试
test_loss, test_acc = model.evaluate_generator(test_generator)
print('test acc:', test_acc)
