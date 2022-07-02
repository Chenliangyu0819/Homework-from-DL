"""
作者：  cly
日期：  2022年06月21日
代码调整PEP8：Ctrl+Alt+L
"""
from keras.models import load_model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16

# 加载模型
save_path = r'cats_and_dogs_small_1.h5'  # 注意名称
# save_path = r'models\simplify_AlexNet_model_100.h5'  # 注意名称
VGG16_model = load_model(save_path)

# 获取模型结构状况
print(VGG16_model.summary())

# 获取VGG16模型框架
conv_base = VGG16(weights='imagenet',  # 加载在ImageNet上预训练的权值
				  include_top=False,  # 是否包括全连接分类器，显然在ImageNet中有上千分类，在我们这里是不需要的
				  input_shape=(150, 150, 3))

# 获取模型权重
weights = np.array(VGG16_model.get_weights())
print('权重的维度（层数）为：{}'.format(weights.shape))

# 测试结果
test_dir = r'D:\学习资料\研一下学期课程\深度学习\课程实验作业\2实验二\test1\cats_and_dogs_small\test'
test_datagen = ImageDataGenerator(rescale=1. / 255)  # 只进行归一化，不进行数据增强，用于生成验证集和测试集

# 生成测试集
test_generator = test_datagen.flow_from_directory(
	test_dir,
	target_size=(150, 150),
	batch_size=32,
	class_mode='binary'
)

# 模型评估
test_loss, test_acc = VGG16_model.evaluate_generator(test_generator)
print('test acc:', test_acc)

# 这段代码用来将model.summary() 输出保存为文件
from contextlib import redirect_stdout

with open('DOGSandCATS_model_summary.txt', 'w') as f:
	with redirect_stdout(f):
		VGG16_model.summary(line_length=200, positions=[0.30, 0.60, 0.7, 1.0])

with open('VGG16_model_summary.txt', 'w') as f:
	with redirect_stdout(f):
		conv_base.summary(line_length=200, positions=[0.30, 0.60, 0.7, 1.0])
