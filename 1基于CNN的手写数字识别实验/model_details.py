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

# 加载模型
save_path = r'models\AlexNet_model_20.h5'  # 注意名称
# save_path = r'models\simplify_AlexNet_model_100.h5'  # 注意名称
AlexNet_model = load_model(save_path)

# 获取模型结构状况
print(AlexNet_model.summary())

# 获取模型权重
weights = np.array(AlexNet_model.get_weights())
print('权重的维度（层数）为：{}'.format(weights.shape))

# 测试结果
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape((-1, 28, 28, 1))  # 展成（28，28，1）的数据
x_test = x_test.reshape((-1, 28, 28, 1))
res = AlexNet_model.evaluate(x_test, y_test)
print('模型在测试集上的评估结果为：{}'.format(res))

#这段代码用来将model.summary() 输出保存为文件
from contextlib import redirect_stdout
with open('model_summary.txt', 'w') as f:
    with redirect_stdout(f):
        AlexNet_model.summary(line_length=200,positions=[0.30,0.60,0.7,1.0])
