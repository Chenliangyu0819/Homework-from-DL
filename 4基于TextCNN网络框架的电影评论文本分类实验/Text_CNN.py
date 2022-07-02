"""
作者：  cly
日期：  2022年06月29日
代码调整PEP8：Ctrl+Alt+L
loss曲线--基于训练集和验证集
代码中的词向量在模型训练时是跟随网络参数一起更新的(Non-static)
"""
import gensim
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import *


# 定义两个关于词、词序以及词向量之间联系的函数
def build_word2id(file, save_to_path=None):
	"""
	建立字典：{单词：词序}
	:param file: word2id保存地址
	:param save_to_path: 保存训练语料库中的词组对应的word2vec到本地
	:return: None
	"""
	word2id = {'_PAD_': 0}  # 在数据集中出现，但不在词向量预训练模型中出现的词均记为'_PAD_'
	path = ['./Dataset/train.txt', './Dataset/validation.txt']

	for _path in path:
		with open(_path, encoding='utf-8') as f:
			for line in f.readlines():
				sp = line.strip().split()
				for word in sp[1:]:
					if word not in word2id.keys():
						word2id[word] = len(word2id)
	if save_to_path:  # 如果存在该文件，那么word2id信息将会写入该文件
		with open(file, 'w', encoding='utf-8') as f:
			for w in word2id:
				f.write(w + '\t')
				f.write(str(word2id[w]))
				f.write('\n')

	return word2id


def build_word2vec(fname, word2id, save_to_path=None):
	"""
	:param fname: 预训练的word2vec.
	:param word2id: 语料文本中包含的词汇集.
	:param save_to_path: 保存训练语料库中的词组对应的word2vec到本地
	:return: 语料文本中词汇集对应的word2vec向量{id: word2vec}.
	"""
	n_words = max(word2id.values()) + 1
	model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)  # 加载词向量模型，建立词到向量的映射关系
	word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))  # 初始化词向量矩阵（词以序号代替）
	for word in word2id.keys():
		try:
			word_vecs[word2id[word]] = model[word]  # 将词向量按顺序填入word_vecs列表中
		except KeyError:
			pass
	if save_to_path:  # 如果存在该文件，那么word2vecs信息将会写入该文件
		with open(save_to_path, 'w', encoding='utf-8') as f:
			for vec in word_vecs:
				vec = [str(w) for w in vec]
				f.write(' '.join(vec))
				f.write('\n')
	return word_vecs


# 定义加载数据集和相应标签的函数
def cat_to_id(classes=None):
	"""
	:param classes: 分类标签；默认为0:pos, 1:neg
	:return: {分类标签：id}
	"""
	if not classes:
		classes = ['0', '1']
	cat2id = {cat: idx for (idx, cat) in enumerate(classes)}
	return classes, cat2id  # (['0', '1'], {'0': 0, '1': 1})


def load_corpus(path, word2id, max_sen_len=50):
	"""
	:param path: 样本语料库的文件
	:return: 文本内容contents，以及分类标签labels(onehot形式)
	"""
	_, cat2id = cat_to_id()
	contents, labels = [], []
	with open(path, encoding='utf-8') as f:
		for line in f.readlines():
			sp = line.strip().split()
			label = sp[0]  # 数据集不能有空行，包括文末处!
			# 用单词的序号来记录每句评论
			content = [word2id.get(w, 0) for w in sp[1:]]  # 在数据集中出现，但不在词向量预训练模型中出现的词均记为'_PAD_'
			# 保证每句评论的长度均为max_sen_len，多则删 少则补'_PAD_'
			content = content[0:max_sen_len]  # 如果content本身长度不及max_sen_len，此句相当于copy功能
			if len(content) < max_sen_len:
				content += [word2id['_PAD_']] * (max_sen_len - len(content))

			labels.append(label)
			contents.append(content)
	counter = Counter(labels)  # 计数器，返回字典，标注每个类别的个数
	print('Total sample num：%d' % (len(labels)))
	print('class num：')
	for w in counter:
		print(w, counter[w])

	contents = np.asarray(contents)
	labels = np.array([cat2id[l] for l in labels])  # 本质上是把字符串'0'和'1'变为了整形0和1；

	return contents, labels  # contents.shape = (len(数据集),50)，其中每一个列表中存储的是每句评论中的词序


# 加载词向量
word2id = build_word2id('./Dataset/word2id.txt')  # 建立字典：{'_PAD_': 0, '死囚': 1, '地起': 58953}
# print(word2id)
word2vec = build_word2vec('./Dataset/wiki_word2vec_50.bin', word2id)  # 列表中存放着58954个单词的词向量，词向量维度是50
assert word2vec.shape == (58954, 50)
# print(word2vec)

# 加载数据和标签
print('train set: ')
train_contents, train_labels = load_corpus('./Dataset/train.txt', word2id, max_sen_len=50)
# print(train_contents[0])  # [ 1  2  3  4 ...  0  0  0  0  0]
# print(train_labels[:10])  # [1 1 1 1 1 1 1 1 1 1]

print('\nvalidation set: ')
val_contents, val_labels = load_corpus('./Dataset/validation.txt', word2id, max_sen_len=50)
print('\ntest set: ')
test_contents, test_labels = load_corpus('./Dataset/test.txt', word2id, max_sen_len=50)


class CONFIG():
	update_w2v = True  # 是否在训练中更新w2v
	vocab_size = 58954  # 词汇量，与word2id中的词汇量一致
	n_class = 2  # 分类数：分别为pos和neg
	embedding_dim = 50  # 词向量维度
	drop_keep_prob = 0.5  # dropout层，参数keep的比例
	kernel_num = 64  # 卷积层filter的数量
	kernel_size = [3, 4, 5]  # 卷积核的尺寸
	pretrained_embed = word2vec  # 预训练的词嵌入模型  word2vec.shape = (58954, 50)


class TextCNN(nn.Module):
	def __init__(self, config):
		super(TextCNN, self).__init__()
		update_w2v = config.update_w2v
		vocab_size = config.vocab_size
		n_class = config.n_class
		embedding_dim = config.embedding_dim
		kernel_num = config.kernel_num
		kernel_size = config.kernel_size
		drop_keep_prob = config.drop_keep_prob
		pretrained_embed = config.pretrained_embed

		# 使用预训练的词向量
		# 参考资料：https://www.jianshu.com/p/1e0ebbefe323
		self.embedding = nn.Embedding(vocab_size, embedding_dim)
		self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embed))
		self.embedding.weight.requires_grad = update_w2v  # requires_grad.True指在训练中更新词向量
		# 卷积层：卷积核的宽度 = 词向量维度；长度自定义
		self.conv1 = nn.Conv2d(1, kernel_num, (kernel_size[0], embedding_dim))
		self.conv2 = nn.Conv2d(1, kernel_num, (kernel_size[1], embedding_dim))
		self.conv3 = nn.Conv2d(1, kernel_num, (kernel_size[2], embedding_dim))
		# Dropout
		self.dropout = nn.Dropout(drop_keep_prob)
		# 全连接层：输入维度 = 经过卷积层（实际上还包括最大池层）得到的feature map的拼接维度；输出维度 = 类别数
		self.fc = nn.Linear(len(kernel_size) * kernel_num, n_class)

	@staticmethod  # 定义静态方法：无需实例化也可调用
	def conv_and_pool(x, conv):
		# x: (batch, 1, sentence_length,  )
		x = conv(x)
		# x: (batch, kernel_num, H_out, 1)
		x = F.relu(x.squeeze(3))
		# x: (batch, kernel_num, H_out)
		x = F.max_pool1d(x, x.size(2)).squeeze(2)  # 最大池化过程
		#  (batch, kernel_num)
		return x

	def forward(self, x):
		x = x.to(torch.int64)
		x = self.embedding(x)
		x = x.unsqueeze(1)
		x1 = self.conv_and_pool(x, self.conv1)  # (batch, kernel_num)
		x2 = self.conv_and_pool(x, self.conv2)  # (batch, kernel_num)
		x3 = self.conv_and_pool(x, self.conv3)  # (batch, kernel_num)
		x = torch.cat((x1, x2, x3), 1)  # (batch, 3 * kernel_num) 拼接操作
		x = self.dropout(x)
		x = self.fc(x)
		x = F.log_softmax(x, dim=1)
		return x


# 设置超参数并准备数据
config = CONFIG()  # 配置模型参数
learning_rate = 0.001  # 学习率
BATCH_SIZE = 64  # 训练批量
EPOCHS = 15  # 训练轮数
model_path = None  # 预训练模型路径
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
train_dataset = TensorDataset(torch.from_numpy(train_contents).type(torch.float),
							  torch.from_numpy(train_labels).type(torch.long))
train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
							  shuffle=True, num_workers=0)  # num_workers为进程数

val_dataset = TensorDataset(torch.from_numpy(val_contents).type(torch.float),
							torch.from_numpy(val_labels).type(torch.long))
val_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE,
							shuffle=True, num_workers=0)

# 配置模型，是否继续上一次的训练
model = TextCNN(config)
if model_path:
	model.load_state_dict(torch.load(model_path))
model.to(DEVICE)

# 设置优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 设置损失函数
criterion = nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=5)  # 每5个epoch调整一次学习率


def train(dataloader, epoch):
	# 定义训练过程
	train_loss, train_acc = 0.0, 0.0
	count, correct = 0, 0
	for batch_idx, (x, y) in enumerate(dataloader):
		x, y = x.to(DEVICE), y.to(DEVICE)
		optimizer.zero_grad()  # 梯度置零
		output = model(x)  # 前向预测
		loss = criterion(output, y)  # 计算误差
		loss.backward()  # 反向传播
		optimizer.step()  # 参数更新
		train_loss += loss.item()  # 注意loss是在每个batch上取过均值的
		correct += (output.argmax(1) == y).float().sum().item()
		count += len(x)

		if (batch_idx + 1) % 100 == 0:
			print('train epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
				epoch, batch_idx * len(x), len(dataloader.dataset),
					   100. * batch_idx / len(dataloader), loss.item()))

	train_loss *= BATCH_SIZE
	train_loss /= len(dataloader.dataset)
	train_acc = correct / count
	print('\ntrain epoch: {}\taverage loss: {:.6f}\taccuracy:{:.4f}%\n'.format(epoch, train_loss, 100. * train_acc))
	scheduler.step()  # 更新学习率

	return train_loss, train_acc


def validation(dataloader, epoch):
	model.eval()
	# 验证过程
	val_loss, val_acc = 0.0, 0.0
	count, correct = 0, 0
	for _, (x, y) in enumerate(dataloader):
		x, y = x.to(DEVICE), y.to(DEVICE)
		output = model(x)
		loss = criterion(output, y)
		val_loss += loss.item()
		correct += (output.argmax(1) == y).float().sum().item()
		count += len(x)

	val_loss *= BATCH_SIZE
	val_loss /= len(dataloader.dataset)
	val_acc = correct / count
	# 打印准确率
	print(
		'validation:train epoch: {}\taverage loss: {:.6f}\t accuracy:{:.2f}%\n'.format(epoch, val_loss, 100 * val_acc))

	return val_loss, val_acc


# 正式训练
if __name__ == '__main__':
	train_losses = []
	train_acces = []
	val_losses = []
	val_acces = []

	for epoch in range(1, EPOCHS + 1):
		tr_loss, tr_acc = train(train_dataloader, epoch)
		val_loss, val_acc = validation(val_dataloader, epoch)
		train_losses.append(tr_loss)
		train_acces.append(tr_acc)
		val_losses.append(val_loss)
		val_acces.append(val_acc)

	model_pth = 'model_' + str(EPOCHS) + '.pth'
	torch.save(model.state_dict(), model_pth)

	# 可视化
	plt.plot(train_losses)
	plt.plot(val_losses)
	plt.legend(['train_loss', 'val_loss'])
	plt.title("The loss of Text_CNN model")
	plt.savefig('loss.png')

	plt.figure()
	plt.plot(train_acces)
	plt.plot(val_acces)
	plt.legend(['train_accuracy', 'val_accuracy'])
	plt.title("The accuracy of Text_CNN model")
	plt.savefig('accuracy.png')
