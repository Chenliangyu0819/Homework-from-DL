"""
作者：  cly
日期：  2022年06月29日
代码调整PEP8：Ctrl+Alt+L
"""
import numpy as np
import torch
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import *
from torch.utils.data import DataLoader

BATCH_SIZE = 32
EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 读入预处理的数据
datas = np.load("./tang-实验三/tang.npz", allow_pickle=True)
data = datas['data'][:10000]  # 减少训练的量，选10000首诗进行训练。
ix2word = datas['ix2word'].item()
word2ix = datas['word2ix'].item()

# # 将narray转为torch.Tensor
data = torch.from_numpy(data)
# DataLoader是一个迭代器，方便访问dataset里面的对象
# train_loader中返回的对象（句子）个数seq_len * batch_size = vocab_size
train_loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)  # (seq_len,batch_size)


class PoetryModel(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim):
		super(PoetryModel, self).__init__()  # 继承pytorch中tdata.Dataset这个类（此为抽象类，需要派生一个子类构造数据集）
		self.hidden_dim = hidden_dim
		self.embedding = nn.Embedding(vocab_size, embedding_dim)  # vocab_size:就是ix2word的长度。
		self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=3)
		self.classifier = nn.Sequential(
			nn.Linear(self.hidden_dim, 512),
			nn.ReLU(inplace=True),
			nn.Linear(512, 2048),
			nn.ReLU(inplace=True),
			nn.Linear(2048, vocab_size)  # 最后还是转为vocab_size维度
		)

	def forward(self, input, hidden=None):
		# hidden指的是给定的输入句子，其会转化成这个h0和c0，作为起始的hidden。即hidden=(h0,c0)
		seq_len, batch_size = input.size()

		if hidden is None:
			h_0 = input.data.new(3 * 1, batch_size, self.hidden_dim).fill_(0).float()  # 1表示单向LSTM
			c_0 = input.data.new(3 * 1, batch_size, self.hidden_dim).fill_(0).float()
		else:
			h_0, c_0 = hidden

		embeds = self.embedding(input)  # [batch, seq_len] => [batch, seq_len, embed_dim] 单词转为词向量
		output, hidden = self.lstm(embeds, (h_0, c_0))
		output = self.classifier(output.view(seq_len * batch_size, -1))

		return output, hidden


# 配置模型，是否继续上一次的训练
model = PoetryModel(len(word2ix), embedding_dim=128, hidden_dim=256)
model.to(DEVICE)

# 优化器、损失函数、学习率的调整
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
# optimizer = torch.optim.SGD(model.parameters(), lr=5e-3, momentum=0.9, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=5)  # 每5个epoch调整一次学习率


def train(model, dataloader, ix2word, word2ix, device, optimizer, scheduler, epoch):
	start = time.time()
	model.train()  # 开启训练模式
	train_loss = 0.0

	for batch_idx, data in enumerate(dataloader):  # data=(训练数据，标签)
		data = data.long().transpose(1, 0).contiguous()
		data = data.to(device)
		optimizer.zero_grad()  # 梯度置零
		input, target = data[:-1, :], data[1:, :]  # 返回输入和标签
		output, _ = model(input)  # 前向传播进行预测
		loss = criterion(output, target.view(-1))  # 计算bath_size个样本的预测值和标签值之间的CrossEntropyLoss()
		loss.backward()  # 误差后向传播
		optimizer.step()  # 参数更新
		train_loss += loss.item()  # 注意loss是在每个batch上取过均值的

		if (batch_idx + 1) % 200 == 0:
			print('train epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.6f}'.format(
				epoch, batch_idx * len(data[1]), len(dataloader.dataset),  # 这里的data维度是(len(poem)=124,batch_size)
				# batch_idx=199,len(data[1])=batch_size=32,199*32=6368
					   100. * batch_idx / len(dataloader), loss.item()))  # len(dataloader)=seq_len=10000/batch_size

	end = time.time()
	train_loss *= BATCH_SIZE  # train_loss本身是在batch上取过均值的
	train_loss /= len(train_loader.dataset)
	print('\ntrain epoch: {}\t average loss: {:.6f}\tETA: {:.2f}s\n'.format(epoch, train_loss, end - start))
	scheduler.step()  # 调整学习率

	return train_loss  # 返回的是在每个epoch上单个样本的平均loss


if __name__ == '__main__':
	train_losses = []

	for epoch in range(1, EPOCHS + 1):
		tr_loss = train(model, train_loader, ix2word, word2ix, DEVICE, optimizer, scheduler, epoch)
		train_losses.append(tr_loss)

	# 可视化
	plt.plot(train_losses)
	plt.ylim(ymin=0.5, ymax=6)
	plt.title("The loss of LSTM model")
	plt.savefig('training_process.png')

	# 保存模型
	filename = "model_" + str(EPOCHS)
	torch.save(model, filename)
