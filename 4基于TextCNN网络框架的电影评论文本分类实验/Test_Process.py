"""
作者：  cly
日期：  2022年07月02日
代码调整PEP8：Ctrl+Alt+L
"""
from Text_CNN import *
from torchinfo import summary


# 加载测试集
test_dataset = TensorDataset(torch.from_numpy(test_contents).type(torch.float),
							 torch.from_numpy(test_labels).type(torch.long))
test_dataloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE,
							 shuffle=False, num_workers=0)
# 读取模型
model = TextCNN(config)
model.load_state_dict(torch.load('model_15.pth'))


def test(dataloader):
	model.eval()
	model.to(DEVICE)

	# 测试过程
	count, correct = 0, 0
	for _, (x, y) in enumerate(dataloader):
		x, y = x.to(DEVICE), y.to(DEVICE)
		output = model(x)
		correct += (output.argmax(1) == y).float().sum().item()
		count += len(x)

	# 打印准确率
	print('test accuracy:{:.2f}%.'.format(100 * correct / count))


test(test_dataloader)

# 将summary() 输出保存为文件
from contextlib import redirect_stdout

with open('TextCNN_model_summary.txt', 'w') as f:
	with redirect_stdout(f):
		summary(model)

# 打印模型的 state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
	print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print('===============================================')
print(model)
print('===============================================')