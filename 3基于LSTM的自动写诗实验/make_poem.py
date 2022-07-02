"""
作者：  cly
日期：  2022年06月26日
代码调整PEP8：Ctrl+Alt+L
"""
# 在调用csdn_lstm的时候，一定要在该py文件中加入if __name__ == '__main__'进行封装，
# 否则导入时会默认将Poem_lstm.py文件执行一遍！
from torchinfo import summary
from Poem_lstm import *

# 加载模型
model = PoetryModel(len(word2ix), embedding_dim=128, hidden_dim=256)
model = torch.load('model_20')
model.eval()  # 开启测试模式

# 打印模型的 state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
	print(param_tensor, "\t", model.state_dict()[param_tensor].size())

print('===============================================')
print(model)
print('===============================================')


# 将summary() 输出保存为文件
from contextlib import redirect_stdout

with open('PoetryModel_model_summary.txt', 'w') as f:
	with redirect_stdout(f):
		summary(model)


# 测试评估
def generate(model, start_words, ix2char, char2ix, max_gen_len):
	'''给定首句或几个汉字进行诗句生成'''
	# max_gen_len 生成诗句的最大长度

	print("给定的句子为：", end=" ")
	for i in start_words:
		print(i, end=" ")
	print()

	# 读取唐诗的第一句
	results = list(start_words)
	start_word_len = len(start_words)

	# 设置第一个词为<START>
	input = torch.Tensor([char2ix['<START>']]).view(1, 1).long()
	hidden = None

	# 生成唐诗
	for i in range(max_gen_len):
		output, hidden = model(input, hidden)  # 递归迭代隐藏层hidden，同时将上一个output作为下一个input

		# 读取第一句
		if i < start_word_len:
			w = results[i]
			input = input.data.new([char2ix[w]]).view(1, 1)

		# 生成后面的句子
		else:
			top_index = output.data[0].topk(1)[1][0].item()
			w = ix2char[top_index]
			results.append(w)
			input = input.data.new([top_index]).view(1, 1)

		# 结束标志
		if w == '<EOP>':
			del results[-1]
			break

	return results


def gen_acrostic(model, start_words, ix2char, char2ix):
	'''
	生成藏头诗
	'''
	print("藏头诗中头部单词为：", end=" ")
	for i in start_words:
		print(i, end=" ")
	print()

	# 读取唐诗的“头”
	results = []
	start_word_len = len(start_words)

	# 设置第一个词为<START>
	input = (torch.Tensor([char2ix['<START>']]).view(1, 1).long())
	hidden = None

	index = 0  # 指示已生成了多少句
	pre_word = '<START>'  # 上一个词

	# 生成藏头诗
	for i in range(max_gen_len_acrostic):
		output, hidden = model(input, hidden)
		top_index = output.data[0].topk(1)[1][0].item()
		w = ix2char[top_index]

		# 如果遇到标志一句的结尾，喂入下一个“头”
		if (pre_word in {u'。', u'！', '<START>'}):
			# 如果生成的诗已经包含全部“头”，则结束
			if index == start_word_len:
				break
			# 把“头”作为输入喂入模型
			else:
				w = start_words[index]
				index += 1
				input = (input.data.new([char2ix[w]])).view(1, 1)

		# 否则，把上一次预测作为下一个词输入
		else:
			input = (input.data.new([char2ix[w]])).view(1, 1)
		results.append(w)
		pre_word = w

	return results


# 续写诗句
start_words = '雨'  # 唐诗的第一句
max_gen_len = 72  # 生成唐诗的最长长度
results = generate(model, start_words, ix2word, word2ix, max_gen_len)
print('本次续写生成的诗句总长度为：{}'.format(len(results)))
print("续写如下：\n")

poetry = ''
for word in results:
	poetry += word
	if word == '。' or word == '!':  # 换行
		poetry += '\n'
print(poetry)

# 撰写藏头诗
print()
start_words_acrostic = '花好月圆'  # 唐诗的“头”
max_gen_len_acrostic = 125  # 生成唐诗的最长长度
results_acrostic = gen_acrostic(model, start_words_acrostic, ix2word, word2ix)
print('本次生成的藏头诗总长度为：{}'.format(len(results_acrostic)))
print("生成如下：\n")

poetry = ''
for word in results_acrostic:
	poetry += word
	if word == '。' or word == '!':  # 换行
		poetry += '\n'
print(poetry)
