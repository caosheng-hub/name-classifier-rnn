# -*- coding:utf-8 -*-
# 导入torch工具
import torch
# 导入nn准备构建模型
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# 导入torch的数据源 数据迭代器工具包
from torch.utils.data import Dataset, DataLoader
# 用于获得常见字母及字符规范化
import string
# 导入时间工具包
import time
# 引入制图工具包
import matplotlib.pyplot as plt
# 进度条
from tqdm import tqdm
import json

# todo: 1.获取常用的字符数量: 也就是one-hot编码的去重之后的词汇的总量n
all_letters = string.ascii_letters + " ,;.'"
n_letters = len(all_letters)

# todo:2. 获取国家名种类数
categorys = ['Italian', 'English', 'Arabic', 'Spanish', 'Scottish', 'Irish', 'Chinese', 'Vietnamese', 'Japanese',
             'French', 'Greek', 'Dutch', 'Korean', 'Polish', 'Portuguese', 'Russian', 'Czech', 'German']
# 国家名个数
categorynum = len(categorys)

# todo:3. 读取数据到内存
def read_data(filepath):
    # 3.1 定义俩个空列表my_list_x（存储人名），my_list_y（存储国家名）
    my_list_x,my_list_y = [],[]
    # 3.2 读取文件内容
    with open(filepath, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            if len(line) <= 5:
                continue
            x,y = line.strip().split('\t')
            my_list_x.append(x)
            my_list_y.append(y)
    return my_list_x,my_list_y

# todo:4. 构建Dataset类
class NameDataset(Dataset):
    def __init__(self,my_list_x,my_list_y):
        self.my_list_x = my_list_x
        self.my_list_y = my_list_y
        self.sample_len = len(my_list_x) # 样本个数
    def __len__(self):
        return self.sample_len
    # 根据索引取出元素item:代表索引
    def __getitem__(self, item):
        # 1. 对异常索引进行修正
        item = min(max(item,0),self.sample_len-1)
        # 2. 根据索引取出样本
        x = self.my_list_x[item]
        y = self.my_list_y[item]
        # 3. 将人名变成one-hot编码的张量形式
        # 3.1 初始化全零的张量 （3，57）
        tensor_x = torch.zeros(len(x),n_letters)
        # 3.2 遍历人名的每个字母，进行one-hot编码的赋值
        for idx,letter in enumerate(x):
            tensor_x[idx][all_letters.find(letter)] = 1
        tensor_y = torch.tensor(categorys.index(y),dtype=torch.long)
        return tensor_x,tensor_y

# todo:5. 实例化dataloader对象
def get_dataloader():
    # 读取文档数据
    my_list_x,my_list_y = read_data(filepath='E:/PycharmProjects/MyFirstProject/NLP/RNN案例——人名分类器/name_classfication.txt')
    # 获取dataset对象
    name_dataset = NameDataset(my_list_x,my_list_y)
    # 封装dataset得到dataloader对象：会对数据进行增维
    train_dataloader = DataLoader(name_dataset,batch_size=1,shuffle=True)
    return train_dataloader

# todo:6. 定义RNN层
class NameRNN(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_layers=1):
        super().__init__()
        # input_size代表输入数据的词嵌入维度
        self.input_size = input_size
        # hidden_size代表隐藏层的维度
        self.hidden_size = hidden_size
        # output_size代表输出的维度:18个国家
        self.output_size = output_size
        # num_layers代表RNN的层数
        self.num_layers = num_layers
        # 定义RNN层
        self.rnn = nn.RNN(self.input_size,self.hidden_size,num_layers)
        # 定义输出层
        self.out = nn.Linear(self.hidden_size,self.output_size)
        # 定义激活函数
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self,x,h0):
        # x:代表输入的原始数据维度（seq_len,input_size）
        # h0:代表初始化的值，维度（num_layers,batch_size,hidden_size） （1，1，128）
        # x需要先升维 （seq_len ,input_size） (seq_len,batch_size,input_size）
        x1 = torch.unsqueeze(x,dim=1) # x.unsqueeze(dim=1)
        # 将h0和x1送入RNN模型
        output,hn = self.rnn(x1,h0)
        # 获取最后一个单词的隐藏层张量来代表整个人名的语意
        # 这里也可以用hn代替
        temp = output[-1] # [1,128]
        # 将temp送入输出层：result [1,18]
        result = self.out(temp)
        return self.softmax(result),hn

    def init_hidden(self):
        return torch.zeros(self.num_layers,1,self.hidden_size)

# todo:7. 定义LSTM层
class NameLSTM(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_layers=1):
        super().__init__()
        # input_size代表输入数据的词嵌入维度
        self.input_size = input_size
        # hidden_size代表隐藏层的维度
        self.hidden_size = hidden_size
        # output_size代表输出的维度:18个国家
        self.output_size = output_size
        # num_layers代表LSTM的层数
        self.num_layers = num_layers
        # 定义LSTM层
        self.lstm = nn.LSTM(self.input_size,self.hidden_size,num_layers)
        # 定义输出层
        self.out = nn.Linear(self.hidden_size,self.output_size)
        # 定义激活函数
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self,x0,h0,c0):
        # x0:代表输入的原始数据维度[seq_len, input_size]
        # h0,c0:代表初始化的值, [num_layers, batch_size, hidden_size]-->[1, 1, 128]
        # x0需要先升维: [seq_len, input_size]-->[seq_len, batch_size, input_size]
        x1 = torch.unsqueeze(x0,dim=1)
        output,(hn,cn) = self.lstm(x1,(h0,c0))
        temp = output[-1]
        result = self.out(temp)
        return self.softmax(result),hn,cn

    def init_hidden(self):
        h0 = torch.zeros(self.num_layers,1,self.hidden_size)
        c0 = torch.zeros(self.num_layers,1,self.hidden_size)
        return h0,c0

# todo:8. 定义GRU层
class NameGRU(nn.Module):
    def __init__(self,input_size,hidden_size,output_size,num_layers=1):
        super().__init__()
        # input_size代表输入数据的词嵌入维度
        self.input_size = input_size
        # hidden_size代表隐藏层的维度
        self.hidden_size = hidden_size
        # output_size代表输出的维度:18个国家
        self.output_size = output_size
        # num_layers代表GRU的层数
        self.num_layers = num_layers
        # 定义RNN层
        self.gru = nn.GRU(self.input_size,self.hidden_size,num_layers)
        # 定义输出层
        self.out = nn.Linear(self.hidden_size,self.output_size)
        # 定义激活函数
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self,x,h0):
        # x:代表输入的原始数据维度（seq_len,input_size）
        # h0:代表初始化的值，维度（num_layers,batch_size,hidden_size） （1，1，128）
        # x需要先升维 （seq_len ,input_size） (seq_len,batch_size,input_size）
        x1 = torch.unsqueeze(x,dim=1) # x.unsqueeze(dim=1)
        # 将h0和x1送入GRU模型
        output,hn = self.gru(x1,h0)
        # 获取最后一个单词的隐藏层张量来代表整个人名的语意
        # 这里也可以用hn代替
        temp = output[-1] # [1,128]
        # 将temp送入输出层：result [1,18]
        result = self.out(temp)
        return self.softmax(result),hn

    def init_hidden(self):
        return torch.zeros(self.num_layers,1,self.hidden_size)

mylr = 1e-3
epochs = 1
# todo:9. 定义RNN训练函数
def train_rnn():
    # 1. 读取文档数据
    my_list_x,my_list_y = read_data(filepath='E:/PycharmProjects/MyFirstProject/NLP/RNN案例——人名分类器/name_classfication.txt')
    # 2. 获取dataset对象
    name_dataset = NameDataset(my_list_x,my_list_y)
    # 3. 实例化模型
    input_size = 57
    hidden_size = 128
    output_size = 18
    rnn_model = NameRNN(input_size,hidden_size,output_size)
    # 4. 实例化损失函数对象
    cross_entropy = nn.NLLLoss()
    # 5. 实例化优化器对象
    adam = optim.Adam(rnn_model.parameters(),lr=mylr)
    # 6. 定义训练模型的打印日志的参数
    start_time = time.time()  # 开始的时间
    total_iter_num = 0  # 已经训练的样本的总个数
    total_loss = 0.0  # 已经训练的样本的损失之和
    total_loss_list = []  # 每隔100个样本计算一下平均损失，画图
    total_num_acc = 0  # 已经训练的样本中预测正确的样本的个数
    total_acc_list = []  # 每隔100个样本计算一下平均准确率，画图
    # 7. 开始训练
    # 开始外部epoch的迭代
    for epoch in range(epochs):
        # 实例化dataloader
        train_dataloader = DataLoader(dataset=name_dataset,batch_size=1,shuffle=True)
        # 开始内部数据迭代
        for idx,(x,y) in enumerate(tqdm(train_dataloader)):
            # 将数据送入模型
            h0 = rnn_model.init_hidden()
            output,hn = rnn_model(x[0],h0)
            # 计算损失
            my_loss = cross_entropy(output,y)
            # 梯度清零
            adam.zero_grad()
            # 反向传播
            my_loss.backward()
            # 更新参数
            adam.step()
            # 打印日志参数
            # 获取已经训练的样本的总数
            total_iter_num += 1
            # 获取已经训练的样本的总损失
            total_loss = total_loss + my_loss.item()
            # 获取已经训练的样本中预测正确的个数
            pre_idx = 1 if torch.argmax(output).item() == y.item() else 0
            total_num_acc += pre_idx
            # 每100次训练，求一次平均损失和平均准确率
            if total_iter_num % 100 == 0:
                # 保留平均损失
                avg_loss = total_loss / 100
                total_loss_list.append(avg_loss)
                # 保留平均准确率
                avg_acc = total_num_acc / 100
                total_acc_list.append(avg_acc)
            # 每隔2000步，打印日志
            if total_iter_num % 2000 == 0:
                temp_loss = total_loss / 2000
                temp_acc = total_num_acc / 2000
                print('轮次:%d, 损失:%.6f, 时间:%d，准确率:%.3f' %(epoch+1, temp_loss, time.time() - start_time, temp_acc))
        # 每轮都保留一个模型
        torch.save(rnn_model.state_dict(), './save_model/ai23_rnn_%d.bin' % (epoch + 1))
    # 8. 计算训练的总时间
    all_time = time.time() - start_time
    # 9. 将训练结果保存
    dict1 = {"total_loss_list":total_loss_list,"all_time":all_time,"total_acc_list":total_acc_list}
    with open('rnn_result.json','w') as fw:
        fw.write(json.dumps(dict1))

# todo:10. 定义LSTM训练函数
def train_lstm():
    # 1. 读取文档数据
    my_list_x,my_list_y = read_data(filepath='E:/PycharmProjects/MyFirstProject/NLP/RNN案例——人名分类器/name_classfication.txt')
    # 2. 获取dataset对象
    name_dataset = NameDataset(my_list_x,my_list_y)
    # 3. 实例化模型
    input_size = 57
    hidden_size = 128
    output_size = 18
    lstm_model = NameLSTM(input_size,hidden_size,output_size)
    # 4. 实例化损失函数对象
    cross_entropy = nn.NLLLoss()
    # 5. 实例化优化器对象
    adam = optim.Adam(lstm_model.parameters(),lr=mylr)
    # 6. 定义训练模型的打印日志的参数
    start_time = time.time()  # 开始的时间
    total_iter_num = 0  # 已经训练的样本的总个数
    total_loss = 0.0  # 已经训练的样本的损失之和
    total_loss_list = []  # 每隔100个样本计算一下平均损失，画图
    total_num_acc = 0  # 已经训练的样本中预测正确的样本的个数
    total_acc_list = []  # 每隔100个样本计算一下平均准确率，画图
    # 7. 开始训练
    # 开始外部epoch的迭代
    for epoch in range(epochs):
        # 实例化dataloader
        train_dataloader = DataLoader(dataset=name_dataset,batch_size=1,shuffle=True)
        # 开始内部数据迭代
        for idx,(x,y) in enumerate(tqdm(train_dataloader)):
            # 将数据送入模型
            h0,c0 = lstm_model.init_hidden()
            output,hn,cn = lstm_model(x[0],h0,c0)
            # 计算损失
            my_loss = cross_entropy(output,y)
            # 梯度清零
            adam.zero_grad()
            # 反向传播
            my_loss.backward()
            # 更新参数
            adam.step()
            # 打印日志参数
            # 获取已经训练的样本的总数
            total_iter_num += 1
            # 获取已经训练的样本的总损失
            total_loss = total_loss + my_loss.item()
            # 获取已经训练的样本中预测正确的个数
            pre_idx = 1 if torch.argmax(output).item() == y.item() else 0
            total_num_acc += pre_idx
            # 每100次训练，求一次平均损失和平均准确率
            if total_iter_num % 100 == 0:
                # 保留平均损失
                avg_loss = total_loss / 100
                total_loss_list.append(avg_loss)
                # 保留平均准确率
                avg_acc = total_num_acc / 100
                total_acc_list.append(avg_acc)
            # 每隔2000步，打印日志
            if total_iter_num % 2000 == 0:
                temp_loss = total_loss / 2000
                temp_acc = total_num_acc / 2000
                print('轮次:%d, 损失:%.6f, 时间:%d，准确率:%.3f' %(epoch+1, temp_loss, time.time() - start_time, temp_acc))
        # 每轮都保留一个模型
        torch.save(lstm_model.state_dict(), './save_model/ai23_lstm_%d.bin' % (epoch + 1))
    # 8. 计算训练的总时间
    all_time = time.time() - start_time
    # 9. 将训练结果保存
    dict1 = {"total_loss_list":total_loss_list,"all_time":all_time,"total_acc_list":total_acc_list}
    with open('lstm_result.json','w') as fw:
        fw.write(json.dumps(dict1))

# todo:11. 定义GRU训练函数
def train_gru():
    # 1. 读取文档数据
    my_list_x,my_list_y = read_data(filepath='E:/PycharmProjects/MyFirstProject/NLP/RNN案例——人名分类器/name_classfication.txt')
    # 2. 获取dataset对象
    name_dataset = NameDataset(my_list_x,my_list_y)
    # 3. 实例化模型
    input_size = 57
    hidden_size = 128
    output_size = 18
    gru_model = NameGRU(input_size,hidden_size,output_size)
    # 4. 实例化损失函数对象
    cross_entropy = nn.NLLLoss()
    # 5. 实例化优化器对象
    adam = optim.Adam(gru_model.parameters(),lr=mylr)
    # 6. 定义训练模型的打印日志的参数
    start_time = time.time()  # 开始的时间
    total_iter_num = 0  # 已经训练的样本的总个数
    total_loss = 0.0  # 已经训练的样本的损失之和
    total_loss_list = []  # 每隔100个样本计算一下平均损失，画图
    total_num_acc = 0  # 已经训练的样本中预测正确的样本的个数
    total_acc_list = []  # 每隔100个样本计算一下平均准确率，画图
    # 7. 开始训练
    # 开始外部epoch的迭代
    for epoch in range(epochs):
        # 实例化dataloader
        train_dataloader = DataLoader(dataset=name_dataset,batch_size=1,shuffle=True)
        # 开始内部数据迭代
        for idx,(x,y) in enumerate(tqdm(train_dataloader)):
            # 将数据送入模型
            h0 = gru_model.init_hidden()
            output,hn = gru_model(x[0],h0)
            # 计算损失
            my_loss = cross_entropy(output,y)
            # 梯度清零
            adam.zero_grad()
            # 反向传播
            my_loss.backward()
            # 更新参数
            adam.step()
            # 打印日志参数
            # 获取已经训练的样本的总数
            total_iter_num += 1
            # 获取已经训练的样本的总损失
            total_loss = total_loss + my_loss.item()
            # 获取已经训练的样本中预测正确的个数
            pre_idx = 1 if torch.argmax(output).item() == y.item() else 0
            total_num_acc += pre_idx
            # 每100次训练，求一次平均损失和平均准确率
            if total_iter_num % 100 == 0:
                # 保留平均损失
                avg_loss = total_loss / 100
                total_loss_list.append(avg_loss)
                # 保留平均准确率
                avg_acc = total_num_acc / 100
                total_acc_list.append(avg_acc)
            # 每隔2000步，打印日志
            if total_iter_num % 2000 == 0:
                temp_loss = total_loss / 2000
                temp_acc = total_num_acc / 2000
                print('轮次:%d, 损失:%.6f, 时间:%d，准确率:%.3f' %(epoch+1, temp_loss, time.time() - start_time, temp_acc))
        # 每轮都保留一个模型
        torch.save(gru_model.state_dict(), './save_model/ai23_gru_%d.bin' % (epoch + 1))
    # 8. 计算训练的总时间
    all_time = time.time() - start_time
    # 9. 将训练结果保存
    dict1 = {"total_loss_list":total_loss_list,"all_time":all_time,"total_acc_list":total_acc_list}
    with open('gru_result.json','w') as fw:
        fw.write(json.dumps(dict1))

# todo:12. 绘图对比不同模型的性能
def compare_rnns():
    # 1. 读取rnn的训练结果
    with open('rnn_result.json','r') as fr:
        rnn_dict = json.loads(fr.read())
    # 2. 读取lstm的训练结果
    with open('lstm_result.json','r') as fr:
        lstm_dict = json.loads(fr.read())
    # 3. 读取gru的训练结果
    with open('gru_result.json','r') as fr:
        gru_dict = json.loads(fr.read())
    # 4. 绘图
    # 4.1 绘制损失对比曲线图
    plt.figure(0)
    plt.plot(rnn_dict["total_loss_list"], label='RNN')
    plt.plot(lstm_dict["total_loss_list"], label='LSTM', color='red')
    plt.plot(gru_dict["total_loss_list"], label='GRU', color='blue')
    plt.legend(loc='upper left')
    plt.savefig('./ai23_avg_loss.png')
    plt.show()
    # 4.2 绘制柱状图对比时间
    plt.figure(1)
    x_data = ["RNN", "LSTM", "GRU"]
    y_data = [rnn_dict["all_time"], lstm_dict["all_time"], gru_dict["all_time"]]
    plt.bar(range(len(x_data)), y_data, tick_label=x_data)
    plt.savefig("./ai23_time.png")
    plt.show()
    # 4.3 绘制准确率对比图
    plt.figure(2)
    plt.plot(rnn_dict["total_acc_list"], label='RNN')
    plt.plot(lstm_dict["total_acc_list"], label='LSTM', color='red')
    plt.plot(gru_dict["total_acc_list"], label='GRU', color='blue')
    plt.legend(loc='upper left')
    plt.savefig('./ai23_avg_acc.png')
    plt.show()

# todo:13. 定义将人名转换为向量的函数
def name2tensor(x):
    """
    将x转换成one-hot编码的向量形式
    :param x: 'bai'
    :return:[[0,1,...], [..], [..]]
    """
    # 1. 初始化全零的张量 （3，57）
    tensor_x = torch.zeros(len(x),n_letters)
    # 2. 遍历人名的每一个字母，进行one-hot编码
    for idx,letter in enumerate(x):
        tensor_x[idx][all_letters.find(letter)] = 1
    return tensor_x

# todo:14. 定义rnn模型的预测函数
def rnn_predict(x):
    # 1.将人名转换为向量
    tensor_x = name2tensor(x)
    # 2.实例化模型并加载训练好的模型参数
    input_size = 57
    hidden_size = 128
    output_size = 18
    rnn_model = NameRNN(input_size,hidden_size,output_size)
    rnn_model.load_state_dict(torch.load('./save_model/ai23_rnn_1.bin'))
    # 3.预测
    with torch.no_grad():
        # 将数据送入模型
        h0 = rnn_model.init_hidden()
        output,hn = rnn_model(tensor_x,h0)
        # 取出预测结果中topk3
        topv,topi = torch.topk(output,k=3,dim=1)
        for i in range(3):
            tempv = topv[0][i]
            tempi = topi[0][i]
            str_class = categorys[tempi]
            print(f'当前的人名是：{x}, 预测值是：{tempv:.2f}, 预测真实国家是：{str_class}')

# todo:15. 定义lstm模型的预测函数
def lstm_predict(x):
    # 1.将人名转换为向量
    tensor_x = name2tensor(x)
    # 2.实例化模型并加载训练好的模型参数
    input_size = 57
    hidden_size = 128
    output_size = 18
    lstm_model = NameLSTM(input_size,hidden_size,output_size)
    lstm_model.load_state_dict(torch.load('./save_model/ai23_lstm_1.bin'))
    # 3.预测
    with torch.no_grad():
        # 将数据送入模型
        h0,c0 = lstm_model.init_hidden()
        output,hn,cn = lstm_model(tensor_x,h0,c0)
        # 取出预测结果中topk3
        topv,topi = torch.topk(output,k=3,dim=1)
        for i in range(3):
            tempv = topv[0][i]
            tempi = topi[0][i]
            str_class = categorys[tempi]
            print(f'当前的人名是：{x}, 预测值是：{tempv:.2f}, 预测真实国家是：{str_class}')

# todo: 16.定义gru模型的预测函数
def gru_predict(x):
    # 1.将x--》人名转换为向量
    tensor_x = name2tensor(x)
    # 2. 实例化模型并加载训练好的模型参数
    input_size = 57
    hidden_size = 128
    output_size = 18
    gru_model = NameGRU(input_size, hidden_size, output_size)
    gru_model.load_state_dict(torch.load('./save_model/ai23_gru_1.bin'))
    # 3. 预测
    with torch.no_grad():
        # 将数据送入模型
        h0 = gru_model.init_hidden()
        output, hn = gru_model(tensor_x, h0)
        print(f'output--》{output}')
        # 取出预测结果中topk3
        topv, topi = torch.topk(output, k=3, dim=1)
        print(f'topv--》{topv}')
        print(f'topi--》{topi}')
        print(f'rnn预测的结果')
        for i in range(3):
            tempv = topv[0][i]
            tempi = topi[0][i]
            str_class = categorys[tempi]
            print(f'当前的人名是：{x}, 预测值是：{tempv:.2f}, 预测真实国家是：{str_class}')

if __name__ == '__main__':
    my_list_x,my_list_y = read_data(filepath='E:/PycharmProjects/MyFirstProject/NLP/RNN案例——人名分类器/name_classfication.txt')
    train_dataloader = get_dataloader()
    for tensor_x,tensor_y in train_dataloader:
        print(f'{tensor_x}')
        print(f'{tensor_x.shape}')
        print(f'{tensor_y}')
        break
    input_size = 57
    hidden_size = 128
    output_size = 18
    model = NameRNN(input_size,hidden_size,output_size)
    model = NameLSTM(input_size, hidden_size, output_size)
    model = NameGRU(input_size, hidden_size, output_size)
    print(model)
    for x,y in train_dataloader:
        # x.shape [batch_size,seq_len,input_size],batch_size = 1
        h0 = model.init_hidden()
        output,hn = model(x[0],h0)
        break
    train_rnn()
    train_lstm()
    train_gru()
    compare_rnns()
    result = name2tensor(x='python')
    print(result)
    rnn_predict(x='java')
    lstm_predict(x='kris')
    gru_predict(x='coke')
