import math
import struct
import numpy as np
from pathlib import Path # 处理路径
import matplotlib.pyplot as plt

def tanh(x):
    return np.tanh(x)

def softmax(x):
    exp = np.exp(x - x.max())
    return exp / exp.sum()

def init_paramters_b(layer): # layer 输入层数
    dist = distribution[layer]['b']
    # 数组的大小是dimensions[layer] = [28 * 28, 10]
    # 等价于 randdouble(0, 1) * (r - l) + l
    res = np.random.rand(dimensions[layer]) * (dist[1] - dist[0]) + dist[0]
    # print("深度:" + str(layer) + "初始化 b: " + str(res))
    return res

def init_paramters_w(layer):
    dist = distribution[layer]['w']
    # print("深度:" + str(layer) + "初始化 w 数组大小" + str(dimensions[layer - 1]) + " * " + str(dimensions[layer]))
    # 返回二维的随机数组
    return np.random.rand(dimensions[layer - 1], dimensions[layer]) * (dist[1] - dist[0]) + dist[0]

def init_paraments():
    parameter = []
    for i in range(len(distribution)):
        layer_parameter = {}
        for j in distribution[i].keys():
            if (j == 'b'):
                layer_parameter['b'] = init_paramters_b(i)
            elif (j == 'w'):
                layer_parameter['w'] = init_paramters_w(i)
        parameter.append(layer_parameter)
    return parameter

def predict(img, parameters):
    l_0_in = img + parameters[0]['b']
    l_0_out = activation[0](l_0_in)
    l_1_in = np.dot(l_0_out, parameters[1]['w'] + parameters[1]['b'])
    l_1_out = activation[1](l_1_in)
    return l_1_out

def show_train(index):
    print("lab:", end = ' '), print(train_lab[index])
    plt.imshow(train_img[index].reshape(28, 28), cmap = 'gray')
    plt.show()

def show_test(index):
    print("lab:", end = ' '), print(test_lab[index])
    plt.imshow(test_img[index].reshape(28, 28), cmap = 'gray')
    plt.show()

def show_valid(index):
    print("lab:", end=' '), print(valid_lab[index])
    plt.imshow(valid_img[index].reshape(28, 28), cmap='gray')
    plt.show()

dimensions = [28 * 28, 10] # 存储两个数据 输入数量 m 和 输出数量 n
activation = [tanh, softmax]

distribution = [ # 列表套字典
    {'b' : [0, 0]}, # 第一层的偏置函数和第二层的偏置函数推荐设置为 1
    {'b' : [0, 0], 'w': [-math.sqrt(6 / (dimensions[0] + dimensions[1])), math.sqrt(6 / (dimensions[0] + dimensions[1]))]}, # 假设只有两层 784 -> 10
]

# 初始化
parameters = init_paraments()

# 测试正确性
# print(predict(np.random.rand(784), parameters).argmax())

# 读入测试集
dataset_path = Path('./MNIST')
train_img_path = dataset_path / 'train-images.idx3-ubyte'
train_lab_path = dataset_path / 'train-labels.idx1-ubyte'
test_img_path = dataset_path / 't10k-images.idx3-ubyte'
test_lab_path = dataset_path / 't10k-labels.idx1-ubyte'

train_num = 50000 # 同时给问题和答案 # 过拟合
valid_num = 10000 # 给问题不给答案 # 永远不能看见 # 测验 # 自己可以调整
test_num = 10000 # 只能看到结果的数据

with open(train_img_path, "rb") as f:
    struct.unpack('>4i', f.read(16))
    # train_img = np.fromfile(f, dtype = np.uint8).reshape(-1, 28 * 28)
    tmp_img = np.fromfile(f, dtype = np.uint8).reshape(-1, 28 * 28)
    train_img = tmp_img[:train_num]
    valid_img = tmp_img[train_num:]

with open(test_img_path, "rb") as f:
    struct.unpack('>4i', f.read(16))
    test_img = np.fromfile(f, dtype = np.uint8).reshape(-1, 28 * 28)

with open(train_lab_path, "rb") as f:
    struct.unpack('>2i', f.read(8))
    # train_lab = np.fromfile(f, dtype = np.uint8)
    tmp_lab = np.fromfile(f, dtype = np.uint8)
    train_lab = tmp_lab[:train_num]
    valid_lab = tmp_lab[train_num:]

with open(test_lab_path, "rb") as f:
    struct.unpack('>2i', f.read(8))
    test_lab = np.fromfile(f, dtype = np.uint8)

show_train(np.random.randint(train_num))
show_valid(np.random.randint(valid_num))
show_test(np.random.randint(test_num))





