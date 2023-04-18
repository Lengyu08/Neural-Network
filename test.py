import copy

import numpy as np
import math,struct,pickle
from pathlib import Path
import matplotlib.pyplot as plt
def tanh(x):
    return np.tanh(x)
def softmax(x):
    exp=np.exp(x-x.max())
    return exp/exp.sum()
def d_softmax(data):
    sm=softmax(data)
    return np.diag(sm)-np.outer(sm,sm)

# def d_tanh(data):
#     return np.diag(1/(np.cosh(data))**2)
def d_tanh(data):
    return 1/(np.cosh(data))**2
def init_parameters_b(layer):
    dist=distribution[layer]['b']
    return np.random.rand(dimensions[layer])*(dist[1]-dist[0])+dist[0]
def init_parameters_w(layer):
    dist=distribution[layer]['w']
    return np.random.rand(dimensions[layer-1],dimensions[layer])*(dist[1]-dist[0])+dist[0]
def predict(img,parameters):
    l0_in=img+parameters[0]['b']
    l0_out=activation[0](l0_in)
    l1_in=np.dot(l0_out,parameters[1]['w'])+parameters[1]['b']
    l1_out=activation[1](l1_in)
    return l1_out
def init_parameters():
    parameter=[]
    for i in range(len(distribution)):
        layer_parameter={}
        for j in distribution[i].keys():
            if j=='b':
                layer_parameter['b']=init_parameters_b(i)
                continue
            if j=='w':
                layer_parameter['w']=init_parameters_w(i)
                continue
        parameter.append(layer_parameter)
    return parameter

def show_train(index):
    plt.imshow(train_img[index].reshape(28, 28), cmap='gray')
    print('label : {}'.format(train_lab[index]))


def show_valid(index):
    plt.imshow(valid_img[index].reshape(28, 28), cmap='gray')
    print('label : {}'.format(valid_lab[index]))


def show_test(index):
    plt.imshow(test_img[index].reshape(28, 28), cmap='gray')
    print('label : {}'.format(test_lab[index]))

def sqr_loss(img,lab,parameters):
    y_pred=predict(img,parameters)
    y=onehot[lab]
    diff=y-y_pred
    return np.dot(diff,diff)

def grad_parameters(img, lab, parameters):
    l0_in = img + parameters[0]['b']
    l0_out = activation[0](l0_in)
    l1_in = np.dot(l0_out, parameters[1]['w']) + parameters[1]['b']
    l1_out = activation[1](l1_in)

    diff = onehot[lab] - l1_out
    act1 = np.dot(differential[activation[1]](l1_in), diff)

    grad_b1 = -2 * act1
    grad_w1 = -2 * np.outer(l0_out, act1)
    grad_b0 = -2 * differential[activation[0]](l0_in) * np.dot(parameters[1]['w'], act1)

    return {'w1': grad_w1, 'b1': grad_b1, 'b0': grad_b0}

def valid_loss(parameters):
    loss_accu=0
    for img_i in range(valid_num):
        loss_accu+=sqr_loss(valid_img[img_i],valid_lab[img_i],parameters)
    return loss_accu/(valid_num/10000)
def valid_accuracy(parameters):
    correct=[predict(valid_img[img_i],parameters).argmax()==valid_lab[img_i] for img_i in range(valid_num)]
    return correct.count(True)/len(correct)
# def train_loss(parameters):
#     loss_accu=0
#     for img_i in range(train_num):
#         loss_accu+=sqr_loss(train_img[img_i],train_lab[img_i],parameters)
#     return loss_accu/(train_num/10000)
# def train_accuracy(parameters):
#     correct=[predict(train_img[img_i],parameters).argmax()==train_lab[img_i] for img_i in range(train_num)]
#     return correct.count(True)/len(correct)
def train_batch(current_batch,parameters):
    grad_accu=grad_parameters(train_img[current_batch*batch_size+0],train_lab[current_batch*batch_size+0],parameters)
    for img_i in range(1,batch_size):
        grad_tmp=grad_parameters(train_img[current_batch*batch_size+img_i],train_lab[current_batch*batch_size+img_i],parameters)
        for key in grad_accu.keys():
            grad_accu[key]+=grad_tmp[key]
    for key in grad_accu.keys():
        grad_accu[key]/=batch_size
    return grad_accu
def combine_parameters(parameters, grad, learn_rate):
    parameter_tmp = copy.deepcopy(parameters)
    parameter_tmp[0]['b'] -= learn_rate * grad['b0']
    parameter_tmp[1]['b'] -= learn_rate * grad['b1']
    parameter_tmp[1]['w'] -= learn_rate * grad['w1']
    return parameter_tmp

dimensions=[28*28,10]
activation=[tanh,softmax]
distribution=[
    {'b':[0,0]},
    {'b':[0,0],'w':[-math.sqrt(6/(dimensions[0]+dimensions[1])),math.sqrt(6/(dimensions[0]+dimensions[1]))]},
]
parameters=init_parameters()
predict(np.random.rand(784),parameters).argmax()

dataset_path = Path('./MNIST')
train_img_path = dataset_path / 'train-images.idx3-ubyte'
train_lab_path = dataset_path / 'train-labels.idx1-ubyte'
test_img_path = dataset_path / 't10k-images.idx3-ubyte'
test_lab_path = dataset_path / 't10k-labels.idx1-ubyte'
train_num = 50000
valid_num = 10000
test_num = 10000

with open(train_img_path, 'rb') as f:
    struct.unpack('>4i', f.read(16))
    tmp_img = np.fromfile(f, dtype=np.uint8).reshape(-1, 28 * 28) / 255
    train_img = tmp_img[:train_num]
    valid_img = tmp_img[train_num:]

with open(test_img_path, 'rb') as f:
    struct.unpack('>4i', f.read(16))
    test_img = np.fromfile(f, dtype=np.uint8).reshape(-1, 28 * 28) / 255

with open(train_lab_path, 'rb') as f:
    struct.unpack('>2i', f.read(8))
    tmp_lab = np.fromfile(f, dtype=np.uint8)
    train_lab = tmp_lab[:train_num]
    valid_lab = tmp_lab[train_num:]

with open(test_lab_path, 'rb') as f:
    struct.unpack('>2i', f.read(8))
    test_lab = np.fromfile(f, dtype=np.uint8)
differential={softmax:d_softmax,tanh:d_tanh}
onehot=np.identity(dimensions[-1])
batch_size=100
print(valid_accuracy(parameters))
for i in range(train_num // batch_size):
    if (i % 100 == 99):
        print("runing batch {} / {}".format(i + 1, train_num // batch_size))
    grad_tmp = train_batch(i, parameters)
    parameters = combine_parameters(parameters, grad_tmp, 1)
print(valid_accuracy(parameters))
