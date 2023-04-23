import numpy

class Activation(object):
    def __init__(self):
        pass
    def tanh(self, x):
        return numpy.tanh(x)
    # softmax 函数主要功能是在最后一层求概率
    def softmax(self, x): # x 的类型是 numpy.array([1, 2, 3, 4, ......])
        exp = numpy.exp(x - x.max()) # 这里是防止溢出 # 对分子分母同时操作等于没操作
        return exp / exp.sum()