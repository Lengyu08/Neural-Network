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
    def d_tanh(data):
        # d_tanh([1, 2, 3, 4])
        # return np.diag(1 / (np.cosh(data) ** 2))
        return 1 / (numpy.cosh(data) ** 2)
    def d_softmax(self, data):
        sm = self.softmax(data)
        # 单元对角矩阵 - 行列向量扩充
        # d_softmax(np.array([1, 2, 3, 4]))
        return numpy.diag(sm) - numpy.outer(sm, sm)