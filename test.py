import numpy as np
# import matplotlib.pyplot as plt

# def show_img():
#     # 加载图片
#     img = np.loadtxt('./input.txt', dtype=np.float32).reshape(28, 28)

#     # 展示图片
#     plt.imshow(img, cmap='gray')
#     plt.show()

def tanh(x):
    return np.tanh(x)

def softmax(x):
    exp=np.exp(x - x.max())
    return exp / exp.sum()

def predict(img):
    l_in = img
    l_out = l_in

    l_in = np.dot(l_out, w1) + b1
    l_out = tanh(l_in)

    l_in = np.dot(l_out, w2) + b2
    l_out = softmax(l_in)

    return l_out

img = np.loadtxt('./input.txt', dtype=np.float32).reshape(784)
b1 = np.loadtxt('./p_1_b.txt', dtype=np.float32).reshape(100,)
w1 = np.loadtxt('./p_1_w.txt', dtype=np.float32).reshape(784, 100)
b2 = np.loadtxt('./p_2_b.txt', dtype=np.float32).reshape(10,)
w2 = np.loadtxt('./p_2_w.txt', dtype=np.float32).reshape(100, 10)

# show_img()
print(np.amax(predict(img)))