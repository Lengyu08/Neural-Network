## 深度学习入门 : 基于Python的理论与实现

```bash
conda install matplotlib pathlib copy struct numba
pip uninstall Pillow
pip install Pillow -i https://pypi.tuna.tsinghua.edu.cn/simple some-package
```

### 感知机



#### 感知机实现逻辑门

**与门**

```python
import numpy
# 手动实现的具有与门功能的感知机
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.5 # theta不少于 0.5 不多于 0.9
    tmp = x1 * w1 + x2 * w2
    if (tmp <= theta):
        return 0
    elif (tmp > theta):
        return 1

def NUMPY_AND(x1, x2):
    x = numpy.array([x1, x2])
    w = numpy.array([0.5, 0.5])
    b = -0.5
    tmp = numpy.sum(w * x) + b;
    if (tmp <= 0):
        return 0
    else:
        return 1


if __name__ == "__main__":
    print(AND(0, 0)) 
    print(AND(1, 0)) 
    print(AND(0, 1))
    print(AND(1, 1))

    print(NUMPY_AND(0, 0))
    print(NUMPY_AND(1, 0))
    print(NUMPY_AND(0, 1))
    print(NUMPY_AND(1, 1))
```



**与非门或门**

```python
import numpy

def NAND(x1, x2):
    x = numpy.array([x1, x2])
    w = numpy.array([-0.5, -0.5])
    b = 0.7
    tmp = numpy.sum(w * x) + b
    if (tmp <= 0):
        return 0
    else:
        return 1

def OR(x1, x2):
    x = numpy.array([x1, x2])
    w = numpy.array([0.5, 0.5])
    b = -0.2
    tmp = numpy.sum(w * x) + b
    if (tmp <= 0):
        return 0
    else:
        return 1

def main():
    print(NAND(0, 0))
    print(NAND(1, 0))
    print(NAND(0, 1))
    print(NAND(1, 1))

    print(OR(0, 0))
    print(OR(1, 0))
    print(OR(0, 1))
    print(OR(1, 1))

if __name__ == "__main__":
    main()
```

#### 多重感知机

**前面模型的缺陷: 无法实现异或门**

**注意: $x$ $y$ 轴是 $x_1$ $x_2$**

<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="614.4" height="460.8" viewBox="0 0 460.8 345.6">
    <defs>
        <style>
            *{stroke-linejoin:round;stroke-linecap:butt}
        </stylei>
    </defs>
    <g id="figure_1">
        <path id="patch_1" d="M0 345.6h460.8V0H0z" style="fill:#fff"/>
        <g id="axes_1">
            <path id="patch_2" d="M57.6 307.584h357.12V41.472H57.6z" style="fill:#fff"/>
            <g id="PathCollection_1">
                <defs>
                    <path id="mc744badb38" d="M0 3a3 3 0 1 0 0-6 3 3 0 0 0 0 6z" style="stroke:#1f77b4"/>
                </defs>
                <g clip-path="url(#pd3ea4e76eb)">
                    <use xlink:href="#mc744badb38" x="236.16" y="174.528" style="fill:#1f77b4;stroke:#1f77b4"/>
                    <use xlink:href="#mc744badb38" x="317.324" y="114.048" style="fill:#1f77b4;stroke:#1f77b4"/>
                    <use xlink:href="#mc744badb38" x="236.16" y="114.048" style="fill:#1f77b4;stroke:#1f77b4"/>
                    <use xlink:href="#mc744badb38" x="317.324" y="174.528" style="fill:#1f77b4;stroke:#1f77b4"/>
                </g>
            </g>
            <g id="matplotlib.axis_1">
                <g id="xtick_1">
                    <g id="line2d_1">
                        <defs>
                            <path id="m68b3462da3" d="M0 0v3.5" style="stroke:#000;stroke-width:.8"/>
                        </defs>
                        <use xlink:href="#m68b3462da3" x="73.833" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    </g>
                    <g id="text_1" transform="matrix(.1 0 0 -.1 66.462 322.182)">
                        <defs>
                            <path id="DejaVuSans-2212" d="M678 2272h4006v-531H678v531z" transform="scale(.01563)"/>
                            <path id="DejaVuSans-32" d="M1228 531h2203V0H469v531q359 372 979 998 621 627 780 809 303 340 423 576 121 236 121 464 0 372-261 606-261 235-680 235-297 0-627-103-329-103-704-313v638q381 153 712 231 332 78 607 78 725 0 1156-363 431-362 431-968 0-288-108-546-107-257-392-607-78-91-497-524-418-433-1181-1211z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-2212"/>
                        <use xlink:href="#DejaVuSans-32" x="83.789"/>
                    </g>
                </g>
                <g id="xtick_2">
                    <use xlink:href="#m68b3462da3" id="line2d_2" x="154.996" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_2" transform="matrix(.1 0 0 -.1 147.625 322.182)">
                        <defs>
                            <path id="DejaVuSans-31" d="M794 531h1031v3560L703 3866v575l1116 225h631V531h1031V0H794v531z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-2212"/>
                        <use xlink:href="#DejaVuSans-31" x="83.789"/>
                    </g>
                </g>
                <g id="xtick_3">
                    <use xlink:href="#m68b3462da3" id="line2d_3" x="236.16" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_3" transform="matrix(.1 0 0 -.1 232.979 322.182)">
                        <defs>
                            <path id="DejaVuSans-30" d="M2034 4250q-487 0-733-480-245-479-245-1442 0-959 245-1439 246-480 733-480 491 0 736 480 246 480 246 1439 0 963-246 1442-245 480-736 480zm0 500q785 0 1199-621 414-620 414-1801 0-1178-414-1799Q2819-91 2034-91q-784 0-1198 620-414 621-414 1799 0 1181 414 1801 414 621 1198 621z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-30"/>
                    </g>
                </g>
                <g id="xtick_4">
                    <use xlink:href="#m68b3462da3" id="line2d_4" x="317.324" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <use xlink:href="#DejaVuSans-31" id="text_4" transform="matrix(.1 0 0 -.1 314.142 322.182)"/>
                </g>
                <g id="xtick_5">
                    <use xlink:href="#m68b3462da3" id="line2d_5" x="398.487" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <use xlink:href="#DejaVuSans-32" id="text_5" transform="matrix(.1 0 0 -.1 395.306 322.182)"/>
                </g>
            </g>
            <g id="matplotlib.axis_2">
                <g id="ytick_1">
                    <g id="line2d_6">
                        <defs>
                            <path id="m953609eea7" d="M0 0h-3.5" style="stroke:#000;stroke-width:.8"/>
                        </defs>
                        <use xlink:href="#m953609eea7" x="57.6" y="295.488" style="stroke:#000;stroke-width:.8"/>
                    </g>
                    <g id="text_6" transform="matrix(.1 0 0 -.1 35.858 299.287)">
                        <use xlink:href="#DejaVuSans-2212"/>
                        <use xlink:href="#DejaVuSans-32" x="83.789"/>
                    </g>
                </g>
                <g id="ytick_2">
                    <use xlink:href="#m953609eea7" id="line2d_7" x="57.6" y="235.008" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_7" transform="matrix(.1 0 0 -.1 35.858 238.807)">
                        <use xlink:href="#DejaVuSans-2212"/>
                        <use xlink:href="#DejaVuSans-31" x="83.789"/>
                    </g>
                </g>
                <g id="ytick_3">
                    <use xlink:href="#m953609eea7" id="line2d_8" x="57.6" y="174.528" style="stroke:#000;stroke-width:.8"/>
                    <use xlink:href="#DejaVuSans-30" id="text_8" transform="matrix(.1 0 0 -.1 44.237 178.327)"/>
                </g>
                <g id="ytick_4">
                    <use xlink:href="#m953609eea7" id="line2d_9" x="57.6" y="114.048" style="stroke:#000;stroke-width:.8"/>
                    <use xlink:href="#DejaVuSans-31" id="text_9" transform="matrix(.1 0 0 -.1 44.237 117.847)"/>
                </g>
                <g id="ytick_5">
                    <use xlink:href="#m953609eea7" id="line2d_10" x="57.6" y="53.568" style="stroke:#000;stroke-width:.8"/>
                    <use xlink:href="#DejaVuSans-32" id="text_10" transform="matrix(.1 0 0 -.1 44.237 57.367)"/>
                </g>
            </g>
            <path id="line2d_11" d="m73.833 53.568 324.654 241.92" clip-path="url(#pd3ea4e76eb)" style="fill:none;stroke:#1f77b4;stroke-width:1.5;stroke-linecap:square"/>
            <path id="line2d_12" d="M57.6 174.528h357.12" clip-path="url(#pd3ea4e76eb)" style="fill:none;stroke:#000;stroke-width:1.5;stroke-linecap:square"/>
            <path id="line2d_13" d="M236.16 307.584V41.472" clip-path="url(#pd3ea4e76eb)" style="fill:none;stroke:#000;stroke-width:1.5;stroke-linecap:square"/>
            <path id="patch_3" d="M57.6 307.584V41.472" style="fill:none;stroke:#000;stroke-width:.8;stroke-linejoin:miter;stroke-linecap:square"/>
            <path id="patch_4" d="M414.72 307.584V41.472" style="fill:none;stroke:#000;stroke-width:.8;stroke-linejoin:miter;stroke-linecap:square"/>
            <path id="patch_5" d="M57.6 307.584h357.12" style="fill:none;stroke:#000;stroke-width:.8;stroke-linejoin:miter;stroke-linecap:square"/>
            <path id="patch_6" d="M57.6 41.472h357.12" style="fill:none;stroke:#000;stroke-width:.8;stroke-linejoin:miter;stroke-linecap:square"/>
        </g>
    </g>
    <defs>
        <clipPath id="pd3ea4e76eb">
            <path d="M57.6 41.472h357.12v266.112H57.6z"/>
        </clipPath>
    </defs>
</svg>

这是我们上文提到 $x_1~x_2$ 分别对应 上面这样一个图像，即 (0, 1) (1, 0) (0, 0) (1, 1) 四个点，通过线性图像将空间分为两部分， 一部分会输出正数，一部分会输出负数

我们以或门为例 `x = numpy.array([x1, x2]) w = numpy.array([0.5, 0.5]) b = 0.4` 
$$
y=\left\{
\begin{array}{ll}
    0~~~(-0.4+x_1w_1+x_2w_2\le0)&\\
   	1~~~(-0.4+x_1w_1+x_2w_2>0)&\\
\end{array}\right.\\
y=\left\{
\begin{array}{ll}
    0~~~(x_1+x_2\le0.8)&\\
   	1~~~(x_1+x_2>0.8)&\\
\end{array}\right.\\
$$
观察上述的函数

我们发现，是将 $x_1=-x_2+0.8$ ( $y=x$ 向上平移 $0.1\sim0.9$ 个单位长度)，正好可以覆盖掉 $(0,0)$ 而不覆盖掉其余三个点

但是异或门的点的分布是离散的 ()，所以我们要用 **多重感知机** 用与门和或门实现异或门 $XOR(A,B)=AB'+A'B=(AB)'(A+B)=(A'+B')(A+B)$

```python
import numpy
# 也可以用一个或非门、一个或门、一个与门实现
def AND(x1, x2):
    x = numpy.array([x1, x2])
    w = numpy.array([0.5, 0.5])
    b = -0.5
    tmp = numpy.sum(w * x) + b;
    if (tmp <= 0):
        return 0
    else:
        return 1

def OR(x1, x2):
    x = numpy.array([x1, x2])
    w = numpy.array([0.5, 0.5])
    b = -0.2
    tmp = numpy.sum(w * x) + b
    if (tmp <= 0):
        return 0
    else:
        return 1

def XOR(x1, x2):
    not_x1 = not x1
    not_x2 = not x2
    s1 = AND(x1, not_x2)
    s2 = AND(not_x1, x2)
    return OR(s1, s2)

def main():
    print(XOR(0, 0))
    print(XOR(0, 1))
    print(XOR(1, 0))
    print(XOR(1, 1))

if __name__ == "__main__":
    main()
```

 

由上述推导总结出一下公式
$$
y=\left\{
\begin{array}{ll}
    0~~~(b+x_1w_1+x_2w_2\le0)&\\
   	1~~~(b+x_1w_1+x_2w_2>0)&\\
\end{array}\right.\\
$$
*b*是被称为偏置的参数，用于控制神经元被激活的容易程度；而*w*1和*w*2

是表示各个信号的权重的参数，用于控制各个信号的**重要性**



### 神经网络输入层

#### 工程中一般的取值

$b_0~~~b_1=0,~0$

$w_1\in[-\sqrt{\frac{6}{m+n}},+\sqrt{\frac{6}{m+n}})$

$n$ 输出节点数 $m$ 输入节点数

#### 激活函数

我们将感知机公式写的再简介一些
$$
x=b+x_1w_1+x_2w_2\\
y=h(b+x_1w_1+x_2w_2)\\\\
h(x)=\left\{
\begin{array}{ll}
    0~~~(x\le0)&\\
   	1~~~(x>0)&\\
\end{array}\right.\\
$$


<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="614.4" height="460.8" viewBox="0 0 460.8 345.6">
    <defs>
        <style>
            *{stroke-linejoin:round;stroke-linecap:butt}
        </style>
    </defs>
    <g id="figure_1">
        <path id="patch_1" d="M0 345.6h460.8V0H0z" style="fill:#fff"/>
        <g id="axes_1">
            <path id="patch_2" d="M57.6 307.584h357.12V41.472H57.6z" style="fill:#fff"/>
            <g id="matplotlib.axis_1">
                <g id="xtick_1">
                    <g id="line2d_1">
                        <defs>
                            <path id="mbd36460b59" d="M0 0v3.5" style="stroke:#000;stroke-width:.8"/>
                        </defs>
                        <use xlink:href="#mbd36460b59" x="73.833" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    </g>
                    <g id="text_1" transform="matrix(.1 0 0 -.1 63.28 322.182)">
                        <defs>
                            <path id="DejaVuSans-2212" d="M678 2272h4006v-531H678v531z" transform="scale(.01563)"/>
                            <path id="DejaVuSans-31" d="M794 531h1031v3560L703 3866v575l1116 225h631V531h1031V0H794v531z" transform="scale(.01563)"/>
                            <path id="DejaVuSans-30" d="M2034 4250q-487 0-733-480-245-479-245-1442 0-959 245-1439 246-480 733-480 491 0 736 480 246 480 246 1439 0 963-246 1442-245 480-736 480zm0 500q785 0 1199-621 414-620 414-1801 0-1178-414-1799Q2819-91 2034-91q-784 0-1198 620-414 621-414 1799 0 1181 414 1801 414 621 1198 621z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-2212"/>
                        <use xlink:href="#DejaVuSans-31" x="83.789"/>
                        <use xlink:href="#DejaVuSans-30" x="147.412"/>
                    </g>
                </g>
                <g id="xtick_2">
                    <use xlink:href="#mbd36460b59" id="line2d_2" x="106.298" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_2" transform="matrix(.1 0 0 -.1 98.927 322.182)">
                        <defs>
                            <path id="DejaVuSans-38" d="M2034 2216q-450 0-708-241-257-241-257-662 0-422 257-663 258-241 708-241t709 242q260 243 260 662 0 421-258 662-257 241-711 241zm-631 268q-406 100-633 378-226 279-226 679 0 559 398 884 399 325 1092 325 697 0 1094-325t397-884q0-400-227-679-226-278-629-378 456-106 710-416 255-309 255-755 0-679-414-1042Q2806-91 2034-91q-771 0-1186 362-414 363-414 1042 0 446 256 755 257 310 713 416zm-231 997q0-362 226-565 227-203 636-203 407 0 636 203 230 203 230 565 0 363-230 566-229 203-636 203-409 0-636-203-226-203-226-566z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-2212"/>
                        <use xlink:href="#DejaVuSans-38" x="83.789"/>
                    </g>
                </g>
                <g id="xtick_3">
                    <use xlink:href="#mbd36460b59" id="line2d_3" x="138.764" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_3" transform="matrix(.1 0 0 -.1 131.393 322.182)">
                        <defs>
                            <path id="DejaVuSans-36" d="M2113 2584q-425 0-674-291-248-290-248-796 0-503 248-796 249-292 674-292t673 292q248 293 248 796 0 506-248 796-248 291-673 291zm1253 1979v-575q-238 112-480 171-242 60-480 60-625 0-955-422-329-422-376-1275 184 272 462 417 279 145 613 145 703 0 1111-427 408-426 408-1160 0-719-425-1154Q2819-91 2113-91q-810 0-1238 620-428 621-428 1799 0 1106 525 1764t1409 658q238 0 480-47t505-140z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-2212"/>
                        <use xlink:href="#DejaVuSans-36" x="83.789"/>
                    </g>
                </g>
                <g id="xtick_4">
                    <use xlink:href="#mbd36460b59" id="line2d_4" x="171.229" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_4" transform="matrix(.1 0 0 -.1 163.858 322.182)">
                        <defs>
                            <path id="DejaVuSans-34" d="M2419 4116 825 1625h1594v2491zm-166 550h794V1625h666v-525h-666V0h-628v1100H313v609l1940 2957z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-2212"/>
                        <use xlink:href="#DejaVuSans-34" x="83.789"/>
                    </g>
                </g>
                <g id="xtick_5">
                    <use xlink:href="#mbd36460b59" id="line2d_5" x="203.695" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_5" transform="matrix(.1 0 0 -.1 196.323 322.182)">
                        <defs>
                            <path id="DejaVuSans-32" d="M1228 531h2203V0H469v531q359 372 979 998 621 627 780 809 303 340 423 576 121 236 121 464 0 372-261 606-261 235-680 235-297 0-627-103-329-103-704-313v638q381 153 712 231 332 78 607 78 725 0 1156-363 431-362 431-968 0-288-108-546-107-257-392-607-78-91-497-524-418-433-1181-1211z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-2212"/>
                        <use xlink:href="#DejaVuSans-32" x="83.789"/>
                    </g>
                </g>
                <g id="xtick_6">
                    <use xlink:href="#mbd36460b59" id="line2d_6" x="236.16" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <use xlink:href="#DejaVuSans-30" id="text_6" transform="matrix(.1 0 0 -.1 232.979 322.182)"/>
                </g>
                <g id="xtick_7">
                    <use xlink:href="#mbd36460b59" id="line2d_7" x="268.625" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <use xlink:href="#DejaVuSans-32" id="text_7" transform="matrix(.1 0 0 -.1 265.444 322.182)"/>
                </g>
                <g id="xtick_8">
                    <use xlink:href="#mbd36460b59" id="line2d_8" x="301.091" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <use xlink:href="#DejaVuSans-34" id="text_8" transform="matrix(.1 0 0 -.1 297.91 322.182)"/>
                </g>
                <g id="xtick_9">
                    <use xlink:href="#mbd36460b59" id="line2d_9" x="333.556" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <use xlink:href="#DejaVuSans-36" id="text_9" transform="matrix(.1 0 0 -.1 330.375 322.182)"/>
                </g>
                <g id="xtick_10">
                    <use xlink:href="#mbd36460b59" id="line2d_10" x="366.022" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <use xlink:href="#DejaVuSans-38" id="text_10" transform="matrix(.1 0 0 -.1 362.84 322.182)"/>
                </g>
                <g id="xtick_11">
                    <use xlink:href="#mbd36460b59" id="line2d_11" x="398.487" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_11" transform="matrix(.1 0 0 -.1 392.125 322.182)">
                        <use xlink:href="#DejaVuSans-31"/>
                        <use xlink:href="#DejaVuSans-30" x="63.623"/>
                    </g>
                </g>
                <g id="text_12" transform="matrix(.1 0 0 -.1 233.2 335.86)">
                    <defs>
                        <path id="DejaVuSans-78" d="M3513 3500 2247 1797 3578 0h-678L1881 1375 863 0H184l1360 1831L300 3500h678l928-1247 928 1247h679z" transform="scale(.01563)"/>
                    </defs>
                    <use xlink:href="#DejaVuSans-78"/>
                </g>
            </g>
            <g id="matplotlib.axis_2">
                <g id="ytick_1">
                    <g id="line2d_12">
                        <defs>
                            <path id="mb3e248921b" d="M0 0h-3.5" style="stroke:#000;stroke-width:.8"/>
                        </defs>
                        <use xlink:href="#mb3e248921b" x="57.6" y="295.499" style="stroke:#000;stroke-width:.8"/>
                    </g>
                    <g id="text_13" transform="matrix(.1 0 0 -.1 34.697 299.298)">
                        <defs>
                            <path id="DejaVuSans-2e" d="M684 794h660V0H684v794z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-30"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-30" x="95.41"/>
                    </g>
                </g>
                <g id="ytick_2">
                    <use xlink:href="#mb3e248921b" id="line2d_13" x="57.6" y="247.111" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_14" transform="matrix(.1 0 0 -.1 34.697 250.91)">
                        <use xlink:href="#DejaVuSans-30"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-32" x="95.41"/>
                    </g>
                </g>
                <g id="ytick_3">
                    <use xlink:href="#mb3e248921b" id="line2d_14" x="57.6" y="198.722" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_15" transform="matrix(.1 0 0 -.1 34.697 202.521)">
                        <use xlink:href="#DejaVuSans-30"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-34" x="95.41"/>
                    </g>
                </g>
                <g id="ytick_4">
                    <use xlink:href="#mb3e248921b" id="line2d_15" x="57.6" y="150.334" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_16" transform="matrix(.1 0 0 -.1 34.697 154.133)">
                        <use xlink:href="#DejaVuSans-30"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-36" x="95.41"/>
                    </g>
                </g>
                <g id="ytick_5">
                    <use xlink:href="#mb3e248921b" id="line2d_16" x="57.6" y="101.945" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_17" transform="matrix(.1 0 0 -.1 34.697 105.745)">
                        <use xlink:href="#DejaVuSans-30"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-38" x="95.41"/>
                    </g>
                </g>
                <g id="ytick_6">
                    <use xlink:href="#mb3e248921b" id="line2d_17" x="57.6" y="53.557" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_18" transform="matrix(.1 0 0 -.1 34.697 57.356)">
                        <use xlink:href="#DejaVuSans-31"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-30" x="95.41"/>
                    </g>
                </g>
                <g id="text_19" transform="matrix(0 -.1 -.1 0 28.617 201.05)">
                    <defs>
                        <path id="DejaVuSans-73" d="M2834 3397v-544q-243 125-506 187-262 63-544 63-428 0-642-131t-214-394q0-200 153-314t616-217l197-44q612-131 870-370t258-667q0-488-386-773Q2250-91 1575-91q-281 0-586 55T347 128v594q319-166 628-249 309-82 613-82 406 0 624 139 219 139 219 392 0 234-158 359-157 125-692 241l-200 47q-534 112-772 345-237 233-237 639 0 494 350 762 350 269 994 269 318 0 599-47 282-46 519-140z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-69" d="M603 3500h575V0H603v3500zm0 1363h575v-729H603v729z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-67" d="M2906 1791q0 625-258 968-257 344-723 344-462 0-720-344-258-343-258-968 0-622 258-966t720-344q466 0 723 344 258 344 258 966zm575-1357q0-893-397-1329-396-436-1215-436-303 0-572 45t-522 139v559q253-137 500-202 247-66 503-66 566 0 847 295t281 892v285q-178-310-456-463T1784 0Q1141 0 747 490 353 981 353 1791q0 812 394 1302 394 491 1037 491 388 0 666-153t456-462v531h575V434z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-6d" d="M3328 2828q216 388 516 572t706 184q547 0 844-383 297-382 297-1088V0h-578v2094q0 503-179 746-178 244-543 244-447 0-707-297-259-296-259-809V0h-578v2094q0 506-178 748t-550 242q-441 0-701-298-259-298-259-808V0H581v3500h578v-544q197 322 472 475t653 153q382 0 649-194 267-193 395-562z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-6f" d="M1959 3097q-462 0-731-361t-269-989q0-628 267-989 268-361 733-361 460 0 728 362 269 363 269 988 0 622-269 986-268 364-728 364zm0 487q750 0 1178-488 429-487 429-1349 0-859-429-1349Q2709-91 1959-91q-753 0-1180 489-426 490-426 1349 0 862 426 1349 427 488 1180 488z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-64" d="M2906 2969v1894h575V0h-575v525q-181-312-458-464-276-152-664-152-634 0-1033 506-398 507-398 1332t398 1331q399 506 1033 506 388 0 664-152 277-151 458-463zM947 1747q0-634 261-995t717-361q456 0 718 361 263 361 263 995t-263 995q-262 361-718 361t-717-361q-261-361-261-995z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-28" d="M1984 4856q-418-718-622-1422-203-703-203-1425 0-721 205-1429t620-1424h-500Q1016-109 783 600T550 2009q0 697 231 1403 232 707 703 1444h500z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-29" d="M513 4856h500q468-737 701-1444 233-706 233-1403 0-700-233-1409T1013-844H513q415 716 620 1424t205 1429q0 722-205 1425-205 704-620 1422z" transform="scale(.01563)"/>
                    </defs>
                    <use xlink:href="#DejaVuSans-73"/>
                    <use xlink:href="#DejaVuSans-69" x="52.1"/>
                    <use xlink:href="#DejaVuSans-67" x="79.883"/>
                    <use xlink:href="#DejaVuSans-6d" x="143.359"/>
                    <use xlink:href="#DejaVuSans-6f" x="240.771"/>
                    <use xlink:href="#DejaVuSans-69" x="301.953"/>
                    <use xlink:href="#DejaVuSans-64" x="329.736"/>
                    <use xlink:href="#DejaVuSans-28" x="393.213"/>
                    <use xlink:href="#DejaVuSans-78" x="432.227"/>
                    <use xlink:href="#DejaVuSans-29" x="491.406"/>
                </g>
            </g>
            <path id="line2d_18" d="m73.833 295.488 3.28-.002 3.278-.003 3.28-.004 3.28-.005 3.278-.005 3.28-.007 3.28-.008 3.278-.01 3.28-.013 3.28-.015 3.278-.018 3.28-.023 3.28-.028 3.278-.034 3.28-.041 3.28-.051 3.278-.062 3.28-.076 3.28-.093 3.279-.114 3.279-.139 3.28-.17 3.279-.207 3.279-.254 3.28-.31 3.279-.377 3.279-.461 3.28-.562 3.279-.684 3.279-.831 3.28-1.01 3.279-1.225 3.279-1.48 3.28-1.788 3.279-2.151 3.279-2.58 3.28-3.078 3.279-3.655 3.279-4.315 3.28-5.057 3.279-5.875 3.279-6.76 3.28-7.69 3.279-8.638 3.279-9.562 3.28-10.418 3.279-11.156 3.279-11.725 3.28-12.086 3.279-12.208 3.279-12.086 3.28-11.725 3.279-11.156 3.279-10.418 3.28-9.562 3.279-8.637 3.279-7.69 3.28-6.76 3.279-5.876 3.28-5.057 3.278-4.315 3.28-3.655 3.28-3.079 3.278-2.579 3.28-2.15 3.28-1.788 3.278-1.481 3.28-1.225 3.28-1.01 3.278-.831 3.28-.684 3.28-.562 3.278-.46 3.28-.378 3.28-.31 3.278-.254 3.28-.207 3.28-.17 3.278-.14 3.28-.113 3.28-.093 3.279-.076 3.279-.062 3.28-.05 3.279-.042 3.279-.034 3.28-.028 3.279-.023 3.279-.018 3.28-.015 3.279-.013 3.279-.01 3.28-.008 3.279-.007 3.279-.005 3.28-.005 3.279-.004 3.279-.003 3.28-.002" clip-path="url(#p212bb9af8c)" style="fill:none;stroke:#1f77b4;stroke-width:1.5;stroke-linecap:square"/>
            <path id="line2d_19" d="M57.6 295.499h357.12" clip-path="url(#p212bb9af8c)" style="fill:none;stroke:#000;stroke-width:1.5;stroke-linecap:square"/>
            <path id="line2d_20" d="M236.16 307.584V41.472" clip-path="url(#p212bb9af8c)" style="fill:none;stroke:#000;stroke-width:1.5;stroke-linecap:square"/>
            <path id="patch_3" d="M57.6 307.584V41.472" style="fill:none;stroke:#000;stroke-width:.8;stroke-linejoin:miter;stroke-linecap:square"/>
            <path id="patch_4" d="M414.72 307.584V41.472" style="fill:none;stroke:#000;stroke-width:.8;stroke-linejoin:miter;stroke-linecap:square"/>
            <path id="patch_5" d="M57.6 307.584h357.12" style="fill:none;stroke:#000;stroke-width:.8;stroke-linejoin:miter;stroke-linecap:square"/>
            <path id="patch_6" d="M57.6 41.472h357.12" style="fill:none;stroke:#000;stroke-width:.8;stroke-linejoin:miter;stroke-linecap:square"/>
            <g id="text_20" transform="matrix(.12 0 0 -.12 184.459 35.472)">
                <defs>
                    <path id="DejaVuSans-53" d="M3425 4513v-616q-359 172-678 256-319 85-616 85-515 0-795-200t-280-569q0-310 186-468 186-157 705-254l381-78q706-135 1042-474t336-907q0-679-455-1029Q2797-91 1919-91q-331 0-705 75-373 75-773 222v650q384-215 753-325 369-109 725-109 540 0 834 212 294 213 294 607 0 343-211 537t-692 291l-385 75q-706 140-1022 440-315 300-315 835 0 619 436 975t1201 356q329 0 669-60 341-59 697-177z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-46" d="M628 4666h2681v-532H1259V2759h1850v-531H1259V0H628v4666z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-75" d="M544 1381v2119h575V1403q0-497 193-746 194-248 582-248 465 0 735 297 271 297 271 810v1984h575V0h-575v538q-209-319-486-474-276-155-642-155-603 0-916 375-312 375-312 1097zm1447 2203z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-6e" d="M3513 2113V0h-575v2094q0 497-194 743-194 247-581 247-466 0-735-297-269-296-269-809V0H581v3500h578v-544q207 316 486 472 280 156 646 156 603 0 912-373 310-373 310-1098z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-63" d="M3122 3366v-538q-244 135-489 202t-495 67q-560 0-870-355-309-354-309-995t309-996q310-354 870-354 250 0 495 67t489 202V134Q2881 22 2623-34q-257-57-548-57-791 0-1257 497-465 497-465 1341 0 856 470 1346 471 491 1290 491 265 0 518-55 253-54 491-163z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-74" d="M1172 4494v-994h1184v-447H1172V1153q0-428 117-550t477-122h590V0h-590q-666 0-919 248-253 249-253 905v1900H172v447h422v994h578z" transform="scale(.01563)"/>
                </defs>
                <use xlink:href="#DejaVuSans-53"/>
                <use xlink:href="#DejaVuSans-69" x="63.477"/>
                <use xlink:href="#DejaVuSans-67" x="91.26"/>
                <use xlink:href="#DejaVuSans-6d" x="154.736"/>
                <use xlink:href="#DejaVuSans-6f" x="252.148"/>
                <use xlink:href="#DejaVuSans-69" x="313.33"/>
                <use xlink:href="#DejaVuSans-64" x="341.113"/>
                <use xlink:href="#DejaVuSans-20" x="404.59"/>
                <use xlink:href="#DejaVuSans-46" x="436.377"/>
                <use xlink:href="#DejaVuSans-75" x="488.396"/>
                <use xlink:href="#DejaVuSans-6e" x="551.775"/>
                <use xlink:href="#DejaVuSans-63" x="615.154"/>
                <use xlink:href="#DejaVuSans-74" x="670.135"/>
                <use xlink:href="#DejaVuSans-69" x="709.344"/>
                <use xlink:href="#DejaVuSans-6f" x="737.127"/>
                <use xlink:href="#DejaVuSans-6e" x="798.309"/>
            </g>
        </g>
    </g>
    <defs>
        <clipPath id="p212bb9af8c">
            <path d="M57.6 41.472h357.12v266.112H57.6z"/>
        </clipPath>
    </defs>
</svg>

最常用的一个函数就是 $sigmoid$ 函数 $h(x)=\frac{1}{1+e^{-x}}$

因为它有着一下的特性不会受负数的影响，只有 $0$ 和 $1$ 两个边界

```python
def sigmoid(x):
	return 1 / (1 + np.exp(-x))
```



**跃阶函数**

```python
import matplotlib.pyplot as plt
import numpy as np

# h(x) = {0 (x <= 0) 1 (x > 0)}
def step_function(x):
    check = x > 0
    return check.astype(np.int) # 把布尔类型转化为 int 类型

def main():
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function(x)
    plt.plot(x, y)
    # 添加 x 轴和 y 轴
    plt.axhline(y = 0, color='k')
    plt.axvline(x = 0, color='k')
    plt.show()

if __name__ == "__main__":
    main()
```



<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="614.4" height="460.8" viewBox="0 0 460.8 345.6">
    <defs>
        <style>
            *{stroke-linejoin:round;stroke-linecap:butt}
        </style>
    </defs>
    <g id="figure_1">
        <path id="patch_1" d="M0 345.6h460.8V0H0z" style="fill:#fff"/>
        <g id="axes_1">
            <path id="patch_2" d="M57.6 307.584h357.12V41.472H57.6z" style="fill:#fff"/>
            <g id="matplotlib.axis_1">
                <g id="xtick_1">
                    <g id="line2d_1">
                        <defs>
                            <path id="mb980cc0da2" d="M0 0v3.5" style="stroke:#000;stroke-width:.8"/>
                        </defs>
                        <use xlink:href="#mb980cc0da2" x="106.626" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    </g>
                    <g id="text_1" transform="matrix(.1 0 0 -.1 99.255 322.182)">
                        <defs>
                            <path id="DejaVuSans-2212" d="M678 2272h4006v-531H678v531z" transform="scale(.01563)"/>
                            <path id="DejaVuSans-34" d="M2419 4116 825 1625h1594v2491zm-166 550h794V1625h666v-525h-666V0h-628v1100H313v609l1940 2957z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-2212"/>
                        <use xlink:href="#DejaVuSans-34" x="83.789"/>
                    </g>
                </g>
                <g id="xtick_2">
                    <use xlink:href="#mb980cc0da2" id="line2d_2" x="172.213" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_2" transform="matrix(.1 0 0 -.1 164.842 322.182)">
                        <defs>
                            <path id="DejaVuSans-32" d="M1228 531h2203V0H469v531q359 372 979 998 621 627 780 809 303 340 423 576 121 236 121 464 0 372-261 606-261 235-680 235-297 0-627-103-329-103-704-313v638q381 153 712 231 332 78 607 78 725 0 1156-363 431-362 431-968 0-288-108-546-107-257-392-607-78-91-497-524-418-433-1181-1211z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-2212"/>
                        <use xlink:href="#DejaVuSans-32" x="83.789"/>
                    </g>
                </g>
                <g id="xtick_3">
                    <use xlink:href="#mb980cc0da2" id="line2d_3" x="237.8" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_3" transform="matrix(.1 0 0 -.1 234.618 322.182)">
                        <defs>
                            <path id="DejaVuSans-30" d="M2034 4250q-487 0-733-480-245-479-245-1442 0-959 245-1439 246-480 733-480 491 0 736 480 246 480 246 1439 0 963-246 1442-245 480-736 480zm0 500q785 0 1199-621 414-620 414-1801 0-1178-414-1799Q2819-91 2034-91q-784 0-1198 620-414 621-414 1799 0 1181 414 1801 414 621 1198 621z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-30"/>
                    </g>
                </g>
                <g id="xtick_4">
                    <use xlink:href="#mb980cc0da2" id="line2d_4" x="303.386" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <use xlink:href="#DejaVuSans-32" id="text_4" transform="matrix(.1 0 0 -.1 300.205 322.182)"/>
                </g>
                <g id="xtick_5">
                    <use xlink:href="#mb980cc0da2" id="line2d_5" x="368.973" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <use xlink:href="#DejaVuSans-34" id="text_5" transform="matrix(.1 0 0 -.1 365.792 322.182)"/>
                </g>
            </g>
            <g id="matplotlib.axis_2">
                <g id="ytick_1">
                    <g id="line2d_6">
                        <defs>
                            <path id="mc76dfe3e7f" d="M0 0h-3.5" style="stroke:#000;stroke-width:.8"/>
                        </defs>
                        <use xlink:href="#mc76dfe3e7f" x="57.6" y="285.408" style="stroke:#000;stroke-width:.8"/>
                    </g>
                    <g id="text_6" transform="matrix(.1 0 0 -.1 34.697 289.207)">
                        <defs>
                            <path id="DejaVuSans-2e" d="M684 794h660V0H684v794z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-30"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-30" x="95.41"/>
                    </g>
                </g>
                <g id="ytick_2">
                    <use xlink:href="#mc76dfe3e7f" id="line2d_7" x="57.6" y="241.056" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_7" transform="matrix(.1 0 0 -.1 34.697 244.855)">
                        <use xlink:href="#DejaVuSans-30"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-32" x="95.41"/>
                    </g>
                </g>
                <g id="ytick_3">
                    <use xlink:href="#mc76dfe3e7f" id="line2d_8" x="57.6" y="196.704" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_8" transform="matrix(.1 0 0 -.1 34.697 200.503)">
                        <use xlink:href="#DejaVuSans-30"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-34" x="95.41"/>
                    </g>
                </g>
                <g id="ytick_4">
                    <use xlink:href="#mc76dfe3e7f" id="line2d_9" x="57.6" y="152.352" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_9" transform="matrix(.1 0 0 -.1 34.697 156.151)">
                        <defs>
                            <path id="DejaVuSans-36" d="M2113 2584q-425 0-674-291-248-290-248-796 0-503 248-796 249-292 674-292t673 292q248 293 248 796 0 506-248 796-248 291-673 291zm1253 1979v-575q-238 112-480 171-242 60-480 60-625 0-955-422-329-422-376-1275 184 272 462 417 279 145 613 145 703 0 1111-427 408-426 408-1160 0-719-425-1154Q2819-91 2113-91q-810 0-1238 620-428 621-428 1799 0 1106 525 1764t1409 658q238 0 480-47t505-140z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-30"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-36" x="95.41"/>
                    </g>
                </g>
                <g id="ytick_5">
                    <use xlink:href="#mc76dfe3e7f" id="line2d_10" x="57.6" y="108" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_10" transform="matrix(.1 0 0 -.1 34.697 111.8)">
                        <defs>
                            <path id="DejaVuSans-38" d="M2034 2216q-450 0-708-241-257-241-257-662 0-422 257-663 258-241 708-241t709 242q260 243 260 662 0 421-258 662-257 241-711 241zm-631 268q-406 100-633 378-226 279-226 679 0 559 398 884 399 325 1092 325 697 0 1094-325t397-884q0-400-227-679-226-278-629-378 456-106 710-416 255-309 255-755 0-679-414-1042Q2806-91 2034-91q-771 0-1186 362-414 363-414 1042 0 446 256 755 257 310 713 416zm-231 997q0-362 226-565 227-203 636-203 407 0 636 203 230 203 230 565 0 363-230 566-229 203-636 203-409 0-636-203-226-203-226-566z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-30"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-38" x="95.41"/>
                    </g>
                </g>
                <g id="ytick_6">
                    <use xlink:href="#mc76dfe3e7f" id="line2d_11" x="57.6" y="63.648" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_11" transform="matrix(.1 0 0 -.1 34.697 67.447)">
                        <defs>
                            <path id="DejaVuSans-31" d="M794 531h1031v3560L703 3866v575l1116 225h631V531h1031V0H794v531z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-31"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-30" x="95.41"/>
                    </g>
                </g>
            </g>
            <path id="line2d_12" d="M73.833 285.408H237.8l3.279-221.76h157.409" clip-path="url(#p2e7aeea040)" style="fill:none;stroke:#1f77b4;stroke-width:1.5;stroke-linecap:square"/>
            <path id="line2d_13" d="M57.6 285.408h357.12" clip-path="url(#p2e7aeea040)" style="fill:none;stroke:#000;stroke-width:1.5;stroke-linecap:square"/>
            <path id="line2d_14" d="M237.8 307.584V41.472" clip-path="url(#p2e7aeea040)" style="fill:none;stroke:#000;stroke-width:1.5;stroke-linecap:square"/>
            <path id="patch_3" d="M57.6 307.584V41.472" style="fill:none;stroke:#000;stroke-width:.8;stroke-linejoin:miter;stroke-linecap:square"/>
            <path id="patch_4" d="M414.72 307.584V41.472" style="fill:none;stroke:#000;stroke-width:.8;stroke-linejoin:miter;stroke-linecap:square"/>
            <path id="patch_5" d="M57.6 307.584h357.12" style="fill:none;stroke:#000;stroke-width:.8;stroke-linejoin:miter;stroke-linecap:square"/>
            <path id="patch_6" d="M57.6 41.472h357.12" style="fill:none;stroke:#000;stroke-width:.8;stroke-linejoin:miter;stroke-linecap:square"/>
        </g>
    </g>
    <defs>
        <clipPath id="p2e7aeea040">
            <path d="M57.6 41.472h357.12v266.112H57.6z"/>
        </clipPath>
    </defs>
</svg>




**对比跃阶函数和 sigmoid 函数** 

$sigmoid$ 函数的平滑性对神经网络的学习具有重要意义

阶跃函数和 $sigmoid$ 函数共同点，就是两者均为非线性函数

**神经网络的激活函数必须使用非线性函数** ，因为如果使用线性函数就无法实现一些功能，加深层数也变得没有意义，正如在感知机里面的两层异或门



#### ReLU函数

优点 $x>0$ 的时候，函数的导数直接就是 $1$，不存在梯度衰减的问题

$ReLU$ 的另一个优点就是计算非常简单，只需要使用阈值判断即可，导数也是几乎不用计算。基于以上两个优点，ReLU的收敛速度要远远快于 $sigmoid$ 和 $tanh$ 。
  $ReLU$ 的第三大优点就是可以产生稀疏性，可以看到小于 $0$ 的部分直接设置为 $0$ ，这就使得神经网络的中间输出是稀疏的，有一定的$Droupout$ 的作用，也就能够在一定程度上防止过拟合。


$$
x=b+x_1w_1+x_2w_2\\
y=h(b+x_1w_1+x_2w_2)\\\\
h(x)=\left\{
\begin{array}{ll}
    0~~~(x\le0)&\\
   	x~~~(x>0)&\\
\end{array}\right.\\
$$



```python
def relu(x):
	return np.maximum(0, x)
```



<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="614.4" height="460.8" viewBox="0 0 460.8 345.6">
    <defs>
        <style>
            *{stroke-linejoin:round;stroke-linecap:butt}
        </style>
    </defs>
    <g id="figure_1">
        <path id="patch_1" d="M0 345.6h460.8V0H0z" style="fill:#fff"/>
        <g id="axes_1">
            <path id="patch_2" d="M57.6 307.584h357.12V41.472H57.6z" style="fill:#fff"/>
            <g id="matplotlib.axis_1">
                <g id="xtick_1">
                    <g id="line2d_1">
                        <defs>
                            <path id="m4da4b1e8fe" d="M0 0v3.5" style="stroke:#000;stroke-width:.8"/>
                        </defs>
                        <use xlink:href="#m4da4b1e8fe" x="73.833" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    </g>
                    <g id="text_1" transform="matrix(.1 0 0 -.1 63.28 322.182)">
                        <defs>
                            <path id="DejaVuSans-2212" d="M678 2272h4006v-531H678v531z" transform="scale(.01563)"/>
                            <path id="DejaVuSans-31" d="M794 531h1031v3560L703 3866v575l1116 225h631V531h1031V0H794v531z" transform="scale(.01563)"/>
                            <path id="DejaVuSans-30" d="M2034 4250q-487 0-733-480-245-479-245-1442 0-959 245-1439 246-480 733-480 491 0 736 480 246 480 246 1439 0 963-246 1442-245 480-736 480zm0 500q785 0 1199-621 414-620 414-1801 0-1178-414-1799Q2819-91 2034-91q-784 0-1198 620-414 621-414 1799 0 1181 414 1801 414 621 1198 621z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-2212"/>
                        <use xlink:href="#DejaVuSans-31" x="83.789"/>
                        <use xlink:href="#DejaVuSans-30" x="147.412"/>
                    </g>
                </g>
                <g id="xtick_2">
                    <use xlink:href="#m4da4b1e8fe" id="line2d_2" x="106.298" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_2" transform="matrix(.1 0 0 -.1 98.927 322.182)">
                        <defs>
                            <path id="DejaVuSans-38" d="M2034 2216q-450 0-708-241-257-241-257-662 0-422 257-663 258-241 708-241t709 242q260 243 260 662 0 421-258 662-257 241-711 241zm-631 268q-406 100-633 378-226 279-226 679 0 559 398 884 399 325 1092 325 697 0 1094-325t397-884q0-400-227-679-226-278-629-378 456-106 710-416 255-309 255-755 0-679-414-1042Q2806-91 2034-91q-771 0-1186 362-414 363-414 1042 0 446 256 755 257 310 713 416zm-231 997q0-362 226-565 227-203 636-203 407 0 636 203 230 203 230 565 0 363-230 566-229 203-636 203-409 0-636-203-226-203-226-566z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-2212"/>
                        <use xlink:href="#DejaVuSans-38" x="83.789"/>
                    </g>
                </g>
                <g id="xtick_3">
                    <use xlink:href="#m4da4b1e8fe" id="line2d_3" x="138.764" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_3" transform="matrix(.1 0 0 -.1 131.393 322.182)">
                        <defs>
                            <path id="DejaVuSans-36" d="M2113 2584q-425 0-674-291-248-290-248-796 0-503 248-796 249-292 674-292t673 292q248 293 248 796 0 506-248 796-248 291-673 291zm1253 1979v-575q-238 112-480 171-242 60-480 60-625 0-955-422-329-422-376-1275 184 272 462 417 279 145 613 145 703 0 1111-427 408-426 408-1160 0-719-425-1154Q2819-91 2113-91q-810 0-1238 620-428 621-428 1799 0 1106 525 1764t1409 658q238 0 480-47t505-140z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-2212"/>
                        <use xlink:href="#DejaVuSans-36" x="83.789"/>
                    </g>
                </g>
                <g id="xtick_4">
                    <use xlink:href="#m4da4b1e8fe" id="line2d_4" x="171.229" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_4" transform="matrix(.1 0 0 -.1 163.858 322.182)">
                        <defs>
                            <path id="DejaVuSans-34" d="M2419 4116 825 1625h1594v2491zm-166 550h794V1625h666v-525h-666V0h-628v1100H313v609l1940 2957z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-2212"/>
                        <use xlink:href="#DejaVuSans-34" x="83.789"/>
                    </g>
                </g>
                <g id="xtick_5">
                    <use xlink:href="#m4da4b1e8fe" id="line2d_5" x="203.695" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_5" transform="matrix(.1 0 0 -.1 196.323 322.182)">
                        <defs>
                            <path id="DejaVuSans-32" d="M1228 531h2203V0H469v531q359 372 979 998 621 627 780 809 303 340 423 576 121 236 121 464 0 372-261 606-261 235-680 235-297 0-627-103-329-103-704-313v638q381 153 712 231 332 78 607 78 725 0 1156-363 431-362 431-968 0-288-108-546-107-257-392-607-78-91-497-524-418-433-1181-1211z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-2212"/>
                        <use xlink:href="#DejaVuSans-32" x="83.789"/>
                    </g>
                </g>
                <g id="xtick_6">
                    <use xlink:href="#m4da4b1e8fe" id="line2d_6" x="236.16" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <use xlink:href="#DejaVuSans-30" id="text_6" transform="matrix(.1 0 0 -.1 232.979 322.182)"/>
                </g>
                <g id="xtick_7">
                    <use xlink:href="#m4da4b1e8fe" id="line2d_7" x="268.625" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <use xlink:href="#DejaVuSans-32" id="text_7" transform="matrix(.1 0 0 -.1 265.444 322.182)"/>
                </g>
                <g id="xtick_8">
                    <use xlink:href="#m4da4b1e8fe" id="line2d_8" x="301.091" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <use xlink:href="#DejaVuSans-34" id="text_8" transform="matrix(.1 0 0 -.1 297.91 322.182)"/>
                </g>
                <g id="xtick_9">
                    <use xlink:href="#m4da4b1e8fe" id="line2d_9" x="333.556" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <use xlink:href="#DejaVuSans-36" id="text_9" transform="matrix(.1 0 0 -.1 330.375 322.182)"/>
                </g>
                <g id="xtick_10">
                    <use xlink:href="#m4da4b1e8fe" id="line2d_10" x="366.022" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <use xlink:href="#DejaVuSans-38" id="text_10" transform="matrix(.1 0 0 -.1 362.84 322.182)"/>
                </g>
                <g id="xtick_11">
                    <use xlink:href="#m4da4b1e8fe" id="line2d_11" x="398.487" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_11" transform="matrix(.1 0 0 -.1 392.125 322.182)">
                        <use xlink:href="#DejaVuSans-31"/>
                        <use xlink:href="#DejaVuSans-30" x="63.623"/>
                    </g>
                </g>
                <g id="text_12" transform="matrix(.1 0 0 -.1 233.2 335.86)">
                    <defs>
                        <path id="DejaVuSans-78" d="M3513 3500 2247 1797 3578 0h-678L1881 1375 863 0H184l1360 1831L300 3500h678l928-1247 928 1247h679z" transform="scale(.01563)"/>
                    </defs>
                    <use xlink:href="#DejaVuSans-78"/>
                </g>
            </g>
            <g id="matplotlib.axis_2">
                <g id="ytick_1">
                    <g id="line2d_12">
                        <defs>
                            <path id="m940595a305" d="M0 0h-3.5" style="stroke:#000;stroke-width:.8"/>
                        </defs>
                        <use xlink:href="#m940595a305" x="57.6" y="295.488" style="stroke:#000;stroke-width:.8"/>
                    </g>
                    <use xlink:href="#DejaVuSans-30" id="text_13" transform="matrix(.1 0 0 -.1 44.237 299.287)"/>
                </g>
                <g id="ytick_2">
                    <use xlink:href="#m940595a305" id="line2d_13" x="57.6" y="247.104" style="stroke:#000;stroke-width:.8"/>
                    <use xlink:href="#DejaVuSans-32" id="text_14" transform="matrix(.1 0 0 -.1 44.237 250.903)"/>
                </g>
                <g id="ytick_3">
                    <use xlink:href="#m940595a305" id="line2d_14" x="57.6" y="198.72" style="stroke:#000;stroke-width:.8"/>
                    <use xlink:href="#DejaVuSans-34" id="text_15" transform="matrix(.1 0 0 -.1 44.237 202.52)"/>
                </g>
                <g id="ytick_4">
                    <use xlink:href="#m940595a305" id="line2d_15" x="57.6" y="150.336" style="stroke:#000;stroke-width:.8"/>
                    <use xlink:href="#DejaVuSans-36" id="text_16" transform="matrix(.1 0 0 -.1 44.237 154.135)"/>
                </g>
                <g id="ytick_5">
                    <use xlink:href="#m940595a305" id="line2d_16" x="57.6" y="101.952" style="stroke:#000;stroke-width:.8"/>
                    <use xlink:href="#DejaVuSans-38" id="text_17" transform="matrix(.1 0 0 -.1 44.237 105.751)"/>
                </g>
                <g id="ytick_6">
                    <use xlink:href="#m940595a305" id="line2d_17" x="57.6" y="53.568" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_18" transform="matrix(.1 0 0 -.1 37.875 57.367)">
                        <use xlink:href="#DejaVuSans-31"/>
                        <use xlink:href="#DejaVuSans-30" x="63.623"/>
                    </g>
                </g>
                <g id="text_19" transform="matrix(0 -.1 -.1 0 31.795 193.91)">
                    <defs>
                        <path id="DejaVuSans-52" d="M2841 2188q203-69 395-294t386-619L4263 0h-679l-596 1197q-232 469-449 622t-592 153h-688V0H628v4666h1425q800 0 1194-335 394-334 394-1009 0-441-205-732-205-290-595-402zM1259 4147V2491h794q456 0 689 211t233 620q0 409-233 617t-689 208h-794z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-65" d="M3597 1894v-281H953q38-594 358-905t892-311q331 0 642 81t618 244V178Q3153 47 2828-22t-659-69q-838 0-1327 487-489 488-489 1320 0 859 464 1363 464 505 1252 505 706 0 1117-455 411-454 411-1235zm-575 169q-6 471-264 752-258 282-683 282-481 0-770-272t-333-766l2050 4z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-4c" d="M628 4666h631V531h2272V0H628v4666z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-55" d="M556 4666h635V1831q0-750 271-1080 272-329 882-329 606 0 878 329 272 330 272 1080v2835h634V1753q0-912-452-1378Q3225-91 2344-91q-885 0-1337 466-451 466-451 1378v2913z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-28" d="M1984 4856q-418-718-622-1422-203-703-203-1425 0-721 205-1429t620-1424h-500Q1016-109 783 600T550 2009q0 697 231 1403 232 707 703 1444h500z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-29" d="M513 4856h500q468-737 701-1444 233-706 233-1403 0-700-233-1409T1013-844H513q415 716 620 1424t205 1429q0 722-205 1425-205 704-620 1422z" transform="scale(.01563)"/>
                    </defs>
                    <use xlink:href="#DejaVuSans-52"/>
                    <use xlink:href="#DejaVuSans-65" x="64.982"/>
                    <use xlink:href="#DejaVuSans-4c" x="126.506"/>
                    <use xlink:href="#DejaVuSans-55" x="177.219"/>
                    <use xlink:href="#DejaVuSans-28" x="250.412"/>
                    <use xlink:href="#DejaVuSans-78" x="289.426"/>
                    <use xlink:href="#DejaVuSans-29" x="348.605"/>
                </g>
            </g>
            <path id="line2d_18" d="M73.833 295.488H234.52l3.279-2.444 3.279-4.887 3.28-4.887 3.279-4.887 3.279-4.888 3.28-4.887 3.279-4.887 3.279-4.888 3.28-4.887 3.279-4.887 3.28-4.887 3.278-4.888 3.28-4.887 3.28-4.887 3.278-4.887 3.28-4.888 3.28-4.887 3.278-4.887 3.28-4.888 3.28-4.887 3.278-4.887 3.28-4.887 3.28-4.888 3.278-4.887 3.28-4.887 3.28-4.887 3.278-4.888 3.28-4.887 3.28-4.887 3.278-4.888 3.28-4.887 3.28-4.887 3.279-4.887 3.279-4.888 3.28-4.887 3.279-4.887 3.279-4.887 3.28-4.888 3.279-4.887 3.279-4.887 3.28-4.888 3.279-4.887 3.279-4.887 3.28-4.887 3.279-4.888 3.279-4.887 3.28-4.887 3.279-4.887 3.279-4.888 3.28-4.887" clip-path="url(#pec10fe9d12)" style="fill:none;stroke:#1f77b4;stroke-width:1.5;stroke-linecap:square"/>
            <path id="line2d_19" d="M57.6 295.488h357.12" clip-path="url(#pec10fe9d12)" style="fill:none;stroke:#000;stroke-width:1.5;stroke-linecap:square"/>
            <path id="line2d_20" d="M236.16 307.584V41.472" clip-path="url(#pec10fe9d12)" style="fill:none;stroke:#000;stroke-width:1.5;stroke-linecap:square"/>
            <path id="patch_3" d="M57.6 307.584V41.472" style="fill:none;stroke:#000;stroke-width:.8;stroke-linejoin:miter;stroke-linecap:square"/>
            <path id="patch_4" d="M414.72 307.584V41.472" style="fill:none;stroke:#000;stroke-width:.8;stroke-linejoin:miter;stroke-linecap:square"/>
            <path id="patch_5" d="M57.6 307.584h357.12" style="fill:none;stroke:#000;stroke-width:.8;stroke-linejoin:miter;stroke-linecap:square"/>
            <path id="patch_6" d="M57.6 41.472h357.12" style="fill:none;stroke:#000;stroke-width:.8;stroke-linejoin:miter;stroke-linecap:square"/>
            <g id="text_20" transform="matrix(.12 0 0 -.12 193.71 35.472)">
                <defs>
                    <path id="DejaVuSans-46" d="M628 4666h2681v-532H1259V2759h1850v-531H1259V0H628v4666z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-75" d="M544 1381v2119h575V1403q0-497 193-746 194-248 582-248 465 0 735 297 271 297 271 810v1984h575V0h-575v538q-209-319-486-474-276-155-642-155-603 0-916 375-312 375-312 1097zm1447 2203z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-6e" d="M3513 2113V0h-575v2094q0 497-194 743-194 247-581 247-466 0-735-297-269-296-269-809V0H581v3500h578v-544q207 316 486 472 280 156 646 156 603 0 912-373 310-373 310-1098z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-63" d="M3122 3366v-538q-244 135-489 202t-495 67q-560 0-870-355-309-354-309-995t309-996q310-354 870-354 250 0 495 67t489 202V134Q2881 22 2623-34q-257-57-548-57-791 0-1257 497-465 497-465 1341 0 856 470 1346 471 491 1290 491 265 0 518-55 253-54 491-163z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-74" d="M1172 4494v-994h1184v-447H1172V1153q0-428 117-550t477-122h590V0h-590q-666 0-919 248-253 249-253 905v1900H172v447h422v994h578z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-69" d="M603 3500h575V0H603v3500zm0 1363h575v-729H603v729z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-6f" d="M1959 3097q-462 0-731-361t-269-989q0-628 267-989 268-361 733-361 460 0 728 362 269 363 269 988 0 622-269 986-268 364-728 364zm0 487q750 0 1178-488 429-487 429-1349 0-859-429-1349Q2709-91 1959-91q-753 0-1180 489-426 490-426 1349 0 862 426 1349 427 488 1180 488z" transform="scale(.01563)"/>
                </defs>
                <use xlink:href="#DejaVuSans-52"/>
                <use xlink:href="#DejaVuSans-65" x="64.982"/>
                <use xlink:href="#DejaVuSans-4c" x="126.506"/>
                <use xlink:href="#DejaVuSans-55" x="177.219"/>
                <use xlink:href="#DejaVuSans-20" x="250.412"/>
                <use xlink:href="#DejaVuSans-46" x="282.199"/>
                <use xlink:href="#DejaVuSans-75" x="334.219"/>
                <use xlink:href="#DejaVuSans-6e" x="397.598"/>
                <use xlink:href="#DejaVuSans-63" x="460.977"/>
                <use xlink:href="#DejaVuSans-74" x="515.957"/>
                <use xlink:href="#DejaVuSans-69" x="555.166"/>
                <use xlink:href="#DejaVuSans-6f" x="582.949"/>
                <use xlink:href="#DejaVuSans-6e" x="644.131"/>
            </g>
        </g>
    </g>
    <defs>
        <clipPath id="pec10fe9d12">
            <path d="M57.6 41.472h357.12v266.112H57.6z"/>
        </clipPath>
    </defs>
</svg>



#### 多维数组实现神经网络

**矩阵乘法**

```python
import matplotlib.pyplot as plt
import numpy as np

def main():
    A = np.array([[1, 2, 3], [4, 5, 6]])
    # print(A.shape) # 2 * 3
    # print(A.ndim) # shape 的数量 2 (维度)
    B = np.array([[1, 2], [3, 4], [5, 6]])
    print(np.dot(A, B))

if __name__ == "__main__":
    main()
```

在逻辑门的实现中 我们实现了 

$b + \left(
\begin{array}{cccc} 
   x_1~~~x_2\\  
\end{array}
\right) \times \left(
\begin{array}{cccc} 
   w_1\\
   w_2\\
\end{array}
\right)$ $\le$ 0 ( $x_1~~x_2$ 对于列向量的变幻可以广播之后其余项为 0，可以更好把变量 $x$ 抽象 $w_1~~w_2$ 定值列向量的变幻，后续易于扩充)



#### 激活函数的应用

```python
import numpy as np

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def identity_function(x): # 恒等函数
    return x
# 三层神经网络
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y) # [ 0.31682708, 0.69627909 ]
```



### 神经网络输出层

#### softmax

**回归问题用恒等函数，分类问题用softmax函数**

把值转化为概率 (数值越大特征越明显)，对问题进行分类

```python
def softmax(a):
 	exp_a = np.exp(a)
 	sum_exp_a = np.sum(exp_a)
 	y = exp_a / sum_exp_a
 	return y
a = np.array([0.3, 2.9, 4.0])
print(softmax(a)) # [ 0.01821127 0.24519181 0.73659691 ] 和为 1
```



**softmax函数的缺陷**

缺陷就是溢出问题，$softmax$ 函数的实现中要进行指 数函数的运算，但是此时指数函数的值很容易变得非常大

比如，$e^{10}$的值会超过 $20000$ ，$e^{100}$ 会变成一个后面有 $40$ 多个 $0$ 的超大值

$e^{1000}$ 的结果会返回 一个表示无穷大的 $inf$



**改进**

```python
def softmax(a):
 	c = np.max(a)
 	exp_a = np.exp(a - c) # 溢出对策
 	sum_exp_a = np.sum(exp_a)
 	y = exp_a / sum_exp_a
 	return y
```



<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="614.4" height="460.8" viewBox="0 0 460.8 345.6">
    <defs>
        <style>
            *{stroke-linejoin:round;stroke-linecap:butt}
        </style>
    </defs>
    <g id="figure_1">
        <path id="patch_1" d="M0 345.6h460.8V0H0z" style="fill:#fff"/>
        <g id="axes_1">
            <path id="patch_2" d="M57.6 307.584h357.12V41.472H57.6z" style="fill:#fff"/>
            <g id="matplotlib.axis_1">
                <g id="xtick_1">
                    <g id="line2d_1">
                        <defs>
                            <path id="m46898e0d8c" d="M0 0v3.5" style="stroke:#000;stroke-width:.8"/>
                        </defs>
                        <use xlink:href="#m46898e0d8c" x="73.833" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    </g>
                    <g id="text_1" transform="matrix(.1 0 0 -.1 61.691 322.182)">
                        <defs>
                            <path id="DejaVuSans-2212" d="M678 2272h4006v-531H678v531z" transform="scale(.01563)"/>
                            <path id="DejaVuSans-32" d="M1228 531h2203V0H469v531q359 372 979 998 621 627 780 809 303 340 423 576 121 236 121 464 0 372-261 606-261 235-680 235-297 0-627-103-329-103-704-313v638q381 153 712 231 332 78 607 78 725 0 1156-363 431-362 431-968 0-288-108-546-107-257-392-607-78-91-497-524-418-433-1181-1211z" transform="scale(.01563)"/>
                            <path id="DejaVuSans-2e" d="M684 794h660V0H684v794z" transform="scale(.01563)"/>
                            <path id="DejaVuSans-30" d="M2034 4250q-487 0-733-480-245-479-245-1442 0-959 245-1439 246-480 733-480 491 0 736 480 246 480 246 1439 0 963-246 1442-245 480-736 480zm0 500q785 0 1199-621 414-620 414-1801 0-1178-414-1799Q2819-91 2034-91q-784 0-1198 620-414 621-414 1799 0 1181 414 1801 414 621 1198 621z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-2212"/>
                        <use xlink:href="#DejaVuSans-32" x="83.789"/>
                        <use xlink:href="#DejaVuSans-2e" x="147.412"/>
                        <use xlink:href="#DejaVuSans-30" x="179.199"/>
                    </g>
                </g>
                <g id="xtick_2">
                    <use xlink:href="#m46898e0d8c" id="line2d_2" x="114.415" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_2" transform="matrix(.1 0 0 -.1 102.273 322.182)">
                        <defs>
                            <path id="DejaVuSans-31" d="M794 531h1031v3560L703 3866v575l1116 225h631V531h1031V0H794v531z" transform="scale(.01563)"/>
                            <path id="DejaVuSans-35" d="M691 4666h2478v-532H1269V2991q137 47 274 70 138 23 276 23 781 0 1237-428 457-428 457-1159 0-753-469-1171Q2575-91 1722-91q-294 0-599 50Q819 9 494 109v635q281-153 581-228t634-75q541 0 856 284 316 284 316 772 0 487-316 771-315 285-856 285-253 0-505-56-251-56-513-175v2344z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-2212"/>
                        <use xlink:href="#DejaVuSans-31" x="83.789"/>
                        <use xlink:href="#DejaVuSans-2e" x="147.412"/>
                        <use xlink:href="#DejaVuSans-35" x="179.199"/>
                    </g>
                </g>
                <g id="xtick_3">
                    <use xlink:href="#m46898e0d8c" id="line2d_3" x="154.996" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_3" transform="matrix(.1 0 0 -.1 142.855 322.182)">
                        <use xlink:href="#DejaVuSans-2212"/>
                        <use xlink:href="#DejaVuSans-31" x="83.789"/>
                        <use xlink:href="#DejaVuSans-2e" x="147.412"/>
                        <use xlink:href="#DejaVuSans-30" x="179.199"/>
                    </g>
                </g>
                <g id="xtick_4">
                    <use xlink:href="#m46898e0d8c" id="line2d_4" x="195.578" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_4" transform="matrix(.1 0 0 -.1 183.437 322.182)">
                        <use xlink:href="#DejaVuSans-2212"/>
                        <use xlink:href="#DejaVuSans-30" x="83.789"/>
                        <use xlink:href="#DejaVuSans-2e" x="147.412"/>
                        <use xlink:href="#DejaVuSans-35" x="179.199"/>
                    </g>
                </g>
                <g id="xtick_5">
                    <use xlink:href="#m46898e0d8c" id="line2d_5" x="236.16" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_5" transform="matrix(.1 0 0 -.1 228.208 322.182)">
                        <use xlink:href="#DejaVuSans-30"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-30" x="95.41"/>
                    </g>
                </g>
                <g id="xtick_6">
                    <use xlink:href="#m46898e0d8c" id="line2d_6" x="276.742" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_6" transform="matrix(.1 0 0 -.1 268.79 322.182)">
                        <use xlink:href="#DejaVuSans-30"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-35" x="95.41"/>
                    </g>
                </g>
                <g id="xtick_7">
                    <use xlink:href="#m46898e0d8c" id="line2d_7" x="317.324" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_7" transform="matrix(.1 0 0 -.1 309.372 322.182)">
                        <use xlink:href="#DejaVuSans-31"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-30" x="95.41"/>
                    </g>
                </g>
                <g id="xtick_8">
                    <use xlink:href="#m46898e0d8c" id="line2d_8" x="357.905" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_8" transform="matrix(.1 0 0 -.1 349.954 322.182)">
                        <use xlink:href="#DejaVuSans-31"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-35" x="95.41"/>
                    </g>
                </g>
                <g id="xtick_9">
                    <use xlink:href="#m46898e0d8c" id="line2d_9" x="398.487" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_9" transform="matrix(.1 0 0 -.1 390.536 322.182)">
                        <use xlink:href="#DejaVuSans-32"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-30" x="95.41"/>
                    </g>
                </g>
                <g id="text_10" transform="matrix(.1 0 0 -.1 233.2 335.86)">
                    <defs>
                        <path id="DejaVuSans-78" d="M3513 3500 2247 1797 3578 0h-678L1881 1375 863 0H184l1360 1831L300 3500h678l928-1247 928 1247h679z" transform="scale(.01563)"/>
                    </defs>
                    <use xlink:href="#DejaVuSans-78"/>
                </g>
            </g>
            <g id="matplotlib.axis_2">
                <g id="ytick_1">
                    <g id="line2d_10">
                        <defs>
                            <path id="mf8bca79cbd" d="M0 0h-3.5" style="stroke:#000;stroke-width:.8"/>
                        </defs>
                        <use xlink:href="#mf8bca79cbd" x="57.6" y="300.002" style="stroke:#000;stroke-width:.8"/>
                    </g>
                    <g id="text_11" transform="matrix(.1 0 0 -.1 21.972 303.8)">
                        <use xlink:href="#DejaVuSans-30"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-30" x="95.41"/>
                        <use xlink:href="#DejaVuSans-30" x="159.033"/>
                        <use xlink:href="#DejaVuSans-30" x="222.656"/>
                    </g>
                </g>
                <g id="ytick_2">
                    <use xlink:href="#mf8bca79cbd" id="line2d_11" x="57.6" y="269.433" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_12" transform="matrix(.1 0 0 -.1 21.972 273.232)">
                        <use xlink:href="#DejaVuSans-30"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-30" x="95.41"/>
                        <use xlink:href="#DejaVuSans-30" x="159.033"/>
                        <use xlink:href="#DejaVuSans-35" x="222.656"/>
                    </g>
                </g>
                <g id="ytick_3">
                    <use xlink:href="#mf8bca79cbd" id="line2d_12" x="57.6" y="238.864" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_13" transform="matrix(.1 0 0 -.1 21.972 242.663)">
                        <use xlink:href="#DejaVuSans-30"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-30" x="95.41"/>
                        <use xlink:href="#DejaVuSans-31" x="159.033"/>
                        <use xlink:href="#DejaVuSans-30" x="222.656"/>
                    </g>
                </g>
                <g id="ytick_4">
                    <use xlink:href="#mf8bca79cbd" id="line2d_13" x="57.6" y="208.294" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_14" transform="matrix(.1 0 0 -.1 21.972 212.094)">
                        <use xlink:href="#DejaVuSans-30"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-30" x="95.41"/>
                        <use xlink:href="#DejaVuSans-31" x="159.033"/>
                        <use xlink:href="#DejaVuSans-35" x="222.656"/>
                    </g>
                </g>
                <g id="ytick_5">
                    <use xlink:href="#mf8bca79cbd" id="line2d_14" x="57.6" y="177.725" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_15" transform="matrix(.1 0 0 -.1 21.972 181.525)">
                        <use xlink:href="#DejaVuSans-30"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-30" x="95.41"/>
                        <use xlink:href="#DejaVuSans-32" x="159.033"/>
                        <use xlink:href="#DejaVuSans-30" x="222.656"/>
                    </g>
                </g>
                <g id="ytick_6">
                    <use xlink:href="#mf8bca79cbd" id="line2d_15" x="57.6" y="147.156" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_16" transform="matrix(.1 0 0 -.1 21.972 150.956)">
                        <use xlink:href="#DejaVuSans-30"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-30" x="95.41"/>
                        <use xlink:href="#DejaVuSans-32" x="159.033"/>
                        <use xlink:href="#DejaVuSans-35" x="222.656"/>
                    </g>
                </g>
                <g id="ytick_7">
                    <use xlink:href="#mf8bca79cbd" id="line2d_16" x="57.6" y="116.587" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_17" transform="matrix(.1 0 0 -.1 21.972 120.387)">
                        <defs>
                            <path id="DejaVuSans-33" d="M2597 2516q453-97 707-404 255-306 255-756 0-690-475-1069Q2609-91 1734-91q-293 0-604 58T488 141v609q262-153 574-231 313-78 654-78 593 0 904 234t311 681q0 413-289 645-289 233-804 233h-544v519h569q465 0 712 186t247 536q0 359-255 551-254 193-729 193-260 0-557-57-297-56-653-174v562q360 100 674 150t592 50q719 0 1137-327 419-326 419-882 0-388-222-655t-631-370z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-30"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-30" x="95.41"/>
                        <use xlink:href="#DejaVuSans-33" x="159.033"/>
                        <use xlink:href="#DejaVuSans-30" x="222.656"/>
                    </g>
                </g>
                <g id="ytick_8">
                    <use xlink:href="#mf8bca79cbd" id="line2d_17" x="57.6" y="86.018" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_18" transform="matrix(.1 0 0 -.1 21.972 89.818)">
                        <use xlink:href="#DejaVuSans-30"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-30" x="95.41"/>
                        <use xlink:href="#DejaVuSans-33" x="159.033"/>
                        <use xlink:href="#DejaVuSans-35" x="222.656"/>
                    </g>
                </g>
                <g id="ytick_9">
                    <use xlink:href="#mf8bca79cbd" id="line2d_18" x="57.6" y="55.449" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_19" transform="matrix(.1 0 0 -.1 21.972 59.248)">
                        <defs>
                            <path id="DejaVuSans-34" d="M2419 4116 825 1625h1594v2491zm-166 550h794V1625h666v-525h-666V0h-628v1100H313v609l1940 2957z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-30"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-30" x="95.41"/>
                        <use xlink:href="#DejaVuSans-34" x="159.033"/>
                        <use xlink:href="#DejaVuSans-30" x="222.656"/>
                    </g>
                </g>
                <g id="text_20" transform="matrix(0 -.1 -.1 0 15.892 201.58)">
                    <defs>
                        <path id="DejaVuSans-73" d="M2834 3397v-544q-243 125-506 187-262 63-544 63-428 0-642-131t-214-394q0-200 153-314t616-217l197-44q612-131 870-370t258-667q0-488-386-773Q2250-91 1575-91q-281 0-586 55T347 128v594q319-166 628-249 309-82 613-82 406 0 624 139 219 139 219 392 0 234-158 359-157 125-692 241l-200 47q-534 112-772 345-237 233-237 639 0 494 350 762 350 269 994 269 318 0 599-47 282-46 519-140z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-6f" d="M1959 3097q-462 0-731-361t-269-989q0-628 267-989 268-361 733-361 460 0 728 362 269 363 269 988 0 622-269 986-268 364-728 364zm0 487q750 0 1178-488 429-487 429-1349 0-859-429-1349Q2709-91 1959-91q-753 0-1180 489-426 490-426 1349 0 862 426 1349 427 488 1180 488z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-66" d="M2375 4863v-479h-550q-309 0-430-125-120-125-120-450v-309h947v-447h-947V0H697v3053H147v447h550v244q0 584 272 851 272 268 862 268h544z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-74" d="M1172 4494v-994h1184v-447H1172V1153q0-428 117-550t477-122h590V0h-590q-666 0-919 248-253 249-253 905v1900H172v447h422v994h578z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-6d" d="M3328 2828q216 388 516 572t706 184q547 0 844-383 297-382 297-1088V0h-578v2094q0 503-179 746-178 244-543 244-447 0-707-297-259-296-259-809V0h-578v2094q0 506-178 748t-550 242q-441 0-701-298-259-298-259-808V0H581v3500h578v-544q197 322 472 475t653 153q382 0 649-194 267-193 395-562z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-61" d="M2194 1759q-697 0-966-159t-269-544q0-306 202-486 202-179 548-179 479 0 768 339t289 901v128h-572zm1147 238V0h-575v531q-197-318-491-470T1556-91q-537 0-855 302-317 302-317 808 0 590 395 890 396 300 1180 300h807v57q0 397-261 614t-733 217q-300 0-585-72-284-72-546-216v532q315 122 612 182 297 61 578 61 760 0 1135-394 375-393 375-1193z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-28" d="M1984 4856q-418-718-622-1422-203-703-203-1425 0-721 205-1429t620-1424h-500Q1016-109 783 600T550 2009q0 697 231 1403 232 707 703 1444h500z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-29" d="M513 4856h500q468-737 701-1444 233-706 233-1403 0-700-233-1409T1013-844H513q415 716 620 1424t205 1429q0 722-205 1425-205 704-620 1422z" transform="scale(.01563)"/>
                    </defs>
                    <use xlink:href="#DejaVuSans-73"/>
                    <use xlink:href="#DejaVuSans-6f" x="52.1"/>
                    <use xlink:href="#DejaVuSans-66" x="113.281"/>
                    <use xlink:href="#DejaVuSans-74" x="146.736"/>
                    <use xlink:href="#DejaVuSans-6d" x="185.945"/>
                    <use xlink:href="#DejaVuSans-61" x="283.357"/>
                    <use xlink:href="#DejaVuSans-78" x="344.637"/>
                    <use xlink:href="#DejaVuSans-28" x="403.816"/>
                    <use xlink:href="#DejaVuSans-78" x="442.83"/>
                    <use xlink:href="#DejaVuSans-29" x="502.01"/>
                </g>
            </g>
            <path id="line2d_19" d="m73.833 295.488 3.28-.186 3.278-.194 3.28-.202 3.28-.21 3.278-.218 3.28-.228 3.28-.237 3.278-.247 3.28-.257 3.28-.268 3.278-.279 3.28-.29 3.28-.302 3.278-.315 3.28-.328 3.28-.34 3.278-.356 3.28-.37 3.28-.385 3.279-.401 3.279-.418 3.28-.434 3.279-.453 3.279-.471 3.28-.491 3.279-.511 3.279-.532 3.28-.554 3.279-.577 3.279-.6 3.28-.626 3.279-.651 3.279-.678 3.28-.706 3.279-.736 3.279-.765 3.28-.797 3.279-.83 3.279-.864 3.28-.9 3.279-.936 3.279-.976 3.28-1.015 3.279-1.058 3.279-1.101 3.28-1.147 3.279-1.193 3.279-1.243 3.28-1.295 3.279-1.347 3.279-1.403 3.28-1.461 3.279-1.522 3.279-1.584 3.28-1.649 3.279-1.717 3.279-1.788 3.28-1.862 3.279-1.939 3.28-2.018 3.278-2.102 3.28-2.189 3.28-2.278 3.278-2.373 3.28-2.47 3.28-2.573 3.278-2.678 3.28-2.789 3.28-2.904 3.278-3.023 3.28-3.148 3.28-3.278 3.278-3.413 3.28-3.554 3.28-3.7 3.278-3.853 3.28-4.012 3.28-4.178 3.278-4.349 3.28-4.529 3.28-4.715 3.279-4.91 3.279-5.113 3.28-5.323 3.279-5.543 3.279-5.77 3.28-6.01 3.279-6.257 3.279-6.515 3.28-6.783 3.279-7.063 3.279-7.355 3.28-7.658 3.279-7.973 3.279-8.302 3.28-8.645 3.279-9 3.279-9.373 3.28-9.758" clip-path="url(#p027f1e1dce)" style="fill:none;stroke:#1f77b4;stroke-width:1.5;stroke-linecap:square"/>
            <path id="line2d_20" d="M236.16 307.584V41.472" clip-path="url(#p027f1e1dce)" style="fill:none;stroke:#000;stroke-width:1.5;stroke-linecap:square"/>
            <path id="patch_3" d="M57.6 307.584V41.472" style="fill:none;stroke:#000;stroke-width:.8;stroke-linejoin:miter;stroke-linecap:square"/>
            <path id="patch_4" d="M414.72 307.584V41.472" style="fill:none;stroke:#000;stroke-width:.8;stroke-linejoin:miter;stroke-linecap:square"/>
            <path id="patch_5" d="M57.6 307.584h357.12" style="fill:none;stroke:#000;stroke-width:.8;stroke-linejoin:miter;stroke-linecap:square"/>
            <path id="patch_6" d="M57.6 41.472h357.12" style="fill:none;stroke:#000;stroke-width:.8;stroke-linejoin:miter;stroke-linecap:square"/>
            <g id="text_21" transform="matrix(.12 0 0 -.12 183.823 35.472)">
                <defs>
                    <path id="DejaVuSans-53" d="M3425 4513v-616q-359 172-678 256-319 85-616 85-515 0-795-200t-280-569q0-310 186-468 186-157 705-254l381-78q706-135 1042-474t336-907q0-679-455-1029Q2797-91 1919-91q-331 0-705 75-373 75-773 222v650q384-215 753-325 369-109 725-109 540 0 834 212 294 213 294 607 0 343-211 537t-692 291l-385 75q-706 140-1022 440-315 300-315 835 0 619 436 975t1201 356q329 0 669-60 341-59 697-177z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-46" d="M628 4666h2681v-532H1259V2759h1850v-531H1259V0H628v4666z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-75" d="M544 1381v2119h575V1403q0-497 193-746 194-248 582-248 465 0 735 297 271 297 271 810v1984h575V0h-575v538q-209-319-486-474-276-155-642-155-603 0-916 375-312 375-312 1097zm1447 2203z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-6e" d="M3513 2113V0h-575v2094q0 497-194 743-194 247-581 247-466 0-735-297-269-296-269-809V0H581v3500h578v-544q207 316 486 472 280 156 646 156 603 0 912-373 310-373 310-1098z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-63" d="M3122 3366v-538q-244 135-489 202t-495 67q-560 0-870-355-309-354-309-995t309-996q310-354 870-354 250 0 495 67t489 202V134Q2881 22 2623-34q-257-57-548-57-791 0-1257 497-465 497-465 1341 0 856 470 1346 471 491 1290 491 265 0 518-55 253-54 491-163z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-69" d="M603 3500h575V0H603v3500zm0 1363h575v-729H603v729z" transform="scale(.01563)"/>
                </defs>
                <use xlink:href="#DejaVuSans-53"/>
                <use xlink:href="#DejaVuSans-6f" x="63.477"/>
                <use xlink:href="#DejaVuSans-66" x="124.658"/>
                <use xlink:href="#DejaVuSans-74" x="158.113"/>
                <use xlink:href="#DejaVuSans-6d" x="197.322"/>
                <use xlink:href="#DejaVuSans-61" x="294.734"/>
                <use xlink:href="#DejaVuSans-78" x="356.014"/>
                <use xlink:href="#DejaVuSans-20" x="415.193"/>
                <use xlink:href="#DejaVuSans-46" x="446.98"/>
                <use xlink:href="#DejaVuSans-75" x="499"/>
                <use xlink:href="#DejaVuSans-6e" x="562.379"/>
                <use xlink:href="#DejaVuSans-63" x="625.758"/>
                <use xlink:href="#DejaVuSans-74" x="680.738"/>
                <use xlink:href="#DejaVuSans-69" x="719.947"/>
                <use xlink:href="#DejaVuSans-6f" x="747.73"/>
                <use xlink:href="#DejaVuSans-6e" x="808.912"/>
            </g>
        </g>
    </g>
    <defs>
        <clipPath id="p027f1e1dce">
            <path d="M57.6 41.472h357.12v266.112H57.6z"/>
        </clipPath>
    </defs>
</svg>



**和 $sigmoid$ 公式一样** 具有以下优点

放大效果良好，没有边界问题不会有负数

可导，这在这使得在使用反向传播算法进行模型训练时更加方便





#### Tanh

$Tanh$ 的诞生比 $Sigmoid$ 晚一些，$sigmoid$ 函数我们提到过有一个缺点就是输出不以 $0$ 为中心，使得收敛变慢的问题。而Tanh则就是解决了这个问题。$Tanh$ 就是双曲正切函数。等于双曲余弦除双曲正弦。函数表达式和图像见下图。这个函数是一个奇函数。

$tanh(x)=\frac{sinh(x)}{cosh(x)}=\frac{e^x-e^{-x}}{e^x+e^{-x}}$

$tanh'(x)=1-tanh^2(x)$



```python
def tanh(x):
    return np.tanh(x)
```



#### 损失函数

##### 均方误差

可以用作损失函数的函数有很多，其中最有名的是**均方误差**（$mean$  $squared$  $error$）

比如在识别手写数字的程序中，希望得到这样一组数 $(0,~1,~0,~...,~0)$ 唯一确定这个数一定是 $2$

```python
import numpy as np
# 而除以 2 的作用是为了在计算梯度时将平方项的系数 2 和除以 2 相互抵消, 从而简化计算
def mean_squared_error(y, t):
 	return 0.5 * np.sum((y - t) ** 2)

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] # 手写识别为 2 的理想模型
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0] # 需要调整的模型
ans = mean_squared_error(np.array(y), np.array(t))
print(ans) # 0.097500000000000031
```

<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="614.4" height="460.8" viewBox="0 0 460.8 345.6">
    <defs>
        <style>
            *{stroke-linejoin:round;stroke-linecap:butt}
        </style>
    </defs>
    <g id="figure_1">
        <path id="patch_1" d="M0 345.6h460.8V0H0z" style="fill:#fff"/>
        <g id="axes_1">
            <path id="patch_2" d="M57.6 307.584h357.12V41.472H57.6z" style="fill:#fff"/>
            <g id="matplotlib.axis_1">
                <g id="xtick_1">
                    <g id="line2d_1">
                        <defs>
                            <path id="m1be7033cca" d="M0 0v3.5" style="stroke:#000;stroke-width:.8"/>
                        </defs>
                        <use xlink:href="#m1be7033cca" x="73.833" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    </g>
                    <g id="text_1" transform="matrix(.1 0 0 -.1 65.881 322.182)">
                        <defs>
                            <path id="DejaVuSans-30" d="M2034 4250q-487 0-733-480-245-479-245-1442 0-959 245-1439 246-480 733-480 491 0 736 480 246 480 246 1439 0 963-246 1442-245 480-736 480zm0 500q785 0 1199-621 414-620 414-1801 0-1178-414-1799Q2819-91 2034-91q-784 0-1198 620-414 621-414 1799 0 1181 414 1801 414 621 1198 621z" transform="scale(.01563)"/>
                            <path id="DejaVuSans-2e" d="M684 794h660V0H684v794z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-30"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-30" x="95.41"/>
                    </g>
                </g>
                <g id="xtick_2">
                    <use xlink:href="#m1be7033cca" id="line2d_2" x="114.415" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_2" transform="matrix(.1 0 0 -.1 106.463 322.182)">
                        <defs>
                            <path id="DejaVuSans-35" d="M691 4666h2478v-532H1269V2991q137 47 274 70 138 23 276 23 781 0 1237-428 457-428 457-1159 0-753-469-1171Q2575-91 1722-91q-294 0-599 50Q819 9 494 109v635q281-153 581-228t634-75q541 0 856 284 316 284 316 772 0 487-316 771-315 285-856 285-253 0-505-56-251-56-513-175v2344z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-30"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-35" x="95.41"/>
                    </g>
                </g>
                <g id="xtick_3">
                    <use xlink:href="#m1be7033cca" id="line2d_3" x="154.996" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_3" transform="matrix(.1 0 0 -.1 147.045 322.182)">
                        <defs>
                            <path id="DejaVuSans-31" d="M794 531h1031v3560L703 3866v575l1116 225h631V531h1031V0H794v531z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-31"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-30" x="95.41"/>
                    </g>
                </g>
                <g id="xtick_4">
                    <use xlink:href="#m1be7033cca" id="line2d_4" x="195.578" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_4" transform="matrix(.1 0 0 -.1 187.627 322.182)">
                        <use xlink:href="#DejaVuSans-31"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-35" x="95.41"/>
                    </g>
                </g>
                <g id="xtick_5">
                    <use xlink:href="#m1be7033cca" id="line2d_5" x="236.16" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_5" transform="matrix(.1 0 0 -.1 228.208 322.182)">
                        <defs>
                            <path id="DejaVuSans-32" d="M1228 531h2203V0H469v531q359 372 979 998 621 627 780 809 303 340 423 576 121 236 121 464 0 372-261 606-261 235-680 235-297 0-627-103-329-103-704-313v638q381 153 712 231 332 78 607 78 725 0 1156-363 431-362 431-968 0-288-108-546-107-257-392-607-78-91-497-524-418-433-1181-1211z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-32"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-30" x="95.41"/>
                    </g>
                </g>
                <g id="xtick_6">
                    <use xlink:href="#m1be7033cca" id="line2d_6" x="276.742" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_6" transform="matrix(.1 0 0 -.1 268.79 322.182)">
                        <use xlink:href="#DejaVuSans-32"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-35" x="95.41"/>
                    </g>
                </g>
                <g id="xtick_7">
                    <use xlink:href="#m1be7033cca" id="line2d_7" x="317.324" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_7" transform="matrix(.1 0 0 -.1 309.372 322.182)">
                        <defs>
                            <path id="DejaVuSans-33" d="M2597 2516q453-97 707-404 255-306 255-756 0-690-475-1069Q2609-91 1734-91q-293 0-604 58T488 141v609q262-153 574-231 313-78 654-78 593 0 904 234t311 681q0 413-289 645-289 233-804 233h-544v519h569q465 0 712 186t247 536q0 359-255 551-254 193-729 193-260 0-557-57-297-56-653-174v562q360 100 674 150t592 50q719 0 1137-327 419-326 419-882 0-388-222-655t-631-370z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-33"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-30" x="95.41"/>
                    </g>
                </g>
                <g id="xtick_8">
                    <use xlink:href="#m1be7033cca" id="line2d_8" x="357.905" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_8" transform="matrix(.1 0 0 -.1 349.954 322.182)">
                        <use xlink:href="#DejaVuSans-33"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-35" x="95.41"/>
                    </g>
                </g>
                <g id="xtick_9">
                    <use xlink:href="#m1be7033cca" id="line2d_9" x="398.487" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_9" transform="matrix(.1 0 0 -.1 390.536 322.182)">
                        <defs>
                            <path id="DejaVuSans-34" d="M2419 4116 825 1625h1594v2491zm-166 550h794V1625h666v-525h-666V0h-628v1100H313v609l1940 2957z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-34"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-30" x="95.41"/>
                    </g>
                </g>
                <g id="text_10" transform="matrix(.1 0 0 -.1 232.07 335.86)">
                    <defs>
                        <path id="DejaVuSans-77" d="M269 3500h575l719-2731 715 2731h678l719-2731 716 2731h575L4050 0h-678l-753 2869L1863 0h-679L269 3500z" transform="scale(.01563)"/>
                    </defs>
                    <use xlink:href="#DejaVuSans-77"/>
                </g>
            </g>
            <g id="matplotlib.axis_2">
                <g id="ytick_1">
                    <g id="line2d_10">
                        <defs>
                            <path id="m4e69dbd42d" d="M0 0h-3.5" style="stroke:#000;stroke-width:.8"/>
                        </defs>
                        <use xlink:href="#m4e69dbd42d" x="57.6" y="302.228" style="stroke:#000;stroke-width:.8"/>
                    </g>
                    <use xlink:href="#DejaVuSans-30" id="text_11" transform="matrix(.1 0 0 -.1 44.237 306.028)"/>
                </g>
                <g id="ytick_2">
                    <use xlink:href="#m4e69dbd42d" id="line2d_11" x="57.6" y="269.893" style="stroke:#000;stroke-width:.8"/>
                    <use xlink:href="#DejaVuSans-35" id="text_12" transform="matrix(.1 0 0 -.1 44.237 273.692)"/>
                </g>
                <g id="ytick_3">
                    <use xlink:href="#m4e69dbd42d" id="line2d_12" x="57.6" y="237.557" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_13" transform="matrix(.1 0 0 -.1 37.875 241.356)">
                        <use xlink:href="#DejaVuSans-31"/>
                        <use xlink:href="#DejaVuSans-30" x="63.623"/>
                    </g>
                </g>
                <g id="ytick_4">
                    <use xlink:href="#m4e69dbd42d" id="line2d_13" x="57.6" y="205.221" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_14" transform="matrix(.1 0 0 -.1 37.875 209.02)">
                        <use xlink:href="#DejaVuSans-31"/>
                        <use xlink:href="#DejaVuSans-35" x="63.623"/>
                    </g>
                </g>
                <g id="ytick_5">
                    <use xlink:href="#m4e69dbd42d" id="line2d_14" x="57.6" y="172.886" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_15" transform="matrix(.1 0 0 -.1 37.875 176.685)">
                        <use xlink:href="#DejaVuSans-32"/>
                        <use xlink:href="#DejaVuSans-30" x="63.623"/>
                    </g>
                </g>
                <g id="ytick_6">
                    <use xlink:href="#m4e69dbd42d" id="line2d_15" x="57.6" y="140.55" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_16" transform="matrix(.1 0 0 -.1 37.875 144.35)">
                        <use xlink:href="#DejaVuSans-32"/>
                        <use xlink:href="#DejaVuSans-35" x="63.623"/>
                    </g>
                </g>
                <g id="ytick_7">
                    <use xlink:href="#m4e69dbd42d" id="line2d_16" x="57.6" y="108.214" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_17" transform="matrix(.1 0 0 -.1 37.875 112.013)">
                        <use xlink:href="#DejaVuSans-33"/>
                        <use xlink:href="#DejaVuSans-30" x="63.623"/>
                    </g>
                </g>
                <g id="ytick_8">
                    <use xlink:href="#m4e69dbd42d" id="line2d_17" x="57.6" y="75.878" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_18" transform="matrix(.1 0 0 -.1 37.875 79.678)">
                        <use xlink:href="#DejaVuSans-33"/>
                        <use xlink:href="#DejaVuSans-35" x="63.623"/>
                    </g>
                </g>
                <g id="ytick_9">
                    <use xlink:href="#m4e69dbd42d" id="line2d_18" x="57.6" y="43.543" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_19" transform="matrix(.1 0 0 -.1 37.875 47.342)">
                        <use xlink:href="#DejaVuSans-34"/>
                        <use xlink:href="#DejaVuSans-30" x="63.623"/>
                    </g>
                </g>
                <g id="text_20" transform="matrix(0 -.1 -.1 0 31.795 185.176)">
                    <defs>
                        <path id="DejaVuSans-4d" d="M628 4666h941l1190-3175 1197 3175h941V0h-616v4097L3078 897h-634L1241 4097V0H628v4666z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-53" d="M3425 4513v-616q-359 172-678 256-319 85-616 85-515 0-795-200t-280-569q0-310 186-468 186-157 705-254l381-78q706-135 1042-474t336-907q0-679-455-1029Q2797-91 1919-91q-331 0-705 75-373 75-773 222v650q384-215 753-325 369-109 725-109 540 0 834 212 294 213 294 607 0 343-211 537t-692 291l-385 75q-706 140-1022 440-315 300-315 835 0 619 436 975t1201 356q329 0 669-60 341-59 697-177z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-45" d="M628 4666h2950v-532H1259V2753h2222v-531H1259V531h2375V0H628v4666z" transform="scale(.01563)"/>
                    </defs>
                    <use xlink:href="#DejaVuSans-4d"/>
                    <use xlink:href="#DejaVuSans-53" x="86.279"/>
                    <use xlink:href="#DejaVuSans-45" x="149.756"/>
                </g>
            </g>
            <path id="line2d_19" d="m73.833 53.568 6.625 18.648 6.626 17.9 6.626 17.153 6.625 16.405 6.626 15.657 6.625 14.91 6.626 14.163 6.626 13.414 6.625 12.668 6.626 11.92 6.625 11.171 6.626 10.425 6.626 9.676 6.625 8.93 6.626 8.181 6.625 7.434 6.626 6.687 6.626 5.939 6.625 5.19 6.626 4.444 6.625 3.697 6.626 2.948 6.626 2.2 6.625 1.454 6.626.706 6.625-.042 6.626-.79 6.626-1.536 6.625-2.285 6.626-3.032 6.625-3.78 6.626-4.527 6.626-5.275 6.625-6.023 6.626-6.77 6.625-7.518 6.626-8.265 6.626-9.013 6.625-9.76 6.626-10.509 6.625-11.255 6.626-12.004 6.626-12.75 6.625-13.499 6.626-14.246 6.625-14.994 6.626-15.74 6.626-16.49 6.625-17.236" clip-path="url(#pccbfe9029a)" style="fill:none;stroke:#1f77b4;stroke-width:1.5;stroke-linecap:square"/>
            <path id="line2d_20" d="M57.6 302.228h357.12" clip-path="url(#pccbfe9029a)" style="fill:none;stroke:#000;stroke-width:1.5;stroke-linecap:square"/>
            <path id="line2d_21" d="M73.833 307.584V41.472" clip-path="url(#pccbfe9029a)" style="fill:none;stroke:#000;stroke-width:1.5;stroke-linecap:square"/>
            <path id="patch_3" d="M57.6 307.584V41.472" style="fill:none;stroke:#000;stroke-width:.8;stroke-linejoin:miter;stroke-linecap:square"/>
            <path id="patch_4" d="M414.72 307.584V41.472" style="fill:none;stroke:#000;stroke-width:.8;stroke-linejoin:miter;stroke-linecap:square"/>
            <path id="patch_5" d="M57.6 307.584h357.12" style="fill:none;stroke:#000;stroke-width:.8;stroke-linejoin:miter;stroke-linecap:square"/>
            <path id="patch_6" d="M57.6 41.472h357.12" style="fill:none;stroke:#000;stroke-width:.8;stroke-linejoin:miter;stroke-linecap:square"/>
            <g id="text_21" transform="matrix(.12 0 0 -.12 180.89 35.472)">
                <defs>
                    <path id="DejaVuSans-4c" d="M628 4666h631V531h2272V0H628v4666z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-6f" d="M1959 3097q-462 0-731-361t-269-989q0-628 267-989 268-361 733-361 460 0 728 362 269 363 269 988 0 622-269 986-268 364-728 364zm0 487q750 0 1178-488 429-487 429-1349 0-859-429-1349Q2709-91 1959-91q-753 0-1180 489-426 490-426 1349 0 862 426 1349 427 488 1180 488z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-73" d="M2834 3397v-544q-243 125-506 187-262 63-544 63-428 0-642-131t-214-394q0-200 153-314t616-217l197-44q612-131 870-370t258-667q0-488-386-773Q2250-91 1575-91q-281 0-586 55T347 128v594q319-166 628-249 309-82 613-82 406 0 624 139 219 139 219 392 0 234-158 359-157 125-692 241l-200 47q-534 112-772 345-237 233-237 639 0 494 350 762 350 269 994 269 318 0 599-47 282-46 519-140z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-46" d="M628 4666h2681v-532H1259V2759h1850v-531H1259V0H628v4666z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-75" d="M544 1381v2119h575V1403q0-497 193-746 194-248 582-248 465 0 735 297 271 297 271 810v1984h575V0h-575v538q-209-319-486-474-276-155-642-155-603 0-916 375-312 375-312 1097zm1447 2203z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-6e" d="M3513 2113V0h-575v2094q0 497-194 743-194 247-581 247-466 0-735-297-269-296-269-809V0H581v3500h578v-544q207 316 486 472 280 156 646 156 603 0 912-373 310-373 310-1098z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-63" d="M3122 3366v-538q-244 135-489 202t-495 67q-560 0-870-355-309-354-309-995t309-996q310-354 870-354 250 0 495 67t489 202V134Q2881 22 2623-34q-257-57-548-57-791 0-1257 497-465 497-465 1341 0 856 470 1346 471 491 1290 491 265 0 518-55 253-54 491-163z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-74" d="M1172 4494v-994h1184v-447H1172V1153q0-428 117-550t477-122h590V0h-590q-666 0-919 248-253 249-253 905v1900H172v447h422v994h578z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-69" d="M603 3500h575V0H603v3500zm0 1363h575v-729H603v729z" transform="scale(.01563)"/>
                </defs>
                <use xlink:href="#DejaVuSans-4d"/>
                <use xlink:href="#DejaVuSans-53" x="86.279"/>
                <use xlink:href="#DejaVuSans-45" x="149.756"/>
                <use xlink:href="#DejaVuSans-20" x="212.939"/>
                <use xlink:href="#DejaVuSans-4c" x="244.727"/>
                <use xlink:href="#DejaVuSans-6f" x="298.689"/>
                <use xlink:href="#DejaVuSans-73" x="359.871"/>
                <use xlink:href="#DejaVuSans-73" x="411.971"/>
                <use xlink:href="#DejaVuSans-20" x="464.07"/>
                <use xlink:href="#DejaVuSans-46" x="495.857"/>
                <use xlink:href="#DejaVuSans-75" x="547.877"/>
                <use xlink:href="#DejaVuSans-6e" x="611.256"/>
                <use xlink:href="#DejaVuSans-63" x="674.635"/>
                <use xlink:href="#DejaVuSans-74" x="729.615"/>
                <use xlink:href="#DejaVuSans-69" x="768.824"/>
                <use xlink:href="#DejaVuSans-6f" x="796.607"/>
                <use xlink:href="#DejaVuSans-6e" x="857.789"/>
            </g>
        </g>
    </g>
    <defs>
        <clipPath id="pccbfe9029a">
            <path d="M57.6 41.472h357.12v266.112H57.6z"/>
        </clipPath>
    </defs>
</svg>

**通过改变模型的参数来减小误差 (在均方误差中只要判断所在的位置下降即可)**



##### 梯度下降

**如果上述问题变成两个变量图像就变成三维图像这时就需要梯度下降算法**

<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="614.4" height="460.8" viewBox="0 0 460.8 345.6">
    <defs>
        <style>
            *{stroke-linejoin:round;stroke-linecap:butt}
        </style>
    </defs>
    <g id="figure_1">
        <path id="patch_1" d="M0 345.6h460.8V0H0z" style="fill:#fff"/>
        <path id="patch_2" d="M103.104 307.584h266.112V41.472H103.104z" style="fill:#fff"/>
        <g id="pane3d_1">
            <path id="patch_3" d="m123.197 241.97 87.88-73.663-1.222-106.233-92.085 67.199" style="fill:#f2f2f2;opacity:.5;stroke:#f2f2f2;stroke-linejoin:miter"/>
        </g>
        <g id="pane3d_2">
            <path id="patch_4" d="m211.076 168.307 141.014 40.988 5.033-109.893-147.268-37.328" style="fill:#e6e6e6;opacity:.5;stroke:#e6e6e6;stroke-linejoin:miter"/>
        </g>
        <g id="pane3d_3">
            <path id="patch_5" d="m123.197 241.97 149.482 48.82 79.411-81.495-141.014-40.988" style="fill:#ececec;opacity:.5;stroke:#ececec;stroke-linejoin:miter"/>
        </g>
        <g id="axis3d_1">
            <path id="line2d_1" d="m123.197 241.97 149.482 48.82" style="fill:none;stroke:#000;stroke-width:.8;stroke-linecap:square"/>
            <g id="text_1" transform="matrix(.1 0 0 -.1 172.406 303.768)">
                <defs>
                    <path id="DejaVuSans-77" d="M269 3500h575l719-2731 715 2731h678l719-2731 716 2731h575L4050 0h-678l-753 2869L1863 0h-679L269 3500z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-31" d="M794 531h1031v3560L703 3866v575l1116 225h631V531h1031V0H794v531z" transform="scale(.01563)"/>
                </defs>
                <use xlink:href="#DejaVuSans-77"/>
                <use xlink:href="#DejaVuSans-31" x="81.787"/>
            </g>
            <g id="Line3DCollection_1">
                <path d="m132.25 244.926 87.402-74.126-.859-106.46" style="fill:none;stroke:#b0b0b0;stroke-width:.8"/>
                <path d="m163.747 255.213 85.705-75.751.419-107.245" style="fill:none;stroke:#b0b0b0;stroke-width:.8"/>
                <path d="m195.968 265.736 83.912-77.43 1.752-108.038" style="fill:none;stroke:#b0b0b0;stroke-width:.8"/>
                <path d="m228.939 276.504 82.017-79.165L314.1 88.497" style="fill:none;stroke:#b0b0b0;stroke-width:.8"/>
                <path d="m262.686 287.526 80.016-80.96L347.3 96.912" style="fill:none;stroke:#b0b0b0;stroke-width:.8"/>
            </g>
            <g id="xtick_1">
                <path id="line2d_2" d="m133.012 244.28-2.287 1.94" style="fill:none;stroke:#000;stroke-width:.8;stroke-linecap:square"/>
                <g id="text_2" transform="matrix(.1 0 0 -.1 121.54 268.235)">
                    <defs>
                        <path id="DejaVuSans-30" d="M2034 4250q-487 0-733-480-245-479-245-1442 0-959 245-1439 246-480 733-480 491 0 736 480 246 480 246 1439 0 963-246 1442-245 480-736 480zm0 500q785 0 1199-621 414-620 414-1801 0-1178-414-1799Q2819-91 2034-91q-784 0-1198 620-414 621-414 1799 0 1181 414 1801 414 621 1198 621z" transform="scale(.01563)"/>
                    </defs>
                    <use xlink:href="#DejaVuSans-30"/>
                </g>
            </g>
            <g id="xtick_2">
                <path id="line2d_3" d="m164.494 254.553-2.244 1.983" style="fill:none;stroke:#000;stroke-width:.8;stroke-linecap:square"/>
                <use xlink:href="#DejaVuSans-31" id="text_3" transform="matrix(.1 0 0 -.1 153.07 278.74)"/>
            </g>
            <g id="xtick_3">
                <path id="line2d_4" d="m196.7 265.06-2.2 2.03" style="fill:none;stroke:#000;stroke-width:.8;stroke-linecap:square"/>
                <g id="text_4" transform="matrix(.1 0 0 -.1 185.326 289.487)">
                    <defs>
                        <path id="DejaVuSans-32" d="M1228 531h2203V0H469v531q359 372 979 998 621 627 780 809 303 340 423 576 121 236 121 464 0 372-261 606-261 235-680 235-297 0-627-103-329-103-704-313v638q381 153 712 231 332 78 607 78 725 0 1156-363 431-362 431-968 0-288-108-546-107-257-392-607-78-91-497-524-418-433-1181-1211z" transform="scale(.01563)"/>
                    </defs>
                    <use xlink:href="#DejaVuSans-32"/>
                </g>
            </g>
            <g id="xtick_4">
                <path id="line2d_5" d="m229.655 275.813-2.152 2.077" style="fill:none;stroke:#000;stroke-width:.8;stroke-linecap:square"/>
                <g id="text_5" transform="matrix(.1 0 0 -.1 218.337 300.486)">
                    <defs>
                        <path id="DejaVuSans-33" d="M2597 2516q453-97 707-404 255-306 255-756 0-690-475-1069Q2609-91 1734-91q-293 0-604 58T488 141v609q262-153 574-231 313-78 654-78 593 0 904 234t311 681q0 413-289 645-289 233-804 233h-544v519h569q465 0 712 186t247 536q0 359-255 551-254 193-729 193-260 0-557-57-297-56-653-174v562q360 100 674 150t592 50q719 0 1137-327 419-326 419-882 0-388-222-655t-631-370z" transform="scale(.01563)"/>
                    </defs>
                    <use xlink:href="#DejaVuSans-33"/>
                </g>
            </g>
            <g id="xtick_5">
                <path id="line2d_6" d="m263.386 286.819-2.102 2.126" style="fill:none;stroke:#000;stroke-width:.8;stroke-linecap:square"/>
                <g id="text_6" transform="matrix(.1 0 0 -.1 252.128 311.744)">
                    <defs>
                        <path id="DejaVuSans-34" d="M2419 4116 825 1625h1594v2491zm-166 550h794V1625h666v-525h-666V0h-628v1100H313v609l1940 2957z" transform="scale(.01563)"/>
                    </defs>
                    <use xlink:href="#DejaVuSans-34"/>
                </g>
            </g>
        </g>
        <g id="axis3d_2">
            <path id="line2d_7" d="m352.09 209.295-79.41 81.495" style="fill:none;stroke:#000;stroke-width:.8;stroke-linecap:square"/>
            <g id="text_7" transform="matrix(.1 0 0 -.1 333.862 278.989)">
                <use xlink:href="#DejaVuSans-77"/>
                <use xlink:href="#DejaVuSans-32" x="81.787"/>
            </g>
            <g id="Line3DCollection_2">
                <path d="m124.138 124.626 5.115 112.267 148.92 48.258" style="fill:none;stroke:#b0b0b0;stroke-width:.8"/>
                <path d="m145.412 109.1 4.095 110.816 147.02 46.4" style="fill:none;stroke:#b0b0b0;stroke-width:.8"/>
                <path d="m165.86 94.179 3.148 109.39 145.156 44.648" style="fill:none;stroke:#b0b0b0;stroke-width:.8"/>
                <path d="m185.53 79.825 2.268 107.995 143.325 42.992" style="fill:none;stroke:#b0b0b0;stroke-width:.8"/>
                <path d="m204.465 66.007 1.449 106.628 141.531 41.427" style="fill:none;stroke:#b0b0b0;stroke-width:.8"/>
            </g>
            <g id="xtick_6">
                <path id="line2d_8" d="m276.919 284.745 3.768 1.22" style="fill:none;stroke:#000;stroke-width:.8;stroke-linecap:square"/>
                <use xlink:href="#DejaVuSans-30" id="text_8" transform="matrix(.1 0 0 -.1 287.873 306.173)"/>
            </g>
            <g id="xtick_7">
                <path id="line2d_9" d="m295.29 265.925 3.716 1.173" style="fill:none;stroke:#000;stroke-width:.8;stroke-linecap:square"/>
                <use xlink:href="#DejaVuSans-31" id="text_9" transform="matrix(.1 0 0 -.1 305.98 287.059)"/>
            </g>
            <g id="xtick_8">
                <path id="line2d_10" d="m312.943 247.842 3.665 1.127" style="fill:none;stroke:#000;stroke-width:.8;stroke-linecap:square"/>
                <use xlink:href="#DejaVuSans-32" id="text_10" transform="matrix(.1 0 0 -.1 323.38 268.692)"/>
            </g>
            <g id="xtick_9">
                <path id="line2d_11" d="m329.919 230.45 3.616 1.086" style="fill:none;stroke:#000;stroke-width:.8;stroke-linecap:square"/>
                <use xlink:href="#DejaVuSans-33" id="text_11" transform="matrix(.1 0 0 -.1 340.111 251.03)"/>
            </g>
            <g id="xtick_10">
                <path id="line2d_12" d="m346.257 213.714 3.567 1.045" style="fill:none;stroke:#000;stroke-width:.8;stroke-linecap:square"/>
                <use xlink:href="#DejaVuSans-34" id="text_12" transform="matrix(.1 0 0 -.1 356.213 234.034)"/>
            </g>
        </g>
        <g id="axis3d_3">
            <path id="line2d_13" d="m352.09 209.295 5.033-109.893" style="fill:none;stroke:#000;stroke-width:.8;stroke-linecap:square"/>
            <g id="text_13" transform="matrix(.1 0 0 -.1 384.28 152.902)">
                <defs>
                    <path id="DejaVuSans-4d" d="M628 4666h941l1190-3175 1197 3175h941V0h-616v4097L3078 897h-634L1241 4097V0H628v4666z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-53" d="M3425 4513v-616q-359 172-678 256-319 85-616 85-515 0-795-200t-280-569q0-310 186-468 186-157 705-254l381-78q706-135 1042-474t336-907q0-679-455-1029Q2797-91 1919-91q-331 0-705 75-373 75-773 222v650q384-215 753-325 369-109 725-109 540 0 834 212 294 213 294 607 0 343-211 537t-692 291l-385 75q-706 140-1022 440-315 300-315 835 0 619 436 975t1201 356q329 0 669-60 341-59 697-177z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-45" d="M628 4666h2950v-532H1259V2753h2222v-531H1259V531h2375V0H628v4666z" transform="scale(.01563)"/>
                </defs>
                <use xlink:href="#DejaVuSans-4d"/>
                <use xlink:href="#DejaVuSans-53" x="86.279"/>
                <use xlink:href="#DejaVuSans-45" x="149.756"/>
            </g>
            <g id="Line3DCollection_3">
                <path d="m353.125 186.711-142.3-40.271-88.741 72.4" style="fill:none;stroke:#b0b0b0;stroke-width:.8"/>
                <path d="m354.187 163.503-143.62-39.517-89.629 71.07" style="fill:none;stroke:#b0b0b0;stroke-width:.8"/>
                <path d="m355.27 139.855-144.966-38.727-90.534 69.674" style="fill:none;stroke:#b0b0b0;stroke-width:.8"/>
                <path d="m356.374 115.755-146.338-37.9-91.457 68.212" style="fill:none;stroke:#b0b0b0;stroke-width:.8"/>
            </g>
            <g id="xtick_11">
                <path id="line2d_14" d="m351.93 186.373 3.587 1.016" style="fill:none;stroke:#000;stroke-width:.8;stroke-linecap:square"/>
                <g id="text_14" transform="matrix(.1 0 0 -.1 365.175 191.75)">
                    <use xlink:href="#DejaVuSans-32"/>
                    <use xlink:href="#DejaVuSans-30" x="63.623"/>
                </g>
            </g>
            <g id="xtick_12">
                <path id="line2d_15" d="m352.98 163.17 3.623.998" style="fill:none;stroke:#000;stroke-width:.8;stroke-linecap:square"/>
                <g id="text_15" transform="matrix(.1 0 0 -.1 366.407 168.588)">
                    <use xlink:href="#DejaVuSans-34"/>
                    <use xlink:href="#DejaVuSans-30" x="63.623"/>
                </g>
            </g>
            <g id="xtick_13">
                <path id="line2d_16" d="m354.052 139.53 3.658.977" style="fill:none;stroke:#000;stroke-width:.8;stroke-linecap:square"/>
                <g id="text_16" transform="matrix(.1 0 0 -.1 367.662 144.989)">
                    <defs>
                        <path id="DejaVuSans-36" d="M2113 2584q-425 0-674-291-248-290-248-796 0-503 248-796 249-292 674-292t673 292q248 293 248 796 0 506-248 796-248 291-673 291zm1253 1979v-575q-238 112-480 171-242 60-480 60-625 0-955-422-329-422-376-1275 184 272 462 417 279 145 613 145 703 0 1111-427 408-426 408-1160 0-719-425-1154Q2819-91 2113-91q-810 0-1238 620-428 621-428 1799 0 1106 525 1764t1409 658q238 0 480-47t505-140z" transform="scale(.01563)"/>
                    </defs>
                    <use xlink:href="#DejaVuSans-36"/>
                    <use xlink:href="#DejaVuSans-30" x="63.623"/>
                </g>
            </g>
            <g id="xtick_14">
                <path id="line2d_17" d="m355.143 115.437 3.695.956" style="fill:none;stroke:#000;stroke-width:.8;stroke-linecap:square"/>
                <g id="text_17" transform="matrix(.1 0 0 -.1 368.94 120.94)">
                    <defs>
                        <path id="DejaVuSans-38" d="M2034 2216q-450 0-708-241-257-241-257-662 0-422 257-663 258-241 708-241t709 242q260 243 260 662 0 421-258 662-257 241-711 241zm-631 268q-406 100-633 378-226 279-226 679 0 559 398 884 399 325 1092 325 697 0 1094-325t397-884q0-400-227-679-226-278-629-378 456-106 710-416 255-309 255-755 0-679-414-1042Q2806-91 2034-91q-771 0-1186 362-414 363-414 1042 0 446 256 755 257 310 713 416zm-231 997q0-362 226-565 227-203 636-203 407 0 636 203 230 203 230 565 0 363-230 566-229 203-636 203-409 0-636-203-226-203-226-566z" transform="scale(.01563)"/>
                    </defs>
                    <use xlink:href="#DejaVuSans-38"/>
                    <use xlink:href="#DejaVuSans-30" x="63.623"/>
                </g>
            </g>
        </g>
        <g id="axes_1">
            <g id="Poly3DCollection_1">
                <path d="m214.935 119.861 2.499 2.914 1.448-4.358-2.499-2.913z" clip-path="url(#p94a45562aa)" style="fill:#edd1c2"/>
                <path d="m217.434 122.775 2.498 2.78 1.448-4.359-2.498-2.78z" clip-path="url(#p94a45562aa)" style="fill:#e9d5cb"/>
                <path d="m212.434 116.814 2.5 3.047 1.45-4.357-2.5-3.047z" clip-path="url(#p94a45562aa)" style="fill:#f1ccb8"/>
                <path d="m219.932 125.555 2.498 2.648 1.447-4.36-2.497-2.647z" clip-path="url(#p94a45562aa)" style="fill:#e4d9d2"/>
                <path d="m222.43 128.203 2.497 2.516 1.446-4.361-2.496-2.515z" clip-path="url(#p94a45562aa)" style="fill:#dfdbd9"/>
                <path d="m213.484 124.087 2.5 2.915 1.45-4.227-2.5-2.914z" clip-path="url(#p94a45562aa)" style="fill:#e7d7ce"/>
                <path d="m215.984 127.002 2.499 2.781 1.45-4.228-2.5-2.78z" clip-path="url(#p94a45562aa)" style="fill:#e1dad6"/>
                <path d="m210.982 121.04 2.502 3.047 1.45-4.226-2.5-3.047z" clip-path="url(#p94a45562aa)" style="fill:#ecd3c5"/>
                <path d="m218.483 129.783 2.498 2.65 1.449-4.23-2.498-2.648z" clip-path="url(#p94a45562aa)" style="fill:#dcdddd"/>
                <path d="m224.927 130.719 2.497 2.384 1.446-4.363-2.497-2.382z" clip-path="url(#p94a45562aa)" style="fill:#dbdcde"/>
                <path d="m220.98 132.432 2.499 2.517 1.448-4.23-2.497-2.516z" clip-path="url(#p94a45562aa)" style="fill:#d7dce3"/>
                <path d="m227.424 133.103 2.498 2.252 1.446-4.364-2.498-2.251z" clip-path="url(#p94a45562aa)" style="fill:#d6dce4"/>
                <path d="m212.03 128.184 2.502 2.915 1.452-4.097-2.5-2.915z" clip-path="url(#p94a45562aa)" style="fill:#dfdbd9"/>
                <path d="m214.532 131.099 2.5 2.782 1.45-4.098-2.498-2.781z" clip-path="url(#p94a45562aa)" style="fill:#dadce0"/>
                <path d="m209.528 125.135 2.503 3.049 1.453-4.097-2.502-3.047z" clip-path="url(#p94a45562aa)" style="fill:#e5d8d1"/>
                <path d="m223.479 134.95 2.497 2.385 1.448-4.232-2.497-2.384z" clip-path="url(#p94a45562aa)" style="fill:#d3dbe7"/>
                <path d="m217.031 133.881 2.5 2.65 1.45-4.099-2.498-2.649z" clip-path="url(#p94a45562aa)" style="fill:#d4dbe6"/>
                <path d="m219.53 136.532 2.499 2.519 1.45-4.102-2.498-2.517z" clip-path="url(#p94a45562aa)" style="fill:#cedaeb"/>
                <path d="m229.922 135.355 2.499 2.12 1.445-4.365-2.498-2.12z" clip-path="url(#p94a45562aa)" style="fill:#d2dbe8"/>
                <path d="m225.976 137.335 2.499 2.254 1.447-4.234-2.498-2.252z" clip-path="url(#p94a45562aa)" style="fill:#cdd9ec"/>
                <path d="m222.029 139.05 2.498 2.388 1.45-4.103-2.498-2.386z" clip-path="url(#p94a45562aa)" style="fill:#cad8ef"/>
                <path d="m210.575 132.15 2.502 2.916 1.455-3.967-2.501-2.915z" clip-path="url(#p94a45562aa)" style="fill:#d8dce2"/>
                <path d="m213.077 135.066 2.5 2.784 1.454-3.969-2.5-2.782z" clip-path="url(#p94a45562aa)" style="fill:#d2dbe8"/>
                <path d="m208.071 129.1 2.504 3.05 1.456-3.966-2.503-3.049z" clip-path="url(#p94a45562aa)" style="fill:#dedcdb"/>
                <path d="m215.578 137.85 2.5 2.652 1.452-3.97-2.499-2.65z" clip-path="url(#p94a45562aa)" style="fill:#ccd9ed"/>
                <path d="m228.475 139.589 2.499 2.123 1.447-4.236-2.499-2.121z" clip-path="url(#p94a45562aa)" style="fill:#c9d7f0"/>
                <path d="m218.077 140.502 2.5 2.52 1.452-3.971-2.499-2.52z" clip-path="url(#p94a45562aa)" style="fill:#c6d6f1"/>
                <path d="m232.42 137.476 2.5 1.99 1.446-4.368-2.5-1.988z" clip-path="url(#p94a45562aa)" style="fill:#cedaeb"/>
                <path d="m224.527 141.438 2.499 2.256 1.449-4.105-2.499-2.254z" clip-path="url(#p94a45562aa)" style="fill:#c5d6f2"/>
                <path d="m220.577 143.022 2.499 2.39 1.451-3.974-2.498-2.387z" clip-path="url(#p94a45562aa)" style="fill:#c0d4f5"/>
                <path d="m209.116 135.987 2.504 2.917 1.457-3.838-2.502-2.916z" clip-path="url(#p94a45562aa)" style="fill:#d1dae9"/>
                <path d="m211.62 138.904 2.501 2.786 1.457-3.84-2.501-2.784z" clip-path="url(#p94a45562aa)" style="fill:#cad8ef"/>
                <path d="m206.611 132.936 2.505 3.05 1.459-3.836-2.504-3.05z" clip-path="url(#p94a45562aa)" style="fill:#d7dce3"/>
                <path d="m230.974 141.712 2.5 1.992 1.447-4.238-2.5-1.99z" clip-path="url(#p94a45562aa)" style="fill:#c5d6f2"/>
                <path d="m227.026 143.694 2.5 2.124 1.448-4.106-2.5-2.123z" clip-path="url(#p94a45562aa)" style="fill:#c0d4f5"/>
                <path d="m214.121 141.69 2.501 2.653 1.455-3.84-2.5-2.653z" clip-path="url(#p94a45562aa)" style="fill:#c4d5f3"/>
                <path d="m234.92 139.466 2.502 1.859 1.446-4.37-2.502-1.857z" clip-path="url(#p94a45562aa)" style="fill:#cbd8ee"/>
                <path d="m223.076 145.411 2.5 2.258 1.45-3.975-2.499-2.256z" clip-path="url(#p94a45562aa)" style="fill:#bbd1f8"/>
                <path d="m216.622 144.343 2.5 2.522 1.455-3.843-2.5-2.52z" clip-path="url(#p94a45562aa)" style="fill:#bed2f6"/>
                <path d="m219.122 146.865 2.5 2.391 1.454-3.845-2.5-2.389z" clip-path="url(#p94a45562aa)" style="fill:#b7cff9"/>
                <path d="m229.525 145.818 2.5 1.995 1.449-4.11-2.5-1.991z" clip-path="url(#p94a45562aa)" style="fill:#bbd1f8"/>
                <path d="m233.474 143.704 2.502 1.861 1.446-4.24-2.501-1.86z" clip-path="url(#p94a45562aa)" style="fill:#c1d4f4"/>
                <path d="m225.575 147.669 2.5 2.127 1.45-3.978-2.5-2.124z" clip-path="url(#p94a45562aa)" style="fill:#b6cefa"/>
                <path d="m207.655 139.695 2.504 2.919 1.46-3.71-2.503-2.917z" clip-path="url(#p94a45562aa)" style="fill:#c9d7f0"/>
                <path d="m210.16 142.614 2.502 2.787 1.46-3.711-2.502-2.786z" clip-path="url(#p94a45562aa)" style="fill:#c3d5f4"/>
                <path d="m205.148 136.643 2.507 3.052 1.461-3.708-2.505-3.05z" clip-path="url(#p94a45562aa)" style="fill:#cfdaea"/>
                <path d="m212.662 145.401 2.502 2.655 1.458-3.713-2.5-2.653z" clip-path="url(#p94a45562aa)" style="fill:#bbd1f8"/>
                <path d="m237.422 141.325 2.504 1.728 1.445-4.373-2.503-1.726z" clip-path="url(#p94a45562aa)" style="fill:#c7d7f0"/>
                <path d="m221.622 149.256 2.5 2.26 1.453-3.847-2.5-2.258z" clip-path="url(#p94a45562aa)" style="fill:#b2ccfb"/>
                <path d="m215.164 148.056 2.502 2.524 1.456-3.715-2.5-2.522z" clip-path="url(#p94a45562aa)" style="fill:#b5cdfa"/>
                <path d="m232.026 147.813 2.502 1.863 1.448-4.11-2.502-1.862z" clip-path="url(#p94a45562aa)" style="fill:#b7cff9"/>
                <path d="m228.075 149.796 2.501 1.996 1.45-3.98-2.5-1.994z" clip-path="url(#p94a45562aa)" style="fill:#b2ccfb"/>
                <path d="m217.666 150.58 2.5 2.393 1.456-3.717-2.5-2.39z" clip-path="url(#p94a45562aa)" style="fill:#afcafc"/>
                <path d="m235.976 145.565 2.503 1.731 1.447-4.243-2.504-1.728z" clip-path="url(#p94a45562aa)" style="fill:#bed2f6"/>
                <path d="m224.122 151.516 2.501 2.129 1.452-3.849-2.5-2.127z" clip-path="url(#p94a45562aa)" style="fill:#adc9fd"/>
                <path d="m206.19 143.274 2.506 2.921 1.463-3.581-2.504-2.92z" clip-path="url(#p94a45562aa)" style="fill:#c1d4f4"/>
                <path d="m208.696 146.195 2.504 2.789 1.462-3.583-2.503-2.787z" clip-path="url(#p94a45562aa)" style="fill:#bad0f8"/>
                <path d="m203.682 140.22 2.508 3.054 1.465-3.58-2.507-3.051z" clip-path="url(#p94a45562aa)" style="fill:#c9d7f0"/>
                <path d="m239.926 143.053 2.506 1.598 1.446-4.376-2.507-1.595z" clip-path="url(#p94a45562aa)" style="fill:#c4d5f3"/>
                <path d="m220.167 152.973 2.5 2.262 1.455-3.72-2.5-2.259z" clip-path="url(#p94a45562aa)" style="fill:#aac7fd"/>
                <path d="m211.2 148.984 2.504 2.657 1.46-3.585-2.502-2.655z" clip-path="url(#p94a45562aa)" style="fill:#b3cdfb"/>
                <path d="m230.576 151.792 2.503 1.867 1.449-3.983-2.502-1.863z" clip-path="url(#p94a45562aa)" style="fill:#aec9fc"/>
                <path d="m234.528 149.676 2.504 1.734 1.447-4.114-2.503-1.73z" clip-path="url(#p94a45562aa)" style="fill:#b3cdfb"/>
                <path d="m226.623 153.645 2.502 1.999 1.451-3.852-2.5-1.996z" clip-path="url(#p94a45562aa)" style="fill:#a9c6fd"/>
                <path d="m213.704 151.64 2.502 2.527 1.46-3.587-2.502-2.524z" clip-path="url(#p94a45562aa)" style="fill:#adc9fd"/>
                <path d="m238.48 147.296 2.505 1.6 1.447-4.245-2.506-1.598z" clip-path="url(#p94a45562aa)" style="fill:#bad0f8"/>
                <path d="m222.668 155.235 2.501 2.131 1.454-3.721-2.5-2.13z" clip-path="url(#p94a45562aa)" style="fill:#a5c3fe"/>
                <path d="m216.206 154.167 2.502 2.394 1.459-3.588-2.501-2.393z" clip-path="url(#p94a45562aa)" style="fill:#a7c5fe"/>
                <path d="m233.079 153.659 2.504 1.735 1.449-3.984-2.504-1.734z" clip-path="url(#p94a45562aa)" style="fill:#aac7fd"/>
                <path d="m229.125 155.644 2.503 1.868 1.45-3.853-2.502-1.867z" clip-path="url(#p94a45562aa)" style="fill:#a5c3fe"/>
                <path d="m204.722 146.726 2.507 2.922 1.467-3.453-2.506-2.92z" clip-path="url(#p94a45562aa)" style="fill:#bad0f8"/>
                <path d="m242.432 144.65 2.509 1.467 1.446-4.378-2.51-1.464z" clip-path="url(#p94a45562aa)" style="fill:#c1d4f4"/>
                <path d="m218.708 156.561 2.502 2.264 1.458-3.59-2.501-2.262z" clip-path="url(#p94a45562aa)" style="fill:#a1c0ff"/>
                <path d="m207.23 149.648 2.505 2.79 1.465-3.454-2.504-2.789z" clip-path="url(#p94a45562aa)" style="fill:#b2ccfb"/>
                <path d="m202.212 143.67 2.51 3.056 1.468-3.452-2.508-3.053z" clip-path="url(#p94a45562aa)" style="fill:#c1d4f4"/>
                <path d="m237.032 151.41 2.506 1.602 1.447-4.116-2.506-1.6z" clip-path="url(#p94a45562aa)" style="fill:#b1cbfc"/>
                <path d="m225.17 157.366 2.502 2.001 1.453-3.723-2.502-1.999z" clip-path="url(#p94a45562aa)" style="fill:#9fbfff"/>
                <path d="m209.735 152.439 2.505 2.659 1.464-3.457-2.504-2.657z" clip-path="url(#p94a45562aa)" style="fill:#abc8fd"/>
                <path d="m212.24 155.098 2.503 2.528 1.463-3.46-2.502-2.525z" clip-path="url(#p94a45562aa)" style="fill:#a5c3fe"/>
                <path d="m240.985 148.896 2.51 1.47 1.446-4.249-2.51-1.466z" clip-path="url(#p94a45562aa)" style="fill:#b7cff9"/>
                <path d="m221.21 158.825 2.503 2.134 1.456-3.593-2.501-2.131z" clip-path="url(#p94a45562aa)" style="fill:#9bbcff"/>
                <path d="m214.743 157.626 2.504 2.397 1.461-3.462-2.502-2.394z" clip-path="url(#p94a45562aa)" style="fill:#9fbfff"/>
                <path d="m231.628 157.512 2.505 1.739 1.45-3.857-2.504-1.735z" clip-path="url(#p94a45562aa)" style="fill:#a1c0ff"/>
                <path d="M235.583 155.394 238.09 157l1.448-3.988-2.506-1.602z" clip-path="url(#p94a45562aa)" style="fill:#a6c4fe"/>
                <path d="m227.672 159.367 2.503 1.871 1.453-3.726-2.503-1.868z" clip-path="url(#p94a45562aa)" style="fill:#9bbcff"/>
                <path d="m244.94 146.117 2.513 1.337 1.446-4.382-2.512-1.333z" clip-path="url(#p94a45562aa)" style="fill:#bfd3f6"/>
                <path d="m217.247 160.023 2.503 2.266 1.46-3.464-2.502-2.264z" clip-path="url(#p94a45562aa)" style="fill:#98b9ff"/>
                <path d="m239.538 153.012 2.509 1.473 1.447-4.12-2.509-1.469z" clip-path="url(#p94a45562aa)" style="fill:#aec9fc"/>
                <path d="m223.713 160.96 2.503 2.003 1.456-3.596-2.503-2.001z" clip-path="url(#p94a45562aa)" style="fill:#97b8ff"/>
                <path d="m203.25 150.05 2.509 2.924 1.47-3.326-2.507-2.922z" clip-path="url(#p94a45562aa)" style="fill:#b2ccfb"/>
                <path d="m205.759 152.974 2.507 2.792 1.469-3.327-2.506-2.79z" clip-path="url(#p94a45562aa)" style="fill:#abc8fd"/>
                <path d="m200.738 146.992 2.511 3.057 1.473-3.323-2.51-3.055z" clip-path="url(#p94a45562aa)" style="fill:#bad0f8"/>
                <path d="m208.266 155.766 2.506 2.661 1.468-3.33-2.505-2.658z" clip-path="url(#p94a45562aa)" style="fill:#a3c2fe"/>
                <path d="m210.772 158.427 2.506 2.53 1.465-3.331-2.503-2.528z" clip-path="url(#p94a45562aa)" style="fill:#9dbdff"/>
                <path d="m243.494 150.366 2.512 1.34 1.447-4.252-2.512-1.337z" clip-path="url(#p94a45562aa)" style="fill:#b5cdfa"/>
                <path d="m219.75 162.289 2.504 2.136 1.459-3.466-2.503-2.134z" clip-path="url(#p94a45562aa)" style="fill:#93b5fe"/>
                <path d="m234.133 159.25 2.507 1.609 1.45-3.86-2.507-1.605z" clip-path="url(#p94a45562aa)" style="fill:#9dbdff"/>
                <path d="m230.175 161.238 2.506 1.741 1.452-3.728-2.505-1.739z" clip-path="url(#p94a45562aa)" style="fill:#97b8ff"/>
                <path d="m238.09 157 2.509 1.475 1.448-3.99-2.509-1.473z" clip-path="url(#p94a45562aa)" style="fill:#a3c2fe"/>
                <path d="m226.216 162.963 2.505 1.873 1.454-3.598-2.503-1.87z" clip-path="url(#p94a45562aa)" style="fill:#93b5fe"/>
                <path d="m213.278 160.957 2.504 2.4 1.465-3.334-2.504-2.397z" clip-path="url(#p94a45562aa)" style="fill:#97b8ff"/>
                <path d="m242.047 154.485 2.512 1.342 1.447-4.122-2.512-1.34z" clip-path="url(#p94a45562aa)" style="fill:#abc8fd"/>
                <path d="m222.254 164.425 2.504 2.006 1.458-3.468-2.503-2.004z" clip-path="url(#p94a45562aa)" style="fill:#8fb1fe"/>
                <path d="m247.453 147.454 2.516 1.205 1.446-4.384-2.516-1.203z" clip-path="url(#p94a45562aa)" style="fill:#bed2f6"/>
                <path d="m215.782 163.357 2.505 2.268 1.463-3.336-2.503-2.266z" clip-path="url(#p94a45562aa)" style="fill:#92b4fe"/>
                <path d="m201.773 153.245 2.511 2.927 1.475-3.198-2.51-2.925z" clip-path="url(#p94a45562aa)" style="fill:#abc8fd"/>
                <path d="m204.284 156.172 2.51 2.794 1.472-3.2-2.507-2.792z" clip-path="url(#p94a45562aa)" style="fill:#a3c2fe"/>
                <path d="m232.68 162.98 2.508 1.61 1.452-3.731-2.507-1.608z" clip-path="url(#p94a45562aa)" style="fill:#94b6ff"/>
                <path d="m199.26 150.186 2.513 3.059 1.476-3.196-2.51-3.057z" clip-path="url(#p94a45562aa)" style="fill:#b3cdfb"/>
                <path d="m206.794 158.966 2.508 2.664 1.47-3.203-2.506-2.66z" clip-path="url(#p94a45562aa)" style="fill:#9dbdff"/>
                <path d="m236.64 160.859 2.51 1.478 1.449-3.862-2.51-1.475z" clip-path="url(#p94a45562aa)" style="fill:#9abbff"/>
                <path d="m228.72 164.836 2.507 1.744 1.454-3.6-2.506-1.742z" clip-path="url(#p94a45562aa)" style="fill:#8fb1fe"/>
                <path d="m246.006 151.705 2.516 1.21 1.447-4.256-2.516-1.205z" clip-path="url(#p94a45562aa)" style="fill:#b3cdfb"/>
                <path d="m218.287 165.625 2.505 2.139 1.462-3.339-2.504-2.136z" clip-path="url(#p94a45562aa)" style="fill:#8caffe"/>
                <path d="m209.302 161.63 2.507 2.532 1.469-3.205-2.506-2.53z" clip-path="url(#p94a45562aa)" style="fill:#96b7ff"/>
                <path d="m240.599 158.475 2.512 1.346 1.448-3.994-2.512-1.342z" clip-path="url(#p94a45562aa)" style="fill:#a1c0ff"/>
                <path d="m224.758 166.431 2.506 1.876 1.457-3.47-2.505-1.874z" clip-path="url(#p94a45562aa)" style="fill:#8badfd"/>
                <path d="m211.809 164.162 2.506 2.402 1.467-3.207-2.504-2.4z" clip-path="url(#p94a45562aa)" style="fill:#90b2fe"/>
                <path d="m244.559 155.827 2.516 1.212 1.447-4.125-2.516-1.209z" clip-path="url(#p94a45562aa)" style="fill:#a9c6fd"/>
                <path d="m220.792 167.764 2.506 2.008 1.46-3.34-2.504-2.007z" clip-path="url(#p94a45562aa)" style="fill:#86a9fc"/>
                <path d="m235.188 164.59 2.51 1.481 1.451-3.734-2.51-1.478z" clip-path="url(#p94a45562aa)" style="fill:#92b4fe"/>
                <path d="m231.227 166.58 2.508 1.614 1.453-3.604-2.507-1.61z" clip-path="url(#p94a45562aa)" style="fill:#8badfd"/>
                <path d="m249.969 148.66 2.52 1.074 1.447-4.388-2.52-1.071z" clip-path="url(#p94a45562aa)" style="fill:#bcd2f7"/>
                <path d="m214.315 166.564 2.506 2.27 1.466-3.209-2.505-2.268z" clip-path="url(#p94a45562aa)" style="fill:#89acfd"/>
                <path d="m239.15 162.337 2.512 1.349 1.449-3.865-2.512-1.346z" clip-path="url(#p94a45562aa)" style="fill:#97b8ff"/>
                <path d="m227.264 168.307 2.507 1.747 1.456-3.474-2.506-1.744z" clip-path="url(#p94a45562aa)" style="fill:#86a9fc"/>
                <path d="m200.293 156.313 2.513 2.93 1.478-3.071-2.51-2.927z" clip-path="url(#p94a45562aa)" style="fill:#a5c3fe"/>
                <path d="m202.806 159.242 2.511 2.797 1.477-3.073-2.51-2.794z" clip-path="url(#p94a45562aa)" style="fill:#9dbdff"/>
                <path d="m197.778 153.252 2.515 3.061 1.48-3.068-2.513-3.059z" clip-path="url(#p94a45562aa)" style="fill:#adc9fd"/>
                <path d="m205.317 162.04 2.51 2.665 1.475-3.075-2.508-2.664z" clip-path="url(#p94a45562aa)" style="fill:#96b7ff"/>
                <path d="m248.522 152.914 2.52 1.078 1.447-4.258-2.52-1.075z" clip-path="url(#p94a45562aa)" style="fill:#b2ccfb"/>
                <path d="m216.82 168.835 2.507 2.14 1.465-3.211-2.505-2.139z" clip-path="url(#p94a45562aa)" style="fill:#85a8fc"/>
                <path d="m243.11 159.82 2.517 1.216 1.448-3.997-2.516-1.212z" clip-path="url(#p94a45562aa)" style="fill:#9fbfff"/>
                <path d="m223.298 169.772 2.506 1.88 1.46-3.345-2.506-1.876z" clip-path="url(#p94a45562aa)" style="fill:#82a6fb"/>
                <path d="m207.827 164.705 2.509 2.535 1.473-3.078-2.507-2.532z" clip-path="url(#p94a45562aa)" style="fill:#8fb1fe"/>
                <path d="m233.735 168.194 2.51 1.484 1.453-3.607-2.51-1.48z" clip-path="url(#p94a45562aa)" style="fill:#88abfd"/>
                <path d="m210.336 167.24 2.508 2.404 1.47-3.08-2.505-2.402z" clip-path="url(#p94a45562aa)" style="fill:#89acfd"/>
                <path d="m237.698 166.071 2.513 1.352 1.45-3.737-2.512-1.349z" clip-path="url(#p94a45562aa)" style="fill:#8fb1fe"/>
                <path d="m229.77 170.054 2.51 1.616 1.455-3.476-2.508-1.614z" clip-path="url(#p94a45562aa)" style="fill:#82a6fb"/>
                <path d="m247.075 157.04 2.52 1.081 1.447-4.129-2.52-1.078z" clip-path="url(#p94a45562aa)" style="fill:#a7c5fe"/>
                <path d="m219.327 170.976 2.507 2.011 1.464-3.215-2.506-2.008z" clip-path="url(#p94a45562aa)" style="fill:#80a3fa"/>
                <path d="m241.662 163.686 2.515 1.218 1.45-3.868-2.516-1.215z" clip-path="url(#p94a45562aa)" style="fill:#96b7ff"/>
                <path d="m225.804 171.651 2.508 1.75 1.459-3.347-2.507-1.747z" clip-path="url(#p94a45562aa)" style="fill:#7ea1fa"/>
                <path d="m252.489 149.734 2.524.944 1.448-4.392-2.525-.94z" clip-path="url(#p94a45562aa)" style="fill:#bbd1f8"/>
                <path d="m212.844 169.644 2.507 2.274 1.47-3.083-2.506-2.271z" clip-path="url(#p94a45562aa)" style="fill:#82a6fb"/>
                <path d="m198.808 159.255 2.515 2.931 1.483-2.944-2.513-2.929z" clip-path="url(#p94a45562aa)" style="fill:#9ebeff"/>
                <path d="m201.323 162.186 2.514 2.8 1.48-2.947-2.51-2.797z" clip-path="url(#p94a45562aa)" style="fill:#97b8ff"/>
                <path d="m245.627 161.036 2.519 1.085 1.448-4-2.52-1.082z" clip-path="url(#p94a45562aa)" style="fill:#9ebeff"/>
                <path d="m196.29 156.191 2.518 3.064 1.485-2.942-2.515-3.06z" clip-path="url(#p94a45562aa)" style="fill:#a6c4fe"/>
                <path d="m221.834 172.987 2.508 1.882 1.462-3.218-2.506-1.879z" clip-path="url(#p94a45562aa)" style="fill:#7b9ff9"/>
                <path d="m251.042 153.992 2.524.948 1.447-4.262-2.524-.944z" clip-path="url(#p94a45562aa)" style="fill:#b1cbfc"/>
                <path d="m215.351 171.918 2.508 2.143 1.468-3.085-2.506-2.141z" clip-path="url(#p94a45562aa)" style="fill:#7ea1fa"/>
                <path d="m203.837 164.986 2.511 2.668 1.479-2.949-2.51-2.666z" clip-path="url(#p94a45562aa)" style="fill:#90b2fe"/>
                <path d="m236.246 169.678 2.513 1.354 1.452-3.61-2.513-1.35z" clip-path="url(#p94a45562aa)" style="fill:#85a8fc"/>
                <path d="m232.28 171.67 2.511 1.487 1.455-3.48-2.511-1.483z" clip-path="url(#p94a45562aa)" style="fill:#80a3fa"/>
                <path d="m206.348 167.654 2.51 2.537 1.478-2.951-2.509-2.535z" clip-path="url(#p94a45562aa)" style="fill:#89acfd"/>
                <path d="m240.211 167.423 2.516 1.221 1.45-3.74-2.515-1.218z" clip-path="url(#p94a45562aa)" style="fill:#8caffe"/>
                <path d="m228.312 173.4 2.51 1.62 1.458-3.35-2.51-1.616z" clip-path="url(#p94a45562aa)" style="fill:#7b9ff9"/>
                <path d="m208.859 170.191 2.51 2.407 1.475-2.954-2.508-2.404z" clip-path="url(#p94a45562aa)" style="fill:#82a6fb"/>
                <path d="m249.594 158.121 2.524.951 1.448-4.132-2.524-.948z" clip-path="url(#p94a45562aa)" style="fill:#a6c4fe"/>
                <path d="m217.86 174.061 2.507 2.014 1.467-3.088-2.507-2.011z" clip-path="url(#p94a45562aa)" style="fill:#799cf8"/>
                <path d="m244.177 164.904 2.52 1.088 1.449-3.871-2.52-1.085z" clip-path="url(#p94a45562aa)" style="fill:#94b6ff"/>
                <path d="m224.342 174.869 2.51 1.751 1.46-3.22-2.508-1.749z" clip-path="url(#p94a45562aa)" style="fill:#779af7"/>
                <path d="m255.013 150.678 2.53.813 1.447-4.395-2.53-.81z" clip-path="url(#p94a45562aa)" style="fill:#bbd1f8"/>
                <path d="m211.369 172.598 2.509 2.276 1.473-2.956-2.507-2.274zM234.791 173.157l2.514 1.358 1.454-3.483-2.513-1.354z" clip-path="url(#p94a45562aa)" style="fill:#7da0f9"/>
                <path d="m248.146 162.121 2.524.955 1.448-4.004-2.524-.95z" clip-path="url(#p94a45562aa)" style="fill:#9dbdff"/>
                <path d="m220.367 176.075 2.51 1.884 1.465-3.09-2.508-1.882z" clip-path="url(#p94a45562aa)" style="fill:#7597f6"/>
                <path d="m238.759 171.032 2.517 1.225 1.451-3.613-2.516-1.221z" clip-path="url(#p94a45562aa)" style="fill:#84a7fc"/>
                <path d="m230.822 175.02 2.512 1.49 1.457-3.353-2.511-1.487z" clip-path="url(#p94a45562aa)" style="fill:#799cf8"/>
                <path d="m197.319 162.069 2.517 2.934 1.487-2.817-2.515-2.931z" clip-path="url(#p94a45562aa)" style="fill:#98b9ff"/>
                <path d="m199.836 165.003 2.516 2.802 1.485-2.82-2.514-2.799z" clip-path="url(#p94a45562aa)" style="fill:#92b4fe"/>
                <path d="m253.566 154.94 2.528.817 1.448-4.266-2.529-.813z" clip-path="url(#p94a45562aa)" style="fill:#b1cbfc"/>
                <path d="m213.878 174.874 2.51 2.147 1.471-2.96-2.508-2.143z" clip-path="url(#p94a45562aa)" style="fill:#779af7"/>
                <path d="m194.799 159.003 2.52 3.066 1.49-2.814-2.518-3.064z" clip-path="url(#p94a45562aa)" style="fill:#a1c0ff"/>
                <path d="m202.352 167.805 2.513 2.67 1.483-2.821-2.511-2.668z" clip-path="url(#p94a45562aa)" style="fill:#89acfd"/>
                <path d="m242.727 168.644 2.52 1.092 1.45-3.744-2.52-1.088z" clip-path="url(#p94a45562aa)" style="fill:#8badfd"/>
                <path d="m226.851 176.62 2.511 1.623 1.46-3.223-2.51-1.62z" clip-path="url(#p94a45562aa)" style="fill:#7597f6"/>
                <path d="m204.865 170.476 2.513 2.54 1.48-2.825-2.51-2.537z" clip-path="url(#p94a45562aa)" style="fill:#82a6fb"/>
                <path d="m252.118 159.072 2.528.82 1.448-4.135-2.528-.817z" clip-path="url(#p94a45562aa)" style="fill:#a6c4fe"/>
                <path d="m216.387 177.02 2.51 2.017 1.47-2.962-2.508-2.014z" clip-path="url(#p94a45562aa)" style="fill:#7295f4"/>
                <path d="m246.697 165.992 2.524.959 1.449-3.875-2.524-.955z" clip-path="url(#p94a45562aa)" style="fill:#93b5fe"/>
                <path d="m222.877 177.96 2.51 1.754 1.464-3.094-2.51-1.751z" clip-path="url(#p94a45562aa)" style="fill:#7093f3"/>
                <path d="m207.378 173.016 2.512 2.41 1.479-2.828-2.51-2.407z" clip-path="url(#p94a45562aa)" style="fill:#7da0f9"/>
                <path d="m237.305 174.515 2.517 1.227 1.454-3.485-2.517-1.225z" clip-path="url(#p94a45562aa)" style="fill:#7b9ff9"/>
                <path d="m233.334 176.51 2.515 1.36 1.456-3.355-2.514-1.358z" clip-path="url(#p94a45562aa)" style="fill:#7699f6"/>
                <path d="m257.542 151.491 2.535.682 1.449-4.4-2.536-.677z" clip-path="url(#p94a45562aa)" style="fill:#bbd1f8"/>
                <path d="m209.89 175.425 2.51 2.28 1.478-2.83-2.51-2.277z" clip-path="url(#p94a45562aa)" style="fill:#779af7"/>
                <path d="m241.276 172.257 2.52 1.095 1.451-3.616-2.52-1.092z" clip-path="url(#p94a45562aa)" style="fill:#82a6fb"/>
                <path d="m229.362 178.243 2.513 1.493 1.46-3.226-2.513-1.49z" clip-path="url(#p94a45562aa)" style="fill:#7093f3"/>
                <path d="m250.67 163.076 2.528.824 1.448-4.007-2.528-.82z" clip-path="url(#p94a45562aa)" style="fill:#9dbdff"/>
                <path d="m218.897 179.037 2.511 1.887 1.469-2.965-2.51-1.884z" clip-path="url(#p94a45562aa)" style="fill:#6e90f2"/>
                <path d="m245.247 169.736 2.524.962 1.45-3.747-2.524-.959z" clip-path="url(#p94a45562aa)" style="fill:#89acfd"/>
                <path d="m225.387 179.714 2.512 1.625 1.463-3.096-2.51-1.623z" clip-path="url(#p94a45562aa)" style="fill:#6e90f2"/>
                <path d="m256.094 155.757 2.534.686 1.449-4.27-2.535-.682z" clip-path="url(#p94a45562aa)" style="fill:#b1cbfc"/>
                <path d="m212.4 177.704 2.512 2.15 1.475-2.833-2.509-2.147z" clip-path="url(#p94a45562aa)" style="fill:#7295f4"/>
                <path d="m195.824 164.756 2.52 2.937 1.492-2.69-2.517-2.934z" clip-path="url(#p94a45562aa)" style="fill:#93b5fe"/>
                <path d="m198.344 167.693 2.518 2.805 1.49-2.693-2.516-2.802z" clip-path="url(#p94a45562aa)" style="fill:#8caffe"/>
                <path d="m193.302 161.688 2.522 3.068 1.495-2.687-2.52-3.066z" clip-path="url(#p94a45562aa)" style="fill:#9bbcff"/>
                <path d="m200.862 170.498 2.516 2.673 1.487-2.695-2.513-2.671z" clip-path="url(#p94a45562aa)" style="fill:#84a7fc"/>
                <path d="m203.378 173.171 2.514 2.543 1.486-2.698-2.513-2.54z" clip-path="url(#p94a45562aa)" style="fill:#7da0f9"/>
                <path d="m249.22 166.95 2.53.829 1.448-3.88-2.528-.823z" clip-path="url(#p94a45562aa)" style="fill:#93b5fe"/>
                <path d="m221.408 180.924 2.512 1.758 1.467-2.968-2.51-1.755z" clip-path="url(#p94a45562aa)" style="fill:#6a8bef"/>
                <path d="m235.85 177.87 2.517 1.231 1.455-3.359-2.517-1.227z" clip-path="url(#p94a45562aa)" style="fill:#7396f5"/>
                <path d="m254.646 159.893 2.534.69 1.448-4.14-2.534-.686z" clip-path="url(#p94a45562aa)" style="fill:#a6c4fe"/>
                <path d="m214.912 179.853 2.512 2.02 1.473-2.836-2.51-2.016z" clip-path="url(#p94a45562aa)" style="fill:#6c8ff1"/>
                <path d="m239.822 175.742 2.521 1.099 1.453-3.49-2.52-1.094z" clip-path="url(#p94a45562aa)" style="fill:#7a9df8"/>
                <path d="m231.875 179.736 2.516 1.363 1.458-3.229-2.515-1.36z" clip-path="url(#p94a45562aa)" style="fill:#6f92f3"/>
                <path d="m205.892 175.714 2.514 2.412 1.484-2.7-2.512-2.41z" clip-path="url(#p94a45562aa)" style="fill:#779af7"/>
                <path d="m243.796 173.352 2.524.965 1.451-3.62-2.524-.961z" clip-path="url(#p94a45562aa)" style="fill:#81a4fb"/>
                <path d="m227.9 181.34 2.514 1.495 1.461-3.1-2.513-1.492z" clip-path="url(#p94a45562aa)" style="fill:#6a8bef"/>
                <path d="m260.077 152.173 2.54.55 1.45-4.404-2.541-.546z" clip-path="url(#p94a45562aa)" style="fill:#bbd1f8"/>
                <path d="m208.406 178.126 2.513 2.282 1.482-2.704-2.511-2.279z" clip-path="url(#p94a45562aa)" style="fill:#7295f4"/>
                <path d="m253.198 163.9 2.533.693 1.45-4.01-2.535-.69z" clip-path="url(#p94a45562aa)" style="fill:#9bbcff"/>
                <path d="m217.424 181.873 2.512 1.89 1.472-2.839-2.51-1.887z" clip-path="url(#p94a45562aa)" style="fill:#688aef"/>
                <path d="m247.771 170.698 2.528.831 1.45-3.75-2.528-.828z" clip-path="url(#p94a45562aa)" style="fill:#89acfd"/>
                <path d="m223.92 182.682 2.513 1.628 1.466-2.97-2.512-1.626z" clip-path="url(#p94a45562aa)" style="fill:#6788ee"/>
                <path d="m258.628 156.443 2.54.554 1.45-4.274-2.541-.55z" clip-path="url(#p94a45562aa)" style="fill:#b1cbfc"/>
                <path d="m210.92 180.408 2.513 2.152 1.479-2.707-2.511-2.149z" clip-path="url(#p94a45562aa)" style="fill:#6c8ff1"/>
                <path d="m194.325 167.317 2.522 2.94 1.497-2.564-2.52-2.937z" clip-path="url(#p94a45562aa)" style="fill:#8fb1fe"/>
                <path d="m238.367 179.101 2.521 1.101 1.455-3.361-2.52-1.099z" clip-path="url(#p94a45562aa)" style="fill:#7295f4"/>
                <path d="m234.391 181.099 2.519 1.234 1.457-3.232-2.518-1.23z" clip-path="url(#p94a45562aa)" style="fill:#6c8ff1"/>
                <path d="m196.847 170.256 2.52 2.808 1.495-2.566-2.518-2.805z" clip-path="url(#p94a45562aa)" style="fill:#86a9fc"/>
                <path d="m191.8 164.245 2.525 3.072 1.5-2.56-2.523-3.07z" clip-path="url(#p94a45562aa)" style="fill:#97b8ff"/>
                <path d="m199.367 173.064 2.518 2.677 1.493-2.57-2.516-2.673z" clip-path="url(#p94a45562aa)" style="fill:#80a3fa"/>
                <path d="m242.343 176.84 2.525.969 1.452-3.492-2.524-.965z" clip-path="url(#p94a45562aa)" style="fill:#799cf8"/>
                <path d="m230.414 182.835 2.516 1.367 1.461-3.103-2.516-1.363z" clip-path="url(#p94a45562aa)" style="fill:#688aef"/>
                <path d="m251.75 167.779 2.532.697 1.45-3.883-2.534-.693z" clip-path="url(#p94a45562aa)" style="fill:#92b4fe"/>
                <path d="m219.936 183.763 2.513 1.76 1.47-2.841-2.511-1.758z" clip-path="url(#p94a45562aa)" style="fill:#6485ec"/>
                <path d="m201.885 175.74 2.517 2.546 1.49-2.572-2.514-2.543z" clip-path="url(#p94a45562aa)" style="fill:#799cf8"/>
                <path d="m257.18 160.583 2.54.558 1.448-4.144-2.54-.554z" clip-path="url(#p94a45562aa)" style="fill:#a6c4fe"/>
                <path d="m213.433 182.56 2.513 2.023 1.478-2.71-2.512-2.02z" clip-path="url(#p94a45562aa)" style="fill:#6788ee"/>
                <path d="m246.32 174.317 2.529.835 1.45-3.623-2.528-.831z" clip-path="url(#p94a45562aa)" style="fill:#81a4fb"/>
                <path d="m226.433 184.31 2.516 1.498 1.465-2.973-2.515-1.496z" clip-path="url(#p94a45562aa)" style="fill:#6485ec"/>
                <path d="m204.402 178.286 2.516 2.415 1.488-2.575-2.514-2.412z" clip-path="url(#p94a45562aa)" style="fill:#7295f4"/>
                <path d="m262.617 152.723 2.547.419 1.45-4.409-2.547-.414z" clip-path="url(#p94a45562aa)" style="fill:#bbd1f8"/>
                <path d="m206.918 180.701 2.516 2.285 1.485-2.578-2.513-2.282z" clip-path="url(#p94a45562aa)" style="fill:#6c8ff1"/>
                <path d="m255.731 164.593 2.54.563 1.448-4.015-2.539-.558z" clip-path="url(#p94a45562aa)" style="fill:#9dbdff"/>
                <path d="m215.946 184.583 2.514 1.892 1.476-2.712-2.512-1.89z" clip-path="url(#p94a45562aa)" style="fill:#6384eb"/>
                <path d="m250.3 171.529 2.533.701 1.45-3.754-2.534-.697z" clip-path="url(#p94a45562aa)" style="fill:#89acfd"/>
                <path d="m222.45 185.523 2.514 1.631 1.47-2.844-2.514-1.628z" clip-path="url(#p94a45562aa)" style="fill:#6282ea"/>
                <path d="m236.91 182.333 2.522 1.104 1.456-3.235-2.52-1.101z" clip-path="url(#p94a45562aa)" style="fill:#6b8df0"/>
                <path d="m240.888 180.202 2.526.972 1.454-3.365-2.525-.968z" clip-path="url(#p94a45562aa)" style="fill:#7093f3"/>
                <path d="m232.93 184.202 2.52 1.237 1.46-3.106-2.519-1.234z" clip-path="url(#p94a45562aa)" style="fill:#6687ed"/>
                <path d="m261.168 156.997 2.546.423 1.45-4.278-2.547-.419z" clip-path="url(#p94a45562aa)" style="fill:#b1cbfc"/>
                <path d="m209.434 182.986 2.515 2.155 1.484-2.58-2.514-2.153z" clip-path="url(#p94a45562aa)" style="fill:#6788ee"/>
                <path d="m244.868 177.809 2.529.838 1.452-3.495-2.529-.835z" clip-path="url(#p94a45562aa)" style="fill:#799cf8"/>
                <path d="m228.949 185.808 2.518 1.37 1.463-2.976-2.516-1.367z" clip-path="url(#p94a45562aa)" style="fill:#6282ea"/>
                <path d="m192.82 169.75 2.525 2.943 1.502-2.437-2.522-2.94z" clip-path="url(#p94a45562aa)" style="fill:#89acfd"/>
                <path d="m195.345 172.693 2.522 2.81 1.5-2.439-2.52-2.808z" clip-path="url(#p94a45562aa)" style="fill:#82a6fb"/>
                <path d="m190.292 166.676 2.528 3.074 1.505-2.433-2.525-3.072z" clip-path="url(#p94a45562aa)" style="fill:#92b4fe"/>
                <path d="m254.282 168.476 2.54.566 1.448-3.886-2.539-.563z" clip-path="url(#p94a45562aa)" style="fill:#93b5fe"/>
                <path d="m218.46 186.475 2.515 1.764 1.474-2.716-2.513-1.76z" clip-path="url(#p94a45562aa)" style="fill:#5f7fe8"/>
                <path d="m197.867 175.504 2.521 2.68 1.497-2.443-2.518-2.677z" clip-path="url(#p94a45562aa)" style="fill:#7b9ff9"/>
                <path d="m248.849 175.152 2.533.704 1.45-3.626-2.533-.701z" clip-path="url(#p94a45562aa)" style="fill:#81a4fb"/>
                <path d="m224.964 187.154 2.517 1.502 1.468-2.848-2.516-1.498z" clip-path="url(#p94a45562aa)" style="fill:#5f7fe8"/>
                <path d="m259.72 161.141 2.545.427 1.449-4.148-2.546-.423z" clip-path="url(#p94a45562aa)" style="fill:#a7c5fe"/>
                <path d="m211.949 185.141 2.515 2.025 1.482-2.583-2.513-2.023z" clip-path="url(#p94a45562aa)" style="fill:#6384eb"/>
                <path d="m200.388 178.183 2.52 2.549 1.494-2.446-2.517-2.545z" clip-path="url(#p94a45562aa)" style="fill:#7597f6"/>
                <path d="m202.907 180.732 2.519 2.418 1.492-2.449-2.516-2.415z" clip-path="url(#p94a45562aa)" style="fill:#6e90f2"/>
                <path d="m239.432 183.437 2.526.975 1.456-3.238-2.526-.972z" clip-path="url(#p94a45562aa)" style="fill:#6a8bef"/>
                <path d="m235.45 185.439 2.523 1.107 1.459-3.109-2.522-1.104z" clip-path="url(#p94a45562aa)" style="fill:#6485ec"/>
                <path d="m252.833 172.23 2.538.57 1.45-3.758-2.539-.566z" clip-path="url(#p94a45562aa)" style="fill:#89acfd"/>
                <path d="m220.975 188.239 2.517 1.634 1.472-2.719-2.515-1.63z" clip-path="url(#p94a45562aa)" style="fill:#5d7ce6"/>
                <path d="m243.414 181.174 2.53.842 1.453-3.369-2.53-.838z" clip-path="url(#p94a45562aa)" style="fill:#7093f3"/>
                <path d="m231.467 187.178 2.52 1.24 1.463-2.98-2.52-1.236z" clip-path="url(#p94a45562aa)" style="fill:#6180e9"/>
                <path d="m258.27 165.156 2.545.432 1.45-4.02-2.546-.427z" clip-path="url(#p94a45562aa)" style="fill:#9dbdff"/>
                <path d="m214.464 187.166 2.516 1.896 1.48-2.587-2.514-1.892z" clip-path="url(#p94a45562aa)" style="fill:#5f7fe8"/>
                <path d="m265.164 153.142 2.553.287 1.45-4.414-2.553-.282z" clip-path="url(#p94a45562aa)" style="fill:#bcd2f7"/>
                <path d="m205.426 183.15 2.517 2.288 1.49-2.452-2.515-2.285z" clip-path="url(#p94a45562aa)" style="fill:#688aef"/>
                <path d="m247.397 178.647 2.534.708 1.451-3.499-2.533-.704z" clip-path="url(#p94a45562aa)" style="fill:#799cf8"/>
                <path d="m227.481 188.656 2.52 1.372 1.466-2.85-2.518-1.37z" clip-path="url(#p94a45562aa)" style="fill:#5d7ce6"/>
                <path d="m263.714 157.42 2.552.291 1.45-4.282-2.552-.287z" clip-path="url(#p94a45562aa)" style="fill:#b2ccfb"/>
                <path d="m207.943 185.438 2.518 2.158 1.488-2.455-2.515-2.155z" clip-path="url(#p94a45562aa)" style="fill:#6384eb"/>
                <path d="m256.821 169.042 2.545.436 1.45-3.89-2.546-.432z" clip-path="url(#p94a45562aa)" style="fill:#93b5fe"/>
                <path d="m216.98 189.062 2.517 1.766 1.478-2.59-2.515-1.763z" clip-path="url(#p94a45562aa)" style="fill:#5b7ae5"/>
                <path d="m191.309 172.057 2.528 2.946 1.508-2.31-2.525-2.943z" clip-path="url(#p94a45562aa)" style="fill:#86a9fc"/>
                <path d="m251.382 175.856 2.539.575 1.45-3.63-2.538-.571z" clip-path="url(#p94a45562aa)" style="fill:#81a4fb"/>
                <path d="m193.837 175.003 2.525 2.814 1.505-2.313-2.522-2.811z" clip-path="url(#p94a45562aa)" style="fill:#7ea1fa"/>
                <path d="m223.492 189.873 2.518 1.504 1.471-2.721-2.517-1.502z" clip-path="url(#p94a45562aa)" style="fill:#5a78e4"/>
                <path d="m188.778 168.98 2.53 3.077 1.512-2.307-2.528-3.074z" clip-path="url(#p94a45562aa)" style="fill:#8db0fe"/>
                <path d="m196.362 177.817 2.523 2.683 1.503-2.317-2.52-2.68z" clip-path="url(#p94a45562aa)" style="fill:#779af7"/>
                <path d="m237.973 186.546 2.527.978 1.458-3.112-2.526-.975z" clip-path="url(#p94a45562aa)" style="fill:#6384eb"/>
                <path d="m262.265 161.568 2.551.296 1.45-4.153-2.552-.29z" clip-path="url(#p94a45562aa)" style="fill:#a9c6fd"/>
                <path d="m210.46 187.596 2.518 2.028 1.486-2.458-2.515-2.025z" clip-path="url(#p94a45562aa)" style="fill:#5f7fe8"/>
                <path d="m241.958 184.412 2.53.845 1.455-3.241-2.53-.842z" clip-path="url(#p94a45562aa)" style="fill:#6a8bef"/>
                <path d="m233.988 188.418 2.524 1.11 1.461-2.982-2.523-1.107z" clip-path="url(#p94a45562aa)" style="fill:#5e7de7"/>
                <path d="m198.885 180.5 2.522 2.552 1.5-2.32-2.519-2.549zM245.943 182.016l2.534.711 1.454-3.372-2.534-.708z" clip-path="url(#p94a45562aa)" style="fill:#7093f3"/>
                <path d="m230 190.028 2.523 1.243 1.465-2.853-2.521-1.24z" clip-path="url(#p94a45562aa)" style="fill:#5a78e4"/>
                <path d="m201.407 183.052 2.521 2.42 1.498-2.322-2.519-2.418z" clip-path="url(#p94a45562aa)" style="fill:#6a8bef"/>
                <path d="m255.371 172.8 2.545.44 1.45-3.762-2.545-.436z" clip-path="url(#p94a45562aa)" style="fill:#8badfd"/>
                <path d="m219.497 190.828 2.519 1.637 1.476-2.592-2.517-1.634z" clip-path="url(#p94a45562aa)" style="fill:#5875e1"/>
                <path d="m260.815 165.588 2.551.3 1.45-4.024-2.551-.296z" clip-path="url(#p94a45562aa)" style="fill:#9ebeff"/>
                <path d="m212.978 189.624 2.518 1.899 1.484-2.461-2.516-1.896z" clip-path="url(#p94a45562aa)" style="fill:#5a78e4"/>
                <path d="m267.717 153.429 2.56.154 1.451-4.418-2.56-.15z" clip-path="url(#p94a45562aa)" style="fill:#bfd3f6"/>
                <path d="m203.928 185.473 2.52 2.29 1.495-2.325-2.517-2.288z" clip-path="url(#p94a45562aa)" style="fill:#6485ec"/>
                <path d="m249.93 179.355 2.54.578 1.451-3.502-2.539-.575z" clip-path="url(#p94a45562aa)" style="fill:#799cf8"/>
                <path d="m226.01 191.377 2.521 1.376 1.47-2.725-2.52-1.372z" clip-path="url(#p94a45562aa)" style="fill:#5875e1"/>
                <path d="m266.266 157.711 2.56.16 1.45-4.288-2.56-.154z" clip-path="url(#p94a45562aa)" style="fill:#b5cdfa"/>
                <path d="m206.448 187.764 2.52 2.16 1.493-2.328-2.518-2.158z" clip-path="url(#p94a45562aa)" style="fill:#5f7fe8"/>
                <path d="m240.5 187.524 2.53.849 1.458-3.116-2.53-.845z" clip-path="url(#p94a45562aa)" style="fill:#6384eb"/>
                <path d="m236.512 189.529 2.528.98 1.46-2.985-2.527-.978z" clip-path="url(#p94a45562aa)" style="fill:#5e7de7"/>
                <path d="m259.366 169.478 2.55.304 1.45-3.895-2.55-.3z" clip-path="url(#p94a45562aa)" style="fill:#94b6ff"/>
                <path d="m215.496 191.523 2.519 1.77 1.482-2.465-2.517-1.766z" clip-path="url(#p94a45562aa)" style="fill:#5875e1"/>
                <path d="m253.921 176.43 2.544.444 1.45-3.634-2.544-.44z" clip-path="url(#p94a45562aa)" style="fill:#81a4fb"/>
                <path d="m222.016 192.465 2.52 1.508 1.474-2.596-2.518-1.504z" clip-path="url(#p94a45562aa)" style="fill:#5572df"/>
                <path d="m244.488 185.257 2.535.715 1.454-3.245-2.534-.711z" clip-path="url(#p94a45562aa)" style="fill:#6a8bef"/>
                <path d="m232.523 191.271 2.525 1.114 1.464-2.856-2.524-1.111z" clip-path="url(#p94a45562aa)" style="fill:#5977e3"/>
                <path d="m189.792 174.237 2.53 2.95 1.515-2.184-2.528-2.946z" clip-path="url(#p94a45562aa)" style="fill:#82a6fb"/>
                <path d="m192.323 177.186 2.528 2.818 1.51-2.187-2.524-2.814z" clip-path="url(#p94a45562aa)" style="fill:#7a9df8"/>
                <path d="m187.258 171.156 2.534 3.081 1.517-2.18-2.53-3.078z" clip-path="url(#p94a45562aa)" style="fill:#8badfd"/>
                <path d="m264.816 161.864 2.559.164 1.45-4.158-2.559-.159z" clip-path="url(#p94a45562aa)" style="fill:#aac7fd"/>
                <path d="m194.85 180.004 2.527 2.686 1.508-2.19-2.523-2.683z" clip-path="url(#p94a45562aa)" style="fill:#7396f5"/>
                <path d="m208.968 189.925 2.52 2.031 1.49-2.332-2.517-2.028z" clip-path="url(#p94a45562aa)" style="fill:#5b7ae5"/>
                <path d="m248.477 182.727 2.54.582 1.453-3.376-2.54-.578z" clip-path="url(#p94a45562aa)" style="fill:#7093f3"/>
                <path d="m228.531 192.753 2.524 1.246 1.468-2.728-2.522-1.243z" clip-path="url(#p94a45562aa)" style="fill:#5572df"/>
                <path d="m197.377 182.69 2.525 2.555 1.505-2.193-2.522-2.552z" clip-path="url(#p94a45562aa)" style="fill:#6c8ff1"/>
                <path d="m257.916 173.24 2.55.308 1.45-3.766-2.55-.304z" clip-path="url(#p94a45562aa)" style="fill:#8caffe"/>
                <path d="m218.015 193.292 2.52 1.64 1.48-2.467-2.518-1.637z" clip-path="url(#p94a45562aa)" style="fill:#5470de"/>
                <path d="m199.902 185.245 2.523 2.424 1.503-2.196-2.52-2.421z" clip-path="url(#p94a45562aa)" style="fill:#6788ee"/>
                <path d="m252.47 179.933 2.544.447 1.451-3.506-2.544-.443z" clip-path="url(#p94a45562aa)" style="fill:#799cf8"/>
                <path d="m224.536 193.973 2.522 1.378 1.473-2.598-2.52-1.376z" clip-path="url(#p94a45562aa)" style="fill:#536edd"/>
                <path d="m263.366 165.887 2.558.169 1.45-4.028-2.558-.164z" clip-path="url(#p94a45562aa)" style="fill:#a1c0ff"/>
                <path d="m211.487 191.956 2.52 1.902 1.49-2.335-2.519-1.899z" clip-path="url(#p94a45562aa)" style="fill:#5673e0"/>
                <path d="m270.276 153.583 2.568.02 1.453-4.422-2.569-.016z" clip-path="url(#p94a45562aa)" style="fill:#c0d4f5"/>
                <path d="m202.425 187.67 2.523 2.293 1.5-2.2-2.52-2.29z" clip-path="url(#p94a45562aa)" style="fill:#6282ea"/>
                <path d="m239.04 190.51 2.532.851 1.459-2.988-2.531-.849z" clip-path="url(#p94a45562aa)" style="fill:#5d7ce6"/>
                <path d="m243.03 188.373 2.536.718 1.457-3.119-2.535-.715z" clip-path="url(#p94a45562aa)" style="fill:#6384eb"/>
                <path d="m235.048 192.385 2.529.984 1.463-2.86-2.528-.98z" clip-path="url(#p94a45562aa)" style="fill:#5875e1"/>
                <path d="m247.023 185.972 2.54.585 1.454-3.248-2.54-.582z" clip-path="url(#p94a45562aa)" style="fill:#6a8bef"/>
                <path d="m231.055 193.999 2.526 1.117 1.467-2.73-2.525-1.115z" clip-path="url(#p94a45562aa)" style="fill:#5470de"/>
                <path d="m256.465 176.874 2.551.312 1.45-3.638-2.55-.308z" clip-path="url(#p94a45562aa)" style="fill:#82a6fb"/>
                <path d="m268.825 157.87 2.567.026 1.452-4.292-2.568-.021z" clip-path="url(#p94a45562aa)" style="fill:#b6cefa"/>
                <path d="m220.535 194.932 2.522 1.51 1.479-2.47-2.52-1.507z" clip-path="url(#p94a45562aa)" style="fill:#516ddb"/>
                <path d="m204.948 189.963 2.522 2.165 1.498-2.203-2.52-2.161z" clip-path="url(#p94a45562aa)" style="fill:#5d7ce6"/>
                <path d="m261.917 169.782 2.557.172 1.45-3.898-2.558-.169z" clip-path="url(#p94a45562aa)" style="fill:#97b8ff"/>
                <path d="m214.008 193.858 2.52 1.772 1.487-2.338-2.519-1.77z" clip-path="url(#p94a45562aa)" style="fill:#5470de"/>
                <path d="m251.017 183.309 2.545.45 1.452-3.379-2.544-.447z" clip-path="url(#p94a45562aa)" style="fill:#7295f4"/>
                <path d="m227.058 195.351 2.525 1.25 1.472-2.602-2.524-1.246z" clip-path="url(#p94a45562aa)" style="fill:#516ddb"/>
                <path d="m188.269 176.29 2.534 2.953 1.52-2.057-2.531-2.949z" clip-path="url(#p94a45562aa)" style="fill:#80a3fa"/>
                <path d="m190.803 179.243 2.53 2.82 1.518-2.06-2.528-2.817z" clip-path="url(#p94a45562aa)" style="fill:#779af7"/>
                <path d="m267.375 162.028 2.565.03 1.452-4.162-2.567-.026z" clip-path="url(#p94a45562aa)" style="fill:#abc8fd"/>
                <path d="m185.732 173.205 2.537 3.085 1.523-2.053-2.534-3.081z" clip-path="url(#p94a45562aa)" style="fill:#88abfd"/>
                <path d="m207.47 192.128 2.522 2.034 1.495-2.206-2.52-2.031z" clip-path="url(#p94a45562aa)" style="fill:#5875e1"/>
                <path d="m193.334 182.063 2.529 2.69 1.514-2.063-2.526-2.686z" clip-path="url(#p94a45562aa)" style="fill:#7093f3"/>
                <path d="m195.863 184.753 2.527 2.558 1.512-2.066-2.525-2.555z" clip-path="url(#p94a45562aa)" style="fill:#6a8bef"/>
                <path d="m260.467 173.548 2.557.177 1.45-3.77-2.557-.173z" clip-path="url(#p94a45562aa)" style="fill:#8db0fe"/>
                <path d="m216.529 195.63 2.522 1.643 1.484-2.34-2.52-1.64z" clip-path="url(#p94a45562aa)" style="fill:#506bda"/>
                <path d="m255.014 180.38 2.55.316 1.452-3.51-2.55-.312z" clip-path="url(#p94a45562aa)" style="fill:#7a9df8"/>
                <path d="m223.057 196.443 2.524 1.381 1.477-2.473-2.522-1.378z" clip-path="url(#p94a45562aa)" style="fill:#4f69d9"/>
                <path d="m241.572 191.361 2.536.722 1.458-2.992-2.535-.718z" clip-path="url(#p94a45562aa)" style="fill:#5d7ce6"/>
                <path d="m237.577 193.37 2.533.854 1.462-2.863-2.532-.851z" clip-path="url(#p94a45562aa)" style="fill:#5875e1"/>
                <path d="m198.39 187.311 2.527 2.428 1.508-2.07-2.523-2.424z" clip-path="url(#p94a45562aa)" style="fill:#6485ec"/>
                <path d="m245.566 189.09 2.54.59 1.457-3.123-2.54-.585z" clip-path="url(#p94a45562aa)" style="fill:#6384eb"/>
                <path d="m233.581 195.116 2.53.987 1.466-2.734-2.529-.984z" clip-path="url(#p94a45562aa)" style="fill:#536edd"/>
                <path d="m265.924 166.056 2.565.035 1.451-4.032-2.565-.031z" clip-path="url(#p94a45562aa)" style="fill:#a2c1ff"/>
                <path d="m209.992 194.162 2.522 1.905 1.494-2.21-2.52-1.9z" clip-path="url(#p94a45562aa)" style="fill:#5470de"/>
                <path d="m272.844 153.604 2.575-.112 1.454-4.429-2.576.118z" clip-path="url(#p94a45562aa)" style="fill:#c3d5f4"/>
                <path d="m200.917 189.74 2.525 2.297 1.506-2.074-2.523-2.294z" clip-path="url(#p94a45562aa)" style="fill:#5e7de7"/>
                <path d="m249.563 186.557 2.545.455 1.454-3.253-2.545-.45z" clip-path="url(#p94a45562aa)" style="fill:#6a8bef"/>
                <path d="m229.583 196.6 2.528 1.12 1.47-2.604-2.526-1.117z" clip-path="url(#p94a45562aa)" style="fill:#506bda"/>
                <path d="m259.016 177.186 2.557.18 1.45-3.641-2.556-.177z" clip-path="url(#p94a45562aa)" style="fill:#85a8fc"/>
                <path d="m219.05 197.273 2.525 1.514 1.482-2.344-2.522-1.51z" clip-path="url(#p94a45562aa)" style="fill:#4e68d8"/>
                <path d="m264.474 169.954 2.565.04 1.45-3.903-2.565-.035z" clip-path="url(#p94a45562aa)" style="fill:#98b9ff"/>
                <path d="m212.514 196.067 2.524 1.775 1.49-2.212-2.52-1.772z" clip-path="url(#p94a45562aa)" style="fill:#506bda"/>
                <path d="m271.392 157.896 2.574-.106 1.453-4.298-2.575.112z" clip-path="url(#p94a45562aa)" style="fill:#b9d0f9"/>
                <path d="m203.442 192.037 2.525 2.167 1.503-2.076-2.522-2.165z" clip-path="url(#p94a45562aa)" style="fill:#5977e3"/>
                <path d="m253.562 183.76 2.55.32 1.453-3.384-2.55-.316z" clip-path="url(#p94a45562aa)" style="fill:#7396f5"/>
                <path d="m225.581 197.824 2.527 1.252 1.475-2.475-2.525-1.25z" clip-path="url(#p94a45562aa)" style="fill:#4e68d8"/>
                <path d="m240.11 194.224 2.537.725 1.46-2.866-2.535-.722z" clip-path="url(#p94a45562aa)" style="fill:#5875e1"/>
                <path d="m269.94 162.059 2.574-.102 1.452-4.167-2.574.106z" clip-path="url(#p94a45562aa)" style="fill:#aec9fc"/>
                <path d="m205.967 194.204 2.524 2.038 1.5-2.08-2.521-2.034z" clip-path="url(#p94a45562aa)" style="fill:#5572df"/>
                <path d="m244.108 192.083 2.541.592 1.458-2.996-2.54-.588z" clip-path="url(#p94a45562aa)" style="fill:#5d7ce6"/>
                <path d="m236.111 196.103 2.534.858 1.465-2.737-2.533-.855z" clip-path="url(#p94a45562aa)" style="fill:#536edd"/>
                <path d="m186.74 178.216 2.536 2.956 1.527-1.93-2.534-2.952z" clip-path="url(#p94a45562aa)" style="fill:#7da0f9"/>
                <path d="m189.276 181.172 2.535 2.825 1.523-1.934-2.531-2.82z" clip-path="url(#p94a45562aa)" style="fill:#7597f6"/>
                <path d="m184.2 175.127 2.54 3.089 1.529-1.926-2.537-3.085z" clip-path="url(#p94a45562aa)" style="fill:#85a8fc"/>
                <path d="m191.81 183.997 2.533 2.693 1.52-1.937-2.53-2.69z" clip-path="url(#p94a45562aa)" style="fill:#6e90f2"/>
                <path d="m263.024 173.725 2.564.044 1.45-3.774-2.564-.04z" clip-path="url(#p94a45562aa)" style="fill:#8fb1fe"/>
                <path d="m215.038 197.842 2.524 1.646 1.489-2.215-2.522-1.643z" clip-path="url(#p94a45562aa)" style="fill:#4e68d8"/>
                <path d="m257.565 180.696 2.557.185 1.451-3.514-2.557-.18z" clip-path="url(#p94a45562aa)" style="fill:#7da0f9"/>
                <path d="m221.575 198.787 2.526 1.384 1.48-2.347-2.524-1.381z" clip-path="url(#p94a45562aa)" style="fill:#4b64d5"/>
                <path d="m248.107 189.68 2.546.457 1.455-3.125-2.545-.455z" clip-path="url(#p94a45562aa)" style="fill:#6485ec"/>
                <path d="m232.111 197.72 2.532.99 1.468-2.607-2.53-.987z" clip-path="url(#p94a45562aa)" style="fill:#4f69d9"/>
                <path d="m194.343 186.69 2.53 2.562 1.517-1.94-2.527-2.559z" clip-path="url(#p94a45562aa)" style="fill:#6788ee"/>
                <path d="m268.49 166.091 2.572-.097 1.452-4.037-2.574.102z" clip-path="url(#p94a45562aa)" style="fill:#a5c3fe"/>
                <path d="m208.491 196.242 2.525 1.908 1.498-2.083-2.522-1.905z" clip-path="url(#p94a45562aa)" style="fill:#516ddb"/>
                <path d="m196.873 189.252 2.53 2.431 1.514-1.944-2.527-2.428z" clip-path="url(#p94a45562aa)" style="fill:#6282ea"/>
                <path d="m252.108 187.012 2.551.323 1.454-3.256-2.551-.32z" clip-path="url(#p94a45562aa)" style="fill:#6b8df0"/>
                <path d="m228.108 199.076 2.53 1.123 1.473-2.479-2.528-1.12z" clip-path="url(#p94a45562aa)" style="fill:#4b64d5"/>
                <path d="m275.419 153.492 2.583-.246 1.456-4.434-2.585.251z" clip-path="url(#p94a45562aa)" style="fill:#c5d6f2"/>
                <path d="m199.402 191.683 2.528 2.301 1.512-1.947-2.525-2.298z" clip-path="url(#p94a45562aa)" style="fill:#5b7ae5"/>
                <path d="m261.573 177.367 2.564.049 1.451-3.647-2.564-.044z" clip-path="url(#p94a45562aa)" style="fill:#86a9fc"/>
                <path d="m217.562 199.488 2.526 1.517 1.487-2.218-2.524-1.514z" clip-path="url(#p94a45562aa)" style="fill:#4a63d3"/>
                <path d="m256.113 184.08 2.557.188 1.452-3.387-2.557-.185z" clip-path="url(#p94a45562aa)" style="fill:#7597f6"/>
                <path d="m224.1 200.171 2.53 1.255 1.478-2.35-2.527-1.252z" clip-path="url(#p94a45562aa)" style="fill:#4a63d3"/>
                <path d="m267.039 169.995 2.572-.093 1.451-3.908-2.573.097z" clip-path="url(#p94a45562aa)" style="fill:#9bbcff"/>
                <path d="m211.016 198.15 2.526 1.779 1.496-2.087-2.524-1.775z" clip-path="url(#p94a45562aa)" style="fill:#4e68d8"/>
                <path d="m242.647 194.949 2.542.595 1.46-2.87-2.541-.59z" clip-path="url(#p94a45562aa)" style="fill:#5875e1"/>
                <path d="m238.645 196.96 2.539.729 1.463-2.74-2.537-.725z" clip-path="url(#p94a45562aa)" style="fill:#536edd"/>
                <path d="m273.966 157.79 2.582-.24 1.454-4.304-2.583.246z" clip-path="url(#p94a45562aa)" style="fill:#bbd1f8"/>
                <path d="m201.93 193.984 2.528 2.17 1.509-1.95-2.525-2.167z" clip-path="url(#p94a45562aa)" style="fill:#5875e1"/>
                <path d="m246.65 192.675 2.546.46 1.457-2.998-2.546-.458z" clip-path="url(#p94a45562aa)" style="fill:#5e7de7"/>
                <path d="m234.643 198.71 2.535.862 1.467-2.611-2.534-.858z" clip-path="url(#p94a45562aa)" style="fill:#4e68d8"/>
                <path d="m250.653 190.137 2.551.327 1.455-3.129-2.551-.323z" clip-path="url(#p94a45562aa)" style="fill:#6485ec"/>
                <path d="m230.638 200.2 2.533.993 1.472-2.482-2.532-.99z" clip-path="url(#p94a45562aa)" style="fill:#4b64d5"/>
                <path d="m272.514 161.957 2.581-.235 1.453-4.173-2.582.24z" clip-path="url(#p94a45562aa)" style="fill:#b2ccfb"/>
                <path d="m204.458 196.155 2.527 2.04 1.506-1.953-2.524-2.038z" clip-path="url(#p94a45562aa)" style="fill:#536edd"/>
                <path d="m260.122 180.881 2.564.053 1.451-3.518-2.564-.05z" clip-path="url(#p94a45562aa)" style="fill:#7ea1fa"/>
                <path d="m220.088 201.005 2.528 1.388 1.485-2.222-2.526-1.384z" clip-path="url(#p94a45562aa)" style="fill:#4961d2"/>
                <path d="m185.203 180.014 2.54 2.96 1.533-1.802-2.537-2.956z" clip-path="url(#p94a45562aa)" style="fill:#7a9df8"/>
                <path d="m265.588 173.77 2.572-.088 1.45-3.78-2.571.093z" clip-path="url(#p94a45562aa)" style="fill:#92b4fe"/>
                <path d="m187.743 182.974 2.538 2.829 1.53-1.806-2.535-2.825z" clip-path="url(#p94a45562aa)" style="fill:#7396f5"/>
                <path d="m213.542 199.929 2.526 1.649 1.494-2.09-2.524-1.646z" clip-path="url(#p94a45562aa)" style="fill:#4a63d3"/>
                <path d="m182.66 176.921 2.543 3.093 1.536-1.798-2.54-3.09z" clip-path="url(#p94a45562aa)" style="fill:#82a6fb"/>
                <path d="m190.28 185.803 2.536 2.697 1.527-1.81-2.532-2.693z" clip-path="url(#p94a45562aa)" style="fill:#6c8ff1"/>
                <path d="m254.66 187.335 2.557.193 1.453-3.26-2.557-.189z" clip-path="url(#p94a45562aa)" style="fill:#6e90f2"/>
                <path d="m226.63 201.426 2.53 1.126 1.478-2.353-2.53-1.123z" clip-path="url(#p94a45562aa)" style="fill:#4961d2"/>
                <path d="m192.816 188.5 2.534 2.565 1.523-1.813-2.53-2.562z" clip-path="url(#p94a45562aa)" style="fill:#6687ed"/>
                <path d="m271.062 165.994 2.58-.23 1.453-4.042-2.581.235z" clip-path="url(#p94a45562aa)" style="fill:#a7c5fe"/>
                <path d="m206.985 198.196 2.527 1.911 1.504-1.957-2.525-1.908z" clip-path="url(#p94a45562aa)" style="fill:#4f69d9"/>
                <path d="m195.35 191.065 2.532 2.435 1.52-1.817-2.529-2.431z" clip-path="url(#p94a45562aa)" style="fill:#5f7fe8"/>
                <path d="m241.184 197.689 2.543.598 1.462-2.743-2.542-.595z" clip-path="url(#p94a45562aa)" style="fill:#536edd"/>
                <path d="m245.19 195.544 2.546.464 1.46-2.872-2.547-.461z" clip-path="url(#p94a45562aa)" style="fill:#5977e3"/>
                <path d="m237.178 199.572 2.54.73 1.466-2.613-2.539-.728z" clip-path="url(#p94a45562aa)" style="fill:#4e68d8"/>
                <path d="m264.137 177.416 2.571-.084 1.452-3.65-2.572.087z" clip-path="url(#p94a45562aa)" style="fill:#89acfd"/>
                <path d="m216.068 201.578 2.528 1.52 1.492-2.093-2.526-1.517z" clip-path="url(#p94a45562aa)" style="fill:#485fd1"/>
                <path d="m278.002 153.246 2.593-.38 1.456-4.44-2.593.386z" clip-path="url(#p94a45562aa)" style="fill:#c9d7f0"/>
                <path d="m197.882 193.5 2.53 2.305 1.518-1.821-2.528-2.301z" clip-path="url(#p94a45562aa)" style="fill:#5a78e4"/>
                <path d="m258.67 184.268 2.564.057 1.452-3.39-2.564-.054z" clip-path="url(#p94a45562aa)" style="fill:#779af7"/>
                <path d="m222.616 202.393 2.53 1.258 1.483-2.225-2.528-1.255z" clip-path="url(#p94a45562aa)" style="fill:#465ecf"/>
                <path d="m249.196 193.136 2.552.33 1.456-3.002-2.551-.327z" clip-path="url(#p94a45562aa)" style="fill:#5f7fe8"/>
                <path d="m233.17 201.193 2.537.863 1.47-2.484-2.534-.861z" clip-path="url(#p94a45562aa)" style="fill:#4a63d3"/>
                <path d="m269.61 169.902 2.58-.225 1.453-3.913-2.581.23z" clip-path="url(#p94a45562aa)" style="fill:#9ebeff"/>
                <path d="m209.512 200.107 2.529 1.782 1.5-1.96-2.525-1.779z" clip-path="url(#p94a45562aa)" style="fill:#4b64d5"/>
                <path d="m276.548 157.55 2.591-.376 1.456-4.308-2.593.38z" clip-path="url(#p94a45562aa)" style="fill:#bfd3f6"/>
                <path d="m200.413 195.805 2.53 2.174 1.515-1.824-2.528-2.171z" clip-path="url(#p94a45562aa)" style="fill:#5572df"/>
                <path d="m253.204 190.464 2.558.196 1.455-3.132-2.558-.193z" clip-path="url(#p94a45562aa)" style="fill:#6788ee"/>
                <path d="m229.16 202.552 2.535.997 1.476-2.356-2.533-.994z" clip-path="url(#p94a45562aa)" style="fill:#485fd1"/>
                <path d="m262.686 180.934 2.571-.079 1.451-3.523-2.57.084z" clip-path="url(#p94a45562aa)" style="fill:#81a4fb"/>
                <path d="m218.596 203.097 2.53 1.39 1.49-2.094-2.528-1.388z" clip-path="url(#p94a45562aa)" style="fill:#455cce"/>
                <path d="m275.095 161.722 2.59-.37 1.454-4.178-2.59.375z" clip-path="url(#p94a45562aa)" style="fill:#b5cdfa"/>
                <path d="m202.943 197.979 2.53 2.044 1.512-1.827-2.527-2.041z" clip-path="url(#p94a45562aa)" style="fill:#516ddb"/>
                <path d="m268.16 173.682 2.58-.221 1.45-3.784-2.58.225z" clip-path="url(#p94a45562aa)" style="fill:#96b7ff"/>
                <path d="m212.04 201.889 2.53 1.652 1.498-1.963-2.526-1.65z" clip-path="url(#p94a45562aa)" style="fill:#4961d2"/>
                <path d="m257.217 187.528 2.564.06 1.453-3.263-2.564-.057z" clip-path="url(#p94a45562aa)" style="fill:#6f92f3"/>
                <path d="m225.146 203.65 2.533 1.13 1.482-2.228-2.532-1.126z" clip-path="url(#p94a45562aa)" style="fill:#455cce"/>
                <path d="m183.66 181.685 2.543 2.964 1.54-1.675-2.54-2.96z" clip-path="url(#p94a45562aa)" style="fill:#799cf8"/>
                <path d="m186.203 184.65 2.541 2.832 1.537-1.68-2.538-2.828z" clip-path="url(#p94a45562aa)" style="fill:#7295f4"/>
                <path d="m181.112 178.588 2.547 3.097 1.544-1.67-2.544-3.094z" clip-path="url(#p94a45562aa)" style="fill:#81a4fb"/>
                <path d="m188.744 187.482 2.54 2.7 1.532-1.682-2.535-2.697z" clip-path="url(#p94a45562aa)" style="fill:#6a8bef"/>
                <path d="m243.727 198.287 2.548.468 1.461-2.747-2.547-.464z" clip-path="url(#p94a45562aa)" style="fill:#5470de"/>
                <path d="m239.718 200.303 2.544.6 1.465-2.616-2.543-.598z" clip-path="url(#p94a45562aa)" style="fill:#4f69d9"/>
                <path d="m191.283 190.182 2.537 2.57 1.53-1.687-2.534-2.565z" clip-path="url(#p94a45562aa)" style="fill:#6485ec"/>
                <path d="m247.736 196.008 2.553.334 1.459-2.875-2.552-.331z" clip-path="url(#p94a45562aa)" style="fill:#5a78e4"/>
                <path d="m235.707 202.056 2.541.735 1.47-2.488-2.54-.731z" clip-path="url(#p94a45562aa)" style="fill:#4a63d3"/>
                <path d="m273.643 165.764 2.589-.364 1.453-4.048-2.59.37z" clip-path="url(#p94a45562aa)" style="fill:#abc8fd"/>
                <path d="m205.473 200.023 2.53 1.915 1.51-1.831-2.528-1.911z" clip-path="url(#p94a45562aa)" style="fill:#4c66d6"/>
                <path d="m193.82 192.752 2.535 2.438 1.527-1.69-2.532-2.435z" clip-path="url(#p94a45562aa)" style="fill:#5e7de7"/>
                <path d="m251.748 193.467 2.558.199 1.456-3.006-2.558-.196z" clip-path="url(#p94a45562aa)" style="fill:#6180e9"/>
                <path d="m231.695 203.549 2.538.866 1.474-2.359-2.536-.863z" clip-path="url(#p94a45562aa)" style="fill:#465ecf"/>
                <path d="m266.708 177.332 2.58-.216 1.451-3.655-2.58.22z" clip-path="url(#p94a45562aa)" style="fill:#8caffe"/>
                <path d="m214.57 203.541 2.53 1.523 1.496-1.967-2.528-1.52z" clip-path="url(#p94a45562aa)" style="fill:#455cce"/>
                <path d="m261.234 184.325 2.571-.075 1.452-3.395-2.571.08z" clip-path="url(#p94a45562aa)" style="fill:#7a9df8"/>
                <path d="m221.127 204.488 2.532 1.261 1.487-2.098-2.53-1.258z" clip-path="url(#p94a45562aa)" style="fill:#445acc"/>
                <path d="m280.595 152.866 2.6-.515 1.459-4.446-2.603.521z" clip-path="url(#p94a45562aa)" style="fill:#ccd9ed"/>
                <path d="m196.355 195.19 2.534 2.308 1.524-1.693-2.531-2.305z" clip-path="url(#p94a45562aa)" style="fill:#5977e3"/>
                <path d="m272.19 169.677 2.59-.359 1.452-3.918-2.59.364z" clip-path="url(#p94a45562aa)" style="fill:#a2c1ff"/>
                <path d="m208.003 201.938 2.531 1.785 1.507-1.834-2.529-1.782z" clip-path="url(#p94a45562aa)" style="fill:#4a63d3"/>
                <path d="m255.762 190.66 2.565.065 1.454-3.137-2.564-.06z" clip-path="url(#p94a45562aa)" style="fill:#688aef"/>
                <path d="m227.68 204.78 2.536.999 1.48-2.23-2.535-.997z" clip-path="url(#p94a45562aa)" style="fill:#445acc"/>
                <path d="m279.14 157.174 2.6-.509 1.456-4.314-2.601.515z" clip-path="url(#p94a45562aa)" style="fill:#c3d5f4"/>
                <path d="m198.89 197.498 2.533 2.178 1.52-1.697-2.53-2.174z" clip-path="url(#p94a45562aa)" style="fill:#5470de"/>
                <path d="m265.257 180.855 2.58-.212 1.45-3.527-2.579.216z" clip-path="url(#p94a45562aa)" style="fill:#84a7fc"/>
                <path d="m217.1 205.064 2.533 1.393 1.494-1.97-2.53-1.39z" clip-path="url(#p94a45562aa)" style="fill:#445acc"/>
                <path d="m242.262 200.904 2.55.47 1.463-2.62-2.548-.467z" clip-path="url(#p94a45562aa)" style="fill:#4f69d9"/>
                <path d="m246.275 198.755 2.554.337 1.46-2.75-2.553-.334z" clip-path="url(#p94a45562aa)" style="fill:#5572df"/>
                <path d="m238.248 202.79 2.546.605 1.468-2.491-2.544-.601z" clip-path="url(#p94a45562aa)" style="fill:#4a63d3"/>
                <path d="m259.78 187.588 2.572-.07 1.453-3.268-2.571.075z" clip-path="url(#p94a45562aa)" style="fill:#7295f4"/>
                <path d="m223.659 205.749 2.535 1.132 1.485-2.102-2.533-1.128z" clip-path="url(#p94a45562aa)" style="fill:#4358cb"/>
                <path d="m270.74 173.461 2.587-.354 1.452-3.789-2.588.359z" clip-path="url(#p94a45562aa)" style="fill:#98b9ff"/>
                <path d="m210.534 203.723 2.532 1.655 1.504-1.837-2.53-1.652z" clip-path="url(#p94a45562aa)" style="fill:#465ecf"/>
                <path d="m277.685 161.352 2.6-.503 1.454-4.184-2.6.51z" clip-path="url(#p94a45562aa)" style="fill:#b9d0f9"/>
                <path d="m201.423 199.676 2.533 2.048 1.517-1.7-2.53-2.045z" clip-path="url(#p94a45562aa)" style="fill:#4f69d9"/>
                <path d="m250.29 196.342 2.559.203 1.457-2.879-2.558-.2z" clip-path="url(#p94a45562aa)" style="fill:#5b7ae5"/>
                <path d="m234.233 204.415 2.543.737 1.472-2.361-2.54-.735z" clip-path="url(#p94a45562aa)" style="fill:#465ecf"/>
                <path d="m182.109 183.228 2.547 2.969 1.547-1.548-2.544-2.964z" clip-path="url(#p94a45562aa)" style="fill:#779af7"/>
                <path d="m184.656 186.197 2.545 2.836 1.543-1.551-2.54-2.833z" clip-path="url(#p94a45562aa)" style="fill:#7093f3"/>
                <path d="m179.558 180.127 2.55 3.101 1.551-1.543-2.547-3.097z" clip-path="url(#p94a45562aa)" style="fill:#80a3fa"/>
                <path d="m187.2 189.033 2.543 2.705 1.54-1.556-2.539-2.7z" clip-path="url(#p94a45562aa)" style="fill:#6a8bef"/>
                <path d="m254.306 193.666 2.565.068 1.456-3.01-2.565-.064z" clip-path="url(#p94a45562aa)" style="fill:#6384eb"/>
                <path d="m230.216 205.779 2.54.87 1.477-2.234-2.538-.866z" clip-path="url(#p94a45562aa)" style="fill:#445acc"/>
                <path d="m276.232 165.4 2.598-.498 1.454-4.053-2.599.503z" clip-path="url(#p94a45562aa)" style="fill:#afcafc"/>
                <path d="m189.743 191.738 2.54 2.573 1.537-1.56-2.537-2.569z" clip-path="url(#p94a45562aa)" style="fill:#6384eb"/>
                <path d="m203.956 201.724 2.533 1.918 1.514-1.704-2.53-1.915z" clip-path="url(#p94a45562aa)" style="fill:#4b64d5"/>
                <path d="m263.805 184.25 2.579-.208 1.452-3.399-2.579.212z" clip-path="url(#p94a45562aa)" style="fill:#7da0f9"/>
                <path d="m219.633 206.457 2.534 1.264 1.492-1.972-2.532-1.261z" clip-path="url(#p94a45562aa)" style="fill:#4257c9"/>
                <path d="m269.288 177.116 2.587-.35 1.452-3.66-2.588.355z" clip-path="url(#p94a45562aa)" style="fill:#90b2fe"/>
                <path d="m213.066 205.378 2.533 1.526 1.501-1.84-2.53-1.523z" clip-path="url(#p94a45562aa)" style="fill:#445acc"/>
                <path d="m192.283 194.311 2.539 2.443 1.533-1.564-2.535-2.438z" clip-path="url(#p94a45562aa)" style="fill:#5d7ce6"/>
                <path d="m258.327 190.725 2.571-.068 1.454-3.14-2.571.071z" clip-path="url(#p94a45562aa)" style="fill:#6b8df0"/>
                <path d="m226.194 206.88 2.539 1.003 1.483-2.104-2.537-1z" clip-path="url(#p94a45562aa)" style="fill:#4257c9"/>
                <path d="m283.196 152.35 2.61-.65 1.46-4.452-2.612.657z" clip-path="url(#p94a45562aa)" style="fill:#cfdaea"/>
                <path d="m194.822 196.754 2.537 2.311 1.53-1.567-2.534-2.308z" clip-path="url(#p94a45562aa)" style="fill:#5875e1"/>
                <path d="m274.78 169.318 2.597-.493 1.453-3.923-2.598.498z" clip-path="url(#p94a45562aa)" style="fill:#a6c4fe"/>
                <path d="m206.489 203.642 2.533 1.788 1.512-1.707-2.53-1.785z" clip-path="url(#p94a45562aa)" style="fill:#485fd1"/>
                <path d="m244.811 201.375 2.555.34 1.463-2.623-2.554-.337z" clip-path="url(#p94a45562aa)" style="fill:#506bda"/>
                <path d="m240.794 203.395 2.55.473 1.467-2.493-2.55-.471z" clip-path="url(#p94a45562aa)" style="fill:#4b64d5"/>
                <path d="m248.829 199.092 2.56.206 1.46-2.753-2.56-.203z" clip-path="url(#p94a45562aa)" style="fill:#5673e0"/>
                <path d="m236.776 205.152 2.547.607 1.471-2.364-2.546-.604z" clip-path="url(#p94a45562aa)" style="fill:#465ecf"/>
                <path d="m281.74 156.665 2.61-.645 1.457-4.32-2.611.65z" clip-path="url(#p94a45562aa)" style="fill:#c6d6f1"/>
                <path d="m197.36 199.065 2.536 2.182 1.527-1.57-2.534-2.179z" clip-path="url(#p94a45562aa)" style="fill:#536edd"/>
                <path d="m267.836 180.643 2.587-.345 1.452-3.531-2.587.35z" clip-path="url(#p94a45562aa)" style="fill:#88abfd"/>
                <path d="m215.599 206.904 2.535 1.396 1.499-1.843-2.533-1.393z" clip-path="url(#p94a45562aa)" style="fill:#4257c9"/>
                <path d="m262.352 187.517 2.58-.203 1.452-3.272-2.579.208z" clip-path="url(#p94a45562aa)" style="fill:#7699f6"/>
                <path d="m222.167 207.721 2.538 1.135 1.49-1.975-2.536-1.132z" clip-path="url(#p94a45562aa)" style="fill:#4055c8"/>
                <path d="m252.849 196.545 2.565.072 1.457-2.883-2.565-.068z" clip-path="url(#p94a45562aa)" style="fill:#5d7ce6"/>
                <path d="m232.756 206.648 2.544.74 1.476-2.236-2.543-.737z" clip-path="url(#p94a45562aa)" style="fill:#445acc"/>
                <path d="m273.327 173.107 2.597-.489 1.453-3.793-2.598.493z" clip-path="url(#p94a45562aa)" style="fill:#9dbdff"/>
                <path d="m209.022 205.43 2.534 1.659 1.51-1.711-2.532-1.655z" clip-path="url(#p94a45562aa)" style="fill:#455cce"/>
                <path d="m280.284 160.849 2.609-.64 1.456-4.189-2.61.645z" clip-path="url(#p94a45562aa)" style="fill:#bed2f6"/>
                <path d="m199.896 201.247 2.536 2.051 1.524-1.574-2.533-2.048z" clip-path="url(#p94a45562aa)" style="fill:#4f69d9"/>
                <path d="m256.871 193.734 2.572-.064 1.455-3.013-2.571.068z" clip-path="url(#p94a45562aa)" style="fill:#6687ed"/>
                <path d="m228.733 207.883 2.541.872 1.482-2.107-2.54-.87z" clip-path="url(#p94a45562aa)" style="fill:#4257c9"/>
                <path d="m180.55 184.643 2.552 2.973 1.554-1.42-2.547-2.968z" clip-path="url(#p94a45562aa)" style="fill:#779af7"/>
                <path d="m183.102 187.616 2.548 2.84 1.55-1.423-2.544-2.836z" clip-path="url(#p94a45562aa)" style="fill:#6f92f3"/>
                <path d="m177.995 181.537 2.555 3.106 1.559-1.415-2.551-3.101z" clip-path="url(#p94a45562aa)" style="fill:#80a3fa"/>
                <path d="m185.65 190.457 2.546 2.709 1.547-1.428-2.542-2.705z" clip-path="url(#p94a45562aa)" style="fill:#688aef"/>
                <path d="m266.384 184.042 2.587-.34 1.452-3.404-2.587.345z" clip-path="url(#p94a45562aa)" style="fill:#80a3fa"/>
                <path d="m278.83 164.902 2.608-.634 1.455-4.058-2.609.639z" clip-path="url(#p94a45562aa)" style="fill:#b3cdfb"/>
                <path d="m218.134 208.3 2.537 1.267 1.496-1.846-2.534-1.264z" clip-path="url(#p94a45562aa)" style="fill:#4055c8"/>
                <path d="m202.432 203.298 2.536 1.921 1.52-1.577-2.532-1.918z" clip-path="url(#p94a45562aa)" style="fill:#4a63d3"/>
                <path d="m188.196 193.166 2.544 2.577 1.543-1.432-2.54-2.573z" clip-path="url(#p94a45562aa)" style="fill:#6282ea"/>
                <path d="m271.875 176.767 2.597-.484 1.452-3.665-2.597.489z" clip-path="url(#p94a45562aa)" style="fill:#94b6ff"/>
                <path d="m211.556 207.089 2.536 1.529 1.507-1.714-2.533-1.526z" clip-path="url(#p94a45562aa)" style="fill:#4358cb"/>
                <path d="m243.345 203.868 2.556.344 1.465-2.497-2.555-.34z" clip-path="url(#p94a45562aa)" style="fill:#4c66d6"/>
                <path d="m260.898 190.657 2.58-.2 1.453-3.143-2.579.203z" clip-path="url(#p94a45562aa)" style="fill:#6f92f3"/>
                <path d="m224.705 208.856 2.54 1.005 1.488-1.978-2.539-1.002z" clip-path="url(#p94a45562aa)" style="fill:#3f53c6"/>
                <path d="m247.366 201.715 2.561.21 1.462-2.627-2.56-.206z" clip-path="url(#p94a45562aa)" style="fill:#516ddb"/>
                <path d="m239.323 205.76 2.552.476 1.47-2.368-2.551-.473z" clip-path="url(#p94a45562aa)" style="fill:#485fd1"/>
                <path d="m190.74 195.743 2.542 2.446 1.54-1.435-2.539-2.443z" clip-path="url(#p94a45562aa)" style="fill:#5d7ce6"/>
                <path d="m251.389 199.298 2.566.075 1.46-2.756-2.566-.072z" clip-path="url(#p94a45562aa)" style="fill:#5977e3"/>
                <path d="m235.3 207.388 2.549.61 1.474-2.239-2.547-.607z" clip-path="url(#p94a45562aa)" style="fill:#445acc"/>
                <path d="m277.377 168.825 2.606-.628 1.455-3.929-2.608.634z" clip-path="url(#p94a45562aa)" style="fill:#aac7fd"/>
                <path d="m285.807 151.7 2.62-.787 1.462-4.458-2.623.793z" clip-path="url(#p94a45562aa)" style="fill:#d4dbe6"/>
                <path d="m204.968 205.22 2.536 1.79 1.518-1.58-2.533-1.788z" clip-path="url(#p94a45562aa)" style="fill:#485fd1"/>
                <path d="m193.282 198.19 2.54 2.315 1.537-1.44-2.537-2.311z" clip-path="url(#p94a45562aa)" style="fill:#5673e0"/>
                <path d="m255.414 196.617 2.573-.06 1.456-2.887-2.572.064z" clip-path="url(#p94a45562aa)" style="fill:#5f7fe8"/>
                <path d="m231.274 208.755 2.546.743 1.48-2.11-2.544-.74z" clip-path="url(#p94a45562aa)" style="fill:#4055c8"/>
                <path d="m270.423 180.298 2.596-.479 1.453-3.536-2.597.484z" clip-path="url(#p94a45562aa)" style="fill:#8caffe"/>
                <path d="m214.092 208.618 2.537 1.4 1.505-1.718-2.535-1.396z" clip-path="url(#p94a45562aa)" style="fill:#4055c8"/>
                <path d="m264.931 187.314 2.587-.337 1.453-3.276-2.587.341z" clip-path="url(#p94a45562aa)" style="fill:#799cf8"/>
                <path d="m220.67 209.567 2.54 1.138 1.495-1.85-2.538-1.134z" clip-path="url(#p94a45562aa)" style="fill:#3e51c5"/>
                <path d="m284.35 156.02 2.619-.78 1.459-4.327-2.621.787z" clip-path="url(#p94a45562aa)" style="fill:#cbd8ee"/>
                <path d="m195.822 200.505 2.54 2.185 1.534-1.443-2.537-2.182z" clip-path="url(#p94a45562aa)" style="fill:#536edd"/>
                <path d="m275.924 172.618 2.606-.623 1.453-3.798-2.606.628z" clip-path="url(#p94a45562aa)" style="fill:#a2c1ff"/>
                <path d="m207.504 207.01 2.537 1.663 1.515-1.584-2.534-1.659z" clip-path="url(#p94a45562aa)" style="fill:#445acc"/>
                <path d="m259.443 193.67 2.58-.196 1.455-3.017-2.58.2z" clip-path="url(#p94a45562aa)" style="fill:#688aef"/>
                <path d="m227.245 209.86 2.544.876 1.485-1.98-2.541-.873z" clip-path="url(#p94a45562aa)" style="fill:#3f53c6"/>
                <path d="m282.893 160.21 2.618-.775 1.458-4.195-2.62.78z" clip-path="url(#p94a45562aa)" style="fill:#c1d4f4"/>
                <path d="m198.362 202.69 2.54 2.055 1.53-1.447-2.536-2.051zM245.9 204.212l2.563.212 1.464-2.5-2.56-.21z" clip-path="url(#p94a45562aa)" style="fill:#4e68d8"/>
                <path d="m241.875 206.236 2.557.346 1.469-2.37-2.556-.344z" clip-path="url(#p94a45562aa)" style="fill:#4961d2"/>
                <path d="m249.927 201.924 2.567.078 1.461-2.63-2.566-.074z" clip-path="url(#p94a45562aa)" style="fill:#5470de"/>
                <path d="m237.849 207.998 2.553.48 1.473-2.242-2.552-.477z" clip-path="url(#p94a45562aa)" style="fill:#455cce"/>
                <path d="m268.971 183.701 2.596-.474 1.452-3.408-2.596.48z" clip-path="url(#p94a45562aa)" style="fill:#84a7fc"/>
                <path d="m216.63 210.017 2.539 1.27 1.502-1.72-2.537-1.267z" clip-path="url(#p94a45562aa)" style="fill:#3e51c5"/>
                <path d="m178.984 185.93 2.555 2.977 1.563-1.29-2.552-2.974z" clip-path="url(#p94a45562aa)" style="fill:#779af7"/>
                <path d="m181.54 188.907 2.552 2.845 1.558-1.295-2.548-2.84z" clip-path="url(#p94a45562aa)" style="fill:#6f92f3"/>
                <path d="m176.425 182.82 2.559 3.11 1.566-1.287-2.555-3.106z" clip-path="url(#p94a45562aa)" style="fill:#80a3fa"/>
                <path d="m184.092 191.752 2.55 2.714 1.554-1.3-2.546-2.71z" clip-path="url(#p94a45562aa)" style="fill:#688aef"/>
                <path d="m281.438 164.268 2.617-.769 1.456-4.064-2.618.775z" clip-path="url(#p94a45562aa)" style="fill:#b9d0f9"/>
                <path d="m200.901 204.745 2.54 1.925 1.527-1.45-2.536-1.922z" clip-path="url(#p94a45562aa)" style="fill:#4a63d3"/>
                <path d="m263.478 190.457 2.587-.332 1.453-3.148-2.587.337z" clip-path="url(#p94a45562aa)" style="fill:#7295f4"/>
                <path d="m223.21 210.705 2.543 1.008 1.492-1.852-2.54-1.005z" clip-path="url(#p94a45562aa)" style="fill:#3e51c5"/>
                <path d="m274.472 176.283 2.605-.618 1.453-3.67-2.606.623z" clip-path="url(#p94a45562aa)" style="fill:#98b9ff"/>
                <path d="m210.041 208.673 2.539 1.532 1.512-1.587-2.536-1.53z" clip-path="url(#p94a45562aa)" style="fill:#4257c9"/>
                <path d="m253.955 199.373 2.574-.057 1.458-2.76-2.573.06z" clip-path="url(#p94a45562aa)" style="fill:#5b7ae5"/>
                <path d="m233.82 209.498 2.55.613 1.479-2.113-2.549-.61z" clip-path="url(#p94a45562aa)" style="fill:#4257c9"/>
                <path d="m186.641 194.466 2.548 2.581 1.55-1.304-2.543-2.577z" clip-path="url(#p94a45562aa)" style="fill:#6282ea"/>
                <path d="m189.189 197.047 2.545 2.45 1.548-1.308-2.542-2.446z" clip-path="url(#p94a45562aa)" style="fill:#5d7ce6"/>
                <path d="m257.987 196.556 2.58-.192 1.456-2.89-2.58.196z" clip-path="url(#p94a45562aa)" style="fill:#6384eb"/>
                <path d="m229.789 210.736 2.548.746 1.483-1.984-2.546-.743z" clip-path="url(#p94a45562aa)" style="fill:#3f53c6"/>
                <path d="m279.983 168.197 2.617-.764 1.455-3.934-2.617.77z" clip-path="url(#p94a45562aa)" style="fill:#afcafc"/>
                <path d="m203.44 206.67 2.54 1.795 1.524-1.454-2.536-1.792z" clip-path="url(#p94a45562aa)" style="fill:#465ecf"/>
                <path d="m288.428 150.913 2.632-.924 1.462-4.465-2.633.93z" clip-path="url(#p94a45562aa)" style="fill:#d8dce2"/>
                <path d="m191.734 199.498 2.544 2.32 1.544-1.313-2.54-2.316z" clip-path="url(#p94a45562aa)" style="fill:#5673e0"/>
                <path d="m267.518 186.977 2.596-.47 1.453-3.28-2.596.474z" clip-path="url(#p94a45562aa)" style="fill:#7da0f9"/>
                <path d="m219.169 211.287 2.542 1.14 1.5-1.722-2.54-1.138z" clip-path="url(#p94a45562aa)" style="fill:#3d50c3"/>
                <path d="m273.02 179.82 2.604-.614 1.453-3.54-2.605.617z" clip-path="url(#p94a45562aa)" style="fill:#90b2fe"/>
                <path d="m212.58 210.205 2.54 1.402 1.51-1.59-2.538-1.4z" clip-path="url(#p94a45562aa)" style="fill:#3f53c6"/>
                <path d="m286.969 155.24 2.63-.918 1.46-4.333-2.631.924z" clip-path="url(#p94a45562aa)" style="fill:#cfdaea"/>
                <path d="m194.278 201.817 2.544 2.189 1.54-1.316-2.54-2.185z" clip-path="url(#p94a45562aa)" style="fill:#516ddb"/>
                <path d="m262.023 193.474 2.587-.329 1.455-3.02-2.587.332z" clip-path="url(#p94a45562aa)" style="fill:#6c8ff1"/>
                <path d="m225.753 211.713 2.546.878 1.49-1.855-2.544-.875z" clip-path="url(#p94a45562aa)" style="fill:#3d50c3"/>
                <path d="m244.432 206.582 2.563.215 1.468-2.373-2.562-.212z" clip-path="url(#p94a45562aa)" style="fill:#4a63d3"/>
                <path d="m248.463 204.424 2.568.08 1.463-2.502-2.567-.078z" clip-path="url(#p94a45562aa)" style="fill:#506bda"/>
                <path d="m240.402 208.478 2.559.349 1.471-2.245-2.557-.346z" clip-path="url(#p94a45562aa)" style="fill:#455cce"/>
                <path d="m278.53 171.995 2.616-.758 1.454-3.804-2.617.764z" clip-path="url(#p94a45562aa)" style="fill:#a6c4fe"/>
                <path d="m205.98 208.465 2.54 1.665 1.521-1.457-2.537-1.662z" clip-path="url(#p94a45562aa)" style="fill:#445acc"/>
                <path d="m252.494 202.002 2.574-.054 1.46-2.632-2.573.057z" clip-path="url(#p94a45562aa)" style="fill:#5673e0"/>
                <path d="m236.37 210.11 2.556.483 1.476-2.115-2.553-.48z" clip-path="url(#p94a45562aa)" style="fill:#4257c9"/>
                <path d="m285.511 159.435 2.63-.912 1.458-4.201-2.63.918z" clip-path="url(#p94a45562aa)" style="fill:#c6d6f1"/>
                <path d="m196.822 204.006 2.542 2.058 1.537-1.319-2.539-2.055z" clip-path="url(#p94a45562aa)" style="fill:#4e68d8"/>
                <path d="m271.567 183.227 2.604-.61 1.453-3.411-2.605.613z" clip-path="url(#p94a45562aa)" style="fill:#89acfd"/>
                <path d="m215.12 211.607 2.542 1.273 1.507-1.593-2.54-1.27z" clip-path="url(#p94a45562aa)" style="fill:#3e51c5"/>
                <path d="m256.529 199.316 2.58-.19 1.458-2.762-2.58.192z" clip-path="url(#p94a45562aa)" style="fill:#5e7de7"/>
                <path d="m232.337 211.482 2.552.615 1.482-1.986-2.55-.613z" clip-path="url(#p94a45562aa)" style="fill:#3f53c6"/>
                <path d="m266.065 190.125 2.595-.467 1.454-3.152-2.596.47z" clip-path="url(#p94a45562aa)" style="fill:#7699f6"/>
                <path d="m221.71 212.427 2.546 1.01 1.497-1.724-2.543-1.008z" clip-path="url(#p94a45562aa)" style="fill:#3c4ec2"/>
                <path d="m177.41 187.088 2.559 2.982 1.57-1.163-2.555-2.977z" clip-path="url(#p94a45562aa)" style="fill:#779af7"/>
                <path d="m179.969 190.07 2.556 2.85 1.567-1.168-2.553-2.845z" clip-path="url(#p94a45562aa)" style="fill:#6f92f3"/>
                <path d="m277.077 175.665 2.615-.753 1.454-3.675-2.616.758z" clip-path="url(#p94a45562aa)" style="fill:#9ebeff"/>
                <path d="m208.52 210.13 2.542 1.535 1.518-1.46-2.539-1.532z" clip-path="url(#p94a45562aa)" style="fill:#4055c8"/>
                <path d="m174.846 183.972 2.563 3.116 1.575-1.158-2.559-3.11z" clip-path="url(#p94a45562aa)" style="fill:#80a3fa"/>
                <path d="m284.055 163.499 2.628-.906 1.457-4.07-2.629.912z" clip-path="url(#p94a45562aa)" style="fill:#bed2f6"/>
                <path d="m199.364 206.064 2.543 1.929 1.534-1.323-2.54-1.925z" clip-path="url(#p94a45562aa)" style="fill:#4a63d3"/>
                <path d="m182.525 192.92 2.554 2.717 1.562-1.171-2.55-2.714z" clip-path="url(#p94a45562aa)" style="fill:#688aef"/>
                <path d="m185.079 195.637 2.551 2.586 1.559-1.176-2.548-2.581z" clip-path="url(#p94a45562aa)" style="fill:#6282ea"/>
                <path d="m260.567 196.364 2.588-.326 1.455-2.893-2.587.33z" clip-path="url(#p94a45562aa)" style="fill:#6687ed"/>
                <path d="m228.299 212.59 2.55.749 1.488-1.857-2.548-.746z" clip-path="url(#p94a45562aa)" style="fill:#3d50c3"/>
                <path d="m187.63 198.223 2.55 2.455 1.554-1.18-2.545-2.45z" clip-path="url(#p94a45562aa)" style="fill:#5d7ce6"/>
                <path d="m282.6 167.433 2.627-.9 1.456-3.94-2.628.906z" clip-path="url(#p94a45562aa)" style="fill:#b5cdfa"/>
                <path d="m201.907 207.993 2.542 1.798 1.531-1.326-2.54-1.795z" clip-path="url(#p94a45562aa)" style="fill:#465ecf"/>
                <path d="m246.995 206.797 2.57.084 1.466-2.376-2.568-.081z" clip-path="url(#p94a45562aa)" style="fill:#4c66d6"/>
                <path d="m242.96 208.827 2.565.218 1.47-2.248-2.563-.215z" clip-path="url(#p94a45562aa)" style="fill:#485fd1"/>
                <path d="m270.114 186.506 2.604-.604 1.453-3.284-2.604.609z" clip-path="url(#p94a45562aa)" style="fill:#81a4fb"/>
                <path d="m217.662 212.88 2.544 1.143 1.505-1.596-2.542-1.14z" clip-path="url(#p94a45562aa)" style="fill:#3c4ec2"/>
                <path d="m291.06 149.99 2.642-1.063 1.464-4.472-2.644 1.069z" clip-path="url(#p94a45562aa)" style="fill:#dcdddd"/>
                <path d="m190.18 200.678 2.547 2.323 1.551-1.184-2.544-2.32z" clip-path="url(#p94a45562aa)" style="fill:#5875e1"/>
                <path d="m251.03 204.505 2.576-.051 1.462-2.506-2.574.054z" clip-path="url(#p94a45562aa)" style="fill:#536edd"/>
                <path d="m275.624 179.206 2.615-.749 1.453-3.545-2.615.753z" clip-path="url(#p94a45562aa)" style="fill:#96b7ff"/>
                <path d="m238.926 210.593 2.56.351 1.475-2.117-2.559-.35z" clip-path="url(#p94a45562aa)" style="fill:#4358cb"/>
                <path d="m211.062 211.665 2.542 1.405 1.516-1.463-2.54-1.402z" clip-path="url(#p94a45562aa)" style="fill:#3f53c6"/>
                <path d="m264.61 193.145 2.596-.463 1.454-3.024-2.595.467z" clip-path="url(#p94a45562aa)" style="fill:#7093f3"/>
                <path d="m224.256 213.438 2.548.88 1.495-1.727-2.546-.878z" clip-path="url(#p94a45562aa)" style="fill:#3c4ec2"/>
                <path d="m255.068 201.948 2.582-.186 1.46-2.636-2.581.19z" clip-path="url(#p94a45562aa)" style="fill:#5977e3"/>
                <path d="m234.889 212.097 2.557.485 1.48-1.99-2.555-.481z" clip-path="url(#p94a45562aa)" style="fill:#3f53c6"/>
                <path d="m289.599 154.322 2.641-1.055 1.462-4.34-2.642 1.062z" clip-path="url(#p94a45562aa)" style="fill:#d4dbe6"/>
                <path d="m192.727 203.001 2.547 2.193 1.548-1.188-2.544-2.189z" clip-path="url(#p94a45562aa)" style="fill:#536edd"/>
                <path d="m281.146 171.237 2.626-.894 1.455-3.81-2.627.9z" clip-path="url(#p94a45562aa)" style="fill:#abc8fd"/>
                <path d="m204.45 209.791 2.543 1.668 1.527-1.33-2.54-1.664z" clip-path="url(#p94a45562aa)" style="fill:#4358cb"/>
                <path d="m259.11 199.126 2.588-.322 1.457-2.766-2.588.326z" clip-path="url(#p94a45562aa)" style="fill:#6180e9"/>
                <path d="m230.849 213.339 2.554.618 1.486-1.86-2.552-.615z" clip-path="url(#p94a45562aa)" style="fill:#3d50c3"/>
                <path d="m288.14 158.523 2.64-1.049 1.46-4.207-2.641 1.055z" clip-path="url(#p94a45562aa)" style="fill:#ccd9ed"/>
                <path d="m195.274 205.194 2.546 2.062 1.544-1.192-2.542-2.058z" clip-path="url(#p94a45562aa)" style="fill:#4e68d8"/>
                <path d="m274.171 182.618 2.615-.744 1.453-3.417-2.615.749z" clip-path="url(#p94a45562aa)" style="fill:#8db0fe"/>
                <path d="m213.604 213.07 2.545 1.276 1.513-1.466-2.542-1.273z" clip-path="url(#p94a45562aa)" style="fill:#3d50c3"/>
                <path d="m268.66 189.658 2.605-.6 1.453-3.156-2.604.604z" clip-path="url(#p94a45562aa)" style="fill:#7b9ff9"/>
                <path d="m220.206 214.023 2.548 1.013 1.502-1.598-2.545-1.01z" clip-path="url(#p94a45562aa)" style="fill:#3c4ec2"/>
                <path d="m279.692 174.912 2.625-.89 1.455-3.68-2.626.895z" clip-path="url(#p94a45562aa)" style="fill:#a3c2fe"/>
                <path d="m206.993 211.46 2.544 1.538 1.525-1.333-2.542-1.535z" clip-path="url(#p94a45562aa)" style="fill:#4055c8"/>
                <path d="m263.155 196.038 2.596-.459 1.455-2.897-2.596.463z" clip-path="url(#p94a45562aa)" style="fill:#6a8bef"/>
                <path d="m226.804 214.319 2.552.75 1.493-1.73-2.55-.748z" clip-path="url(#p94a45562aa)" style="fill:#3c4ec2"/>
                <path d="m286.683 162.593 2.638-1.042 1.459-4.077-2.64 1.05z" clip-path="url(#p94a45562aa)" style="fill:#c3d5f4"/>
                <path d="m175.827 188.116 2.563 2.987 1.579-1.033-2.56-2.982z" clip-path="url(#p94a45562aa)" style="fill:#799cf8"/>
                <path d="m197.82 207.256 2.546 1.932 1.54-1.195-2.542-1.929z" clip-path="url(#p94a45562aa)" style="fill:#4a63d3"/>
                <path d="m178.39 191.103 2.56 2.855 1.575-1.038-2.556-2.85z" clip-path="url(#p94a45562aa)" style="fill:#7093f3"/>
                <path d="m245.525 209.045 2.571.086 1.469-2.25-2.57-.084z" clip-path="url(#p94a45562aa)" style="fill:#4961d2"/>
                <path d="m173.26 184.996 2.567 3.12 1.582-1.028-2.563-3.116z" clip-path="url(#p94a45562aa)" style="fill:#80a3fa"/>
                <path d="m180.95 193.958 2.558 2.722 1.57-1.043-2.553-2.717z" clip-path="url(#p94a45562aa)" style="fill:#6a8bef"/>
                <path d="m249.565 206.881 2.576-.048 1.465-2.379-2.575.05z" clip-path="url(#p94a45562aa)" style="fill:#4f69d9"/>
                <path d="m241.486 210.944 2.566.221 1.473-2.12-2.564-.218z" clip-path="url(#p94a45562aa)" style="fill:#455cce"/>
                <path d="m183.508 196.68 2.556 2.59 1.566-1.047-2.551-2.586z" clip-path="url(#p94a45562aa)" style="fill:#6384eb"/>
                <path d="m253.606 204.454 2.582-.183 1.462-2.509-2.582.186z" clip-path="url(#p94a45562aa)" style="fill:#5572df"/>
                <path d="m237.446 212.582 2.562.354 1.478-1.992-2.56-.351z" clip-path="url(#p94a45562aa)" style="fill:#4055c8"/>
                <path d="m272.718 185.902 2.614-.74 1.454-3.288-2.615.744z" clip-path="url(#p94a45562aa)" style="fill:#86a9fc"/>
                <path d="m186.064 199.27 2.553 2.46 1.562-1.052-2.549-2.455z" clip-path="url(#p94a45562aa)" style="fill:#5d7ce6"/>
                <path d="m216.15 214.346 2.546 1.146 1.51-1.469-2.544-1.143z" clip-path="url(#p94a45562aa)" style="fill:#3c4ec2"/>
                <path d="m285.227 166.533 2.637-1.037 1.457-3.945-2.638 1.042z" clip-path="url(#p94a45562aa)" style="fill:#bad0f8"/>
                <path d="m200.366 209.188 2.546 1.802 1.537-1.199-2.542-1.798z" clip-path="url(#p94a45562aa)" style="fill:#465ecf"/>
                <path d="m267.206 192.682 2.605-.596 1.454-3.028-2.605.6z" clip-path="url(#p94a45562aa)" style="fill:#7597f6"/>
                <path d="m222.754 215.036 2.55.884 1.5-1.601-2.548-.881z" clip-path="url(#p94a45562aa)" style="fill:#3b4cc0"/>
                <path d="m257.65 201.762 2.589-.319 1.459-2.639-2.589.322z" clip-path="url(#p94a45562aa)" style="fill:#5d7ce6"/>
                <path d="m278.239 178.457 2.624-.884 1.454-3.55-2.625.889z" clip-path="url(#p94a45562aa)" style="fill:#9bbcff"/>
                <path d="m233.403 213.957 2.558.487 1.485-1.862-2.557-.485zM209.537 212.998l2.546 1.408 1.521-1.336-2.542-1.405z" clip-path="url(#p94a45562aa)" style="fill:#3e51c5"/>
                <path d="m293.702 148.927 2.654-1.2 1.466-4.48-2.656 1.208z" clip-path="url(#p94a45562aa)" style="fill:#e1dad6"/>
                <path d="m188.617 201.73 2.552 2.327 1.558-1.056-2.548-2.323z" clip-path="url(#p94a45562aa)" style="fill:#5875e1"/>
                <path d="m292.24 153.267 2.653-1.194 1.463-4.346-2.654 1.2z" clip-path="url(#p94a45562aa)" style="fill:#d9dce1"/>
                <path d="m261.698 198.804 2.596-.456 1.457-2.769-2.596.46z" clip-path="url(#p94a45562aa)" style="fill:#6485ec"/>
                <path d="m229.356 215.07 2.556.62 1.49-1.733-2.553-.618z" clip-path="url(#p94a45562aa)" style="fill:#3c4ec2"/>
                <path d="m191.169 204.057 2.55 2.197 1.555-1.06-2.547-2.193z" clip-path="url(#p94a45562aa)" style="fill:#536edd"/>
                <path d="m283.772 170.343 2.636-1.032 1.456-3.815-2.637 1.037z" clip-path="url(#p94a45562aa)" style="fill:#b2ccfb"/>
                <path d="m202.912 210.99 2.547 1.671 1.534-1.202-2.544-1.668z" clip-path="url(#p94a45562aa)" style="fill:#445acc"/>
                <path d="m271.265 189.058 2.614-.736 1.453-3.16-2.614.74z" clip-path="url(#p94a45562aa)" style="fill:#80a3fa"/>
                <path d="m218.696 215.492 2.55 1.016 1.508-1.472-2.548-1.013z" clip-path="url(#p94a45562aa)" style="fill:#3b4cc0"/>
                <path d="m276.786 181.874 2.624-.88 1.453-3.421-2.624.884z" clip-path="url(#p94a45562aa)" style="fill:#93b5fe"/>
                <path d="m212.083 214.406 2.548 1.279 1.518-1.339-2.545-1.276z" clip-path="url(#p94a45562aa)" style="fill:#3d50c3"/>
                <path d="m290.78 157.474 2.651-1.187 1.462-4.214-2.653 1.194z" clip-path="url(#p94a45562aa)" style="fill:#d1dae9"/>
                <path d="m248.096 209.131 2.577-.045 1.468-2.253-2.576.048z" clip-path="url(#p94a45562aa)" style="fill:#4b64d5"/>
                <path d="m244.052 211.165 2.572.09 1.472-2.124-2.57-.086z" clip-path="url(#p94a45562aa)" style="fill:#465ecf"/>
                <path d="m193.72 206.254 2.549 2.066 1.551-1.064-2.546-2.062z" clip-path="url(#p94a45562aa)" style="fill:#4f69d9"/>
                <path d="m252.141 206.833 2.583-.18 1.464-2.382-2.582.183z" clip-path="url(#p94a45562aa)" style="fill:#516ddb"/>
                <path d="m240.008 212.936 2.567.223 1.477-1.994-2.566-.22z" clip-path="url(#p94a45562aa)" style="fill:#4358cb"/>
                <path d="m265.75 195.58 2.606-.594 1.455-2.9-2.605.596z" clip-path="url(#p94a45562aa)" style="fill:#6f92f3"/>
                <path d="m225.304 215.92 2.555.753 1.497-1.604-2.552-.75z" clip-path="url(#p94a45562aa)" style="fill:#3b4cc0"/>
                <path d="m282.317 174.022 2.636-1.026 1.455-3.685-2.636 1.032z" clip-path="url(#p94a45562aa)" style="fill:#a9c6fd"/>
                <path d="m205.459 212.661 2.547 1.542 1.531-1.205-2.544-1.539z" clip-path="url(#p94a45562aa)" style="fill:#4055c8"/>
                <path d="m256.188 204.27 2.59-.315 1.46-2.512-2.588.319z" clip-path="url(#p94a45562aa)" style="fill:#5977e3"/>
                <path d="m235.961 214.444 2.564.357 1.483-1.865-2.562-.354z" clip-path="url(#p94a45562aa)" style="fill:#3f53c6"/>
                <path d="m289.321 161.55 2.65-1.18 1.46-4.083-2.651 1.187z" clip-path="url(#p94a45562aa)" style="fill:#c9d7f0"/>
                <path d="m196.269 208.32 2.55 1.935 1.547-1.067-2.546-1.932z" clip-path="url(#p94a45562aa)" style="fill:#4a63d3"/>
                <path d="m174.235 189.015 2.568 2.992 1.587-.904-2.563-2.987z" clip-path="url(#p94a45562aa)" style="fill:#799cf8"/>
                <path d="m176.803 192.007 2.565 2.86 1.583-.91-2.56-2.854z" clip-path="url(#p94a45562aa)" style="fill:#7295f4"/>
                <path d="m171.663 185.89 2.572 3.125 1.592-.899-2.568-3.12z" clip-path="url(#p94a45562aa)" style="fill:#81a4fb"/>
                <path d="m179.368 194.867 2.562 2.727 1.578-.914-2.557-2.722z" clip-path="url(#p94a45562aa)" style="fill:#6b8df0"/>
                <path d="m260.239 201.443 2.597-.452 1.458-2.643-2.596.456z" clip-path="url(#p94a45562aa)" style="fill:#6180e9"/>
                <path d="m231.912 215.69 2.561.49 1.488-1.736-2.558-.487z" clip-path="url(#p94a45562aa)" style="fill:#3d50c3"/>
                <path d="m181.93 197.594 2.559 2.595 1.575-.919-2.556-2.59z" clip-path="url(#p94a45562aa)" style="fill:#6485ec"/>
                <path d="m275.332 185.162 2.624-.875 1.454-3.293-2.624.88z" clip-path="url(#p94a45562aa)" style="fill:#8caffe"/>
                <path d="m214.63 215.685 2.55 1.148 1.516-1.341-2.547-1.146z" clip-path="url(#p94a45562aa)" style="fill:#3c4ec2"/>
                <path d="m269.81 192.086 2.615-.732 1.454-3.032-2.614.736z" clip-path="url(#p94a45562aa)" style="fill:#7a9df8"/>
                <path d="m221.246 216.508 2.554.886 1.504-1.474-2.55-.884z" clip-path="url(#p94a45562aa)" style="fill:#3b4cc0"/>
                <path d="m287.864 165.496 2.649-1.175 1.458-3.951-2.65 1.18z" clip-path="url(#p94a45562aa)" style="fill:#c0d4f5"/>
                <path d="m184.489 200.189 2.557 2.463 1.57-.923-2.552-2.459z" clip-path="url(#p94a45562aa)" style="fill:#5e7de7"/>
                <path d="m198.818 210.255 2.55 1.805 1.544-1.07-2.546-1.802z" clip-path="url(#p94a45562aa)" style="fill:#485fd1"/>
                <path d="m280.863 177.573 2.635-1.021 1.455-3.556-2.636 1.026z" clip-path="url(#p94a45562aa)" style="fill:#a1c0ff"/>
                <path d="m208.006 214.203 2.55 1.411 1.527-1.208-2.546-1.408z" clip-path="url(#p94a45562aa)" style="fill:#3f53c6"/>
                <path d="m296.356 147.727 2.667-1.34 1.467-4.486-2.668 1.347z" clip-path="url(#p94a45562aa)" style="fill:#e6d7cf"/>
                <path d="m187.046 202.652 2.556 2.332 1.567-.927-2.552-2.328z" clip-path="url(#p94a45562aa)" style="fill:#5977e3"/>
                <path d="m264.294 198.348 2.605-.59 1.457-2.772-2.605.593z" clip-path="url(#p94a45562aa)" style="fill:#6a8bef"/>
                <path d="m227.859 216.673 2.558.623 1.495-1.606-2.556-.62z" clip-path="url(#p94a45562aa)" style="fill:#3b4cc0"/>
                <path d="m246.624 211.254 2.579-.043 1.47-2.125-2.577.045z" clip-path="url(#p94a45562aa)" style="fill:#4961d2"/>
                <path d="m250.673 209.086 2.585-.178 1.466-2.255-2.583.18z" clip-path="url(#p94a45562aa)" style="fill:#4f69d9"/>
                <path d="m242.575 213.159 2.574.092 1.475-1.997-2.572-.089z" clip-path="url(#p94a45562aa)" style="fill:#445acc"/>
                <path d="m286.408 169.311 2.648-1.169 1.457-3.82-2.649 1.174z" clip-path="url(#p94a45562aa)" style="fill:#b7cff9"/>
                <path d="m294.893 152.073 2.664-1.333 1.466-4.353-2.667 1.34z" clip-path="url(#p94a45562aa)" style="fill:#dedcdb"/>
                <path d="m201.368 212.06 2.55 1.675 1.54-1.074-2.546-1.671z" clip-path="url(#p94a45562aa)" style="fill:#445acc"/>
                <path d="m189.602 204.984 2.554 2.2 1.563-.93-2.55-2.197z" clip-path="url(#p94a45562aa)" style="fill:#5470de"/>
                <path d="m254.724 206.653 2.591-.313 1.463-2.385-2.59.316z" clip-path="url(#p94a45562aa)" style="fill:#5572df"/>
                <path d="m238.525 214.8 2.57.226 1.48-1.867-2.567-.223z" clip-path="url(#p94a45562aa)" style="fill:#4055c8"/>
                <path d="m273.879 188.322 2.623-.87 1.454-3.165-2.624.875z" clip-path="url(#p94a45562aa)" style="fill:#85a8fc"/>
                <path d="m217.18 216.833 2.554 1.02 1.512-1.345-2.55-1.016z" clip-path="url(#p94a45562aa)" style="fill:#3b4cc0"/>
                <path d="m279.41 180.994 2.634-1.016 1.454-3.426-2.635 1.02z" clip-path="url(#p94a45562aa)" style="fill:#9abbff"/>
                <path d="m210.555 215.614 2.551 1.282 1.525-1.211-2.548-1.279z" clip-path="url(#p94a45562aa)" style="fill:#3d50c3"/>
                <path d="m268.356 194.986 2.614-.728 1.455-2.904-2.614.732z" clip-path="url(#p94a45562aa)" style="fill:#7396f5"/>
                <path d="m223.8 217.394 2.556.756 1.503-1.477-2.555-.753z" clip-path="url(#p94a45562aa)" style="fill:#3b4cc0"/>
                <path d="m293.431 156.287 2.663-1.326 1.463-4.22-2.664 1.332z" clip-path="url(#p94a45562aa)" style="fill:#d6dce4"/>
                <path d="m258.778 203.955 2.598-.45 1.46-2.514-2.597.452z" clip-path="url(#p94a45562aa)" style="fill:#5d7ce6"/>
                <path d="m234.473 216.18 2.566.358 1.486-1.737-2.564-.357z" clip-path="url(#p94a45562aa)" style="fill:#3e51c5"/>
                <path d="m192.156 207.184 2.554 2.07 1.559-.934-2.55-2.066z" clip-path="url(#p94a45562aa)" style="fill:#506bda"/>
                <path d="m284.953 172.996 2.647-1.163 1.456-3.69-2.648 1.168z" clip-path="url(#p94a45562aa)" style="fill:#afcafc"/>
                <path d="m203.918 213.735 2.55 1.545 1.538-1.077-2.547-1.542z" clip-path="url(#p94a45562aa)" style="fill:#4257c9"/>
                <path d="m262.836 200.99 2.606-.586 1.457-2.645-2.605.59z" clip-path="url(#p94a45562aa)" style="fill:#6485ec"/>
                <path d="m230.417 217.296 2.563.492 1.493-1.609-2.56-.49z" clip-path="url(#p94a45562aa)" style="fill:#3c4ec2"/>
                <path d="m291.971 160.37 2.662-1.32 1.461-4.089-2.663 1.326z" clip-path="url(#p94a45562aa)" style="fill:#cedaeb"/>
                <path d="m194.71 209.254 2.553 1.94 1.555-.939-2.55-1.935z" clip-path="url(#p94a45562aa)" style="fill:#4b64d5"/>
                <path d="m172.634 189.784 2.573 2.997 1.596-.774-2.568-2.992z" clip-path="url(#p94a45562aa)" style="fill:#7b9ff9"/>
                <path d="m175.207 192.781 2.569 2.865 1.592-.78-2.565-2.859z" clip-path="url(#p94a45562aa)" style="fill:#7396f5"/>
                <path d="m170.058 186.652 2.576 3.132 1.6-.769-2.571-3.126z" clip-path="url(#p94a45562aa)" style="fill:#84a7fc"/>
                <path d="m277.956 184.287 2.634-1.012 1.454-3.297-2.634 1.016z" clip-path="url(#p94a45562aa)" style="fill:#93b5fe"/>
                <path d="m213.106 216.896 2.553 1.151 1.522-1.214-2.55-1.148z" clip-path="url(#p94a45562aa)" style="fill:#3c4ec2"/>
                <path d="m272.425 191.354 2.623-.867 1.454-3.036-2.623.871z" clip-path="url(#p94a45562aa)" style="fill:#80a3fa"/>
                <path d="m219.734 217.852 2.555.889 1.51-1.347-2.553-.886z" clip-path="url(#p94a45562aa)" style="fill:#3b4cc0"/>
                <path d="m177.776 195.646 2.566 2.732 1.588-.784-2.562-2.727z" clip-path="url(#p94a45562aa)" style="fill:#6c8ff1"/>
                <path d="m180.342 198.378 2.564 2.6 1.583-.79-2.56-2.594z" clip-path="url(#p94a45562aa)" style="fill:#6687ed"/>
                <path d="m249.203 211.211 2.586-.175 1.469-2.128-2.585.178z" clip-path="url(#p94a45562aa)" style="fill:#4b64d5"/>
                <path d="m245.149 213.25 2.58-.04 1.474-1.999-2.579.043z" clip-path="url(#p94a45562aa)" style="fill:#465ecf"/>
                <path d="m283.498 176.552 2.646-1.159 1.456-3.56-2.647 1.163z" clip-path="url(#p94a45562aa)" style="fill:#a7c5fe"/>
                <path d="m206.469 215.28 2.552 1.415 1.534-1.08-2.549-1.412z" clip-path="url(#p94a45562aa)" style="fill:#3f53c6"/>
                <path d="m290.513 164.321 2.66-1.313 1.46-3.958-2.662 1.32z" clip-path="url(#p94a45562aa)" style="fill:#c6d6f1"/>
                <path d="m197.263 211.194 2.553 1.808 1.552-.942-2.55-1.805z" clip-path="url(#p94a45562aa)" style="fill:#4961d2"/>
                <path d="m266.9 197.759 2.614-.725 1.456-2.776-2.614.728z" clip-path="url(#p94a45562aa)" style="fill:#6e90f2"/>
                <path d="m226.356 218.15 2.561.625 1.5-1.48-2.558-.622z" clip-path="url(#p94a45562aa)" style="fill:#3b4cc0"/>
                <path d="m253.258 208.908 2.592-.31 1.465-2.258-2.59.313z" clip-path="url(#p94a45562aa)" style="fill:#516ddb"/>
                <path d="m241.095 215.026 2.575.094 1.479-1.87-2.574-.091z" clip-path="url(#p94a45562aa)" style="fill:#4358cb"/>
                <path d="m182.906 200.977 2.561 2.468 1.58-.793-2.558-2.463z" clip-path="url(#p94a45562aa)" style="fill:#5f7fe8"/>
                <path d="m257.315 206.34 2.599-.447 1.462-2.388-2.598.45z" clip-path="url(#p94a45562aa)" style="fill:#5977e3"/>
                <path d="m237.039 216.538 2.571.228 1.485-1.74-2.57-.225z" clip-path="url(#p94a45562aa)" style="fill:#3f53c6"/>
                <path d="m299.023 146.387 2.678-1.48 1.47-4.494-2.68 1.488z" clip-path="url(#p94a45562aa)" style="fill:#ebd3c6"/>
                <path d="m185.467 203.445 2.56 2.336 1.575-.797-2.556-2.332z" clip-path="url(#p94a45562aa)" style="fill:#5a78e4"/>
                <path d="m276.502 187.451 2.634-1.007 1.454-3.169-2.634 1.012z" clip-path="url(#p94a45562aa)" style="fill:#8caffe"/>
                <path d="m289.056 168.142 2.659-1.308 1.458-3.826-2.66 1.313z" clip-path="url(#p94a45562aa)" style="fill:#bed2f6"/>
                <path d="m215.66 218.047 2.555 1.022 1.519-1.217-2.553-1.019z" clip-path="url(#p94a45562aa)" style="fill:#3b4cc0"/>
                <path d="m199.816 213.002 2.554 1.679 1.548-.946-2.55-1.675z" clip-path="url(#p94a45562aa)" style="fill:#455cce"/>
                <path d="m297.557 150.74 2.677-1.473 1.467-4.36-2.678 1.48z" clip-path="url(#p94a45562aa)" style="fill:#e4d9d2"/>
                <path d="m188.027 205.781 2.559 2.205 1.57-.802-2.554-2.2z" clip-path="url(#p94a45562aa)" style="fill:#5572df"/>
                <path d="m261.376 203.505 2.606-.583 1.46-2.518-2.606.587z" clip-path="url(#p94a45562aa)" style="fill:#6180e9"/>
                <path d="m232.98 217.788 2.568.361 1.491-1.61-2.566-.36z" clip-path="url(#p94a45562aa)" style="fill:#3d50c3"/>
                <path d="m270.97 194.258 2.624-.863 1.454-2.908-2.623.867z" clip-path="url(#p94a45562aa)" style="fill:#799cf8"/>
                <path d="m222.29 218.74 2.559.759 1.507-1.35-2.556-.755z" clip-path="url(#p94a45562aa)" style="fill:#3b4cc0"/>
                <path d="m282.044 179.978 2.646-1.154 1.454-3.43-2.646 1.158z" clip-path="url(#p94a45562aa)" style="fill:#a1c0ff"/>
                <path d="m209.021 216.695 2.554 1.284 1.531-1.083-2.55-1.282z" clip-path="url(#p94a45562aa)" style="fill:#3e51c5"/>
                <path d="m296.094 154.961 2.675-1.466 1.465-4.228-2.677 1.473z" clip-path="url(#p94a45562aa)" style="fill:#dcdddd"/>
                <path d="m190.586 207.986 2.557 2.074 1.567-.806-2.554-2.07z" clip-path="url(#p94a45562aa)" style="fill:#516ddb"/>
                <path d="m265.442 200.404 2.615-.721 1.457-2.649-2.615.725z" clip-path="url(#p94a45562aa)" style="fill:#6a8bef"/>
                <path d="m228.917 218.775 2.566.494 1.497-1.481-2.563-.492z" clip-path="url(#p94a45562aa)" style="fill:#3b4cc0"/>
                <path d="m287.6 171.833 2.658-1.303 1.457-3.696-2.66 1.308z" clip-path="url(#p94a45562aa)" style="fill:#b6cefa"/>
                <path d="m202.37 214.68 2.554 1.548 1.545-.948-2.551-1.545z" clip-path="url(#p94a45562aa)" style="fill:#4358cb"/>
                <path d="m247.73 213.21 2.586-.173 1.473-2-2.586.174z" clip-path="url(#p94a45562aa)" style="fill:#4a63d3"/>
                <path d="m251.789 211.036 2.593-.308 1.468-2.13-2.592.31z" clip-path="url(#p94a45562aa)" style="fill:#4f69d9"/>
                <path d="m243.67 215.12 2.582-.038 1.477-1.872-2.58.04z" clip-path="url(#p94a45562aa)" style="fill:#455cce"/>
                <path d="m294.633 159.05 2.674-1.46 1.462-4.095-2.675 1.466z" clip-path="url(#p94a45562aa)" style="fill:#d5dbe5"/>
                <path d="m193.143 210.06 2.557 1.943 1.563-.81-2.553-1.939z" clip-path="url(#p94a45562aa)" style="fill:#4e68d8"/>
                <path d="m275.048 190.487 2.634-1.003 1.454-3.04-2.634 1.007z" clip-path="url(#p94a45562aa)" style="fill:#85a8fc"/>
                <path d="m218.215 219.069 2.558.89 1.516-1.218-2.555-.889z" clip-path="url(#p94a45562aa)" style="fill:#3b4cc0"/>
                <path d="m280.59 183.275 2.645-1.149 1.455-3.302-2.646 1.154z" clip-path="url(#p94a45562aa)" style="fill:#98b9ff"/>
                <path d="m211.575 217.979 2.556 1.154 1.528-1.086-2.553-1.151z" clip-path="url(#p94a45562aa)" style="fill:#3d50c3"/>
                <path d="m255.85 208.598 2.6-.444 1.464-2.26-2.599.446z" clip-path="url(#p94a45562aa)" style="fill:#5572df"/>
                <path d="m239.61 216.766 2.578.096 1.482-1.742-2.575-.094z" clip-path="url(#p94a45562aa)" style="fill:#4257c9"/>
                <path d="m171.024 190.421 2.577 3.003 1.606-.643-2.573-2.997z" clip-path="url(#p94a45562aa)" style="fill:#7da0f9"/>
                <path d="m173.601 193.424 2.574 2.87 1.6-.648-2.568-2.865z" clip-path="url(#p94a45562aa)" style="fill:#7699f6"/>
                <path d="m168.443 187.284 2.58 3.137 1.611-.637-2.576-3.132z" clip-path="url(#p94a45562aa)" style="fill:#85a8fc"/>
                <path d="m176.175 196.294 2.57 2.737 1.597-.653-2.566-2.732z" clip-path="url(#p94a45562aa)" style="fill:#6e90f2"/>
                <path d="m269.514 197.034 2.624-.86 1.456-2.78-2.624.864z" clip-path="url(#p94a45562aa)" style="fill:#7396f5"/>
                <path d="m224.849 219.499 2.563.627 1.505-1.351-2.56-.625z" clip-path="url(#p94a45562aa)" style="fill:#3b4cc0"/>
                <path d="m286.144 175.393 2.658-1.297 1.456-3.566-2.658 1.303z" clip-path="url(#p94a45562aa)" style="fill:#aec9fc"/>
                <path d="m178.746 199.031 2.568 2.605 1.592-.659-2.564-2.6z" clip-path="url(#p94a45562aa)" style="fill:#688aef"/>
                <path d="m204.924 216.228 2.556 1.418 1.541-.951-2.552-1.415z" clip-path="url(#p94a45562aa)" style="fill:#4055c8"/>
                <path d="m259.914 205.893 2.607-.58 1.461-2.391-2.606.583z" clip-path="url(#p94a45562aa)" style="fill:#5d7ce6"/>
                <path d="m235.548 218.15 2.574.229 1.488-1.613-2.571-.228z" clip-path="url(#p94a45562aa)" style="fill:#3e51c5"/>
                <path d="m293.173 163.008 2.673-1.454 1.46-3.963-2.673 1.46z" clip-path="url(#p94a45562aa)" style="fill:#cdd9ec"/>
                <path d="m195.7 212.003 2.557 1.812 1.56-.813-2.554-1.808z" clip-path="url(#p94a45562aa)" style="fill:#4a63d3"/>
                <path d="m181.314 201.636 2.566 2.472 1.587-.663-2.561-2.468z" clip-path="url(#p94a45562aa)" style="fill:#6282ea"/>
                <path d="m263.982 202.922 2.616-.718 1.459-2.521-2.615.721z" clip-path="url(#p94a45562aa)" style="fill:#6687ed"/>
                <path d="m231.483 219.27 2.57.362 1.495-1.483-2.568-.361z" clip-path="url(#p94a45562aa)" style="fill:#3c4ec2"/>
                <path d="m301.701 144.906 2.692-1.621 1.471-4.502-2.693 1.63z" clip-path="url(#p94a45562aa)" style="fill:#efcfbf"/>
                <path d="m183.88 204.108 2.564 2.34 1.583-.667-2.56-2.336z" clip-path="url(#p94a45562aa)" style="fill:#5d7ce6"/>
                <path d="m279.136 186.444 2.645-1.145 1.454-3.173-2.645 1.15z" clip-path="url(#p94a45562aa)" style="fill:#92b4fe"/>
                <path d="m214.131 219.133 2.559 1.024 1.525-1.088-2.556-1.022z" clip-path="url(#p94a45562aa)" style="fill:#3c4ec2"/>
                <path d="m273.594 193.395 2.633-1 1.455-2.911-2.634 1.003z" clip-path="url(#p94a45562aa)" style="fill:#80a3fa"/>
                <path d="m220.773 219.96 2.563.76 1.513-1.221-2.56-.758z" clip-path="url(#p94a45562aa)" style="fill:#3b4cc0"/>
                <path d="m291.715 166.834 2.671-1.447 1.46-3.833-2.673 1.454z" clip-path="url(#p94a45562aa)" style="fill:#c5d6f2"/>
                <path d="m198.257 213.815 2.557 1.682 1.556-.816-2.554-1.679z" clip-path="url(#p94a45562aa)" style="fill:#465ecf"/>
                <path d="m284.69 178.824 2.657-1.292 1.455-3.436-2.658 1.297z" clip-path="url(#p94a45562aa)" style="fill:#a7c5fe"/>
                <path d="m207.48 217.646 2.557 1.287 1.538-.954-2.554-1.284z" clip-path="url(#p94a45562aa)" style="fill:#3f53c6"/>
                <path d="m300.234 149.267 2.69-1.614 1.469-4.368-2.692 1.621z" clip-path="url(#p94a45562aa)" style="fill:#e9d5cb"/>
                <path d="m186.444 206.449 2.563 2.209 1.579-.672-2.559-2.205z" clip-path="url(#p94a45562aa)" style="fill:#5875e1"/>
                <path d="m250.316 213.037 2.595-.305 1.47-2.004-2.592.308z" clip-path="url(#p94a45562aa)" style="fill:#4c66d6"/>
                <path d="m246.252 215.082 2.589-.17 1.475-1.875-2.587.173z" clip-path="url(#p94a45562aa)" style="fill:#485fd1"/>
                <path d="m268.057 199.683 2.624-.857 1.457-2.652-2.624.86z" clip-path="url(#p94a45562aa)" style="fill:#6f92f3"/>
                <path d="m227.412 220.126 2.568.497 1.503-1.354-2.566-.494z" clip-path="url(#p94a45562aa)" style="fill:#3b4cc0"/>
                <path d="m254.382 210.728 2.6-.441 1.468-2.133-2.6.444z" clip-path="url(#p94a45562aa)" style="fill:#536edd"/>
                <path d="m242.188 216.862 2.583-.036 1.481-1.744-2.582.038z" clip-path="url(#p94a45562aa)" style="fill:#445acc"/>
                <path d="m298.77 153.495 2.688-1.607 1.466-4.235-2.69 1.614z" clip-path="url(#p94a45562aa)" style="fill:#e2dad5"/>
                <path d="m189.007 208.658 2.561 2.078 1.575-.676-2.557-2.074z" clip-path="url(#p94a45562aa)" style="fill:#536edd"/>
                <path d="m290.258 170.53 2.67-1.441 1.458-3.702-2.671 1.447z" clip-path="url(#p94a45562aa)" style="fill:#bed2f6"/>
                <path d="m200.814 215.497 2.558 1.551 1.552-.82-2.554-1.547z" clip-path="url(#p94a45562aa)" style="fill:#445acc"/>
                <path d="m258.45 208.154 2.608-.579 1.463-2.263-2.607.581z" clip-path="url(#p94a45562aa)" style="fill:#5a78e4"/>
                <path d="m238.122 218.379 2.579.098 1.487-1.615-2.578-.096z" clip-path="url(#p94a45562aa)" style="fill:#4055c8"/>
                <path d="m277.682 189.484 2.645-1.141 1.454-3.044-2.645 1.145z" clip-path="url(#p94a45562aa)" style="fill:#8caffe"/>
                <path d="m216.69 220.157 2.561.893 1.522-1.09-2.558-.891z" clip-path="url(#p94a45562aa)" style="fill:#3b4cc0"/>
                <path d="m283.235 182.126 2.657-1.287 1.455-3.307-2.657 1.292z" clip-path="url(#p94a45562aa)" style="fill:#9fbfff"/>
                <path d="m210.037 218.933 2.56 1.157 1.534-.957-2.556-1.154z" clip-path="url(#p94a45562aa)" style="fill:#3d50c3"/>
                <path d="m297.307 157.59 2.686-1.6 1.465-4.102-2.689 1.607z" clip-path="url(#p94a45562aa)" style="fill:#dadce0"/>
                <path d="m191.568 210.736 2.561 1.946 1.571-.68-2.557-1.942z" clip-path="url(#p94a45562aa)" style="fill:#4f69d9"/>
                <path d="m272.138 196.174 2.634-.996 1.455-2.783-2.633 1z" clip-path="url(#p94a45562aa)" style="fill:#7a9df8"/>
                <path d="m223.336 220.72 2.566.63 1.51-1.224-2.563-.627z" clip-path="url(#p94a45562aa)" style="fill:#3b4cc0"/>
                <path d="m262.521 205.312 2.616-.715 1.46-2.393-2.615.718z" clip-path="url(#p94a45562aa)" style="fill:#6282ea"/>
                <path d="m234.053 219.632 2.575.232 1.494-1.485-2.574-.23z" clip-path="url(#p94a45562aa)" style="fill:#3e51c5"/>
                <path d="m169.404 190.928 2.582 3.008 1.615-.512-2.577-3.003z" clip-path="url(#p94a45562aa)" style="fill:#80a3fa"/>
                <path d="m171.986 193.936 2.579 2.876 1.61-.518-2.574-2.87z" clip-path="url(#p94a45562aa)" style="fill:#799cf8"/>
                <path d="m166.818 187.785 2.586 3.143 1.62-.507-2.581-3.137z" clip-path="url(#p94a45562aa)" style="fill:#88abfd"/>
                <path d="m174.565 196.812 2.575 2.742 1.606-.523-2.571-2.737z" clip-path="url(#p94a45562aa)" style="fill:#7093f3"/>
                <path d="m288.802 174.096 2.67-1.436 1.456-3.571-2.67 1.441z" clip-path="url(#p94a45562aa)" style="fill:#b6cefa"/>
                <path d="m203.372 217.048 2.56 1.42 1.548-.822-2.556-1.418z" clip-path="url(#p94a45562aa)" style="fill:#4257c9"/>
                <path d="m177.14 199.554 2.573 2.61 1.6-.528-2.567-2.605z" clip-path="url(#p94a45562aa)" style="fill:#6a8bef"/>
                <path d="m295.846 161.554 2.685-1.593 1.462-3.97-2.686 1.6z" clip-path="url(#p94a45562aa)" style="fill:#d3dbe7"/>
                <path d="m194.13 212.682 2.56 1.816 1.567-.683-2.557-1.812z" clip-path="url(#p94a45562aa)" style="fill:#4b64d5"/>
                <path d="m266.598 202.204 2.625-.854 1.458-2.524-2.624.857z" clip-path="url(#p94a45562aa)" style="fill:#6b8df0"/>
                <path d="m229.98 220.623 2.573.365 1.5-1.356-2.57-.363z" clip-path="url(#p94a45562aa)" style="fill:#3c4ec2"/>
                <path d="m179.713 202.163 2.57 2.477 1.597-.532-2.566-2.472z" clip-path="url(#p94a45562aa)" style="fill:#6485ec"/>
                <path d="m248.84 214.911 2.597-.303 1.474-1.876-2.595.305z" clip-path="url(#p94a45562aa)" style="fill:#4b64d5"/>
                <path d="m281.781 185.299 2.656-1.283 1.455-3.177-2.657 1.287z" clip-path="url(#p94a45562aa)" style="fill:#98b9ff"/>
                <path d="m212.597 220.09 2.561 1.027 1.532-.96-2.559-1.024z" clip-path="url(#p94a45562aa)" style="fill:#3d50c3"/>
                <path d="m276.227 192.395 2.645-1.137 1.455-2.915-2.645 1.14z" clip-path="url(#p94a45562aa)" style="fill:#85a8fc"/>
                <path d="m219.251 221.05 2.565.763 1.52-1.093-2.563-.76z" clip-path="url(#p94a45562aa)" style="fill:#3b4cc0"/>
                <path d="m252.91 212.732 2.603-.44 1.47-2.005-2.601.441z" clip-path="url(#p94a45562aa)" style="fill:#506bda"/>
                <path d="m244.771 216.826 2.59-.168 1.48-1.747-2.589.17z" clip-path="url(#p94a45562aa)" style="fill:#465ecf"/>
                <path d="m304.393 143.285 2.705-1.765 1.474-4.51-2.708 1.773z" clip-path="url(#p94a45562aa)" style="fill:#f2cab5"/>
                <path d="m182.283 204.64 2.569 2.345 1.592-.536-2.564-2.341z" clip-path="url(#p94a45562aa)" style="fill:#5f7fe8"/>
                <path d="m256.983 210.287 2.609-.576 1.466-2.136-2.608.579z" clip-path="url(#p94a45562aa)" style="fill:#5673e0"/>
                <path d="m240.7 218.477 2.586-.034 1.485-1.617-2.583.036z" clip-path="url(#p94a45562aa)" style="fill:#4358cb"/>
                <path d="m294.386 165.387 2.684-1.588 1.46-3.838-2.684 1.593z" clip-path="url(#p94a45562aa)" style="fill:#ccd9ed"/>
                <path d="m196.69 214.498 2.561 1.686 1.563-.687-2.557-1.682z" clip-path="url(#p94a45562aa)" style="fill:#4961d2"/>
                <path d="m287.347 177.532 2.668-1.43 1.456-3.442-2.67 1.436z" clip-path="url(#p94a45562aa)" style="fill:#aec9fc"/>
                <path d="m205.932 218.469 2.56 1.29 1.545-.826-2.557-1.287z" clip-path="url(#p94a45562aa)" style="fill:#4055c8"/>
                <path d="m270.681 198.826 2.634-.993 1.457-2.655-2.634.996z" clip-path="url(#p94a45562aa)" style="fill:#7597f6"/>
                <path d="m225.902 221.35 2.57.498 1.508-1.225-2.568-.497z" clip-path="url(#p94a45562aa)" style="fill:#3c4ec2"/>
                <path d="m302.924 147.653 2.703-1.757 1.471-4.376-2.705 1.765z" clip-path="url(#p94a45562aa)" style="fill:#edd1c2"/>
                <path d="m184.852 206.985 2.567 2.214 1.588-.541-2.563-2.21z" clip-path="url(#p94a45562aa)" style="fill:#5a78e4"/>
                <path d="m261.058 207.575 2.617-.712 1.462-2.266-2.616.715z" clip-path="url(#p94a45562aa)" style="fill:#5e7de7"/>
                <path d="m236.628 219.864 2.582.1 1.49-1.487-2.578-.098z" clip-path="url(#p94a45562aa)" style="fill:#3f53c6"/>
                <path d="m301.458 151.888 2.701-1.75 1.468-4.242-2.703 1.757z" clip-path="url(#p94a45562aa)" style="fill:#e8d6cc"/>
                <path d="m187.419 209.199 2.566 2.082 1.583-.545-2.561-2.078z" clip-path="url(#p94a45562aa)" style="fill:#5572df"/>
                <path d="m292.928 169.089 2.683-1.582 1.46-3.708-2.685 1.588z" clip-path="url(#p94a45562aa)" style="fill:#c4d5f3"/>
                <path d="m199.251 216.184 2.562 1.554 1.56-.69-2.559-1.551z" clip-path="url(#p94a45562aa)" style="fill:#465ecf"/>
                <path d="m280.327 188.343 2.655-1.279 1.455-3.048-2.656 1.283z" clip-path="url(#p94a45562aa)" style="fill:#93b5fe"/>
                <path d="m215.158 221.117 2.565.895 1.528-.962-2.561-.893z" clip-path="url(#p94a45562aa)" style="fill:#3c4ec2"/>
                <path d="m265.137 204.597 2.626-.85 1.46-2.397-2.625.854z" clip-path="url(#p94a45562aa)" style="fill:#6788ee"/>
                <path d="m232.553 220.988 2.578.234 1.497-1.358-2.575-.232z" clip-path="url(#p94a45562aa)" style="fill:#3e51c5"/>
                <path d="m274.772 195.178 2.645-1.133 1.455-2.787-2.645 1.137z" clip-path="url(#p94a45562aa)" style="fill:#81a4fb"/>
                <path d="m221.816 221.813 2.57.632 1.516-1.095-2.566-.63z" clip-path="url(#p94a45562aa)" style="fill:#3b4cc0"/>
                <path d="m285.892 180.839 2.668-1.426 1.455-3.312-2.668 1.431z" clip-path="url(#p94a45562aa)" style="fill:#a7c5fe"/>
                <path d="m208.492 219.759 2.563 1.16 1.542-.829-2.56-1.157z" clip-path="url(#p94a45562aa)" style="fill:#3f53c6"/>
                <path d="m299.993 155.99 2.7-1.742 1.466-4.11-2.701 1.75z" clip-path="url(#p94a45562aa)" style="fill:#e0dbd8"/>
                <path d="m189.985 211.281 2.565 1.95 1.58-.549-2.562-1.946z" clip-path="url(#p94a45562aa)" style="fill:#516ddb"/>
                <path d="m269.223 201.35 2.635-.99 1.457-2.527-2.634.993z" clip-path="url(#p94a45562aa)" style="fill:#7093f3"/>
                <path d="m228.472 221.848 2.575.367 1.506-1.227-2.573-.365z" clip-path="url(#p94a45562aa)" style="fill:#3c4ec2"/>
                <path d="m251.437 214.608 2.603-.438 1.473-1.878-2.602.44z" clip-path="url(#p94a45562aa)" style="fill:#4f69d9"/>
                <path d="m247.362 216.658 2.597-.302 1.478-1.748-2.596.303z" clip-path="url(#p94a45562aa)" style="fill:#4a63d3"/>
                <path d="m291.471 172.66 2.682-1.576 1.458-3.577-2.683 1.582z" clip-path="url(#p94a45562aa)" style="fill:#bcd2f7"/>
                <path d="m201.813 217.738 2.563 1.424 1.556-.693-2.56-1.421z" clip-path="url(#p94a45562aa)" style="fill:#445acc"/>
                <path d="m167.774 191.302 2.587 3.014 1.625-.38-2.582-3.008z" clip-path="url(#p94a45562aa)" style="fill:#82a6fb"/>
                <path d="m170.361 194.316 2.584 2.881 1.62-.385-2.579-2.876z" clip-path="url(#p94a45562aa)" style="fill:#7b9ff9"/>
                <path d="m165.183 188.153 2.591 3.149 1.63-.374-2.586-3.143z" clip-path="url(#p94a45562aa)" style="fill:#8badfd"/>
                <path d="m255.513 212.292 2.61-.573 1.469-2.008-2.61.576z" clip-path="url(#p94a45562aa)" style="fill:#5572df"/>
                <path d="m243.286 218.443 2.593-.167 1.483-1.618-2.59.168z" clip-path="url(#p94a45562aa)" style="fill:#455cce"/>
                <path d="m172.945 197.197 2.58 2.748 1.615-.391-2.575-2.742z" clip-path="url(#p94a45562aa)" style="fill:#7597f6"/>
                <path d="m298.53 159.96 2.7-1.735 1.463-3.977-2.7 1.742z" clip-path="url(#p94a45562aa)" style="fill:#d9dce1"/>
                <path d="m192.55 213.232 2.565 1.82 1.575-.554-2.56-1.816z" clip-path="url(#p94a45562aa)" style="fill:#4e68d8"/>
                <path d="m175.525 199.945 2.578 2.614 1.61-.396-2.573-2.61z" clip-path="url(#p94a45562aa)" style="fill:#6e90f2"/>
                <path d="m259.592 209.711 2.618-.71 1.465-2.138-2.617.712z" clip-path="url(#p94a45562aa)" style="fill:#5b7ae5"/>
                <path d="m239.21 219.964 2.587-.032 1.49-1.489-2.586.034z" clip-path="url(#p94a45562aa)" style="fill:#4257c9"/>
                <path d="m278.872 191.258 2.656-1.274 1.454-2.92-2.655 1.279z" clip-path="url(#p94a45562aa)" style="fill:#8caffe"/>
                <path d="m217.723 222.012 2.568.765 1.525-.964-2.565-.763z" clip-path="url(#p94a45562aa)" style="fill:#3c4ec2"/>
                <path d="m284.437 184.016 2.668-1.421 1.455-3.182-2.668 1.426z" clip-path="url(#p94a45562aa)" style="fill:#a1c0ff"/>
                <path d="m211.055 220.918 2.565 1.03 1.538-.831-2.561-1.027z" clip-path="url(#p94a45562aa)" style="fill:#3e51c5"/>
                <path d="m178.103 202.56 2.575 2.481 1.605-.4-2.57-2.478z" clip-path="url(#p94a45562aa)" style="fill:#6788ee"/>
                <path d="m273.315 197.833 2.645-1.13 1.457-2.658-2.645 1.133z" clip-path="url(#p94a45562aa)" style="fill:#7b9ff9"/>
                <path d="m224.385 222.445 2.573.5 1.514-1.097-2.57-.498z" clip-path="url(#p94a45562aa)" style="fill:#3c4ec2"/>
                <path d="m307.098 141.52 2.72-1.907 1.475-4.518-2.721 1.916z" clip-path="url(#p94a45562aa)" style="fill:#f5c2aa"/>
                <path d="m263.675 206.863 2.626-.849 1.462-2.268-2.626.851z" clip-path="url(#p94a45562aa)" style="fill:#6384eb"/>
                <path d="m235.13 221.222 2.584.101 1.496-1.359-2.582-.1z" clip-path="url(#p94a45562aa)" style="fill:#3f53c6"/>
                <path d="m180.678 205.041 2.573 2.35 1.6-.406-2.568-2.345z" clip-path="url(#p94a45562aa)" style="fill:#6282ea"/>
                <path d="m290.015 176.101 2.681-1.57 1.457-3.447-2.682 1.576z" clip-path="url(#p94a45562aa)" style="fill:#b6cefa"/>
                <path d="m204.376 219.162 2.564 1.293 1.552-.696-2.56-1.29z" clip-path="url(#p94a45562aa)" style="fill:#4257c9"/>
                <path d="m297.07 163.8 2.697-1.73 1.462-3.845-2.698 1.736z" clip-path="url(#p94a45562aa)" style="fill:#d3dbe7"/>
                <path d="m195.115 215.051 2.565 1.689 1.571-.556-2.56-1.686z" clip-path="url(#p94a45562aa)" style="fill:#4b64d5"/>
                <path d="m305.627 145.896 2.717-1.9 1.473-4.383-2.719 1.907z" clip-path="url(#p94a45562aa)" style="fill:#f2cbb7"/>
                <path d="m183.25 207.391 2.572 2.218 1.597-.41-2.567-2.214z" clip-path="url(#p94a45562aa)" style="fill:#5d7ce6"/>
                <path d="m267.763 203.746 2.636-.987 1.459-2.399-2.635.99z" clip-path="url(#p94a45562aa)" style="fill:#6c8ff1"/>
                <path d="m231.047 222.215 2.58.236 1.504-1.23-2.578-.233z" clip-path="url(#p94a45562aa)" style="fill:#3e51c5"/>
                <path d="m282.982 187.064 2.668-1.417 1.455-3.052-2.668 1.421z" clip-path="url(#p94a45562aa)" style="fill:#9abbff"/>
                <path d="m213.62 221.947 2.568.898 1.535-.833-2.565-.895z" clip-path="url(#p94a45562aa)" style="fill:#3d50c3"/>
                <path d="m295.611 167.507 2.696-1.723 1.46-3.714-2.697 1.73z" clip-path="url(#p94a45562aa)" style="fill:#cbd8ee"/>
                <path d="m304.159 150.138 2.715-1.892 1.47-4.25-2.717 1.9z" clip-path="url(#p94a45562aa)" style="fill:#edd2c3"/>
                <path d="m277.417 194.045 2.655-1.271 1.456-2.79-2.656 1.274z" clip-path="url(#p94a45562aa)" style="fill:#88abfd"/>
                <path d="m197.68 216.74 2.566 1.558 1.567-.56-2.562-1.554z" clip-path="url(#p94a45562aa)" style="fill:#4961d2"/>
                <path d="m220.291 222.777 2.572.634 1.522-.966-2.569-.632z" clip-path="url(#p94a45562aa)" style="fill:#3c4ec2"/>
                <path d="m185.822 209.61 2.57 2.085 1.593-.414-2.566-2.082z" clip-path="url(#p94a45562aa)" style="fill:#5977e3"/>
                <path d="m249.96 216.356 2.605-.436 1.475-1.75-2.603.438z" clip-path="url(#p94a45562aa)" style="fill:#4c66d6"/>
                <path d="m288.56 179.413 2.68-1.566 1.456-3.317-2.68 1.571z" clip-path="url(#p94a45562aa)" style="fill:#aec9fc"/>
                <path d="m206.94 220.455 2.567 1.162 1.548-.699-2.563-1.16z" clip-path="url(#p94a45562aa)" style="fill:#4055c8"/>
                <path d="m254.04 214.17 2.612-.571 1.472-1.88-2.61.573z" clip-path="url(#p94a45562aa)" style="fill:#536edd"/>
                <path d="m245.879 218.276 2.599-.3 1.481-1.62-2.597.302z" clip-path="url(#p94a45562aa)" style="fill:#4961d2"/>
                <path d="m258.124 211.719 2.619-.709 1.467-2.01-2.618.711z" clip-path="url(#p94a45562aa)" style="fill:#5977e3"/>
                <path d="m241.797 219.932 2.595-.165 1.487-1.49-2.593.166z" clip-path="url(#p94a45562aa)" style="fill:#455cce"/>
                <path d="m271.858 200.36 2.645-1.127 1.457-2.53-2.645 1.13z" clip-path="url(#p94a45562aa)" style="fill:#779af7"/>
                <path d="m226.958 222.945 2.578.37 1.511-1.1-2.575-.367z" clip-path="url(#p94a45562aa)" style="fill:#3d50c3"/>
                <path d="m302.693 154.248 2.714-1.885 1.467-4.117-2.715 1.892z" clip-path="url(#p94a45562aa)" style="fill:#e7d7ce"/>
                <path d="m188.393 211.695 2.569 1.955 1.588-.418-2.565-1.951z" clip-path="url(#p94a45562aa)" style="fill:#5470de"/>
                <path d="m262.21 209 2.628-.846 1.463-2.14-2.626.849z" clip-path="url(#p94a45562aa)" style="fill:#6180e9"/>
                <path d="m237.714 221.323 2.59-.03 1.493-1.361-2.587.032z" clip-path="url(#p94a45562aa)" style="fill:#4257c9"/>
                <path d="m294.153 171.084 2.695-1.718 1.459-3.582-2.696 1.723z" clip-path="url(#p94a45562aa)" style="fill:#c4d5f3"/>
                <path d="m200.246 218.298 2.566 1.426 1.564-.562-2.563-1.424z" clip-path="url(#p94a45562aa)" style="fill:#465ecf"/>
                <path d="m166.134 191.543 2.592 3.02 1.635-.247-2.587-3.014z" clip-path="url(#p94a45562aa)" style="fill:#86a9fc"/>
                <path d="m168.726 194.564 2.589 2.886 1.63-.253-2.584-2.88z" clip-path="url(#p94a45562aa)" style="fill:#7ea1fa"/>
                <path d="m163.537 188.388 2.597 3.155 1.64-.241-2.591-3.149z" clip-path="url(#p94a45562aa)" style="fill:#8fb1fe"/>
                <path d="m281.528 189.984 2.667-1.414 1.455-2.923-2.668 1.417z" clip-path="url(#p94a45562aa)" style="fill:#94b6ff"/>
                <path d="m171.315 197.45 2.585 2.753 1.625-.258-2.58-2.748z" clip-path="url(#p94a45562aa)" style="fill:#779af7"/>
                <path d="m216.188 222.845 2.572.767 1.531-.835-2.568-.765z" clip-path="url(#p94a45562aa)" style="fill:#3d50c3"/>
                <path d="m301.23 158.225 2.711-1.879 1.466-3.983-2.714 1.885z" clip-path="url(#p94a45562aa)" style="fill:#e0dbd8"/>
                <path d="m190.962 213.65 2.57 1.823 1.583-.422-2.565-1.82z" clip-path="url(#p94a45562aa)" style="fill:#516ddb"/>
                <path d="m287.105 182.595 2.68-1.562 1.455-3.186-2.68 1.566z" clip-path="url(#p94a45562aa)" style="fill:#a7c5fe"/>
                <path d="m209.507 221.617 2.568 1.031 1.545-.7-2.565-1.03z" clip-path="url(#p94a45562aa)" style="fill:#3f53c6"/>
                <path d="m173.9 200.203 2.583 2.62 1.62-.264-2.578-2.614z" clip-path="url(#p94a45562aa)" style="fill:#7093f3"/>
                <path d="m266.301 206.014 2.637-.984 1.46-2.27-2.635.986z" clip-path="url(#p94a45562aa)" style="fill:#6a8bef"/>
                <path d="m233.628 222.45 2.586.104 1.5-1.23-2.583-.102z" clip-path="url(#p94a45562aa)" style="fill:#3f53c6"/>
                <path d="m275.96 196.703 2.656-1.267 1.456-2.662-2.655 1.271z" clip-path="url(#p94a45562aa)" style="fill:#82a6fb"/>
                <path d="m222.863 223.41 2.576.503 1.52-.968-2.574-.5z" clip-path="url(#p94a45562aa)" style="fill:#3d50c3"/>
                <path d="m176.483 202.823 2.58 2.487 1.615-.269-2.575-2.482z" clip-path="url(#p94a45562aa)" style="fill:#6b8df0"/>
                <path d="m292.696 174.53 2.694-1.712 1.458-3.452-2.695 1.718z" clip-path="url(#p94a45562aa)" style="fill:#bed2f6"/>
                <path d="m202.812 219.724 2.569 1.296 1.56-.565-2.565-1.293z" clip-path="url(#p94a45562aa)" style="fill:#445acc"/>
                <path d="m309.817 139.613 2.734-2.053 1.478-4.527-2.736 2.062z" clip-path="url(#p94a45562aa)" style="fill:#f7ba9f"/>
                <path d="m299.767 162.07 2.711-1.872 1.463-3.852-2.712 1.879z" clip-path="url(#p94a45562aa)" style="fill:#d9dce1"/>
                <path d="m179.062 205.31 2.578 2.355 1.61-.274-2.572-2.35z" clip-path="url(#p94a45562aa)" style="fill:#6687ed"/>
                <path d="m193.531 215.473 2.57 1.693 1.579-.426-2.565-1.689z" clip-path="url(#p94a45562aa)" style="fill:#4e68d8"/>
                <path d="m270.399 202.76 2.645-1.125 1.459-2.402-2.645 1.127z" clip-path="url(#p94a45562aa)" style="fill:#7396f5"/>
                <path d="m229.536 223.314 2.583.237 1.509-1.1-2.58-.236z" clip-path="url(#p94a45562aa)" style="fill:#3e51c5"/>
                <path d="m252.565 215.92 2.613-.57 1.474-1.751-2.612.571z" clip-path="url(#p94a45562aa)" style="fill:#516ddb"/>
                <path d="m248.478 217.976 2.607-.434 1.48-1.622-2.606.436z" clip-path="url(#p94a45562aa)" style="fill:#4c66d6"/>
                <path d="m308.344 143.996 2.732-2.044 1.475-4.392-2.734 2.053z" clip-path="url(#p94a45562aa)" style="fill:#f5c4ac"/>
                <path d="m256.652 213.599 2.621-.707 1.47-1.882-2.62.709z" clip-path="url(#p94a45562aa)" style="fill:#5875e1"/>
                <path d="m244.392 219.767 2.601-.299 1.485-1.492-2.6.3z" clip-path="url(#p94a45562aa)" style="fill:#485fd1"/>
                <path d="m181.64 207.665 2.576 2.222 1.606-.278-2.571-2.218z" clip-path="url(#p94a45562aa)" style="fill:#6180e9"/>
                <path d="m285.65 185.647 2.68-1.558 1.455-3.056-2.68 1.562z" clip-path="url(#p94a45562aa)" style="fill:#a2c1ff"/>
                <path d="m212.075 222.648 2.572.9 1.541-.703-2.568-.898z" clip-path="url(#p94a45562aa)" style="fill:#3f53c6"/>
                <path d="m280.072 192.774 2.667-1.41 1.456-2.794-2.667 1.414z" clip-path="url(#p94a45562aa)" style="fill:#8fb1fe"/>
                <path d="m218.76 223.612 2.574.636 1.529-.837-2.572-.634z" clip-path="url(#p94a45562aa)" style="fill:#3e51c5"/>
                <path d="m260.743 211.01 2.628-.844 1.467-2.012-2.628.846z" clip-path="url(#p94a45562aa)" style="fill:#5e7de7"/>
                <path d="m240.304 221.293 2.596-.164 1.492-1.362-2.595.165z" clip-path="url(#p94a45562aa)" style="fill:#445acc"/>
                <path d="m298.307 165.784 2.71-1.866 1.461-3.72-2.71 1.872z" clip-path="url(#p94a45562aa)" style="fill:#d3dbe7"/>
                <path d="m196.1 217.166 2.57 1.56 1.576-.428-2.566-1.558z" clip-path="url(#p94a45562aa)" style="fill:#4b64d5"/>
                <path d="m306.874 148.246 2.73-2.037 1.472-4.257-2.732 2.044z" clip-path="url(#p94a45562aa)" style="fill:#f1ccb8"/>
                <path d="m184.216 209.887 2.575 2.09 1.602-.282-2.57-2.086z" clip-path="url(#p94a45562aa)" style="fill:#5b7ae5"/>
                <path d="m291.24 177.847 2.693-1.708 1.457-3.321-2.694 1.712z" clip-path="url(#p94a45562aa)" style="fill:#b6cefa"/>
                <path d="m205.38 221.02 2.57 1.165 1.557-.568-2.567-1.162z" clip-path="url(#p94a45562aa)" style="fill:#4358cb"/>
                <path d="m274.503 199.233 2.656-1.264 1.457-2.533-2.656 1.267z" clip-path="url(#p94a45562aa)" style="fill:#7ea1fa"/>
                <path d="m225.44 223.913 2.58.371 1.516-.97-2.578-.369z" clip-path="url(#p94a45562aa)" style="fill:#3e51c5"/>
                <path d="m264.838 208.154 2.637-.982 1.463-2.142-2.637.984z" clip-path="url(#p94a45562aa)" style="fill:#6687ed"/>
                <path d="m236.214 222.554 2.592-.029 1.498-1.232-2.59.03z" clip-path="url(#p94a45562aa)" style="fill:#4257c9"/>
                <path d="m305.407 152.363 2.728-2.03 1.469-4.124-2.73 2.037z" clip-path="url(#p94a45562aa)" style="fill:#edd2c3"/>
                <path d="m186.791 211.978 2.574 1.959 1.597-.287-2.57-1.955z" clip-path="url(#p94a45562aa)" style="fill:#5875e1"/>
                <path d="m296.848 169.366 2.708-1.86 1.46-3.588-2.709 1.866z" clip-path="url(#p94a45562aa)" style="fill:#ccd9ed"/>
                <path d="m198.67 218.726 2.571 1.43 1.571-.432-2.566-1.426z" clip-path="url(#p94a45562aa)" style="fill:#4961d2"/>
                <path d="m268.938 205.03 2.646-1.122 1.46-2.273-2.645 1.124z" clip-path="url(#p94a45562aa)" style="fill:#6f92f3"/>
                <path d="m232.12 223.551 2.588.105 1.506-1.102-2.586-.103z" clip-path="url(#p94a45562aa)" style="fill:#4055c8"/>
                <path d="m284.195 188.57 2.68-1.553 1.454-2.928-2.68 1.558z" clip-path="url(#p94a45562aa)" style="fill:#9dbdff"/>
                <path d="m214.647 223.548 2.574.77 1.539-.706-2.572-.767z" clip-path="url(#p94a45562aa)" style="fill:#3f53c6"/>
                <path d="m278.616 195.436 2.668-1.407 1.455-2.665-2.667 1.41z" clip-path="url(#p94a45562aa)" style="fill:#89acfd"/>
                <path d="m164.483 191.651 2.598 3.027 1.645-.114-2.592-3.02z" clip-path="url(#p94a45562aa)" style="fill:#8badfd"/>
                <path d="m221.334 224.248 2.58.504 1.525-.839-2.576-.502z" clip-path="url(#p94a45562aa)" style="fill:#3e51c5"/>
                <path d="m289.785 181.033 2.692-1.703 1.456-3.19-2.693 1.707z" clip-path="url(#p94a45562aa)" style="fill:#b1cbfc"/>
                <path d="m167.08 194.678 2.595 2.892 1.64-.12-2.589-2.886z" clip-path="url(#p94a45562aa)" style="fill:#82a6fb"/>
                <path d="m207.95 222.185 2.573 1.034 1.552-.57-2.568-1.032z" clip-path="url(#p94a45562aa)" style="fill:#4257c9"/>
                <path d="m161.88 188.49 2.603 3.161 1.65-.108-2.596-3.155z" clip-path="url(#p94a45562aa)" style="fill:#93b5fe"/>
                <path d="m303.941 156.346 2.727-2.022 1.467-3.99-2.728 2.029z" clip-path="url(#p94a45562aa)" style="fill:#e7d7ce"/>
                <path d="m189.365 213.937 2.574 1.827 1.592-.29-2.569-1.824z" clip-path="url(#p94a45562aa)" style="fill:#5470de"/>
                <path d="m169.675 197.57 2.59 2.76 1.635-.127-2.585-2.753z" clip-path="url(#p94a45562aa)" style="fill:#7b9ff9"/>
                <path d="m251.085 217.542 2.615-.568 1.478-1.623-2.613.57z" clip-path="url(#p94a45562aa)" style="fill:#506bda"/>
                <path d="m172.265 200.33 2.588 2.624 1.63-.13-2.583-2.62z" clip-path="url(#p94a45562aa)" style="fill:#7597f6"/>
                <path d="m255.178 215.35 2.622-.704 1.473-1.754-2.62.707z" clip-path="url(#p94a45562aa)" style="fill:#5673e0"/>
                <path d="m246.993 219.468 2.609-.432 1.483-1.494-2.607.434z" clip-path="url(#p94a45562aa)" style="fill:#4b64d5"/>
                <path d="m174.853 202.954 2.584 2.493 1.625-.137-2.58-2.487z" clip-path="url(#p94a45562aa)" style="fill:#6f92f3"/>
                <path d="m273.044 201.635 2.657-1.262 1.458-2.404-2.656 1.264z" clip-path="url(#p94a45562aa)" style="fill:#7a9df8"/>
                <path d="m228.02 224.284 2.586.239 1.513-.972-2.583-.237z" clip-path="url(#p94a45562aa)" style="fill:#3f53c6"/>
                <path d="m259.273 212.892 2.63-.842 1.468-1.884-2.628.844z" clip-path="url(#p94a45562aa)" style="fill:#5d7ce6"/>
                <path d="m242.9 221.13 2.604-.298 1.489-1.364-2.601.299z" clip-path="url(#p94a45562aa)" style="fill:#485fd1"/>
                <path d="m295.39 172.818 2.708-1.855 1.458-3.457-2.708 1.86z" clip-path="url(#p94a45562aa)" style="fill:#c5d6f2"/>
                <path d="m201.241 220.156 2.572 1.299 1.568-.435-2.569-1.296z" clip-path="url(#p94a45562aa)" style="fill:#485fd1"/>
                <path d="m302.478 160.198 2.725-2.016 1.465-3.858-2.727 2.022z" clip-path="url(#p94a45562aa)" style="fill:#e0dbd8"/>
                <path d="m191.939 215.764 2.573 1.696 1.589-.294-2.57-1.693z" clip-path="url(#p94a45562aa)" style="fill:#516ddb"/>
                <path d="m312.55 137.56 2.75-2.199 1.48-4.535-2.751 2.207z" clip-path="url(#p94a45562aa)" style="fill:#f7b194"/>
                <path d="m177.437 205.447 2.583 2.36 1.62-.142-2.578-2.355z" clip-path="url(#p94a45562aa)" style="fill:#688aef"/>
                <path d="m263.371 210.166 2.639-.98 1.465-2.014-2.637.982z" clip-path="url(#p94a45562aa)" style="fill:#6485ec"/>
                <path d="m238.806 222.525 2.598-.162 1.496-1.234-2.596.164z" clip-path="url(#p94a45562aa)" style="fill:#455cce"/>
                <path d="m282.74 191.364 2.679-1.55 1.455-2.797-2.68 1.553z" clip-path="url(#p94a45562aa)" style="fill:#97b8ff"/>
                <path d="m217.221 224.317 2.578.638 1.535-.707-2.574-.636z" clip-path="url(#p94a45562aa)" style="fill:#3f53c6"/>
                <path d="m288.33 184.09 2.692-1.699 1.455-3.06-2.692 1.702z" clip-path="url(#p94a45562aa)" style="fill:#aac7fd"/>
                <path d="m210.523 223.219 2.575.902 1.549-.573-2.572-.9z" clip-path="url(#p94a45562aa)" style="fill:#4257c9"/>
                <path d="m311.076 141.952 2.747-2.19 1.477-4.4-2.75 2.198z" clip-path="url(#p94a45562aa)" style="fill:#f7bca1"/>
                <path d="m180.02 207.806 2.58 2.227 1.616-.146-2.576-2.222z" clip-path="url(#p94a45562aa)" style="fill:#6485ec"/>
                <path d="m277.16 197.969 2.667-1.404 1.457-2.536-2.668 1.407z" clip-path="url(#p94a45562aa)" style="fill:#85a8fc"/>
                <path d="m223.914 224.752 2.583.372 1.523-.84-2.58-.37z" clip-path="url(#p94a45562aa)" style="fill:#3f53c6"/>
                <path d="m267.475 207.172 2.647-1.12 1.462-2.144-2.646 1.122z" clip-path="url(#p94a45562aa)" style="fill:#6c8ff1"/>
                <path d="m234.708 223.656 2.595-.028 1.503-1.103-2.592.03z" clip-path="url(#p94a45562aa)" style="fill:#4358cb"/>
                <path d="m301.016 163.918 2.724-2.01 1.463-3.726-2.725 2.016z" clip-path="url(#p94a45562aa)" style="fill:#dadce0"/>
                <path d="m194.512 217.46 2.574 1.564 1.584-.298-2.57-1.56z" clip-path="url(#p94a45562aa)" style="fill:#4e68d8"/>
                <path d="m293.933 176.14 2.707-1.85 1.458-3.327-2.708 1.855z" clip-path="url(#p94a45562aa)" style="fill:#bfd3f6"/>
                <path d="m203.813 221.455 2.574 1.167 1.564-.437-2.57-1.165z" clip-path="url(#p94a45562aa)" style="fill:#455cce"/>
                <path d="m309.604 146.21 2.745-2.183 1.474-4.265-2.747 2.19z" clip-path="url(#p94a45562aa)" style="fill:#f4c5ad"/>
                <path d="m182.6 210.033 2.58 2.095 1.611-.15-2.575-2.09z" clip-path="url(#p94a45562aa)" style="fill:#5f7fe8"/>
                <path d="m271.584 203.908 2.658-1.26 1.46-2.275-2.658 1.262z" clip-path="url(#p94a45562aa)" style="fill:#7699f6"/>
                <path d="m230.606 224.523 2.59.106 1.512-.973-2.589-.105z" clip-path="url(#p94a45562aa)" style="fill:#4257c9"/>
                <path d="m308.135 150.333 2.742-2.174 1.472-4.132-2.745 2.182z" clip-path="url(#p94a45562aa)" style="fill:#f1ccb8"/>
                <path d="m185.18 212.128 2.58 1.963 1.605-.154-2.574-1.96z" clip-path="url(#p94a45562aa)" style="fill:#5b7ae5"/>
                <path d="m253.7 216.974 2.624-.703 1.476-1.625-2.622.705z" clip-path="url(#p94a45562aa)" style="fill:#5572df"/>
                <path d="m249.602 219.036 2.617-.567 1.481-1.495-2.615.568z" clip-path="url(#p94a45562aa)" style="fill:#506bda"/>
                <path d="m286.874 187.017 2.692-1.695 1.456-2.93-2.693 1.697z" clip-path="url(#p94a45562aa)" style="fill:#a5c3fe"/>
                <path d="m299.556 167.506 2.723-2.003 1.46-3.594-2.723 2.009z" clip-path="url(#p94a45562aa)" style="fill:#d4dbe6"/>
                <path d="m213.098 224.121 2.578.771 1.545-.575-2.574-.769z" clip-path="url(#p94a45562aa)" style="fill:#4257c9"/>
                <path d="m197.086 219.024 2.575 1.433 1.58-.3-2.57-1.43z" clip-path="url(#p94a45562aa)" style="fill:#4c66d6"/>
                <path d="m257.8 214.646 2.631-.84 1.472-1.756-2.63.842z" clip-path="url(#p94a45562aa)" style="fill:#5b7ae5"/>
                <path d="m245.504 220.832 2.61-.431 1.488-1.365-2.609.432z" clip-path="url(#p94a45562aa)" style="fill:#4b64d5"/>
                <path d="m281.284 194.029 2.679-1.547 1.456-2.668-2.68 1.55z" clip-path="url(#p94a45562aa)" style="fill:#92b4fe"/>
                <path d="m219.8 224.955 2.582.506 1.532-.709-2.58-.504z" clip-path="url(#p94a45562aa)" style="fill:#4055c8"/>
                <path d="m292.477 179.33 2.706-1.845 1.457-3.195-2.707 1.85z" clip-path="url(#p94a45562aa)" style="fill:#b9d0f9"/>
                <path d="m206.387 222.622 2.576 1.036 1.56-.44-2.572-1.033z" clip-path="url(#p94a45562aa)" style="fill:#455cce"/>
                <path d="m261.903 212.05 2.64-.979 1.467-1.885-2.639.98z" clip-path="url(#p94a45562aa)" style="fill:#6282ea"/>
                <path d="m241.404 222.363 2.606-.296 1.494-1.235-2.604.297z" clip-path="url(#p94a45562aa)" style="fill:#485fd1"/>
                <path d="m306.668 154.324 2.74-2.167 1.47-3.998-2.743 2.174z" clip-path="url(#p94a45562aa)" style="fill:#edd2c3"/>
                <path d="m162.82 191.625 2.604 3.033 1.657.02-2.598-3.027z" clip-path="url(#p94a45562aa)" style="fill:#8fb1fe"/>
                <path d="m165.424 194.658 2.6 2.898 1.65.014-2.593-2.892z" clip-path="url(#p94a45562aa)" style="fill:#88abfd"/>
                <path d="m187.76 214.09 2.577 1.832 1.602-.158-2.574-1.827z" clip-path="url(#p94a45562aa)" style="fill:#5875e1"/>
                <path d="m160.213 188.456 2.608 3.169 1.662.026-2.602-3.161z" clip-path="url(#p94a45562aa)" style="fill:#97b8ff"/>
                <path d="m275.701 200.373 2.668-1.401 1.458-2.407-2.668 1.404z" clip-path="url(#p94a45562aa)" style="fill:#81a4fb"/>
                <path d="m226.497 225.124 2.589.24 1.52-.841-2.586-.239z" clip-path="url(#p94a45562aa)" style="fill:#4055c8"/>
                <path d="m168.024 197.556 2.595 2.765 1.646.008-2.59-2.759z" clip-path="url(#p94a45562aa)" style="fill:#80a3fa"/>
                <path d="m170.62 200.321 2.592 2.63 1.64.003-2.587-2.625z" clip-path="url(#p94a45562aa)" style="fill:#7a9df8"/>
                <path d="m266.01 209.186 2.648-1.118 1.464-2.015-2.647 1.12z" clip-path="url(#p94a45562aa)" style="fill:#6a8bef"/>
                <path d="m237.303 223.628 2.6-.16 1.501-1.105-2.598.162z" clip-path="url(#p94a45562aa)" style="fill:#455cce"/>
                <path d="m298.098 170.963 2.721-1.998 1.46-3.462-2.723 2.003z" clip-path="url(#p94a45562aa)" style="fill:#cdd9ec"/>
                <path d="m199.661 220.457 2.576 1.301 1.576-.303-2.572-1.299z" clip-path="url(#p94a45562aa)" style="fill:#4a63d3"/>
                <path d="m173.212 202.952 2.59 2.497 1.635-.002-2.584-2.493z" clip-path="url(#p94a45562aa)" style="fill:#7396f5"/>
                <path d="m305.203 158.182 2.74-2.16 1.466-3.865-2.741 2.167z" clip-path="url(#p94a45562aa)" style="fill:#e7d7ce"/>
                <path d="m190.337 215.922 2.578 1.7 1.597-.162-2.573-1.696z" clip-path="url(#p94a45562aa)" style="fill:#5572df"/>
                <path d="m285.419 189.814 2.691-1.69 1.456-2.802-2.692 1.695z" clip-path="url(#p94a45562aa)" style="fill:#9fbfff"/>
                <path d="m215.676 224.892 2.581.64 1.542-.577-2.578-.638z" clip-path="url(#p94a45562aa)" style="fill:#4257c9"/>
                <path d="m315.3 135.361 2.764-2.345 1.483-4.545-2.767 2.355z" clip-path="url(#p94a45562aa)" style="fill:#f7a889"/>
                <path d="m175.802 205.45 2.588 2.364 1.63-.008-2.583-2.36z" clip-path="url(#p94a45562aa)" style="fill:#6e90f2"/>
                <path d="m270.122 206.053 2.658-1.258 1.462-2.147-2.658 1.26z" clip-path="url(#p94a45562aa)" style="fill:#7396f5"/>
                <path d="m291.022 182.391 2.705-1.84 1.456-3.066-2.706 1.845z" clip-path="url(#p94a45562aa)" style="fill:#b2ccfb"/>
                <path d="m233.197 224.629 2.597-.027 1.509-.974-2.595.028zM208.963 223.658l2.579.905 1.556-.442-2.575-.902z" clip-path="url(#p94a45562aa)" style="fill:#445acc"/>
                <path d="m279.827 196.565 2.68-1.544 1.456-2.539-2.68 1.547z" clip-path="url(#p94a45562aa)" style="fill:#8db0fe"/>
                <path d="m222.382 225.46 2.586.374 1.53-.71-2.584-.372z" clip-path="url(#p94a45562aa)" style="fill:#4257c9"/>
                <path d="m313.823 139.762 2.762-2.338 1.48-4.408-2.765 2.345z" clip-path="url(#p94a45562aa)" style="fill:#f7b396"/>
                <path d="m178.39 207.814 2.586 2.232 1.625-.013-2.581-2.227z" clip-path="url(#p94a45562aa)" style="fill:#688aef"/>
                <path d="m296.64 174.29 2.72-1.993 1.459-3.332-2.721 1.998z" clip-path="url(#p94a45562aa)" style="fill:#c7d7f0"/>
                <path d="m202.237 221.758 2.578 1.17 1.572-.306-2.574-1.167z" clip-path="url(#p94a45562aa)" style="fill:#4961d2"/>
                <path d="m303.74 161.909 2.738-2.155 1.464-3.732-2.74 2.16z" clip-path="url(#p94a45562aa)" style="fill:#e1dad6"/>
                <path d="m252.219 218.47 2.625-.703 1.48-1.496-2.624.703z" clip-path="url(#p94a45562aa)" style="fill:#5470de"/>
                <path d="m192.915 217.621 2.579 1.568 1.592-.165-2.574-1.564z" clip-path="url(#p94a45562aa)" style="fill:#516ddb"/>
                <path d="m256.324 216.27 2.633-.838 1.474-1.627-2.63.84z" clip-path="url(#p94a45562aa)" style="fill:#5a78e4"/>
                <path d="m248.115 220.401 2.618-.566 1.486-1.366-2.617.567z" clip-path="url(#p94a45562aa)" style="fill:#4f69d9"/>
                <path d="m274.242 202.648 2.668-1.398 1.46-2.278-2.669 1.4z" clip-path="url(#p94a45562aa)" style="fill:#7ea1fa"/>
                <path d="m229.086 225.364 2.594.108 1.517-.843-2.591-.106z" clip-path="url(#p94a45562aa)" style="fill:#4358cb"/>
                <path d="m312.349 144.027 2.76-2.329 1.476-4.274-2.762 2.338z" clip-path="url(#p94a45562aa)" style="fill:#f6bda2"/>
                <path d="m180.976 210.046 2.584 2.099 1.62-.017-2.58-2.095z" clip-path="url(#p94a45562aa)" style="fill:#6384eb"/>
                <path d="m260.431 213.805 2.641-.977 1.47-1.757-2.64.979z" clip-path="url(#p94a45562aa)" style="fill:#6180e9"/>
                <path d="m244.01 222.067 2.613-.43 1.492-1.236-2.611.431z" clip-path="url(#p94a45562aa)" style="fill:#4b64d5"/>
                <path d="m264.542 211.071 2.65-1.115 1.466-1.888-2.648 1.118z" clip-path="url(#p94a45562aa)" style="fill:#688aef"/>
                <path d="m289.566 185.322 2.705-1.836 1.456-2.935-2.705 1.84z" clip-path="url(#p94a45562aa)" style="fill:#adc9fd"/>
                <path d="m239.904 223.467 2.608-.295 1.498-1.105-2.606.296z" clip-path="url(#p94a45562aa)" style="fill:#4961d2"/>
                <path d="m211.542 224.563 2.581.773 1.553-.444-2.578-.77z" clip-path="url(#p94a45562aa)" style="fill:#445acc"/>
                <path d="m283.963 192.482 2.692-1.687 1.455-2.671-2.691 1.69z" clip-path="url(#p94a45562aa)" style="fill:#9abbff"/>
                <path d="m218.257 225.532 2.586.507 1.539-.578-2.583-.506z" clip-path="url(#p94a45562aa)" style="fill:#4358cb"/>
                <path d="m310.877 148.159 2.758-2.322 1.474-4.139-2.76 2.33z" clip-path="url(#p94a45562aa)" style="fill:#f4c5ad"/>
                <path d="m183.56 212.145 2.583 1.967 1.616-.021-2.579-1.963z" clip-path="url(#p94a45562aa)" style="fill:#5f7fe8"/>
                <path d="m302.279 165.503 2.737-2.148 1.462-3.6-2.738 2.154z" clip-path="url(#p94a45562aa)" style="fill:#dbdcde"/>
                <path d="m195.494 219.189 2.579 1.436 1.588-.168-2.575-1.433z" clip-path="url(#p94a45562aa)" style="fill:#506bda"/>
                <path d="m295.183 177.485 2.72-1.987 1.457-3.201-2.72 1.993z" clip-path="url(#p94a45562aa)" style="fill:#c1d4f4"/>
                <path d="m204.815 222.928 2.58 1.039 1.568-.309-2.576-1.036z" clip-path="url(#p94a45562aa)" style="fill:#485fd1"/>
                <path d="m278.37 198.972 2.679-1.54 1.457-2.41-2.68 1.543z" clip-path="url(#p94a45562aa)" style="fill:#89acfd"/>
                <path d="m224.968 225.834 2.592.242 1.526-.712-2.589-.24z" clip-path="url(#p94a45562aa)" style="fill:#4358cb"/>
                <path d="m268.658 208.068 2.66-1.255 1.462-2.018-2.658 1.258z" clip-path="url(#p94a45562aa)" style="fill:#7093f3"/>
                <path d="m235.794 224.602 2.604-.16 1.506-.975-2.601.161z" clip-path="url(#p94a45562aa)" style="fill:#465ecf"/>
                <path d="m309.409 152.157 2.756-2.314 1.47-4.006-2.758 2.322z" clip-path="url(#p94a45562aa)" style="fill:#f1ccb8"/>
                <path d="m186.143 214.112 2.583 1.835 1.611-.025-2.578-1.831z" clip-path="url(#p94a45562aa)" style="fill:#5b7ae5"/>
                <path d="m161.148 191.463 2.609 3.04 1.667.155-2.603-3.033z" clip-path="url(#p94a45562aa)" style="fill:#94b6ff"/>
                <path d="m163.757 194.503 2.605 2.905 1.662.148-2.6-2.898z" clip-path="url(#p94a45562aa)" style="fill:#8caffe"/>
                <path d="m158.534 188.288 2.614 3.175 1.673.162-2.608-3.169z" clip-path="url(#p94a45562aa)" style="fill:#9dbdff"/>
                <path d="m166.362 197.408 2.6 2.77 1.657.143-2.595-2.765z" clip-path="url(#p94a45562aa)" style="fill:#85a8fc"/>
                <path d="m168.963 200.178 2.598 2.637 1.651.137-2.593-2.631z" clip-path="url(#p94a45562aa)" style="fill:#7ea1fa"/>
                <path d="m300.819 168.965 2.736-2.142 1.46-3.468-2.736 2.148z" clip-path="url(#p94a45562aa)" style="fill:#d5dbe5"/>
                <path d="m198.073 220.625 2.58 1.304 1.584-.17-2.576-1.302z" clip-path="url(#p94a45562aa)" style="fill:#4e68d8"/>
                <path d="m272.78 204.795 2.67-1.396 1.46-2.15-2.668 1.4z" clip-path="url(#p94a45562aa)" style="fill:#7b9ff9"/>
                <path d="m231.68 225.472 2.6-.026 1.514-.844-2.597.027z" clip-path="url(#p94a45562aa)" style="fill:#455cce"/>
                <path d="m288.11 188.124 2.705-1.833 1.456-2.805-2.705 1.836z" clip-path="url(#p94a45562aa)" style="fill:#a7c5fe"/>
                <path d="m214.123 225.336 2.585.641 1.55-.445-2.582-.64z" clip-path="url(#p94a45562aa)" style="fill:#445acc"/>
                <path d="m171.561 202.815 2.595 2.503 1.646.131-2.59-2.497z" clip-path="url(#p94a45562aa)" style="fill:#799cf8"/>
                <path d="m307.942 156.022 2.755-2.307 1.468-3.872-2.756 2.314z" clip-path="url(#p94a45562aa)" style="fill:#edd2c3"/>
                <path d="m188.726 215.947 2.583 1.703 1.606-.029-2.578-1.7zM254.844 217.767l2.635-.838 1.478-1.497-2.633.839z" clip-path="url(#p94a45562aa)" style="fill:#5977e3"/>
                <path d="m250.733 219.835 2.628-.701 1.483-1.367-2.625.702z" clip-path="url(#p94a45562aa)" style="fill:#5470de"/>
                <path d="m282.506 195.021 2.692-1.684 1.457-2.542-2.692 1.687z" clip-path="url(#p94a45562aa)" style="fill:#96b7ff"/>
                <path d="m220.843 226.039 2.59.375 1.535-.58-2.586-.373z" clip-path="url(#p94a45562aa)" style="fill:#445acc"/>
                <path d="m293.727 180.55 2.719-1.983 1.457-3.07-2.72 1.988z" clip-path="url(#p94a45562aa)" style="fill:#bbd1f8"/>
                <path d="m207.395 223.967 2.583.906 1.564-.31-2.58-.905z" clip-path="url(#p94a45562aa)" style="fill:#485fd1"/>
                <path d="m258.957 215.432 2.642-.976 1.473-1.628-2.64.977z" clip-path="url(#p94a45562aa)" style="fill:#5f7fe8"/>
                <path d="m246.623 221.637 2.621-.565 1.49-1.237-2.62.566z" clip-path="url(#p94a45562aa)" style="fill:#506bda"/>
                <path d="m318.064 133.016 2.78-2.495 1.486-4.553-2.783 2.503z" clip-path="url(#p94a45562aa)" style="fill:#f59c7d"/>
                <path d="m174.156 205.318 2.593 2.37 1.64.126-2.587-2.365z" clip-path="url(#p94a45562aa)" style="fill:#7295f4"/>
                <path d="m263.072 212.828 2.65-1.114 1.47-1.758-2.65 1.115z" clip-path="url(#p94a45562aa)" style="fill:#6788ee"/>
                <path d="m242.512 223.172 2.615-.429 1.496-1.106-2.613.43z" clip-path="url(#p94a45562aa)" style="fill:#4c66d6"/>
                <path d="m316.585 137.424 2.778-2.485 1.482-4.418-2.78 2.495z" clip-path="url(#p94a45562aa)" style="fill:#f7a889"/>
                <path d="m276.91 201.25 2.68-1.539 1.459-2.28-2.68 1.54z" clip-path="url(#p94a45562aa)" style="fill:#85a8fc"/>
                <path d="m227.56 226.076 2.597.108 1.523-.712-2.594-.108z" clip-path="url(#p94a45562aa)" style="fill:#455cce"/>
                <path d="m176.75 207.687 2.59 2.237 1.636.122-2.586-2.232z" clip-path="url(#p94a45562aa)" style="fill:#6e90f2"/>
                <path d="m299.36 172.297 2.735-2.137 1.46-3.337-2.736 2.142z" clip-path="url(#p94a45562aa)" style="fill:#cfdaea"/>
                <path d="m200.653 221.929 2.582 1.173 1.58-.174-2.578-1.17z" clip-path="url(#p94a45562aa)" style="fill:#4c66d6"/>
                <path d="m306.478 159.754 2.753-2.3 1.466-3.739-2.755 2.307z" clip-path="url(#p94a45562aa)" style="fill:#e8d6cc"/>
                <path d="m191.309 217.65 2.583 1.57 1.602-.031-2.579-1.568z" clip-path="url(#p94a45562aa)" style="fill:#5673e0"/>
                <path d="m267.192 209.956 2.66-1.255 1.465-1.888-2.659 1.255z" clip-path="url(#p94a45562aa)" style="fill:#6f92f3"/>
                <path d="m238.398 224.442 2.61-.294 1.504-.976-2.608.295z" clip-path="url(#p94a45562aa)" style="fill:#4a63d3"/>
                <path d="m315.109 141.698 2.776-2.477 1.478-4.282-2.778 2.485z" clip-path="url(#p94a45562aa)" style="fill:#f7b396"/>
                <path d="m179.34 209.924 2.59 2.104 1.63.117-2.584-2.1z" clip-path="url(#p94a45562aa)" style="fill:#688aef"/>
                <path d="m286.655 190.795 2.704-1.83 1.456-2.674-2.705 1.833z" clip-path="url(#p94a45562aa)" style="fill:#a3c2fe"/>
                <path d="m216.708 225.977 2.59.509 1.545-.447-2.586-.507z" clip-path="url(#p94a45562aa)" style="fill:#455cce"/>
                <path d="m292.271 183.486 2.719-1.98 1.456-2.939-2.72 1.984z" clip-path="url(#p94a45562aa)" style="fill:#b6cefa"/>
                <path d="m209.978 224.873 2.585.775 1.56-.312-2.581-.773z" clip-path="url(#p94a45562aa)" style="fill:#485fd1"/>
                <path d="m271.317 206.813 2.67-1.395 1.463-2.02-2.67 1.397z" clip-path="url(#p94a45562aa)" style="fill:#799cf8"/>
                <path d="m234.28 225.446 2.606-.16 1.512-.844-2.604.16z" clip-path="url(#p94a45562aa)" style="fill:#485fd1"/>
                <path d="m281.049 197.431 2.692-1.681 1.457-2.413-2.692 1.684z" clip-path="url(#p94a45562aa)" style="fill:#92b4fe"/>
                <path d="m223.433 226.414 2.595.242 1.532-.58-2.592-.242z" clip-path="url(#p94a45562aa)" style="fill:#455cce"/>
                <path d="m305.016 163.355 2.751-2.295 1.464-3.606-2.753 2.3z" clip-path="url(#p94a45562aa)" style="fill:#e2dad5"/>
                <path d="m313.635 145.837 2.775-2.469 1.475-4.147-2.776 2.477z" clip-path="url(#p94a45562aa)" style="fill:#f7bca1"/>
                <path d="m193.892 219.22 2.583 1.44 1.598-.035-2.58-1.436z" clip-path="url(#p94a45562aa)" style="fill:#5470de"/>
                <path d="m181.93 212.028 2.588 1.97 1.625.114-2.583-1.967z" clip-path="url(#p94a45562aa)" style="fill:#6485ec"/>
                <path d="m297.903 175.498 2.734-2.133 1.458-3.205-2.735 2.137z" clip-path="url(#p94a45562aa)" style="fill:#cad8ef"/>
                <path d="m203.235 223.102 2.584 1.04 1.576-.175-2.58-1.039z" clip-path="url(#p94a45562aa)" style="fill:#4b64d5"/>
                <path d="m253.36 219.134 2.637-.837 1.482-1.368-2.635.838z" clip-path="url(#p94a45562aa)" style="fill:#5977e3"/>
                <path d="m257.479 216.929 2.643-.975 1.477-1.498-2.642.976z" clip-path="url(#p94a45562aa)" style="fill:#5f7fe8"/>
                <path d="m249.244 221.072 2.63-.7 1.487-1.238-2.628.701z" clip-path="url(#p94a45562aa)" style="fill:#5470de"/>
                <path d="m312.165 149.843 2.772-2.462 1.473-4.013-2.775 2.47z" clip-path="url(#p94a45562aa)" style="fill:#f5c4ac"/>
                <path d="m184.518 213.999 2.588 1.839 1.62.109-2.583-1.835z" clip-path="url(#p94a45562aa)" style="fill:#6180e9"/>
                <path d="m275.45 203.399 2.68-1.537 1.46-2.15-2.68 1.538z" clip-path="url(#p94a45562aa)" style="fill:#82a6fb"/>
                <path d="m230.157 226.184 2.603-.025 1.52-.713-2.6.026z" clip-path="url(#p94a45562aa)" style="fill:#485fd1"/>
                <path d="m159.463 191.165 2.615 3.047 1.679.29-2.61-3.04z" clip-path="url(#p94a45562aa)" style="fill:#9abbff"/>
                <path d="m162.078 194.212 2.61 2.911 1.674.285-2.605-2.905z" clip-path="url(#p94a45562aa)" style="fill:#92b4fe"/>
                <path d="m156.843 187.983 2.62 3.182 1.685.298-2.614-3.175z" clip-path="url(#p94a45562aa)" style="fill:#a2c1ff"/>
                <path d="m261.599 214.456 2.652-1.114 1.472-1.628-2.651 1.114z" clip-path="url(#p94a45562aa)" style="fill:#6687ed"/>
                <path d="m245.127 222.743 2.623-.564 1.494-1.107-2.621.565z" clip-path="url(#p94a45562aa)" style="fill:#506bda"/>
                <path d="m164.688 197.123 2.607 2.777 1.668.278-2.601-2.77z" clip-path="url(#p94a45562aa)" style="fill:#8badfd"/>
                <path d="m303.555 166.823 2.75-2.288 1.462-3.475-2.751 2.295z" clip-path="url(#p94a45562aa)" style="fill:#dddcdc"/>
                <path d="m196.475 220.66 2.585 1.307 1.593-.038-2.58-1.304z" clip-path="url(#p94a45562aa)" style="fill:#516ddb"/>
                <path d="m290.815 186.29 2.719-1.975 1.456-2.808-2.719 1.979z" clip-path="url(#p94a45562aa)" style="fill:#b1cbfc"/>
                <path d="m212.563 225.648 2.589.643 1.556-.314-2.585-.641z" clip-path="url(#p94a45562aa)" style="fill:#485fd1"/>
                <path d="m167.295 199.9 2.604 2.642 1.662.273-2.598-2.637z" clip-path="url(#p94a45562aa)" style="fill:#84a7fc"/>
                <path d="m285.198 193.337 2.705-1.826 1.456-2.545-2.704 1.83z" clip-path="url(#p94a45562aa)" style="fill:#9ebeff"/>
                <path d="m219.297 226.486 2.594.376 1.542-.448-2.59-.375z" clip-path="url(#p94a45562aa)" style="fill:#465ecf"/>
                <path d="m265.723 211.714 2.661-1.253 1.468-1.76-2.66 1.255z" clip-path="url(#p94a45562aa)" style="fill:#6e90f2"/>
                <path d="m241.008 224.148 2.618-.429 1.5-.976-2.614.43z" clip-path="url(#p94a45562aa)" style="fill:#4e68d8"/>
                <path d="m296.446 178.567 2.733-2.128 1.458-3.074-2.734 2.133z" clip-path="url(#p94a45562aa)" style="fill:#c4d5f3"/>
                <path d="m205.82 224.142 2.586.909 1.572-.178-2.583-.906z" clip-path="url(#p94a45562aa)" style="fill:#4b64d5"/>
                <path d="m169.899 202.542 2.6 2.509 1.657.267-2.595-2.503z" clip-path="url(#p94a45562aa)" style="fill:#7ea1fa"/>
                <path d="m310.697 153.715 2.77-2.454 1.47-3.88-2.772 2.462z" clip-path="url(#p94a45562aa)" style="fill:#f2cbb7"/>
                <path d="m187.106 215.838 2.587 1.706 1.616.106-2.583-1.703z" clip-path="url(#p94a45562aa)" style="fill:#5d7ce6"/>
                <path d="m279.59 199.711 2.693-1.679 1.458-2.282-2.692 1.681z" clip-path="url(#p94a45562aa)" style="fill:#8fb1fe"/>
                <path d="m226.028 226.656 2.6.11 1.53-.582-2.598-.108z" clip-path="url(#p94a45562aa)" style="fill:#485fd1"/>
                <path d="m320.845 130.521 2.797-2.644 1.488-4.563-2.8 2.654z" clip-path="url(#p94a45562aa)" style="fill:#f18f71"/>
                <path d="m172.5 205.05 2.598 2.375 1.651.262-2.593-2.37z" clip-path="url(#p94a45562aa)" style="fill:#779af7"/>
                <path d="m269.852 208.701 2.67-1.393 1.465-1.89-2.67 1.395z" clip-path="url(#p94a45562aa)" style="fill:#7699f6"/>
                <path d="m236.886 225.287 2.613-.294 1.51-.845-2.611.294z" clip-path="url(#p94a45562aa)" style="fill:#4b64d5"/>
                <path d="m302.095 170.16 2.75-2.283 1.46-3.342-2.75 2.288z" clip-path="url(#p94a45562aa)" style="fill:#d7dce3"/>
                <path d="m199.06 221.967 2.586 1.175 1.59-.04-2.583-1.173z" clip-path="url(#p94a45562aa)" style="fill:#506bda"/>
                <path d="m319.363 134.939 2.795-2.635 1.484-4.427-2.797 2.644z" clip-path="url(#p94a45562aa)" style="fill:#f59c7d"/>
                <path d="m175.098 207.425 2.596 2.242 1.646.257-2.59-2.237z" clip-path="url(#p94a45562aa)" style="fill:#7295f4"/>
                <path d="m309.231 157.454 2.77-2.448 1.466-3.745-2.77 2.454z" clip-path="url(#p94a45562aa)" style="fill:#eed0c0"/>
                <path d="m189.693 217.544 2.588 1.575 1.61.102-2.582-1.571z" clip-path="url(#p94a45562aa)" style="fill:#5a78e4"/>
                <path d="m289.36 188.966 2.718-1.972 1.456-2.679-2.719 1.976z" clip-path="url(#p94a45562aa)" style="fill:#abc8fd"/>
                <path d="m215.152 226.29 2.593.51 1.552-.314-2.589-.509z" clip-path="url(#p94a45562aa)" style="fill:#4961d2"/>
                <path d="m273.987 205.418 2.682-1.534 1.462-2.022-2.681 1.537z" clip-path="url(#p94a45562aa)" style="fill:#80a3fa"/>
                <path d="m232.76 226.16 2.61-.16 1.516-.713-2.606.159z" clip-path="url(#p94a45562aa)" style="fill:#4a63d3"/>
                <path d="m317.885 139.221 2.792-2.626 1.48-4.291-2.794 2.635z" clip-path="url(#p94a45562aa)" style="fill:#f7a889"/>
                <path d="m294.99 181.507 2.733-2.124 1.456-2.944-2.733 2.128z" clip-path="url(#p94a45562aa)" style="fill:#bfd3f6"/>
                <path d="m177.694 209.667 2.595 2.108 1.64.253-2.589-2.104z" clip-path="url(#p94a45562aa)" style="fill:#6e90f2"/>
                <path d="m208.406 225.051 2.589.776 1.568-.179-2.585-.775z" clip-path="url(#p94a45562aa)" style="fill:#4b64d5"/>
                <path d="m283.741 195.75 2.705-1.824 1.457-2.415-2.705 1.826z" clip-path="url(#p94a45562aa)" style="fill:#9bbcff"/>
                <path d="m221.89 226.862 2.6.243 1.538-.449-2.595-.242z" clip-path="url(#p94a45562aa)" style="fill:#485fd1"/>
                <path d="m255.997 218.297 2.646-.975 1.48-1.368-2.644.975z" clip-path="url(#p94a45562aa)" style="fill:#5e7de7"/>
                <path d="m251.873 220.371 2.638-.836 1.486-1.238-2.636.837z" clip-path="url(#p94a45562aa)" style="fill:#5977e3"/>
                <path d="m260.122 215.954 2.654-1.113 1.475-1.499-2.652 1.114z" clip-path="url(#p94a45562aa)" style="fill:#6485ec"/>
                <path d="m247.75 222.179 2.632-.7 1.491-1.108-2.629.7z" clip-path="url(#p94a45562aa)" style="fill:#5572df"/>
                <path d="m307.767 161.06 2.768-2.44 1.465-3.614-2.769 2.448z" clip-path="url(#p94a45562aa)" style="fill:#ead5c9"/>
                <path d="m192.28 219.119 2.589 1.442 1.606.099-2.583-1.44z" clip-path="url(#p94a45562aa)" style="fill:#5977e3"/>
                <path d="m300.637 173.365 2.749-2.278 1.459-3.21-2.75 2.283z" clip-path="url(#p94a45562aa)" style="fill:#d2dbe8"/>
                <path d="m201.646 223.142 2.589 1.043 1.584-.043-2.584-1.04z" clip-path="url(#p94a45562aa)" style="fill:#4f69d9"/>
                <path d="m316.41 143.368 2.79-2.618 1.477-4.155-2.792 2.626z" clip-path="url(#p94a45562aa)" style="fill:#f7b194"/>
                <path d="m180.289 211.775 2.593 1.976 1.636.248-2.588-1.971z" clip-path="url(#p94a45562aa)" style="fill:#6a8bef"/>
                <path d="m264.251 213.342 2.663-1.252 1.47-1.63-2.661 1.254z" clip-path="url(#p94a45562aa)" style="fill:#6c8ff1"/>
                <path d="m243.626 223.72 2.625-.564 1.5-.977-2.624.564z" clip-path="url(#p94a45562aa)" style="fill:#516ddb"/>
                <path d="m278.13 201.862 2.694-1.677 1.46-2.153-2.694 1.68z" clip-path="url(#p94a45562aa)" style="fill:#8badfd"/>
                <path d="m228.628 226.766 2.606-.024 1.526-.583-2.603.025z" clip-path="url(#p94a45562aa)" style="fill:#4961d2"/>
                <path d="m268.384 210.46 2.672-1.391 1.467-1.76-2.671 1.392z" clip-path="url(#p94a45562aa)" style="fill:#7597f6"/>
                <path d="m239.5 224.993 2.62-.428 1.506-.846-2.618.429z" clip-path="url(#p94a45562aa)" style="fill:#4f69d9"/>
                <path d="m314.937 147.381 2.789-2.61 1.474-4.02-2.79 2.617z" clip-path="url(#p94a45562aa)" style="fill:#f7ba9f"/>
                <path d="m182.882 213.751 2.593 1.843 1.63.244-2.587-1.84z" clip-path="url(#p94a45562aa)" style="fill:#6687ed"/>
                <path d="m293.534 184.315 2.732-2.12 1.457-2.812-2.733 2.124z" clip-path="url(#p94a45562aa)" style="fill:#bad0f8"/>
                <path d="m210.995 225.827 2.593.645 1.564-.181-2.589-.643z" clip-path="url(#p94a45562aa)" style="fill:#4b64d5"/>
                <path d="m287.903 191.51 2.718-1.968 1.457-2.548-2.719 1.972z" clip-path="url(#p94a45562aa)" style="fill:#a7c5fe"/>
                <path d="m217.745 226.8 2.597.378 1.549-.316-2.594-.376z" clip-path="url(#p94a45562aa)" style="fill:#4961d2"/>
                <path d="m306.305 164.535 2.767-2.436 1.463-3.48-2.768 2.441z" clip-path="url(#p94a45562aa)" style="fill:#e4d9d2"/>
                <path d="m157.766 190.73 2.62 3.054 1.692.428-2.615-3.047z" clip-path="url(#p94a45562aa)" style="fill:#9fbfff"/>
                <path d="m194.869 220.56 2.589 1.31 1.602.097-2.585-1.307z" clip-path="url(#p94a45562aa)" style="fill:#5673e0"/>
                <path d="m160.387 193.784 2.616 2.918 1.685.421-2.61-2.911z" clip-path="url(#p94a45562aa)" style="fill:#98b9ff"/>
                <path d="m155.14 187.54 2.626 3.19 1.697.435-2.62-3.182z" clip-path="url(#p94a45562aa)" style="fill:#a7c5fe"/>
                <path d="m163.003 196.702 2.613 2.783 1.68.415-2.608-2.777z" clip-path="url(#p94a45562aa)" style="fill:#90b2fe"/>
                <path d="m299.18 176.44 2.747-2.274 1.459-3.079-2.75 2.278z" clip-path="url(#p94a45562aa)" style="fill:#cdd9ec"/>
                <path d="m204.235 224.185 2.59.91 1.58-.044-2.586-.909z" clip-path="url(#p94a45562aa)" style="fill:#4f69d9"/>
                <path d="m165.616 199.485 2.61 2.648 1.673.41-2.604-2.643z" clip-path="url(#p94a45562aa)" style="fill:#89acfd"/>
                <path d="m282.283 198.032 2.706-1.821 1.457-2.285-2.705 1.824z" clip-path="url(#p94a45562aa)" style="fill:#97b8ff"/>
                <path d="m224.49 227.105 2.603.11 1.535-.45-2.6-.109z" clip-path="url(#p94a45562aa)" style="fill:#4a63d3"/>
                <path d="m272.523 207.308 2.683-1.533 1.463-1.891-2.682 1.534z" clip-path="url(#p94a45562aa)" style="fill:#7ea1fa"/>
                <path d="m235.37 226 2.615-.292 1.514-.715-2.613.294z" clip-path="url(#p94a45562aa)" style="fill:#4c66d6"/>
                <path d="m313.467 151.26 2.787-2.603 1.472-3.886-2.789 2.61z" clip-path="url(#p94a45562aa)" style="fill:#f5c2aa"/>
                <path d="m185.475 215.594 2.592 1.71 1.626.24-2.587-1.706z" clip-path="url(#p94a45562aa)" style="fill:#6282ea"/>
                <path d="m168.225 202.133 2.607 2.514 1.668.404-2.601-2.509z" clip-path="url(#p94a45562aa)" style="fill:#84a7fc"/>
                <path d="m323.642 127.877 2.815-2.795 1.49-4.573-2.817 2.805z" clip-path="url(#p94a45562aa)" style="fill:#ec7f63"/>
                <path d="m170.832 204.647 2.603 2.38 1.663.398-2.598-2.374z" clip-path="url(#p94a45562aa)" style="fill:#7ea1fa"/>
                <path d="m304.845 167.877 2.765-2.43 1.462-3.348-2.767 2.436z" clip-path="url(#p94a45562aa)" style="fill:#dfdbd9"/>
                <path d="m197.458 221.87 2.59 1.178 1.598.094-2.586-1.175z" clip-path="url(#p94a45562aa)" style="fill:#5572df"/>
                <path d="m254.511 219.535 2.648-.974 1.484-1.239-2.646.975z" clip-path="url(#p94a45562aa)" style="fill:#5f7fe8"/>
                <path d="m276.67 203.884 2.693-1.676 1.46-2.023-2.692 1.677z" clip-path="url(#p94a45562aa)" style="fill:#89acfd"/>
                <path d="m231.234 226.742 2.612-.159 1.523-.582-2.609.158z" clip-path="url(#p94a45562aa)" style="fill:#4c66d6"/>
                <path d="m258.643 217.322 2.655-1.112 1.478-1.369-2.654 1.113z" clip-path="url(#p94a45562aa)" style="fill:#6485ec"/>
                <path d="m250.382 221.479 2.64-.837 1.49-1.107-2.639.836z" clip-path="url(#p94a45562aa)" style="fill:#5a78e4"/>
                <path d="m322.158 132.304 2.812-2.786 1.487-4.436-2.815 2.795z" clip-path="url(#p94a45562aa)" style="fill:#f18f71"/>
                <path d="m312 155.006 2.785-2.596 1.47-3.753-2.788 2.604z" clip-path="url(#p94a45562aa)" style="fill:#f2c9b4"/>
                <path d="m173.435 207.027 2.602 2.247 1.657.393-2.596-2.242z" clip-path="url(#p94a45562aa)" style="fill:#799cf8"/>
                <path d="m188.067 217.304 2.593 1.578 1.62.237-2.587-1.575z" clip-path="url(#p94a45562aa)" style="fill:#5f7fe8"/>
                <path d="m292.078 186.994 2.732-2.117 1.456-2.682-2.732 2.12z" clip-path="url(#p94a45562aa)" style="fill:#b5cdfa"/>
                <path d="m213.588 226.472 2.596.511 1.56-.182-2.592-.51z" clip-path="url(#p94a45562aa)" style="fill:#4b64d5"/>
                <path d="m262.776 214.841 2.664-1.25 1.474-1.5-2.663 1.251z" clip-path="url(#p94a45562aa)" style="fill:#6b8df0"/>
                <path d="m246.251 223.156 2.634-.7 1.497-.977-2.632.7z" clip-path="url(#p94a45562aa)" style="fill:#5673e0"/>
                <path d="m286.446 193.926 2.719-1.967 1.456-2.417-2.718 1.969z" clip-path="url(#p94a45562aa)" style="fill:#a3c2fe"/>
                <path d="m220.342 227.178 2.601.244 1.546-.317-2.598-.243z" clip-path="url(#p94a45562aa)" style="fill:#4b64d5"/>
                <path d="m297.723 179.383 2.747-2.27 1.457-2.947-2.748 2.273z" clip-path="url(#p94a45562aa)" style="fill:#c7d7f0"/>
                <path d="m206.825 225.095 2.594.779 1.576-.047-2.59-.776z" clip-path="url(#p94a45562aa)" style="fill:#4f69d9"/>
                <path d="m320.677 136.595 2.81-2.778 1.483-4.3-2.812 2.787z" clip-path="url(#p94a45562aa)" style="fill:#f49a7b"/>
                <path d="m176.037 209.274 2.6 2.113 1.652.388-2.595-2.108zM266.914 212.09l2.673-1.39 1.47-1.631-2.673 1.392z" clip-path="url(#p94a45562aa)" style="fill:#7396f5"/>
                <path d="m242.12 224.565 2.628-.564 1.503-.845-2.625.563z" clip-path="url(#p94a45562aa)" style="fill:#536edd"/>
                <path d="m280.824 200.185 2.706-1.82 1.459-2.154-2.706 1.821z" clip-path="url(#p94a45562aa)" style="fill:#94b6ff"/>
                <path d="m227.093 227.216 2.609-.024 1.532-.45-2.606.024z" clip-path="url(#p94a45562aa)" style="fill:#4c66d6"/>
                <path d="m303.386 171.087 2.764-2.425 1.46-3.215-2.765 2.43z" clip-path="url(#p94a45562aa)" style="fill:#dadce0"/>
                <path d="m200.049 223.048 2.592 1.045 1.594.092-2.589-1.043z" clip-path="url(#p94a45562aa)" style="fill:#5470de"/>
                <path d="m310.535 158.62 2.784-2.59 1.466-3.62-2.785 2.596z" clip-path="url(#p94a45562aa)" style="fill:#efcebd"/>
                <path d="m190.66 218.882 2.593 1.445 1.616.234-2.588-1.442z" clip-path="url(#p94a45562aa)" style="fill:#5d7ce6"/>
                <path d="m319.2 140.75 2.808-2.769 1.48-4.164-2.81 2.778z" clip-path="url(#p94a45562aa)" style="fill:#f7a688"/>
                <path d="m271.056 209.069 2.684-1.532 1.466-1.762-2.683 1.533z" clip-path="url(#p94a45562aa)" style="fill:#7da0f9"/>
                <path d="m237.985 225.708 2.623-.428 1.512-.715-2.62.428z" clip-path="url(#p94a45562aa)" style="fill:#506bda"/>
                <path d="m178.637 211.387 2.599 1.98 1.646.384-2.593-1.976z" clip-path="url(#p94a45562aa)" style="fill:#6f92f3"/>
                <path d="m290.621 189.542 2.733-2.114 1.456-2.551-2.732 2.117z" clip-path="url(#p94a45562aa)" style="fill:#b1cbfc"/>
                <path d="m216.184 226.983 2.6.378 1.558-.183-2.597-.377z" clip-path="url(#p94a45562aa)" style="fill:#4c66d6"/>
                <path d="m296.266 182.195 2.747-2.265 1.457-2.816-2.747 2.269z" clip-path="url(#p94a45562aa)" style="fill:#c4d5f3"/>
                <path d="m209.419 225.874 2.596.645 1.573-.047-2.593-.645z" clip-path="url(#p94a45562aa)" style="fill:#4f69d9"/>
                <path d="m317.726 144.77 2.805-2.76 1.477-4.029-2.808 2.77z" clip-path="url(#p94a45562aa)" style="fill:#f7b093"/>
                <path d="m181.236 213.367 2.598 1.847 1.64.38-2.592-1.843z" clip-path="url(#p94a45562aa)" style="fill:#6b8df0"/>
                <path d="m275.206 205.775 2.694-1.674 1.463-1.893-2.694 1.676z" clip-path="url(#p94a45562aa)" style="fill:#86a9fc"/>
                <path d="m233.846 226.583 2.62-.292 1.52-.583-2.617.293z" clip-path="url(#p94a45562aa)" style="fill:#4f69d9"/>
                <path d="m309.072 162.1 2.782-2.585 1.465-3.486-2.784 2.59z" clip-path="url(#p94a45562aa)" style="fill:#ebd3c6"/>
                <path d="m193.253 220.327 2.593 1.313 1.612.23-2.59-1.31z" clip-path="url(#p94a45562aa)" style="fill:#5b7ae5"/>
                <path d="m284.989 196.21 2.718-1.964 1.458-2.287-2.719 1.967z" clip-path="url(#p94a45562aa)" style="fill:#a1c0ff"/>
                <path d="m222.943 227.422 2.607.111 1.543-.317-2.604-.11z" clip-path="url(#p94a45562aa)" style="fill:#4c66d6"/>
                <path d="m156.056 190.157 2.627 3.06 1.704.567-2.621-3.054z" clip-path="url(#p94a45562aa)" style="fill:#a6c4fe"/>
                <path d="m158.683 193.218 2.623 2.925 1.697.559-2.616-2.918z" clip-path="url(#p94a45562aa)" style="fill:#9ebeff"/>
                <path d="m301.927 174.166 2.764-2.42 1.459-3.084-2.764 2.425z" clip-path="url(#p94a45562aa)" style="fill:#d5dbe5"/>
                <path d="m153.424 186.96 2.632 3.197 1.71.573-2.626-3.19z" clip-path="url(#p94a45562aa)" style="fill:#aec9fc"/>
                <path d="m202.641 224.093 2.595.913 1.59.09-2.591-.911z" clip-path="url(#p94a45562aa)" style="fill:#5470de"/>
                <path d="m161.306 196.143 2.619 2.79 1.69.552-2.612-2.783z" clip-path="url(#p94a45562aa)" style="fill:#97b8ff"/>
                <path d="m257.159 218.56 2.657-1.11 1.482-1.24-2.655 1.112z" clip-path="url(#p94a45562aa)" style="fill:#6485ec"/>
                <path d="m253.022 220.642 2.65-.973 1.487-1.108-2.648.974z" clip-path="url(#p94a45562aa)" style="fill:#5f7fe8"/>
                <path d="m163.925 198.932 2.615 2.655 1.685.546-2.61-2.648z" clip-path="url(#p94a45562aa)" style="fill:#90b2fe"/>
                <path d="m261.298 216.21 2.666-1.25 1.476-1.37-2.664 1.251z" clip-path="url(#p94a45562aa)" style="fill:#6b8df0"/>
                <path d="m248.885 222.456 2.643-.837 1.494-.977-2.64.837z" clip-path="url(#p94a45562aa)" style="fill:#5b7ae5"/>
                <path d="m316.254 148.657 2.804-2.753 1.473-3.894-2.805 2.76z" clip-path="url(#p94a45562aa)" style="fill:#f7b89c"/>
                <path d="m183.834 215.214 2.597 1.714 1.636.376-2.592-1.71z" clip-path="url(#p94a45562aa)" style="fill:#688aef"/>
                <path d="m166.54 201.587 2.612 2.52 1.68.54-2.607-2.514z" clip-path="url(#p94a45562aa)" style="fill:#8badfd"/>
                <path d="m279.363 202.208 2.707-1.818 1.46-2.024-2.706 1.819z" clip-path="url(#p94a45562aa)" style="fill:#92b4fe"/>
                <path d="m229.702 227.192 2.615-.158 1.53-.45-2.613.158z" clip-path="url(#p94a45562aa)" style="fill:#4f69d9"/>
                <path d="m265.44 213.59 2.675-1.39 1.472-1.5-2.673 1.39z" clip-path="url(#p94a45562aa)" style="fill:#7396f5"/>
                <path d="m244.748 224.001 2.636-.7 1.501-.845-2.634.7z" clip-path="url(#p94a45562aa)" style="fill:#5875e1"/>
                <path d="m326.457 125.082 2.832-2.948 1.493-4.583-2.835 2.958z" clip-path="url(#p94a45562aa)" style="fill:#e57058"/>
                <path d="m307.61 165.447 2.781-2.579 1.463-3.353-2.782 2.584z" clip-path="url(#p94a45562aa)" style="fill:#e7d7ce"/>
                <path d="m169.152 204.107 2.61 2.385 1.673.535-2.603-2.38z" clip-path="url(#p94a45562aa)" style="fill:#84a7fc"/>
                <path d="m195.846 221.64 2.596 1.18 1.607.228-2.591-1.177z" clip-path="url(#p94a45562aa)" style="fill:#5a78e4"/>
                <path d="m294.81 184.877 2.747-2.262 1.456-2.685-2.747 2.265z" clip-path="url(#p94a45562aa)" style="fill:#bfd3f6"/>
                <path d="m212.015 226.52 2.6.512 1.57-.049-2.597-.511z" clip-path="url(#p94a45562aa)" style="fill:#506bda"/>
                <path d="m269.587 210.7 2.685-1.532 1.468-1.631-2.684 1.532z" clip-path="url(#p94a45562aa)" style="fill:#7b9ff9"/>
                <path d="m240.608 225.28 2.631-.564 1.509-.715-2.628.564z" clip-path="url(#p94a45562aa)" style="fill:#5572df"/>
                <path d="m289.165 191.96 2.732-2.112 1.457-2.42-2.733 2.114z" clip-path="url(#p94a45562aa)" style="fill:#adc9fd"/>
                <path d="m218.785 227.361 2.605.245 1.553-.184-2.601-.244z" clip-path="url(#p94a45562aa)" style="fill:#4f69d9"/>
                <path d="m314.785 152.41 2.802-2.747 1.47-3.76-2.803 2.754z" clip-path="url(#p94a45562aa)" style="fill:#f5c0a7"/>
                <path d="m186.431 216.928 2.598 1.581 1.63.373-2.592-1.578z" clip-path="url(#p94a45562aa)" style="fill:#6687ed"/>
                <path d="m324.97 129.518 2.83-2.939 1.49-4.445-2.833 2.948z" clip-path="url(#p94a45562aa)" style="fill:#ec7f63"/>
                <path d="m171.762 206.492 2.607 2.252 1.668.53-2.602-2.247z" clip-path="url(#p94a45562aa)" style="fill:#7ea1fa"/>
                <path d="m300.47 177.114 2.763-2.417 1.458-2.951-2.764 2.42z" clip-path="url(#p94a45562aa)" style="fill:#d1dae9"/>
                <path d="m205.236 225.006 2.598.78 1.585.088-2.594-.779z" clip-path="url(#p94a45562aa)" style="fill:#5470de"/>
                <path d="m283.53 198.366 2.719-1.963 1.458-2.157-2.718 1.965z" clip-path="url(#p94a45562aa)" style="fill:#9ebeff"/>
                <path d="m225.55 227.533 2.613-.023 1.539-.318-2.61.024z" clip-path="url(#p94a45562aa)" style="fill:#4f69d9"/>
                <path d="m273.74 207.537 2.696-1.674 1.464-1.762-2.694 1.674z" clip-path="url(#p94a45562aa)" style="fill:#85a8fc"/>
                <path d="m236.465 226.29 2.626-.427 1.517-.583-2.623.428z" clip-path="url(#p94a45562aa)" style="fill:#536edd"/>
                <path d="m323.487 133.817 2.828-2.93 1.485-4.308-2.83 2.939z" clip-path="url(#p94a45562aa)" style="fill:#f08b6e"/>
                <path d="m174.369 208.744 2.605 2.118 1.663.525-2.6-2.113z" clip-path="url(#p94a45562aa)" style="fill:#7a9df8"/>
                <path d="m306.15 168.662 2.78-2.573 1.461-3.22-2.781 2.578z" clip-path="url(#p94a45562aa)" style="fill:#e2dad5"/>
                <path d="m198.442 222.82 2.597 1.047 1.602.226-2.592-1.045z" clip-path="url(#p94a45562aa)" style="fill:#5977e3"/>
                <path d="m313.319 156.03 2.8-2.741 1.468-3.626-2.802 2.747z" clip-path="url(#p94a45562aa)" style="fill:#f4c6af"/>
                <path d="m189.029 218.51 2.597 1.447 1.627.37-2.593-1.445z" clip-path="url(#p94a45562aa)" style="fill:#6384eb"/>
                <path d="m322.008 137.981 2.825-2.921 1.482-4.172-2.828 2.93z" clip-path="url(#p94a45562aa)" style="fill:#f4987a"/>
                <path d="m176.974 210.862 2.605 1.984 1.657.521-2.599-1.98z" clip-path="url(#p94a45562aa)" style="fill:#7699f6"/>
                <path d="m277.9 204.1 2.708-1.816 1.462-1.894-2.707 1.818z" clip-path="url(#p94a45562aa)" style="fill:#8fb1fe"/>
                <path d="m232.317 227.034 2.622-.293 1.526-.45-2.619.292z" clip-path="url(#p94a45562aa)" style="fill:#536edd"/>
                <path d="m293.354 187.428 2.746-2.26 1.457-2.553-2.747 2.262z" clip-path="url(#p94a45562aa)" style="fill:#bbd1f8"/>
                <path d="m255.671 219.669 2.66-1.112 1.485-1.108-2.657 1.112z" clip-path="url(#p94a45562aa)" style="fill:#6687ed"/>
                <path d="m214.616 227.032 2.604.38 1.565-.05-2.6-.379z" clip-path="url(#p94a45562aa)" style="fill:#516ddb"/>
                <path d="m259.816 217.45 2.668-1.251 1.48-1.24-2.666 1.251z" clip-path="url(#p94a45562aa)" style="fill:#6b8df0"/>
                <path d="m251.528 221.62 2.651-.974 1.492-.977-2.65.973z" clip-path="url(#p94a45562aa)" style="fill:#6180e9"/>
                <path d="m299.013 179.93 2.763-2.412 1.457-2.82-2.763 2.416z" clip-path="url(#p94a45562aa)" style="fill:#ccd9ed"/>
                <path d="m207.834 225.786 2.6.647 1.581.086-2.596-.645z" clip-path="url(#p94a45562aa)" style="fill:#5470de"/>
                <path d="m287.707 194.246 2.733-2.108 1.457-2.29-2.732 2.111z" clip-path="url(#p94a45562aa)" style="fill:#aac7fd"/>
                <path d="m221.39 227.606 2.61.112 1.55-.185-2.607-.11z" clip-path="url(#p94a45562aa)" style="fill:#506bda"/>
                <path d="m263.964 214.96 2.676-1.39 1.475-1.37-2.675 1.39z" clip-path="url(#p94a45562aa)" style="fill:#7396f5"/>
                <path d="m247.384 223.302 2.645-.837 1.499-.846-2.643.837z" clip-path="url(#p94a45562aa)" style="fill:#5d7ce6"/>
                <path d="m320.531 142.01 2.823-2.913 1.479-4.037-2.825 2.921z" clip-path="url(#p94a45562aa)" style="fill:#f6a385"/>
                <path d="m179.579 212.846 2.603 1.851 1.652.517-2.598-1.847z" clip-path="url(#p94a45562aa)" style="fill:#7295f4"/>
                <path d="m311.854 159.515 2.799-2.734 1.466-3.492-2.8 2.74z" clip-path="url(#p94a45562aa)" style="fill:#f1ccb8"/>
                <path d="m191.626 219.957 2.6 1.316 1.62.367-2.593-1.313z" clip-path="url(#p94a45562aa)" style="fill:#6180e9"/>
                <path d="m304.69 171.746 2.78-2.569 1.46-3.088-2.78 2.573z" clip-path="url(#p94a45562aa)" style="fill:#dedcdb"/>
                <path d="m201.039 223.867 2.6.914 1.597.225-2.595-.913z" clip-path="url(#p94a45562aa)" style="fill:#5977e3"/>
                <path d="m268.115 212.2 2.687-1.531 1.47-1.5-2.685 1.53z" clip-path="url(#p94a45562aa)" style="fill:#7a9df8"/>
                <path d="m243.24 224.716 2.638-.7 1.506-.714-2.636.7z" clip-path="url(#p94a45562aa)" style="fill:#5977e3"/>
                <path d="m154.334 189.445 2.633 3.068 1.716.705-2.627-3.061z" clip-path="url(#p94a45562aa)" style="fill:#adc9fd"/>
                <path d="m156.967 192.513 2.63 2.932 1.709.698-2.623-2.925z" clip-path="url(#p94a45562aa)" style="fill:#a5c3fe"/>
                <path d="m151.696 186.24 2.638 3.205 1.722.712-2.632-3.197z" clip-path="url(#p94a45562aa)" style="fill:#b5cdfa"/>
                <path d="m282.07 200.39 2.72-1.96 1.459-2.027-2.72 1.963z" clip-path="url(#p94a45562aa)" style="fill:#9bbcff"/>
                <path d="m228.163 227.51 2.618-.158 1.536-.318-2.615.158z" clip-path="url(#p94a45562aa)" style="fill:#536edd"/>
                <path d="m159.596 195.445 2.625 2.796 1.704.691-2.619-2.79z" clip-path="url(#p94a45562aa)" style="fill:#9ebeff"/>
                <path d="m162.221 198.241 2.622 2.66 1.697.686-2.615-2.655z" clip-path="url(#p94a45562aa)" style="fill:#97b8ff"/>
                <path d="m319.058 145.904 2.821-2.906 1.475-3.901-2.823 2.913z" clip-path="url(#p94a45562aa)" style="fill:#f7ac8e"/>
                <path d="m182.182 214.697 2.602 1.718 1.647.513-2.597-1.714z" clip-path="url(#p94a45562aa)" style="fill:#6e90f2"/>
                <path d="m272.272 209.168 2.697-1.673 1.467-1.632-2.696 1.674z" clip-path="url(#p94a45562aa)" style="fill:#84a7fc"/>
                <path d="m239.091 225.863 2.634-.564 1.514-.583-2.63.564z" clip-path="url(#p94a45562aa)" style="fill:#5875e1"/>
                <path d="m164.843 200.902 2.618 2.526 1.691.679-2.612-2.52z" clip-path="url(#p94a45562aa)" style="fill:#92b4fe"/>
                <path d="m297.557 182.615 2.762-2.409 1.457-2.688-2.763 2.412z" clip-path="url(#p94a45562aa)" style="fill:#c9d7f0"/>
                <path d="m310.391 162.868 2.798-2.728 1.464-3.359-2.799 2.734z" clip-path="url(#p94a45562aa)" style="fill:#eed0c0"/>
                <path d="m210.435 226.433 2.604.513 1.577.086-2.6-.513z" clip-path="url(#p94a45562aa)" style="fill:#5470de"/>
                <path d="m291.897 189.848 2.747-2.256 1.456-2.423-2.746 2.259z" clip-path="url(#p94a45562aa)" style="fill:#b7cff9"/>
                <path d="m194.225 221.273 2.6 1.182 1.617.365-2.596-1.18z" clip-path="url(#p94a45562aa)" style="fill:#5f7fe8"/>
                <path d="m217.22 227.411 2.61.246 1.56-.05-2.605-.246z" clip-path="url(#p94a45562aa)" style="fill:#536edd"/>
                <path d="m329.29 122.134 2.85-3.103 1.496-4.593-2.854 3.113z" clip-path="url(#p94a45562aa)" style="fill:#dd5f4b"/>
                <path d="m167.46 203.428 2.616 2.391 1.686.673-2.61-2.385z" clip-path="url(#p94a45562aa)" style="fill:#8badfd"/>
                <path d="m303.233 174.697 2.778-2.564 1.46-2.956-2.78 2.569z" clip-path="url(#p94a45562aa)" style="fill:#d9dce1"/>
                <path d="m276.436 205.863 2.708-1.815 1.464-1.764-2.708 1.817z" clip-path="url(#p94a45562aa)" style="fill:#8db0fe"/>
                <path d="m203.638 224.781 2.602.782 1.594.223-2.598-.78z" clip-path="url(#p94a45562aa)" style="fill:#5977e3"/>
                <path d="m234.939 226.741 2.629-.428 1.523-.45-2.626.428z" clip-path="url(#p94a45562aa)" style="fill:#5673e0"/>
                <path d="m317.587 149.663 2.82-2.898 1.472-3.767-2.821 2.906z" clip-path="url(#p94a45562aa)" style="fill:#f7b599"/>
                <path d="m286.249 196.403 2.733-2.107 1.458-2.158-2.733 2.108z" clip-path="url(#p94a45562aa)" style="fill:#a7c5fe"/>
                <path d="m224 227.718 2.617-.024 1.546-.184-2.613.023z" clip-path="url(#p94a45562aa)" style="fill:#536edd"/>
                <path d="m184.784 216.415 2.603 1.585 1.642.51-2.598-1.582z" clip-path="url(#p94a45562aa)" style="fill:#6b8df0"/>
                <path d="m327.8 126.579 2.848-3.093 1.492-4.455-2.85 3.103z" clip-path="url(#p94a45562aa)" style="fill:#e46e56"/>
                <path d="m170.076 205.819 2.613 2.257 1.68.668-2.607-2.252z" clip-path="url(#p94a45562aa)" style="fill:#85a8fc"/>
                <path d="m258.33 218.557 2.67-1.25 1.484-1.108-2.668 1.25z" clip-path="url(#p94a45562aa)" style="fill:#6c8ff1"/>
                <path d="m254.18 220.646 2.66-1.112 1.49-.977-2.659 1.112z" clip-path="url(#p94a45562aa)" style="fill:#6788ee"/>
                <path d="m262.484 216.199 2.678-1.39 1.478-1.24-2.676 1.39z" clip-path="url(#p94a45562aa)" style="fill:#7396f5"/>
                <path d="m250.03 222.465 2.653-.974 1.496-.845-2.651.973z" clip-path="url(#p94a45562aa)" style="fill:#6282ea"/>
                <path d="m326.315 130.888 2.845-3.084 1.488-4.318-2.848 3.093z" clip-path="url(#p94a45562aa)" style="fill:#ea7b60"/>
                <path d="m172.69 208.076 2.61 2.123 1.674.663-2.605-2.118z" clip-path="url(#p94a45562aa)" style="fill:#81a4fb"/>
                <path d="m308.93 166.089 2.797-2.723 1.462-3.226-2.798 2.728z" clip-path="url(#p94a45562aa)" style="fill:#ead4c8"/>
                <path d="m196.825 222.455 2.602 1.05 1.612.362-2.597-1.047z" clip-path="url(#p94a45562aa)" style="fill:#5e7de7"/>
                <path d="m280.608 202.284 2.72-1.96 1.461-1.895-2.72 1.961z" clip-path="url(#p94a45562aa)" style="fill:#98b9ff"/>
                <path d="m230.781 227.352 2.625-.293 1.533-.318-2.622.293z" clip-path="url(#p94a45562aa)" style="fill:#5572df"/>
                <path d="m316.119 153.289 2.818-2.892 1.47-3.632-2.82 2.898z" clip-path="url(#p94a45562aa)" style="fill:#f6bda2"/>
                <path d="m187.387 218 2.603 1.451 1.636.506-2.597-1.448z" clip-path="url(#p94a45562aa)" style="fill:#688aef"/>
                <path d="m266.64 213.57 2.688-1.531 1.474-1.37-2.687 1.53z" clip-path="url(#p94a45562aa)" style="fill:#7a9df8"/>
                <path d="m245.878 224.016 2.647-.837 1.504-.714-2.645.837z" clip-path="url(#p94a45562aa)" style="fill:#5f7fe8"/>
                <path d="m296.1 185.169 2.762-2.406 1.457-2.557-2.762 2.409z" clip-path="url(#p94a45562aa)" style="fill:#c4d5f3"/>
                <path d="m213.04 226.946 2.608.38 1.572.085-2.604-.38z" clip-path="url(#p94a45562aa)" style="fill:#5572df"/>
                <path d="m324.833 135.06 2.843-3.075 1.484-4.181-2.845 3.084z" clip-path="url(#p94a45562aa)" style="fill:#f08a6c"/>
                <path d="m175.3 210.199 2.61 1.989 1.669.658-2.605-1.984z" clip-path="url(#p94a45562aa)" style="fill:#7da0f9"/>
                <path d="m290.44 192.138 2.747-2.255 1.457-2.291-2.747 2.256z" clip-path="url(#p94a45562aa)" style="fill:#b3cdfb"/>
                <path d="m219.83 227.657 2.614.111 1.557-.05-2.61-.112z" clip-path="url(#p94a45562aa)" style="fill:#5572df"/>
                <path d="m301.776 177.518 2.778-2.561 1.457-2.824-2.778 2.564z" clip-path="url(#p94a45562aa)" style="fill:#d5dbe5"/>
                <path d="m206.24 225.563 2.605.648 1.59.222-2.601-.647z" clip-path="url(#p94a45562aa)" style="fill:#5977e3"/>
                <path d="m270.802 210.669 2.698-1.672 1.47-1.502-2.698 1.673z" clip-path="url(#p94a45562aa)" style="fill:#84a7fc"/>
                <path d="m241.725 225.299 2.641-.7 1.512-.583-2.639.7z" clip-path="url(#p94a45562aa)" style="fill:#5b7ae5"/>
                <path d="m284.79 198.43 2.733-2.106 1.459-2.028-2.733 2.107z" clip-path="url(#p94a45562aa)" style="fill:#a5c3fe"/>
                <path d="m314.653 156.781 2.816-2.885 1.468-3.499-2.818 2.892z" clip-path="url(#p94a45562aa)" style="fill:#f5c2aa"/>
                <path d="m226.617 227.694 2.622-.157 1.542-.185-2.618.158z" clip-path="url(#p94a45562aa)" style="fill:#5673e0"/>
                <path d="m323.354 139.097 2.842-3.067 1.48-4.045-2.843 3.075z" clip-path="url(#p94a45562aa)" style="fill:#f39577"/>
                <path d="m189.99 219.451 2.604 1.318 1.631.504-2.599-1.316z" clip-path="url(#p94a45562aa)" style="fill:#6788ee"/>
                <path d="m177.91 212.188 2.609 1.855 1.663.654-2.603-1.85z" clip-path="url(#p94a45562aa)" style="fill:#799cf8"/>
                <path d="m307.47 169.177 2.796-2.718 1.46-3.093-2.796 2.723z" clip-path="url(#p94a45562aa)" style="fill:#e6d7cf"/>
                <path d="m199.427 223.505 2.604.916 1.607.36-2.6-.914z" clip-path="url(#p94a45562aa)" style="fill:#5e7de7"/>
                <path d="m274.97 207.495 2.709-1.815 1.465-1.632-2.708 1.815z" clip-path="url(#p94a45562aa)" style="fill:#8caffe"/>
                <path d="m237.568 226.313 2.637-.564 1.52-.45-2.634.564z" clip-path="url(#p94a45562aa)" style="fill:#5a78e4"/>
                <path d="m152.598 188.592 2.64 3.076 1.73.845-2.634-3.068z" clip-path="url(#p94a45562aa)" style="fill:#b3cdfb"/>
                <path d="m155.238 191.668 2.636 2.939 1.722.838-2.629-2.932z" clip-path="url(#p94a45562aa)" style="fill:#adc9fd"/>
                <path d="m149.953 185.379 2.645 3.213 1.736.853-2.638-3.205z" clip-path="url(#p94a45562aa)" style="fill:#bbd1f8"/>
                <path d="m157.874 194.607 2.631 2.803 1.716.831-2.625-2.796z" clip-path="url(#p94a45562aa)" style="fill:#a5c3fe"/>
                <path d="m321.88 142.998 2.838-3.059 1.478-3.909-2.842 3.067z" clip-path="url(#p94a45562aa)" style="fill:#f59f80"/>
                <path d="m180.519 214.043 2.608 1.721 1.657.651-2.602-1.718z" clip-path="url(#p94a45562aa)" style="fill:#7597f6"/>
                <path d="m160.505 197.41 2.628 2.667 1.71.825-2.622-2.661z" clip-path="url(#p94a45562aa)" style="fill:#9ebeff"/>
                <path d="m294.644 187.592 2.762-2.404 1.456-2.425-2.762 2.406z" clip-path="url(#p94a45562aa)" style="fill:#c1d4f4"/>
                <path d="m215.648 227.326 2.613.246 1.568.085-2.609-.246z" clip-path="url(#p94a45562aa)" style="fill:#5875e1"/>
                <path d="m300.319 180.206 2.777-2.557 1.458-2.692-2.778 2.56z" clip-path="url(#p94a45562aa)" style="fill:#d2dbe8"/>
                <path d="m208.845 226.211 2.61.514 1.584.221-2.604-.513z" clip-path="url(#p94a45562aa)" style="fill:#5977e3"/>
                <path d="m279.144 204.048 2.722-1.96 1.462-1.763-2.72 1.96z" clip-path="url(#p94a45562aa)" style="fill:#97b8ff"/>
                <path d="m233.406 227.06 2.633-.43 1.529-.317-2.629.428z" clip-path="url(#p94a45562aa)" style="fill:#5977e3"/>
                <path d="m313.189 160.14 2.815-2.88 1.465-3.364-2.816 2.885z" clip-path="url(#p94a45562aa)" style="fill:#f2c9b4"/>
                <path d="m163.133 200.077 2.624 2.532 1.704.819-2.618-2.526z" clip-path="url(#p94a45562aa)" style="fill:#98b9ff"/>
                <path d="m256.84 219.534 2.672-1.252 1.488-.976-2.67 1.251z" clip-path="url(#p94a45562aa)" style="fill:#6e90f2"/>
                <path d="m192.594 220.77 2.605 1.184 1.626.501-2.6-1.182z" clip-path="url(#p94a45562aa)" style="fill:#6687ed"/>
                <path d="m261 217.306 2.68-1.39 1.482-1.108-2.678 1.39z" clip-path="url(#p94a45562aa)" style="fill:#7396f5"/>
                <path d="m252.683 221.491 2.664-1.112 1.494-.845-2.662 1.112z" clip-path="url(#p94a45562aa)" style="fill:#688aef"/>
                <path d="m288.982 194.296 2.747-2.252 1.458-2.16-2.747 2.254z" clip-path="url(#p94a45562aa)" style="fill:#b1cbfc"/>
                <path d="m222.444 227.768 2.62-.023 1.553-.05-2.616.023z" clip-path="url(#p94a45562aa)" style="fill:#5875e1"/>
                <path d="m306.011 172.133 2.795-2.714 1.46-2.96-2.796 2.718z" clip-path="url(#p94a45562aa)" style="fill:#e2dad5"/>
                <path d="m332.14 119.03 2.87-3.258 1.5-4.603-2.874 3.269z" clip-path="url(#p94a45562aa)" style="fill:#d1493f"/>
                <path d="m202.03 224.421 2.607.783 1.603.359-2.602-.782z" clip-path="url(#p94a45562aa)" style="fill:#5e7de7"/>
                <path d="m265.162 214.808 2.69-1.53 1.476-1.24-2.688 1.532z" clip-path="url(#p94a45562aa)" style="fill:#7a9df8"/>
                <path d="m248.525 223.18 2.657-.976 1.501-.713-2.654.974z" clip-path="url(#p94a45562aa)" style="fill:#6485ec"/>
                <path d="m165.757 202.61 2.621 2.396 1.698.813-2.615-2.391z" clip-path="url(#p94a45562aa)" style="fill:#93b5fe"/>
                <path d="m320.407 146.765 2.837-3.051 1.474-3.775-2.839 3.06z" clip-path="url(#p94a45562aa)" style="fill:#f7a98b"/>
                <path d="m183.127 215.764 2.608 1.588 1.652.648-2.603-1.585z" clip-path="url(#p94a45562aa)" style="fill:#7295f4"/>
                <path d="m330.648 123.486 2.868-3.249 1.494-4.465-2.87 3.259z" clip-path="url(#p94a45562aa)" style="fill:#da5a49"/>
                <path d="m269.328 212.039 2.7-1.673 1.472-1.37-2.698 1.673z" clip-path="url(#p94a45562aa)" style="fill:#82a6fb"/>
                <path d="m244.366 224.598 2.65-.837 1.51-.582-2.648.837z" clip-path="url(#p94a45562aa)" style="fill:#6180e9"/>
                <path d="m168.378 205.006 2.62 2.263 1.691.807-2.613-2.257z" clip-path="url(#p94a45562aa)" style="fill:#8db0fe"/>
                <path d="m283.328 200.325 2.735-2.105 1.46-1.896-2.734 2.105z" clip-path="url(#p94a45562aa)" style="fill:#a2c1ff"/>
                <path d="m229.239 227.537 2.628-.294 1.54-.184-2.626.293z" clip-path="url(#p94a45562aa)" style="fill:#5977e3"/>
                <path d="m311.727 163.366 2.813-2.875 1.464-3.23-2.815 2.879z" clip-path="url(#p94a45562aa)" style="fill:#f0cdbb"/>
                <path d="m195.199 221.954 2.606 1.052 1.622.499-2.602-1.05z" clip-path="url(#p94a45562aa)" style="fill:#6485ec"/>
                <path d="m329.16 127.804 2.865-3.24 1.49-4.327-2.867 3.249z" clip-path="url(#p94a45562aa)" style="fill:#e36b54"/>
                <path d="m170.997 207.269 2.617 2.127 1.686.803-2.61-2.123z" clip-path="url(#p94a45562aa)" style="fill:#88abfd"/>
                <path d="m273.5 208.997 2.71-1.815 1.469-1.502-2.71 1.815z" clip-path="url(#p94a45562aa)" style="fill:#8caffe"/>
                <path d="m240.205 225.749 2.644-.701 1.517-.45-2.641.7z" clip-path="url(#p94a45562aa)" style="fill:#5f7fe8"/>
                <path d="m298.862 182.763 2.777-2.554 1.457-2.56-2.777 2.557z" clip-path="url(#p94a45562aa)" style="fill:#cedaeb"/>
                <path d="m211.454 226.725 2.613.381 1.58.22-2.608-.38z" clip-path="url(#p94a45562aa)" style="fill:#5a78e4"/>
                <path d="m318.937 150.397 2.835-3.044 1.472-3.64-2.837 3.052z" clip-path="url(#p94a45562aa)" style="fill:#f7b093"/>
                <path d="m185.735 217.352 2.608 1.455 1.647.644-2.603-1.451z" clip-path="url(#p94a45562aa)" style="fill:#6f92f3"/>
                <path d="m293.187 189.883 2.762-2.4 1.457-2.295-2.762 2.404z" clip-path="url(#p94a45562aa)" style="fill:#bed2f6"/>
                <path d="m218.26 227.572 2.619.112 1.565.084-2.615-.111z" clip-path="url(#p94a45562aa)" style="fill:#5977e3"/>
                <path d="m304.554 174.957 2.794-2.71 1.458-2.828-2.795 2.714z" clip-path="url(#p94a45562aa)" style="fill:#dedcdb"/>
                <path d="m204.637 225.204 2.61.65 1.598.357-2.605-.648z" clip-path="url(#p94a45562aa)" style="fill:#5e7de7"/>
                <path d="m327.676 131.985 2.863-3.23 1.486-4.19-2.865 3.239z" clip-path="url(#p94a45562aa)" style="fill:#e9785d"/>
                <path d="m173.614 209.396 2.616 1.994 1.68.798-2.61-1.99z" clip-path="url(#p94a45562aa)" style="fill:#84a7fc"/>
                <path d="m287.523 196.324 2.748-2.251 1.458-2.03-2.747 2.253z" clip-path="url(#p94a45562aa)" style="fill:#aec9fc"/>
                <path d="m225.063 227.745 2.626-.158 1.55-.05-2.622.157z" clip-path="url(#p94a45562aa)" style="fill:#5a78e4"/>
                <path d="m277.679 205.68 2.722-1.958 1.465-1.633-2.722 1.959z" clip-path="url(#p94a45562aa)" style="fill:#96b7ff"/>
                <path d="m236.039 226.63 2.64-.564 1.526-.317-2.637.564z" clip-path="url(#p94a45562aa)" style="fill:#5e7de7"/>
                <path d="m310.266 166.459 2.813-2.87 1.461-3.098-2.813 2.875z" clip-path="url(#p94a45562aa)" style="fill:#edd1c2"/>
                <path d="m197.805 223.006 2.609.918 1.617.497-2.604-.916z" clip-path="url(#p94a45562aa)" style="fill:#6485ec"/>
                <path d="m317.47 153.896 2.833-3.039 1.47-3.504-2.836 3.044z" clip-path="url(#p94a45562aa)" style="fill:#f7b89c"/>
                <path d="m188.343 218.807 2.609 1.32 1.642.642-2.604-1.318z" clip-path="url(#p94a45562aa)" style="fill:#6e90f2"/>
                <path d="m326.196 136.03 2.86-3.222 1.483-4.053-2.863 3.23z" clip-path="url(#p94a45562aa)" style="fill:#ee8468"/>
                <path d="m176.23 211.39 2.614 1.86 1.675.793-2.609-1.855z" clip-path="url(#p94a45562aa)" style="fill:#80a3fa"/>
                <path d="m259.512 218.282 2.683-1.39 1.485-.976-2.68 1.39z" clip-path="url(#p94a45562aa)" style="fill:#7597f6"/>
                <path d="m255.347 220.379 2.674-1.252 1.491-.845-2.671 1.252z" clip-path="url(#p94a45562aa)" style="fill:#6f92f3"/>
                <path d="m263.68 215.916 2.692-1.532 1.48-1.107-2.69 1.531z" clip-path="url(#p94a45562aa)" style="fill:#7b9ff9"/>
                <path d="m251.182 222.204 2.666-1.113 1.499-.712-2.664 1.112z" clip-path="url(#p94a45562aa)" style="fill:#6b8df0"/>
                <path d="m281.866 202.089 2.735-2.104 1.462-1.765-2.735 2.105z" clip-path="url(#p94a45562aa)" style="fill:#a1c0ff"/>
                <path d="m231.867 227.243 2.636-.429 1.536-.184-2.633.43z" clip-path="url(#p94a45562aa)" style="fill:#5d7ce6"/>
                <path d="m297.406 185.188 2.777-2.551 1.456-2.428-2.777 2.554z" clip-path="url(#p94a45562aa)" style="fill:#cbd8ee"/>
                <path d="m214.067 227.106 2.617.246 1.577.22-2.613-.246z" clip-path="url(#p94a45562aa)" style="fill:#5d7ce6"/>
                <path d="m150.85 187.598 2.646 3.083 1.742.987-2.64-3.076z" clip-path="url(#p94a45562aa)" style="fill:#bbd1f8"/>
                <path d="m153.496 190.681 2.642 2.947 1.736.979-2.636-2.94z" clip-path="url(#p94a45562aa)" style="fill:#b3cdfb"/>
                <path d="m267.852 213.277 2.701-1.672 1.475-1.239-2.7 1.673z" clip-path="url(#p94a45562aa)" style="fill:#84a7fc"/>
                <path d="m247.016 223.76 2.66-.975 1.506-.58-2.657.974z" clip-path="url(#p94a45562aa)" style="fill:#6788ee"/>
                <path d="m148.198 184.377 2.651 3.22 1.75.995-2.646-3.213z" clip-path="url(#p94a45562aa)" style="fill:#c3d5f4"/>
                <path d="m303.096 177.649 2.794-2.707 1.458-2.695-2.794 2.71z" clip-path="url(#p94a45562aa)" style="fill:#dadce0"/>
                <path d="m207.247 225.853 2.613.516 1.594.356-2.609-.514z" clip-path="url(#p94a45562aa)" style="fill:#5f7fe8"/>
                <path d="m156.138 193.628 2.638 2.81 1.73.972-2.632-2.803z" clip-path="url(#p94a45562aa)" style="fill:#adc9fd"/>
                <path d="m324.718 139.94 2.858-3.214 1.48-3.918-2.86 3.222z" clip-path="url(#p94a45562aa)" style="fill:#f29072"/>
                <path d="m178.844 213.25 2.614 1.725 1.669.79-2.608-1.722z" clip-path="url(#p94a45562aa)" style="fill:#7da0f9"/>
                <path d="m291.73 192.044 2.761-2.4 1.458-2.162-2.762 2.401z" clip-path="url(#p94a45562aa)" style="fill:#bbd1f8"/>
                <path d="m220.879 227.684 2.623-.023 1.561.084-2.62.023z" clip-path="url(#p94a45562aa)" style="fill:#5b7ae5"/>
                <path d="m158.776 196.438 2.634 2.674 1.723.965-2.628-2.667z" clip-path="url(#p94a45562aa)" style="fill:#a6c4fe"/>
                <path d="m316.004 157.26 2.833-3.032 1.466-3.37-2.834 3.038z" clip-path="url(#p94a45562aa)" style="fill:#f6bea4"/>
                <path d="m190.952 220.128 2.61 1.187 1.637.64-2.605-1.186z" clip-path="url(#p94a45562aa)" style="fill:#6c8ff1"/>
                <path d="m272.028 210.366 2.712-1.815 1.47-1.37-2.71 1.816z" clip-path="url(#p94a45562aa)" style="fill:#8caffe"/>
                <path d="m242.849 225.048 2.653-.839 1.514-.448-2.65.837z" clip-path="url(#p94a45562aa)" style="fill:#6485ec"/>
                <path d="m161.41 199.112 2.63 2.538 1.717.96-2.624-2.533z" clip-path="url(#p94a45562aa)" style="fill:#a1c0ff"/>
                <path d="m308.806 169.42 2.812-2.866 1.46-2.965-2.812 2.87z" clip-path="url(#p94a45562aa)" style="fill:#ead4c8"/>
                <path d="m200.414 223.924 2.611.784 1.612.496-2.606-.783z" clip-path="url(#p94a45562aa)" style="fill:#6485ec"/>
                <path d="m335.01 115.772 2.89-3.416 1.502-4.615-2.893 3.428z" clip-path="url(#p94a45562aa)" style="fill:#c43032"/>
                <path d="m286.063 198.22 2.748-2.25 1.46-1.897-2.748 2.25z" clip-path="url(#p94a45562aa)" style="fill:#adc9fd"/>
                <path d="m227.689 227.587 2.632-.294 1.546-.05-2.628.294z" clip-path="url(#p94a45562aa)" style="fill:#5e7de7"/>
                <path d="m164.04 201.65 2.628 2.403 1.71.953-2.621-2.397z" clip-path="url(#p94a45562aa)" style="fill:#9abbff"/>
                <path d="m323.244 143.714 2.856-3.207 1.476-3.781-2.858 3.213z" clip-path="url(#p94a45562aa)" style="fill:#f49a7b"/>
                <path d="m181.458 214.975 2.613 1.591 1.664.786-2.608-1.588z" clip-path="url(#p94a45562aa)" style="fill:#7a9df8"/>
                <path d="m276.21 207.182 2.724-1.959 1.467-1.501-2.722 1.958z" clip-path="url(#p94a45562aa)" style="fill:#96b7ff"/>
                <path d="m238.678 226.066 2.648-.702 1.523-.316-2.644.7z" clip-path="url(#p94a45562aa)" style="fill:#6282ea"/>
                <path d="m333.516 120.237 2.887-3.406 1.497-4.475-2.89 3.416z" clip-path="url(#p94a45562aa)" style="fill:#cf453c"/>
                <path d="m166.668 204.053 2.625 2.268 1.704.948-2.619-2.263z" clip-path="url(#p94a45562aa)" style="fill:#96b7ff"/>
                <path d="m314.54 160.491 2.832-3.027 1.465-3.236-2.833 3.032z" clip-path="url(#p94a45562aa)" style="fill:#f5c4ac"/>
                <path d="m193.562 221.315 2.611 1.054 1.632.637-2.606-1.052z" clip-path="url(#p94a45562aa)" style="fill:#6b8df0"/>
                <path d="m301.64 180.209 2.793-2.704 1.457-2.563-2.794 2.707z" clip-path="url(#p94a45562aa)" style="fill:#d7dce3"/>
                <path d="m209.86 226.369 2.617.38 1.59.357-2.613-.38z" clip-path="url(#p94a45562aa)" style="fill:#6180e9"/>
                <path d="m295.949 187.482 2.777-2.55 1.457-2.295-2.777 2.551z" clip-path="url(#p94a45562aa)" style="fill:#c7d7f0"/>
                <path d="m216.684 227.352 2.622.112 1.573.22-2.618-.112z" clip-path="url(#p94a45562aa)" style="fill:#5e7de7"/>
                <path d="m332.025 124.565 2.884-3.397 1.494-4.337-2.887 3.406z" clip-path="url(#p94a45562aa)" style="fill:#d85646"/>
                <path d="m169.293 206.32 2.623 2.134 1.698.942-2.617-2.127z" clip-path="url(#p94a45562aa)" style="fill:#90b2fe"/>
                <path d="m321.772 147.353 2.854-3.2 1.474-3.646-2.856 3.207z" clip-path="url(#p94a45562aa)" style="fill:#f6a385"/>
                <path d="m184.071 216.566 2.614 1.458 1.658.783-2.608-1.455z" clip-path="url(#p94a45562aa)" style="fill:#779af7"/>
                <path d="m280.401 203.722 2.736-2.104 1.464-1.633-2.735 2.104z" clip-path="url(#p94a45562aa)" style="fill:#9fbfff"/>
                <path d="m234.503 226.814 2.643-.566 1.532-.182-2.64.564z" clip-path="url(#p94a45562aa)" style="fill:#6282ea"/>
                <path d="m307.348 172.247 2.811-2.861 1.46-2.832-2.813 2.865z" clip-path="url(#p94a45562aa)" style="fill:#e7d7ce"/>
                <path d="m203.025 224.708 2.614.65 1.608.495-2.61-.65z" clip-path="url(#p94a45562aa)" style="fill:#6485ec"/>
                <path d="m290.27 194.073 2.763-2.398 1.458-2.03-2.762 2.399z" clip-path="url(#p94a45562aa)" style="fill:#b9d0f9"/>
                <path d="m223.502 227.66 2.63-.158 1.557.085-2.626.158z" clip-path="url(#p94a45562aa)" style="fill:#5e7de7"/>
                <path d="m258.02 219.127 2.685-1.393 1.49-.843-2.683 1.391z" clip-path="url(#p94a45562aa)" style="fill:#7699f6"/>
                <path d="m262.195 216.891 2.693-1.532 1.484-.975-2.692 1.532z" clip-path="url(#p94a45562aa)" style="fill:#7da0f9"/>
                <path d="m253.848 221.091 2.676-1.253 1.497-.711-2.674 1.252z" clip-path="url(#p94a45562aa)" style="fill:#7295f4"/>
                <path d="m330.539 128.755 2.881-3.388 1.49-4.199-2.885 3.397z" clip-path="url(#p94a45562aa)" style="fill:#e0654f"/>
                <path d="m171.916 208.454 2.622 1.998 1.692.938-2.616-1.994z" clip-path="url(#p94a45562aa)" style="fill:#8caffe"/>
                <path d="m266.372 214.384 2.703-1.673 1.478-1.106-2.701 1.672z" clip-path="url(#p94a45562aa)" style="fill:#84a7fc"/>
                <path d="m249.676 222.785 2.668-1.115 1.504-.579-2.666 1.113z" clip-path="url(#p94a45562aa)" style="fill:#6c8ff1"/>
                <path d="m313.079 163.59 2.83-3.023 1.463-3.103-2.832 3.027z" clip-path="url(#p94a45562aa)" style="fill:#f2c9b4"/>
                <path d="m196.173 222.369 2.614.92 1.627.635-2.609-.918z" clip-path="url(#p94a45562aa)" style="fill:#6b8df0"/>
                <path d="m270.553 211.605 2.714-1.816 1.473-1.238-2.712 1.815z" clip-path="url(#p94a45562aa)" style="fill:#8caffe"/>
                <path d="m245.502 224.21 2.662-.978 1.512-.447-2.66.976z" clip-path="url(#p94a45562aa)" style="fill:#6a8bef"/>
                <path d="m320.303 150.857 2.853-3.193 1.47-3.51-2.854 3.199z" clip-path="url(#p94a45562aa)" style="fill:#f7ac8e"/>
                <path d="m284.6 199.985 2.75-2.25 1.461-1.765-2.748 2.25z" clip-path="url(#p94a45562aa)" style="fill:#abc8fd"/>
                <path d="m230.321 227.293 2.639-.43 1.543-.049-2.636.43z" clip-path="url(#p94a45562aa)" style="fill:#6180e9"/>
                <path d="m186.685 218.024 2.614 1.323 1.653.78-2.61-1.32z" clip-path="url(#p94a45562aa)" style="fill:#7597f6"/>
                <path d="m329.056 132.808 2.879-3.378 1.485-4.063-2.881 3.388z" clip-path="url(#p94a45562aa)" style="fill:#e67259"/>
                <path d="m174.538 210.452 2.62 1.864 1.686.933-2.614-1.86z" clip-path="url(#p94a45562aa)" style="fill:#88abfd"/>
                <path d="m300.183 182.637 2.793-2.701 1.457-2.43-2.794 2.703z" clip-path="url(#p94a45562aa)" style="fill:#d4dbe6"/>
                <path d="m212.477 226.75 2.622.246 1.585.356-2.617-.246z" clip-path="url(#p94a45562aa)" style="fill:#6282ea"/>
                <path d="m274.74 208.551 2.725-1.959 1.47-1.369-2.724 1.959z" clip-path="url(#p94a45562aa)" style="fill:#96b7ff"/>
                <path d="m241.326 225.364 2.656-.84 1.52-.315-2.653.839z" clip-path="url(#p94a45562aa)" style="fill:#6788ee"/>
                <path d="m294.491 189.644 2.778-2.548 1.457-2.163-2.777 2.55z" clip-path="url(#p94a45562aa)" style="fill:#c5d6f2"/>
                <path d="m219.306 227.464 2.628-.024 1.568.22-2.623.024z" clip-path="url(#p94a45562aa)" style="fill:#6180e9"/>
                <path d="m305.89 174.942 2.81-2.858 1.46-2.698-2.812 2.86z" clip-path="url(#p94a45562aa)" style="fill:#e3d9d3"/>
                <path d="m205.64 225.358 2.617.516 1.603.495-2.613-.516z" clip-path="url(#p94a45562aa)" style="fill:#6485ec"/>
                <path d="m149.087 186.46 2.653 3.093 1.756 1.128-2.647-3.083z" clip-path="url(#p94a45562aa)" style="fill:#c3d5f4"/>
                <path d="m151.74 189.553 2.65 2.954 1.748 1.121-2.642-2.947z" clip-path="url(#p94a45562aa)" style="fill:#bbd1f8"/>
                <path d="m146.428 183.231 2.659 3.23 1.762 1.137-2.651-3.221z" clip-path="url(#p94a45562aa)" style="fill:#cad8ef"/>
                <path d="m327.576 136.726 2.877-3.371 1.482-3.925-2.88 3.378z" clip-path="url(#p94a45562aa)" style="fill:#ec7f63"/>
                <path d="m154.39 192.507 2.644 2.817 1.742 1.114-2.638-2.81z" clip-path="url(#p94a45562aa)" style="fill:#b5cdfa"/>
                <path d="m177.158 212.316 2.62 1.729 1.68.93-2.614-1.726z" clip-path="url(#p94a45562aa)" style="fill:#84a7fc"/>
                <path d="m318.837 154.228 2.85-3.187 1.469-3.377-2.853 3.193z" clip-path="url(#p94a45562aa)" style="fill:#f7b396"/>
                <path d="m189.3 219.347 2.614 1.19 1.648.778-2.61-1.187z" clip-path="url(#p94a45562aa)" style="fill:#7396f5"/>
                <path d="m157.034 195.324 2.64 2.68 1.736 1.108-2.634-2.674z" clip-path="url(#p94a45562aa)" style="fill:#aec9fc"/>
                <path d="m288.811 195.97 2.763-2.397 1.46-1.898-2.763 2.398z" clip-path="url(#p94a45562aa)" style="fill:#b7cff9"/>
                <path d="m226.132 227.502 2.636-.295 1.553.086-2.632.294z" clip-path="url(#p94a45562aa)" style="fill:#6282ea"/>
                <path d="m311.618 166.554 2.83-3.018 1.461-2.969-2.83 3.022z" clip-path="url(#p94a45562aa)" style="fill:#f1cdba"/>
                <path d="m198.787 223.288 2.616.786 1.622.634-2.611-.784z" clip-path="url(#p94a45562aa)" style="fill:#6b8df0"/>
                <path d="m278.934 205.223 2.738-2.103 1.465-1.502-2.736 2.104z" clip-path="url(#p94a45562aa)" style="fill:#9fbfff"/>
                <path d="m237.146 226.248 2.65-.703 1.53-.181-2.648.702z" clip-path="url(#p94a45562aa)" style="fill:#6687ed"/>
                <path d="m159.674 198.005 2.637 2.544 1.73 1.101-2.631-2.538z" clip-path="url(#p94a45562aa)" style="fill:#a9c6fd"/>
                <path d="m337.9 112.356 2.91-3.576 1.506-4.626-2.914 3.587z" clip-path="url(#p94a45562aa)" style="fill:#b50927"/>
                <path d="m162.311 200.55 2.634 2.408 1.723 1.095-2.627-2.403z" clip-path="url(#p94a45562aa)" style="fill:#a2c1ff"/>
                <path d="m326.1 140.507 2.875-3.363 1.478-3.789-2.877 3.37z" clip-path="url(#p94a45562aa)" style="fill:#f08a6c"/>
                <path d="m179.777 214.045 2.62 1.595 1.674.926-2.613-1.591z" clip-path="url(#p94a45562aa)" style="fill:#81a4fb"/>
                <path d="m260.705 217.734 2.695-1.533 1.488-.842-2.693 1.532z" clip-path="url(#p94a45562aa)" style="fill:#7ea1fa"/>
                <path d="m256.524 219.838 2.687-1.394 1.494-.71-2.684 1.393z" clip-path="url(#p94a45562aa)" style="fill:#799cf8"/>
                <path d="m264.888 215.359 2.705-1.675 1.482-.973-2.703 1.673z" clip-path="url(#p94a45562aa)" style="fill:#85a8fc"/>
                <path d="m298.726 184.933 2.793-2.7 1.457-2.297-2.793 2.7z" clip-path="url(#p94a45562aa)" style="fill:#d2dbe8"/>
                <path d="m252.344 221.67 2.679-1.254 1.501-.578-2.676 1.253z" clip-path="url(#p94a45562aa)" style="fill:#7597f6"/>
                <path d="m215.099 226.996 2.626.112 1.581.356-2.622-.112z" clip-path="url(#p94a45562aa)" style="fill:#6485ec"/>
                <path d="m336.403 116.831 2.907-3.566 1.5-4.485-2.91 3.576z" clip-path="url(#p94a45562aa)" style="fill:#c12b30"/>
                <path d="m283.137 201.618 2.75-2.249 1.463-1.633-2.75 2.25z" clip-path="url(#p94a45562aa)" style="fill:#aac7fd"/>
                <path d="m304.433 177.505 2.81-2.855 1.458-2.566-2.81 2.858z" clip-path="url(#p94a45562aa)" style="fill:#e0dbd8"/>
                <path d="m232.96 226.863 2.646-.567 1.54-.048-2.643.566zM208.257 225.874l2.622.382 1.598.494-2.617-.381z" clip-path="url(#p94a45562aa)" style="fill:#6687ed"/>
                <path d="m164.945 202.958 2.632 2.274 1.716 1.089-2.625-2.268z" clip-path="url(#p94a45562aa)" style="fill:#9ebeff"/>
                <path d="m317.372 157.464 2.85-3.181 1.466-3.242-2.851 3.187z" clip-path="url(#p94a45562aa)" style="fill:#f7b99e"/>
                <path d="m191.914 220.537 2.617 1.056 1.642.776-2.611-1.054z" clip-path="url(#p94a45562aa)" style="fill:#7295f4"/>
                <path d="m269.075 212.71 2.715-1.816 1.477-1.105-2.714 1.816z" clip-path="url(#p94a45562aa)" style="fill:#8db0fe"/>
                <path d="m248.164 223.232 2.671-1.116 1.51-.446-2.67 1.115z" clip-path="url(#p94a45562aa)" style="fill:#7093f3"/>
                <path d="m293.033 191.675 2.778-2.547 1.458-2.032-2.778 2.548z" clip-path="url(#p94a45562aa)" style="fill:#c3d5f4"/>
                <path d="m221.934 227.44 2.633-.159 1.565.221-2.63.159z" clip-path="url(#p94a45562aa)" style="fill:#6384eb"/>
                <path d="m310.16 169.386 2.828-3.014 1.46-2.836-2.83 3.018z" clip-path="url(#p94a45562aa)" style="fill:#eed0c0"/>
                <path d="m201.403 224.074 2.62.651 1.616.633-2.614-.65z" clip-path="url(#p94a45562aa)" style="fill:#6b8df0"/>
                <path d="m334.91 121.168 2.904-3.556 1.496-4.347-2.907 3.566z" clip-path="url(#p94a45562aa)" style="fill:#cc403a"/>
                <path d="m324.626 144.153 2.874-3.356 1.475-3.653-2.875 3.363z" clip-path="url(#p94a45562aa)" style="fill:#f39475"/>
                <path d="m167.577 205.232 2.629 2.137 1.71 1.085-2.623-2.133z" clip-path="url(#p94a45562aa)" style="fill:#98b9ff"/>
                <path d="m182.396 215.64 2.62 1.46 1.669.924-2.614-1.458z" clip-path="url(#p94a45562aa)" style="fill:#7ea1fa"/>
                <path d="m273.267 209.79 2.726-1.96 1.472-1.238-2.725 1.96z" clip-path="url(#p94a45562aa)" style="fill:#96b7ff"/>
                <path d="m243.982 224.524 2.665-.979 1.517-.313-2.662.977z" clip-path="url(#p94a45562aa)" style="fill:#6e90f2"/>
                <path d="m287.35 197.736 2.764-2.397 1.46-1.766-2.763 2.397z" clip-path="url(#p94a45562aa)" style="fill:#b5cdfa"/>
                <path d="m228.768 227.207 2.642-.43 1.55.086-2.639.43z" clip-path="url(#p94a45562aa)" style="fill:#6687ed"/>
                <path d="m333.42 125.367 2.902-3.546 1.492-4.209-2.905 3.556z" clip-path="url(#p94a45562aa)" style="fill:#d55042"/>
                <path d="m170.206 207.37 2.627 2.002 1.705 1.08-2.622-1.998z" clip-path="url(#p94a45562aa)" style="fill:#94b6ff"/>
                <path d="m315.91 160.567 2.848-3.177 1.464-3.107-2.85 3.181z" clip-path="url(#p94a45562aa)" style="fill:#f6bea4"/>
                <path d="m194.531 221.593 2.619.921 1.637.774-2.614-.92z" clip-path="url(#p94a45562aa)" style="fill:#7295f4"/>
                <path d="m277.465 206.592 2.739-2.103 1.468-1.37-2.738 2.104z" clip-path="url(#p94a45562aa)" style="fill:#9fbfff"/>
                <path d="m239.796 225.545 2.66-.841 1.526-.18-2.656.84z" clip-path="url(#p94a45562aa)" style="fill:#6b8df0"/>
                <path d="m323.156 147.664 2.871-3.349 1.473-3.518-2.874 3.356z" clip-path="url(#p94a45562aa)" style="fill:#f59d7e"/>
                <path d="m185.016 217.1 2.62 1.327 1.663.92-2.614-1.323z" clip-path="url(#p94a45562aa)" style="fill:#7da0f9"/>
                <path d="m302.976 179.936 2.81-2.853 1.457-2.433-2.81 2.855z" clip-path="url(#p94a45562aa)" style="fill:#dddcdc"/>
                <path d="m210.879 226.256 2.625.246 1.595.494-2.622-.246z" clip-path="url(#p94a45562aa)" style="fill:#688aef"/>
                <path d="m297.269 187.096 2.793-2.697 1.457-2.165-2.793 2.699z" clip-path="url(#p94a45562aa)" style="fill:#cedaeb"/>
                <path d="m331.935 129.43 2.9-3.538 1.487-4.07-2.902 3.545z" clip-path="url(#p94a45562aa)" style="fill:#dd5f4b"/>
                <path d="m217.725 227.108 2.632-.025 1.577.357-2.628.024z" clip-path="url(#p94a45562aa)" style="fill:#6687ed"/>
                <path d="m172.833 209.372 2.626 1.868 1.699 1.076-2.62-1.864z" clip-path="url(#p94a45562aa)" style="fill:#90b2fe"/>
                <path d="m308.7 172.084 2.829-3.01 1.459-2.702-2.829 3.014z" clip-path="url(#p94a45562aa)" style="fill:#ebd3c6"/>
                <path d="m204.022 224.725 2.623.517 1.612.632-2.618-.516z" clip-path="url(#p94a45562aa)" style="fill:#6b8df0"/>
                <path d="m281.672 203.12 2.75-2.25 1.465-1.501-2.75 2.25z" clip-path="url(#p94a45562aa)" style="fill:#a9c6fd"/>
                <path d="m235.606 226.296 2.654-.705 1.536-.046-2.65.703z" clip-path="url(#p94a45562aa)" style="fill:#6b8df0"/>
                <path d="m291.574 193.573 2.779-2.546 1.458-1.899-2.778 2.547z" clip-path="url(#p94a45562aa)" style="fill:#c1d4f4"/>
                <path d="m224.567 227.281 2.64-.295 1.56.221-2.635.295z" clip-path="url(#p94a45562aa)" style="fill:#6788ee"/>
                <path d="m259.21 218.444 2.699-1.535 1.491-.708-2.695 1.533z" clip-path="url(#p94a45562aa)" style="fill:#81a4fb"/>
                <path d="m147.31 185.18 2.66 3.1 1.77 1.273-2.653-3.092z" clip-path="url(#p94a45562aa)" style="fill:#cad8ef"/>
                <path d="m330.453 133.355 2.897-3.529 1.484-3.934-2.899 3.538z" clip-path="url(#p94a45562aa)" style="fill:#e36c55"/>
                <path d="m149.97 188.28 2.656 2.962 1.763 1.265-2.649-2.954z" clip-path="url(#p94a45562aa)" style="fill:#c4d5f3"/>
                <path d="m175.46 211.24 2.625 1.734 1.692 1.07-2.62-1.728z" clip-path="url(#p94a45562aa)" style="fill:#8caffe"/>
                <path d="m263.4 216.2 2.708-1.675 1.485-.84-2.705 1.674z" clip-path="url(#p94a45562aa)" style="fill:#86a9fc"/>
                <path d="m255.023 220.416 2.69-1.395 1.498-.577-2.687 1.394z" clip-path="url(#p94a45562aa)" style="fill:#7b9ff9"/>
                <path d="m144.643 181.941 2.666 3.239 1.778 1.28-2.66-3.229z" clip-path="url(#p94a45562aa)" style="fill:#d1dae9"/>
                <path d="m321.688 151.04 2.87-3.342 1.47-3.383-2.872 3.35z" clip-path="url(#p94a45562aa)" style="fill:#f6a586"/>
                <path d="m187.635 218.427 2.62 1.192 1.66.918-2.616-1.19z" clip-path="url(#p94a45562aa)" style="fill:#7b9ff9"/>
                <path d="m152.626 191.242 2.651 2.825 1.757 1.257-2.645-2.817z" clip-path="url(#p94a45562aa)" style="fill:#bcd2f7"/>
                <path d="m314.448 163.536 2.847-3.172 1.463-2.974-2.849 3.177z" clip-path="url(#p94a45562aa)" style="fill:#f5c4ac"/>
                <path d="m197.15 222.514 2.621.787 1.632.773-2.616-.786z" clip-path="url(#p94a45562aa)" style="fill:#7295f4"/>
                <path d="m267.593 213.684 2.717-1.817 1.48-.973-2.715 1.817z" clip-path="url(#p94a45562aa)" style="fill:#8db0fe"/>
                <path d="m250.835 222.116 2.682-1.256 1.506-.444-2.679 1.254z" clip-path="url(#p94a45562aa)" style="fill:#779af7"/>
                <path d="m155.277 194.067 2.648 2.687 1.75 1.25-2.641-2.68z" clip-path="url(#p94a45562aa)" style="fill:#b7cff9"/>
                <path d="m157.925 196.754 2.643 2.551 1.743 1.244-2.637-2.544z" clip-path="url(#p94a45562aa)" style="fill:#b1cbfc"/>
                <path d="m285.887 199.369 2.765-2.397 1.462-1.633-2.764 2.397z" clip-path="url(#p94a45562aa)" style="fill:#b5cdfa"/>
                <path d="m231.41 226.776 2.65-.568 1.546.088-2.646.567z" clip-path="url(#p94a45562aa)" style="fill:#6a8bef"/>
                <path d="m271.79 210.894 2.729-1.96 1.474-1.105-2.726 1.96z" clip-path="url(#p94a45562aa)" style="fill:#96b7ff"/>
                <path d="m246.647 223.545 2.674-1.117 1.514-.312-2.671 1.116z" clip-path="url(#p94a45562aa)" style="fill:#7396f5"/>
                <path d="m328.975 137.144 2.895-3.521 1.48-3.797-2.897 3.529z" clip-path="url(#p94a45562aa)" style="fill:#e9785d"/>
                <path d="m301.52 182.234 2.81-2.85 1.456-2.3-2.81 2.852z" clip-path="url(#p94a45562aa)" style="fill:#dadce0"/>
                <path d="m213.504 226.502 2.631.111 1.59.495-2.626-.112z" clip-path="url(#p94a45562aa)" style="fill:#6a8bef"/>
                <path d="m178.085 212.974 2.625 1.598 1.686 1.068-2.619-1.595z" clip-path="url(#p94a45562aa)" style="fill:#89acfd"/>
                <path d="m160.568 199.305 2.64 2.415 1.737 1.238-2.634-2.409z" clip-path="url(#p94a45562aa)" style="fill:#abc8fd"/>
                <path d="m307.243 174.65 2.828-3.007 1.458-2.57-2.828 3.011z" clip-path="url(#p94a45562aa)" style="fill:#e9d5cb"/>
                <path d="m206.645 225.242 2.626.382 1.608.632-2.622-.382z" clip-path="url(#p94a45562aa)" style="fill:#6c8ff1"/>
                <path d="m295.811 189.128 2.794-2.696 1.457-2.033-2.793 2.697z" clip-path="url(#p94a45562aa)" style="fill:#cdd9ec"/>
                <path d="m220.357 227.083 2.637-.16 1.573.358-2.633.16z" clip-path="url(#p94a45562aa)" style="fill:#6a8bef"/>
                <path d="m320.222 154.283 2.868-3.338 1.468-3.247-2.87 3.343z" clip-path="url(#p94a45562aa)" style="fill:#f7ac8e"/>
                <path d="m275.993 207.83 2.74-2.105 1.47-1.236-2.738 2.103z" clip-path="url(#p94a45562aa)" style="fill:#9fbfff"/>
                <path d="m242.456 224.704 2.668-.98 1.523-.179-2.665.979z" clip-path="url(#p94a45562aa)" style="fill:#7295f4"/>
                <path d="m190.256 219.619 2.622 1.057 1.653.917-2.617-1.056z" clip-path="url(#p94a45562aa)" style="fill:#7a9df8"/>
                <path d="m163.209 201.72 2.638 2.28 1.73 1.232-2.632-2.274z" clip-path="url(#p94a45562aa)" style="fill:#a6c4fe"/>
                <path d="m312.988 166.372 2.847-3.169 1.46-2.84-2.847 3.173z" clip-path="url(#p94a45562aa)" style="fill:#f3c8b2"/>
                <path d="m199.771 223.3 2.624.653 1.627.772-2.619-.651z" clip-path="url(#p94a45562aa)" style="fill:#7295f4"/>
                <path d="m327.5 140.797 2.893-3.514 1.477-3.66-2.895 3.521z" clip-path="url(#p94a45562aa)" style="fill:#ed8366"/>
                <path d="m180.71 214.572 2.625 1.464 1.68 1.065-2.619-1.461z" clip-path="url(#p94a45562aa)" style="fill:#88abfd"/>
                <path d="m290.114 195.339 2.78-2.546 1.459-1.766-2.779 2.546z" clip-path="url(#p94a45562aa)" style="fill:#bfd3f6"/>
                <path d="m227.206 226.986 2.647-.433 1.557.223-2.642.431z" clip-path="url(#p94a45562aa)" style="fill:#6b8df0"/>
                <path d="m165.847 204 2.635 2.142 1.724 1.227-2.63-2.137z" clip-path="url(#p94a45562aa)" style="fill:#a1c0ff"/>
                <path d="m280.204 204.489 2.752-2.25 1.467-1.369-2.751 2.25z" clip-path="url(#p94a45562aa)" style="fill:#a9c6fd"/>
                <path d="m238.26 225.591 2.663-.843 1.533-.044-2.66.841z" clip-path="url(#p94a45562aa)" style="fill:#7093f3"/>
                <path d="m168.482 206.142 2.634 2.008 1.717 1.222-2.627-2.003z" clip-path="url(#p94a45562aa)" style="fill:#9dbdff"/>
                <path d="m318.758 157.39 2.867-3.332 1.465-3.113-2.868 3.338z" clip-path="url(#p94a45562aa)" style="fill:#f7b396"/>
                <path d="m192.878 220.676 2.624.923 1.648.915-2.619-.921z" clip-path="url(#p94a45562aa)" style="fill:#7a9df8"/>
                <path d="m305.786 177.083 2.828-3.005 1.457-2.435-2.828 3.007z" clip-path="url(#p94a45562aa)" style="fill:#e6d7cf"/>
                <path d="m209.27 225.624 2.631.246 1.603.632-2.625-.246z" clip-path="url(#p94a45562aa)" style="fill:#6f92f3"/>
                <path d="m300.062 184.399 2.81-2.849 1.458-2.167-2.81 2.85z" clip-path="url(#p94a45562aa)" style="fill:#d8dce2"/>
                <path d="m216.135 226.613 2.636-.024 1.586.494-2.632.025z" clip-path="url(#p94a45562aa)" style="fill:#6c8ff1"/>
                <path d="m261.909 216.91 2.71-1.679 1.489-.706-2.708 1.676z" clip-path="url(#p94a45562aa)" style="fill:#89acfd"/>
                <path d="m257.712 219.02 2.7-1.536 1.497-.575-2.698 1.535z" clip-path="url(#p94a45562aa)" style="fill:#84a7fc"/>
                <path d="m326.027 144.315 2.892-3.507 1.474-3.525-2.893 3.514z" clip-path="url(#p94a45562aa)" style="fill:#f18d6f"/>
                <path d="m183.335 216.036 2.625 1.329 1.675 1.062-2.62-1.326z" clip-path="url(#p94a45562aa)" style="fill:#85a8fc"/>
                <path d="m284.423 200.87 2.766-2.397 1.463-1.5-2.765 2.396z" clip-path="url(#p94a45562aa)" style="fill:#b3cdfb"/>
                <path d="m234.06 226.208 2.658-.707 1.542.09-2.654.705z" clip-path="url(#p94a45562aa)" style="fill:#6f92f3"/>
                <path d="m266.108 214.525 2.719-1.82 1.483-.838-2.717 1.817z" clip-path="url(#p94a45562aa)" style="fill:#90b2fe"/>
                <path d="m253.517 220.86 2.692-1.397 1.503-.442-2.689 1.395z" clip-path="url(#p94a45562aa)" style="fill:#7ea1fa"/>
                <path d="m311.529 169.074 2.846-3.165 1.46-2.706-2.847 3.169z" clip-path="url(#p94a45562aa)" style="fill:#f2cbb7"/>
                <path d="m202.395 223.953 2.627.517 1.623.772-2.623-.517z" clip-path="url(#p94a45562aa)" style="fill:#7396f5"/>
                <path d="m294.353 191.027 2.794-2.696 1.458-1.9-2.794 2.697z" clip-path="url(#p94a45562aa)" style="fill:#cbd8ee"/>
                <path d="m222.994 226.923 2.643-.297 1.57.36-2.64.295z" clip-path="url(#p94a45562aa)" style="fill:#6c8ff1"/>
                <path d="m171.116 208.15 2.632 1.873 1.711 1.217-2.626-1.868z" clip-path="url(#p94a45562aa)" style="fill:#98b9ff"/>
                <path d="m270.31 211.867 2.73-1.963 1.479-.97-2.729 1.96z" clip-path="url(#p94a45562aa)" style="fill:#97b8ff"/>
                <path d="m249.321 222.428 2.684-1.259 1.512-.309-2.682 1.256z" clip-path="url(#p94a45562aa)" style="fill:#7a9df8"/>
                <path d="m274.519 208.934 2.741-2.107 1.473-1.102-2.74 2.104z" clip-path="url(#p94a45562aa)" style="fill:#9fbfff"/>
                <path d="m245.124 223.724 2.677-1.12 1.52-.176-2.674 1.117z" clip-path="url(#p94a45562aa)" style="fill:#779af7"/>
                <path d="m317.295 160.364 2.867-3.328 1.463-2.978-2.867 3.332z" clip-path="url(#p94a45562aa)" style="fill:#f7b89c"/>
                <path d="m195.502 221.6 2.626.787 1.643.914-2.621-.787z" clip-path="url(#p94a45562aa)" style="fill:#7a9df8"/>
                <path d="m324.558 147.698 2.89-3.502 1.47-3.388-2.89 3.507z" clip-path="url(#p94a45562aa)" style="fill:#f39577"/>
                <path d="m185.96 217.365 2.626 1.194 1.67 1.06-2.62-1.192z" clip-path="url(#p94a45562aa)" style="fill:#84a7fc"/>
                <path d="m173.748 210.023 2.632 1.737 1.705 1.214-2.626-1.734z" clip-path="url(#p94a45562aa)" style="fill:#96b7ff"/>
                <path d="m145.517 183.753 2.668 3.109 1.785 1.418-2.66-3.1z" clip-path="url(#p94a45562aa)" style="fill:#d2dbe8"/>
                <path d="m288.652 196.972 2.78-2.545 1.461-1.634-2.779 2.546z" clip-path="url(#p94a45562aa)" style="fill:#bfd3f6"/>
                <path d="m148.185 186.862 2.663 2.97 1.778 1.41-2.656-2.962z" clip-path="url(#p94a45562aa)" style="fill:#cbd8ee"/>
                <path d="m229.853 226.553 2.653-.57 1.554.225-2.65.568z" clip-path="url(#p94a45562aa)" style="fill:#6f92f3"/>
                <path d="m142.844 180.506 2.673 3.247 1.792 1.427-2.666-3.239z" clip-path="url(#p94a45562aa)" style="fill:#d8dce2"/>
                <path d="m150.848 189.832 2.659 2.832 1.77 1.403-2.651-2.825z" clip-path="url(#p94a45562aa)" style="fill:#c5d6f2"/>
                <path d="m153.507 192.664 2.654 2.695 1.764 1.395-2.648-2.687z" clip-path="url(#p94a45562aa)" style="fill:#bfd3f6"/>
                <path d="m278.733 205.725 2.754-2.252 1.47-1.235-2.753 2.25z" clip-path="url(#p94a45562aa)" style="fill:#a9c6fd"/>
                <path d="m304.33 179.383 2.827-3.003 1.457-2.302-2.828 3.005z" clip-path="url(#p94a45562aa)" style="fill:#e4d9d2"/>
                <path d="m240.923 224.748 2.671-.982 1.53-.042-2.668.98z" clip-path="url(#p94a45562aa)" style="fill:#7699f6"/>
                <path d="m211.901 225.87 2.635.11 1.6.633-2.632-.11z" clip-path="url(#p94a45562aa)" style="fill:#7093f3"/>
                <path d="m156.16 195.359 2.651 2.558 1.757 1.388-2.643-2.55z" clip-path="url(#p94a45562aa)" style="fill:#bad0f8"/>
                <path d="m298.605 186.432 2.81-2.848 1.458-2.034-2.81 2.849z" clip-path="url(#p94a45562aa)" style="fill:#d6dce4"/>
                <path d="m218.77 226.589 2.642-.162 1.582.496-2.637.16z" clip-path="url(#p94a45562aa)" style="fill:#6f92f3"/>
                <path d="m310.071 171.643 2.846-3.162 1.458-2.572-2.846 3.165z" clip-path="url(#p94a45562aa)" style="fill:#f0cdbb"/>
                <path d="m205.022 224.47 2.631.382 1.618.772-2.626-.382z" clip-path="url(#p94a45562aa)" style="fill:#7597f6"/>
                <path d="m176.38 211.76 2.63 1.602 1.7 1.21-2.625-1.598z" clip-path="url(#p94a45562aa)" style="fill:#93b5fe"/>
                <path d="m158.811 197.917 2.648 2.42 1.75 1.383-2.64-2.415z" clip-path="url(#p94a45562aa)" style="fill:#b5cdfa"/>
                <path d="m323.09 150.945 2.888-3.496 1.47-3.253-2.89 3.502z" clip-path="url(#p94a45562aa)" style="fill:#f59d7e"/>
                <path d="m188.586 218.56 2.628 1.059 1.664 1.057-2.622-1.057z" clip-path="url(#p94a45562aa)" style="fill:#82a6fb"/>
                <path d="m292.893 192.793 2.795-2.695 1.46-1.767-2.795 2.696z" clip-path="url(#p94a45562aa)" style="fill:#cad8ef"/>
                <path d="m225.637 226.626 2.65-.434 1.566.361-2.647.433z" clip-path="url(#p94a45562aa)" style="fill:#7093f3"/>
                <path d="m282.956 202.238 2.767-2.398 1.466-1.367-2.766 2.397z" clip-path="url(#p94a45562aa)" style="fill:#b3cdfb"/>
                <path d="m236.718 225.501 2.665-.845 1.54.092-2.663.843z" clip-path="url(#p94a45562aa)" style="fill:#7597f6"/>
                <path d="m315.835 163.203 2.865-3.324 1.462-2.843-2.867 3.328z" clip-path="url(#p94a45562aa)" style="fill:#f6bda2"/>
                <path d="m198.128 222.387 2.63.653 1.637.913-2.624-.652z" clip-path="url(#p94a45562aa)" style="fill:#7a9df8"/>
                <path d="m161.459 200.338 2.644 2.285 1.744 1.376-2.638-2.279z" clip-path="url(#p94a45562aa)" style="fill:#afcafc"/>
                <path d="m260.412 217.484 2.712-1.68 1.494-.573-2.71 1.678z" clip-path="url(#p94a45562aa)" style="fill:#8caffe"/>
                <path d="m264.618 215.231 2.722-1.821 1.487-.705-2.72 1.82z" clip-path="url(#p94a45562aa)" style="fill:#92b4fe"/>
                <path d="m256.209 219.463 2.702-1.54 1.501-.44-2.7 1.538z" clip-path="url(#p94a45562aa)" style="fill:#86a9fc"/>
                <path d="m179.01 213.362 2.631 1.467 1.694 1.207-2.625-1.464z" clip-path="url(#p94a45562aa)" style="fill:#90b2fe"/>
                <path d="m164.103 202.623 2.642 2.148 1.737 1.371-2.635-2.143z" clip-path="url(#p94a45562aa)" style="fill:#aac7fd"/>
                <path d="m268.827 212.705 2.732-1.964 1.481-.837-2.73 1.963z" clip-path="url(#p94a45562aa)" style="fill:#98b9ff"/>
                <path d="m252.005 221.17 2.695-1.4 1.509-.307-2.692 1.397z" clip-path="url(#p94a45562aa)" style="fill:#82a6fb"/>
                <path d="m287.189 198.473 2.78-2.546 1.463-1.5-2.78 2.545z" clip-path="url(#p94a45562aa)" style="fill:#bed2f6"/>
                <path d="m232.506 225.983 2.662-.708 1.55.226-2.658.707z" clip-path="url(#p94a45562aa)" style="fill:#7597f6"/>
                <path d="m302.873 181.55 2.827-3.002 1.457-2.168-2.827 3.003z" clip-path="url(#p94a45562aa)" style="fill:#e1dad6"/>
                <path d="m214.536 225.98 2.64-.025 1.595.634-2.636.024z" clip-path="url(#p94a45562aa)" style="fill:#7396f5"/>
                <path d="m308.614 174.078 2.845-3.16 1.458-2.437-2.846 3.162z" clip-path="url(#p94a45562aa)" style="fill:#eed0c0"/>
                <path d="m321.625 154.058 2.887-3.491 1.466-3.118-2.888 3.496z" clip-path="url(#p94a45562aa)" style="fill:#f6a385"/>
                <path d="m207.653 224.852 2.635.246 1.613.772-2.63-.246z" clip-path="url(#p94a45562aa)" style="fill:#7699f6"/>
                <path d="m273.04 209.904 2.744-2.107 1.476-.97-2.741 2.107z" clip-path="url(#p94a45562aa)" style="fill:#a1c0ff"/>
                <path d="m247.8 222.604 2.688-1.26 1.517-.175-2.684 1.259z" clip-path="url(#p94a45562aa)" style="fill:#7ea1fa"/>
                <path d="m191.214 219.619 2.63.924 1.658 1.056-2.624-.923z" clip-path="url(#p94a45562aa)" style="fill:#82a6fb"/>
                <path d="m166.745 204.771 2.64 2.013 1.731 1.366-2.634-2.008z" clip-path="url(#p94a45562aa)" style="fill:#a6c4fe"/>
                <path d="m297.147 188.331 2.81-2.847 1.459-1.9-2.81 2.848z" clip-path="url(#p94a45562aa)" style="fill:#d5dbe5"/>
                <path d="m221.412 226.427 2.648-.298 1.577.498-2.643.296z" clip-path="url(#p94a45562aa)" style="fill:#7396f5"/>
                <path d="m181.641 214.829 2.632 1.332 1.687 1.204-2.625-1.329z" clip-path="url(#p94a45562aa)" style="fill:#8db0fe"/>
                <path d="m314.375 165.909 2.865-3.32 1.46-2.71-2.865 3.324z" clip-path="url(#p94a45562aa)" style="fill:#f5c0a7"/>
                <path d="m200.757 223.04 2.632.518 1.633.912-2.627-.517z" clip-path="url(#p94a45562aa)" style="fill:#7a9df8"/>
                <path d="m277.26 206.827 2.756-2.252 1.471-1.102-2.754 2.252z" clip-path="url(#p94a45562aa)" style="fill:#aac7fd"/>
                <path d="m243.594 223.766 2.68-1.122 1.527-.04-2.677 1.12z" clip-path="url(#p94a45562aa)" style="fill:#7da0f9"/>
                <path d="m169.386 206.784 2.638 1.877 1.724 1.362-2.632-1.873z" clip-path="url(#p94a45562aa)" style="fill:#a2c1ff"/>
                <path d="m291.432 194.427 2.796-2.696 1.46-1.633-2.795 2.695z" clip-path="url(#p94a45562aa)" style="fill:#c9d7f0"/>
                <path d="m228.288 226.192 2.657-.571 1.561.362-2.653.57z" clip-path="url(#p94a45562aa)" style="fill:#7699f6"/>
                <path d="m281.487 203.473 2.769-2.399 1.467-1.234-2.767 2.398z" clip-path="url(#p94a45562aa)" style="fill:#b3cdfb"/>
                <path d="m320.162 157.036 2.886-3.486 1.464-2.983-2.887 3.49z" clip-path="url(#p94a45562aa)" style="fill:#f7aa8c"/>
                <path d="m239.383 224.656 2.675-.984 1.536.094-2.671.982z" clip-path="url(#p94a45562aa)" style="fill:#7b9ff9"/>
                <path d="m193.843 220.543 2.632.79 1.653 1.054-2.626-.788z" clip-path="url(#p94a45562aa)" style="fill:#82a6fb"/>
                <path d="m184.273 216.16 2.632 1.197 1.681 1.202-2.626-1.194z" clip-path="url(#p94a45562aa)" style="fill:#8caffe"/>
                <path d="m172.024 208.66 2.638 1.742 1.718 1.358-2.632-1.737z" clip-path="url(#p94a45562aa)" style="fill:#9fbfff"/>
                <path d="m143.71 182.18 2.675 3.117 1.8 1.565-2.668-3.109z" clip-path="url(#p94a45562aa)" style="fill:#d9dce1"/>
                <path d="m146.385 185.297 2.67 2.978 1.793 1.557-2.663-2.97z" clip-path="url(#p94a45562aa)" style="fill:#d3dbe7"/>
                <path d="m307.157 176.38 2.845-3.158 1.457-2.303-2.845 3.159z" clip-path="url(#p94a45562aa)" style="fill:#ecd3c5"/>
                <path d="m141.029 178.924 2.681 3.256 1.807 1.573-2.673-3.247z" clip-path="url(#p94a45562aa)" style="fill:#dfdbd9"/>
                <path d="m210.288 225.098 2.64.11 1.608.773-2.635-.111z" clip-path="url(#p94a45562aa)" style="fill:#799cf8"/>
                <path d="m149.056 188.275 2.665 2.84 1.786 1.55-2.659-2.833z" clip-path="url(#p94a45562aa)" style="fill:#cdd9ec"/>
                <path d="m301.416 183.584 2.827-3 1.457-2.036-2.827 3.002z" clip-path="url(#p94a45562aa)" style="fill:#e0dbd8"/>
                <path d="m217.176 225.955 2.646-.163 1.59.635-2.641.162z" clip-path="url(#p94a45562aa)" style="fill:#7699f6"/>
                <path d="m151.721 191.115 2.662 2.703 1.778 1.541-2.654-2.695z" clip-path="url(#p94a45562aa)" style="fill:#c7d7f0"/>
                <path d="m312.917 168.48 2.864-3.317 1.459-2.575-2.865 3.321z" clip-path="url(#p94a45562aa)" style="fill:#f4c5ad"/>
                <path d="m203.39 223.558 2.635.381 1.628.913-2.63-.382z" clip-path="url(#p94a45562aa)" style="fill:#7b9ff9"/>
                <path d="m263.124 215.804 2.724-1.824 1.492-.57-2.722 1.821z" clip-path="url(#p94a45562aa)" style="fill:#94b6ff"/>
                <path d="m258.911 217.923 2.715-1.682 1.498-.437-2.712 1.68z" clip-path="url(#p94a45562aa)" style="fill:#8fb1fe"/>
                <path d="m285.723 199.84 2.782-2.547 1.465-1.366-2.781 2.546z" clip-path="url(#p94a45562aa)" style="fill:#bed2f6"/>
                <path d="m235.168 225.275 2.67-.848 1.545.23-2.665.844z" clip-path="url(#p94a45562aa)" style="fill:#7a9df8"/>
                <path d="m154.383 193.818 2.657 2.564 1.771 1.535-2.65-2.558z" clip-path="url(#p94a45562aa)" style="fill:#c3d5f4"/>
                <path d="m267.34 213.41 2.734-1.966 1.485-.703-2.732 1.964z" clip-path="url(#p94a45562aa)" style="fill:#9bbcff"/>
                <path d="m254.7 219.77 2.705-1.542 1.506-.305-2.702 1.54z" clip-path="url(#p94a45562aa)" style="fill:#8badfd"/>
                <path d="m295.688 190.098 2.811-2.847 1.459-1.767-2.81 2.847z" clip-path="url(#p94a45562aa)" style="fill:#d4dbe6"/>
                <path d="m224.06 226.13 2.655-.437 1.573.5-2.65.433z" clip-path="url(#p94a45562aa)" style="fill:#779af7"/>
                <path d="m174.662 210.402 2.637 1.605 1.712 1.355-2.631-1.602z" clip-path="url(#p94a45562aa)" style="fill:#9bbcff"/>
                <path d="m157.04 196.382 2.655 2.428 1.764 1.528-2.648-2.421z" clip-path="url(#p94a45562aa)" style="fill:#bed2f6"/>
                <path d="m186.905 217.357 2.633 1.061 1.676 1.2-2.628-1.059z" clip-path="url(#p94a45562aa)" style="fill:#8caffe"/>
                <path d="m271.559 210.741 2.745-2.11 1.48-.834-2.744 2.107z" clip-path="url(#p94a45562aa)" style="fill:#a3c2fe"/>
                <path d="m250.488 221.343 2.698-1.402 1.514-.171-2.695 1.4z" clip-path="url(#p94a45562aa)" style="fill:#86a9fc"/>
                <path d="m318.7 159.88 2.885-3.483 1.463-2.847-2.886 3.486z" clip-path="url(#p94a45562aa)" style="fill:#f7b093"/>
                <path d="m196.475 221.332 2.634.654 1.648 1.054-2.629-.653z" clip-path="url(#p94a45562aa)" style="fill:#82a6fb"/>
                <path d="m159.695 198.81 2.651 2.29 1.757 1.523-2.644-2.285z" clip-path="url(#p94a45562aa)" style="fill:#b9d0f9"/>
                <path d="m275.784 207.797 2.757-2.255 1.475-.967-2.756 2.252z" clip-path="url(#p94a45562aa)" style="fill:#abc8fd"/>
                <path d="m246.275 222.644 2.69-1.264 1.523-.037-2.687 1.26z" clip-path="url(#p94a45562aa)" style="fill:#84a7fc"/>
                <path d="m289.97 195.927 2.796-2.697 1.462-1.499-2.796 2.696z" clip-path="url(#p94a45562aa)" style="fill:#c9d7f0"/>
                <path d="m230.945 225.62 2.665-.71 1.558.365-2.662.708z" clip-path="url(#p94a45562aa)" style="fill:#7a9df8"/>
                <path d="m177.299 212.007 2.637 1.47 1.705 1.352-2.63-1.467z" clip-path="url(#p94a45562aa)" style="fill:#9abbff"/>
                <path d="m305.7 178.548 2.845-3.156 1.457-2.17-2.845 3.158z" clip-path="url(#p94a45562aa)" style="fill:#ead4c8"/>
                <path d="m162.346 201.1 2.649 2.154 1.75 1.517-2.642-2.148z" clip-path="url(#p94a45562aa)" style="fill:#b3cdfb"/>
                <path d="m212.928 225.208 2.645-.027 1.603.774-2.64.026z" clip-path="url(#p94a45562aa)" style="fill:#7a9df8"/>
                <path d="m311.46 170.919 2.863-3.316 1.458-2.44-2.864 3.318z" clip-path="url(#p94a45562aa)" style="fill:#f3c7b1"/>
                <path d="m206.025 223.94 2.64.245 1.623.913-2.635-.246z" clip-path="url(#p94a45562aa)" style="fill:#7ea1fa"/>
                <path d="m280.016 204.575 2.77-2.401 1.47-1.1-2.769 2.4z" clip-path="url(#p94a45562aa)" style="fill:#b5cdfa"/>
                <path d="m242.058 223.672 2.684-1.125 1.533.097-2.68 1.122z" clip-path="url(#p94a45562aa)" style="fill:#81a4fb"/>
                <path d="m299.958 185.484 2.828-3 1.457-1.9-2.827 3z" clip-path="url(#p94a45562aa)" style="fill:#dedcdb"/>
                <path d="m219.822 225.792 2.652-.3 1.586.637-2.648.298z" clip-path="url(#p94a45562aa)" style="fill:#7a9df8"/>
                <path d="m189.538 218.418 2.635.926 1.67 1.199-2.63-.924z" clip-path="url(#p94a45562aa)" style="fill:#8badfd"/>
                <path d="m164.995 203.254 2.647 2.018 1.744 1.512-2.64-2.013z" clip-path="url(#p94a45562aa)" style="fill:#afcafc"/>
                <path d="m317.24 162.588 2.884-3.478 1.461-2.713-2.885 3.482z" clip-path="url(#p94a45562aa)" style="fill:#f7b497"/>
                <path d="m199.109 221.986 2.637.517 1.643 1.055-2.632-.518z" clip-path="url(#p94a45562aa)" style="fill:#82a6fb"/>
                <path d="m179.936 213.478 2.637 1.334 1.7 1.349-2.632-1.332z" clip-path="url(#p94a45562aa)" style="fill:#97b8ff"/>
                <path d="m294.228 191.731 2.812-2.847 1.46-1.633-2.812 2.847z" clip-path="url(#p94a45562aa)" style="fill:#d3dbe7"/>
                <path d="m226.715 225.693 2.661-.574 1.57.502-2.658.571z" clip-path="url(#p94a45562aa)" style="fill:#7b9ff9"/>
                <path d="m284.256 201.074 2.783-2.548 1.466-1.233-2.782 2.547z" clip-path="url(#p94a45562aa)" style="fill:#bed2f6"/>
                <path d="m237.837 224.427 2.678-.987 1.543.232-2.675.984z" clip-path="url(#p94a45562aa)" style="fill:#80a3fa"/>
                <path d="m167.642 205.272 2.645 1.881 1.737 1.508-2.638-1.877z" clip-path="url(#p94a45562aa)" style="fill:#abc8fd"/>
                <path d="m261.626 216.241 2.726-1.827 1.496-.434-2.724 1.824z" clip-path="url(#p94a45562aa)" style="fill:#97b8ff"/>
                <path d="m265.848 213.98 2.736-1.97 1.49-.566-2.734 1.966z" clip-path="url(#p94a45562aa)" style="fill:#9ebeff"/>
                <path d="m257.405 218.228 2.717-1.685 1.504-.302-2.715 1.682z" clip-path="url(#p94a45562aa)" style="fill:#93b5fe"/>
                <path d="m192.173 219.344 2.637.79 1.665 1.198-2.632-.789z" clip-path="url(#p94a45562aa)" style="fill:#8badfd"/>
                <path d="m270.074 211.444 2.747-2.113 1.483-.7-2.745 2.11z" clip-path="url(#p94a45562aa)" style="fill:#a5c3fe"/>
                <path d="m253.186 219.94 2.708-1.544 1.511-.168-2.705 1.542z" clip-path="url(#p94a45562aa)" style="fill:#8db0fe"/>
                <path d="m182.573 214.812 2.638 1.199 1.694 1.346-2.632-1.196z" clip-path="url(#p94a45562aa)" style="fill:#96b7ff"/>
                <path d="m310.002 173.222 2.863-3.313 1.458-2.306-2.864 3.316z" clip-path="url(#p94a45562aa)" style="fill:#f2cab5"/>
                <path d="m208.665 224.185 2.645.11 1.618.913-2.64-.11z" clip-path="url(#p94a45562aa)" style="fill:#80a3fa"/>
                <path d="m304.243 180.583 2.845-3.155 1.457-2.036-2.845 3.156z" clip-path="url(#p94a45562aa)" style="fill:#e9d5cb"/>
                <path d="m215.573 225.181 2.65-.164 1.6.775-2.647.163z" clip-path="url(#p94a45562aa)" style="fill:#7ea1fa"/>
                <path d="m288.505 197.293 2.798-2.697 1.463-1.366-2.796 2.697z" clip-path="url(#p94a45562aa)" style="fill:#c7d7f0"/>
                <path d="m170.287 207.153 2.644 1.746 1.73 1.503-2.637-1.741z" clip-path="url(#p94a45562aa)" style="fill:#a9c6fd"/>
                <path d="m233.61 224.91 2.673-.85 1.554.367-2.67.848z" clip-path="url(#p94a45562aa)" style="fill:#80a3fa"/>
                <path d="m274.304 208.631 2.76-2.257 1.477-.832-2.757 2.255z" clip-path="url(#p94a45562aa)" style="fill:#adc9fd"/>
                <path d="m248.965 221.38 2.7-1.405 1.52-.034-2.697 1.402z" clip-path="url(#p94a45562aa)" style="fill:#8badfd"/>
                <path d="m141.887 180.458 2.683 3.126 1.815 1.713-2.675-3.117z" clip-path="url(#p94a45562aa)" style="fill:#e0dbd8"/>
                <path d="m144.57 183.584 2.678 2.987 1.808 1.704-2.67-2.978z" clip-path="url(#p94a45562aa)" style="fill:#dbdcde"/>
                <path d="m315.781 165.163 2.883-3.476 1.46-2.577-2.884 3.478z" clip-path="url(#p94a45562aa)" style="fill:#f7b89c"/>
                <path d="m201.746 222.503 2.641.382 1.638 1.054-2.636-.381z" clip-path="url(#p94a45562aa)" style="fill:#84a7fc"/>
                <path d="m139.198 177.192 2.69 3.266 1.822 1.722-2.681-3.256z" clip-path="url(#p94a45562aa)" style="fill:#e6d7cf"/>
                <path d="m298.5 187.25 2.828-3 1.458-1.766-2.828 3z" clip-path="url(#p94a45562aa)" style="fill:#dddcdc"/>
                <path d="m222.474 225.493 2.659-.438 1.582.638-2.655.436z" clip-path="url(#p94a45562aa)" style="fill:#7ea1fa"/>
                <path d="m147.248 186.571 2.673 2.848 1.8 1.696-2.665-2.84z" clip-path="url(#p94a45562aa)" style="fill:#d6dce4"/>
                <path d="m149.921 189.42 2.669 2.709 1.793 1.689-2.662-2.703z" clip-path="url(#p94a45562aa)" style="fill:#d1dae9"/>
                <path d="m278.541 205.542 2.771-2.403 1.473-.965-2.77 2.4z" clip-path="url(#p94a45562aa)" style="fill:#b6cefa"/>
                <path d="m244.742 222.547 2.694-1.266 1.529.1-2.69 1.263z" clip-path="url(#p94a45562aa)" style="fill:#88abfd"/>
                <path d="m152.59 192.129 2.665 2.571 1.785 1.682-2.657-2.564z" clip-path="url(#p94a45562aa)" style="fill:#cbd8ee"/>
                <path d="m172.931 208.899 2.643 1.609 1.725 1.5-2.637-1.606z" clip-path="url(#p94a45562aa)" style="fill:#a5c3fe"/>
                <path d="m185.21 216.01 2.64 1.064 1.688 1.344-2.633-1.06z" clip-path="url(#p94a45562aa)" style="fill:#94b6ff"/>
                <path d="m194.81 220.134 2.64.654 1.659 1.198-2.634-.654z" clip-path="url(#p94a45562aa)" style="fill:#8badfd"/>
                <path d="m155.255 194.7 2.661 2.434 1.779 1.676-2.655-2.428z" clip-path="url(#p94a45562aa)" style="fill:#c6d6f1"/>
                <path d="m292.766 193.23 2.813-2.848 1.461-1.498-2.812 2.847z" clip-path="url(#p94a45562aa)" style="fill:#d2dbe8"/>
                <path d="m229.376 225.12 2.67-.714 1.564.504-2.665.71z" clip-path="url(#p94a45562aa)" style="fill:#81a4fb"/>
                <path d="m282.785 202.174 2.785-2.55 1.469-1.098-2.783 2.548z" clip-path="url(#p94a45562aa)" style="fill:#bfd3f6"/>
                <path d="m240.515 223.44 2.688-1.128 1.539.235-2.684 1.125z" clip-path="url(#p94a45562aa)" style="fill:#86a9fc"/>
                <path d="m157.916 197.134 2.658 2.297 1.772 1.67-2.651-2.291z" clip-path="url(#p94a45562aa)" style="fill:#c1d4f4"/>
                <path d="m308.545 175.392 2.864-3.312 1.456-2.171-2.863 3.313z" clip-path="url(#p94a45562aa)" style="fill:#f1ccb8"/>
                <path d="m211.31 224.294 2.65-.028 1.613.915-2.645.027z" clip-path="url(#p94a45562aa)" style="fill:#82a6fb"/>
                <path d="m175.574 210.508 2.643 1.473 1.719 1.497-2.637-1.47z" clip-path="url(#p94a45562aa)" style="fill:#a3c2fe"/>
                <path d="m302.786 182.484 2.845-3.155 1.457-1.901-2.845 3.155z" clip-path="url(#p94a45562aa)" style="fill:#e8d6cc"/>
                <path d="m218.223 225.017 2.657-.301 1.594.777-2.652.3z" clip-path="url(#p94a45562aa)" style="fill:#81a4fb"/>
                <path d="m314.323 167.603 2.883-3.473 1.458-2.443-2.883 3.476z" clip-path="url(#p94a45562aa)" style="fill:#f7bca1"/>
                <path d="m204.387 222.885 2.645.245 1.633 1.055-2.64-.246z" clip-path="url(#p94a45562aa)" style="fill:#86a9fc"/>
                <path d="m160.574 199.43 2.656 2.16 1.765 1.664-2.649-2.154z" clip-path="url(#p94a45562aa)" style="fill:#bed2f6"/>
                <path d="m187.85 217.074 2.64.927 1.683 1.343-2.635-.926z" clip-path="url(#p94a45562aa)" style="fill:#94b6ff"/>
                <path d="m287.039 198.526 2.799-2.7 1.465-1.23-2.798 2.697z" clip-path="url(#p94a45562aa)" style="fill:#c9d7f0"/>
                <path d="m236.283 224.06 2.682-.99 1.55.37-2.678.987z" clip-path="url(#p94a45562aa)" style="fill:#86a9fc"/>
                <path d="m264.352 214.414 2.739-1.972 1.493-.431-2.736 1.969z" clip-path="url(#p94a45562aa)" style="fill:#a1c0ff"/>
                <path d="m260.122 216.543 2.73-1.83 1.5-.299-2.726 1.827z" clip-path="url(#p94a45562aa)" style="fill:#9bbcff"/>
                <path d="m297.04 188.884 2.83-3.001 1.458-1.632-2.829 3z" clip-path="url(#p94a45562aa)" style="fill:#dcdddd"/>
                <path d="m225.133 225.055 2.666-.577 1.577.641-2.661.574z" clip-path="url(#p94a45562aa)" style="fill:#82a6fb"/>
                <path d="m197.45 220.788 2.642.518 1.654 1.197-2.637-.517z" clip-path="url(#p94a45562aa)" style="fill:#8caffe"/>
                <path d="m268.584 212.01 2.75-2.114 1.487-.565-2.747 2.113z" clip-path="url(#p94a45562aa)" style="fill:#a7c5fe"/>
                <path d="m255.894 218.396 2.72-1.689 1.508-.164-2.717 1.685z" clip-path="url(#p94a45562aa)" style="fill:#97b8ff"/>
                <path d="m163.23 201.59 2.654 2.023 1.758 1.659-2.647-2.018z" clip-path="url(#p94a45562aa)" style="fill:#b9d0f9"/>
                <path d="m178.217 211.98 2.644 1.338 1.712 1.494-2.637-1.334z" clip-path="url(#p94a45562aa)" style="fill:#a1c0ff"/>
                <path d="m272.82 209.331 2.762-2.26 1.481-.697-2.759 2.257z" clip-path="url(#p94a45562aa)" style="fill:#afcafc"/>
                <path d="m251.665 219.975 2.712-1.548 1.517-.03-2.708 1.544z" clip-path="url(#p94a45562aa)" style="fill:#93b5fe"/>
                <path d="m165.884 203.613 2.652 1.886 1.75 1.654-2.644-1.881z" clip-path="url(#p94a45562aa)" style="fill:#b6cefa"/>
                <path d="m277.063 206.374 2.773-2.405 1.476-.83-2.771 2.403z" clip-path="url(#p94a45562aa)" style="fill:#b7cff9"/>
                <path d="m247.436 221.28 2.703-1.408 1.526.103-2.7 1.405z" clip-path="url(#p94a45562aa)" style="fill:#90b2fe"/>
                <path d="m291.303 194.596 2.814-2.85 1.462-1.364-2.813 2.848z" clip-path="url(#p94a45562aa)" style="fill:#d2dbe8"/>
                <path d="m232.045 224.406 2.677-.852 1.561.506-2.673.85z" clip-path="url(#p94a45562aa)" style="fill:#86a9fc"/>
                <path d="m307.088 177.428 2.864-3.312 1.457-2.036-2.864 3.312z" clip-path="url(#p94a45562aa)" style="fill:#f0cdbb"/>
                <path d="m190.49 218 2.643.792 1.677 1.342-2.637-.79z" clip-path="url(#p94a45562aa)" style="fill:#94b6ff"/>
                <path d="m213.96 224.266 2.655-.165 1.608.916-2.65.164z" clip-path="url(#p94a45562aa)" style="fill:#85a8fc"/>
                <path d="m312.865 169.909 2.883-3.472 1.458-2.307-2.883 3.473z" clip-path="url(#p94a45562aa)" style="fill:#f6bfa6"/>
                <path d="m207.032 223.13 2.65.108 1.628 1.056-2.645-.109z" clip-path="url(#p94a45562aa)" style="fill:#88abfd"/>
                <path d="m180.86 213.318 2.645 1.201 1.706 1.492-2.638-1.199z" clip-path="url(#p94a45562aa)" style="fill:#9fbfff"/>
                <path d="m301.328 184.25 2.846-3.154 1.457-1.767-2.845 3.155z" clip-path="url(#p94a45562aa)" style="fill:#e6d7cf"/>
                <path d="m220.88 224.716 2.663-.44 1.59.779-2.659.438z" clip-path="url(#p94a45562aa)" style="fill:#85a8fc"/>
                <path d="m281.312 203.139 2.787-2.553 1.471-.963-2.785 2.55z" clip-path="url(#p94a45562aa)" style="fill:#c0d4f5"/>
                <path d="m243.203 222.312 2.697-1.27 1.536.239-2.694 1.266z" clip-path="url(#p94a45562aa)" style="fill:#8db0fe"/>
                <path d="m168.536 205.499 2.65 1.75 1.745 1.65-2.644-1.746z" clip-path="url(#p94a45562aa)" style="fill:#b2ccfb"/>
                <path d="m200.092 221.306 2.647.381 1.648 1.198-2.64-.382z" clip-path="url(#p94a45562aa)" style="fill:#8db0fe"/>
                <path d="m140.048 178.587 2.691 3.135 1.831 1.862-2.683-3.126z" clip-path="url(#p94a45562aa)" style="fill:#e8d6cc"/>
                <path d="m142.74 181.722 2.685 2.995 1.823 1.854-2.678-2.987z" clip-path="url(#p94a45562aa)" style="fill:#e3d9d3"/>
                <path d="m137.351 175.311 2.697 3.276 1.84 1.871-2.69-3.266z" clip-path="url(#p94a45562aa)" style="fill:#edd2c3"/>
                <path d="m145.425 184.717 2.68 2.856 1.816 1.846-2.673-2.848z" clip-path="url(#p94a45562aa)" style="fill:#dddcdc"/>
                <path d="m295.58 190.382 2.83-3.002 1.46-1.497-2.83 3z" clip-path="url(#p94a45562aa)" style="fill:#dcdddd"/>
                <path d="m227.799 224.478 2.673-.715 1.573.643-2.669.713z" clip-path="url(#p94a45562aa)" style="fill:#88abfd"/>
                <path d="m148.106 187.573 2.676 2.718 1.808 1.838-2.669-2.71z" clip-path="url(#p94a45562aa)" style="fill:#d8dce2"/>
                <path d="m285.57 199.623 2.8-2.701 1.468-1.096-2.799 2.7z" clip-path="url(#p94a45562aa)" style="fill:#c9d7f0"/>
                <path d="m238.965 223.07 2.691-1.131 1.547.373-2.688 1.128z" clip-path="url(#p94a45562aa)" style="fill:#8db0fe"/>
                <path d="m171.186 207.248 2.65 1.613 1.738 1.647-2.643-1.61z" clip-path="url(#p94a45562aa)" style="fill:#afcafc"/>
                <path d="m150.782 190.29 2.672 2.58 1.8 1.83-2.664-2.571z" clip-path="url(#p94a45562aa)" style="fill:#d4dbe6"/>
                <path d="m193.133 218.792 2.646.654 1.67 1.342-2.64-.654z" clip-path="url(#p94a45562aa)" style="fill:#94b6ff"/>
                <path d="m183.505 214.519 2.645 1.065 1.7 1.49-2.64-1.063z" clip-path="url(#p94a45562aa)" style="fill:#9ebeff"/>
                <path d="m153.454 192.87 2.669 2.44 1.793 1.824-2.661-2.434z" clip-path="url(#p94a45562aa)" style="fill:#cfdaea"/>
                <path d="m262.851 214.713 2.741-1.976 1.499-.295-2.74 1.972z" clip-path="url(#p94a45562aa)" style="fill:#a5c3fe"/>
                <path d="m311.409 172.08 2.882-3.47 1.457-2.173-2.883 3.472z" clip-path="url(#p94a45562aa)" style="fill:#f5c1a9"/>
                <path d="m209.682 223.238 2.654-.03 1.624 1.058-2.65.028z" clip-path="url(#p94a45562aa)" style="fill:#8badfd"/>
                <path d="m267.09 212.442 2.753-2.118 1.49-.428-2.749 2.115z" clip-path="url(#p94a45562aa)" style="fill:#aac7fd"/>
                <path d="m258.614 216.707 2.731-1.833 1.506-.161-2.729 1.83z" clip-path="url(#p94a45562aa)" style="fill:#9fbfff"/>
                <path d="m305.631 179.33 2.864-3.312 1.457-1.902-2.864 3.312z" clip-path="url(#p94a45562aa)" style="fill:#efcfbf"/>
                <path d="m216.615 224.1 2.66-.303 1.605.919-2.657.301z" clip-path="url(#p94a45562aa)" style="fill:#89acfd"/>
                <path d="m271.334 209.896 2.763-2.264 1.485-.56-2.761 2.26z" clip-path="url(#p94a45562aa)" style="fill:#b1cbfc"/>
                <path d="m254.377 218.427 2.723-1.692 1.514-.028-2.72 1.69z" clip-path="url(#p94a45562aa)" style="fill:#9bbcff"/>
                <path d="m156.123 195.31 2.665 2.303 1.786 1.818-2.658-2.297z" clip-path="url(#p94a45562aa)" style="fill:#cbd8ee"/>
                <path d="m202.739 221.687 2.65.245 1.643 1.198-2.645-.245z" clip-path="url(#p94a45562aa)" style="fill:#8fb1fe"/>
                <path d="m289.838 195.826 2.815-2.851 1.464-1.229-2.814 2.85z" clip-path="url(#p94a45562aa)" style="fill:#d3dbe7"/>
                <path d="m234.722 223.554 2.686-.994 1.557.51-2.682.99z" clip-path="url(#p94a45562aa)" style="fill:#8caffe"/>
                <path d="m173.836 208.861 2.65 1.476 1.731 1.644-2.643-1.473z" clip-path="url(#p94a45562aa)" style="fill:#adc9fd"/>
                <path d="m299.87 185.883 2.846-3.156 1.458-1.631-2.846 3.155z" clip-path="url(#p94a45562aa)" style="fill:#e6d7cf"/>
                <path d="m223.543 224.276 2.67-.58 1.586.782-2.666.577z" clip-path="url(#p94a45562aa)" style="fill:#8badfd"/>
                <path d="m275.582 207.071 2.775-2.408 1.48-.694-2.774 2.405z" clip-path="url(#p94a45562aa)" style="fill:#b9d0f9"/>
                <path d="m250.14 219.872 2.714-1.552 1.523.107-2.712 1.548z" clip-path="url(#p94a45562aa)" style="fill:#97b8ff"/>
                <path d="m158.788 197.613 2.663 2.165 1.78 1.812-2.657-2.16z" clip-path="url(#p94a45562aa)" style="fill:#c6d6f1"/>
                <path d="m186.15 215.584 2.646.928 1.695 1.489-2.641-.927z" clip-path="url(#p94a45562aa)" style="fill:#9ebeff"/>
                <path d="m195.779 219.446 2.648.518 1.665 1.342-2.642-.518z" clip-path="url(#p94a45562aa)" style="fill:#96b7ff"/>
                <path d="m279.836 203.969 2.789-2.556 1.474-.827-2.787 2.553z" clip-path="url(#p94a45562aa)" style="fill:#c1d4f4"/>
                <path d="m245.9 221.043 2.707-1.413 1.532.242-2.703 1.409z" clip-path="url(#p94a45562aa)" style="fill:#96b7ff"/>
                <path d="m161.45 199.778 2.662 2.027 1.772 1.808-2.654-2.023z" clip-path="url(#p94a45562aa)" style="fill:#c3d5f4"/>
                <path d="m176.486 210.337 2.65 1.34 1.725 1.641-2.644-1.337z" clip-path="url(#p94a45562aa)" style="fill:#abc8fd"/>
                <path d="m294.117 191.746 2.831-3.003 1.461-1.363-2.83 3.002z" clip-path="url(#p94a45562aa)" style="fill:#dbdcde"/>
                <path d="m230.472 223.763 2.681-.856 1.57.647-2.678.852z" clip-path="url(#p94a45562aa)" style="fill:#8db0fe"/>
                <path d="m309.952 174.116 2.882-3.47 1.457-2.036-2.882 3.47z" clip-path="url(#p94a45562aa)" style="fill:#f5c4ac"/>
                <path d="m212.336 223.209 2.66-.168 1.619 1.06-2.655.165z" clip-path="url(#p94a45562aa)" style="fill:#8fb1fe"/>
                <path d="m284.099 200.586 2.802-2.704 1.47-.96-2.8 2.701z" clip-path="url(#p94a45562aa)" style="fill:#cad8ef"/>
                <path d="m241.656 221.939 2.7-1.273 1.544.377-2.697 1.27z" clip-path="url(#p94a45562aa)" style="fill:#94b6ff"/>
                <path d="m205.389 221.932 2.654.107 1.639 1.2-2.65-.109z" clip-path="url(#p94a45562aa)" style="fill:#92b4fe"/>
                <path d="m164.112 201.805 2.658 1.891 1.766 1.803-2.652-1.886z" clip-path="url(#p94a45562aa)" style="fill:#bfd3f6"/>
                <path d="m188.796 216.512 2.649.792 1.688 1.488-2.642-.791z" clip-path="url(#p94a45562aa)" style="fill:#9ebeff"/>
                <path d="m304.174 181.096 2.864-3.312 1.457-1.766-2.864 3.311z" clip-path="url(#p94a45562aa)" style="fill:#eed0c0"/>
                <path d="m219.276 223.797 2.667-.443 1.6.922-2.663.44z" clip-path="url(#p94a45562aa)" style="fill:#8db0fe"/>
                <path d="m179.136 211.677 2.65 1.203 1.719 1.639-2.644-1.201z" clip-path="url(#p94a45562aa)" style="fill:#a9c6fd"/>
                <path d="m198.427 219.964 2.652.381 1.66 1.342-2.647-.38z" clip-path="url(#p94a45562aa)" style="fill:#96b7ff"/>
                <path d="m298.41 187.38 2.847-3.156 1.459-1.497-2.847 3.156z" clip-path="url(#p94a45562aa)" style="fill:#e5d8d1"/>
                <path d="m166.77 203.696 2.658 1.754 1.758 1.798-2.65-1.75z" clip-path="url(#p94a45562aa)" style="fill:#bcd2f7"/>
                <path d="m226.213 223.697 2.677-.72 1.582.786-2.673.715z" clip-path="url(#p94a45562aa)" style="fill:#8fb1fe"/>
                <path d="m288.37 196.922 2.817-2.854 1.466-1.093-2.815 2.851z" clip-path="url(#p94a45562aa)" style="fill:#d3dbe7"/>
                <path d="m237.408 222.56 2.695-1.135 1.553.514-2.69 1.131z" clip-path="url(#p94a45562aa)" style="fill:#93b5fe"/>
                <path d="m265.592 212.737 2.755-2.122 1.496-.291-2.752 2.118z" clip-path="url(#p94a45562aa)" style="fill:#aec9fc"/>
                <path d="m261.345 214.874 2.744-1.98 1.503-.157-2.74 1.976z" clip-path="url(#p94a45562aa)" style="fill:#a9c6fd"/>
                <path d="m269.843 210.324 2.765-2.267 1.489-.425-2.763 2.264z" clip-path="url(#p94a45562aa)" style="fill:#b5cdfa"/>
                <path d="m257.1 216.735 2.734-1.837 1.511-.024-2.731 1.833z" clip-path="url(#p94a45562aa)" style="fill:#a3c2fe"/>
                <path d="m138.193 176.563 2.699 3.145 1.847 2.014-2.69-3.135z" clip-path="url(#p94a45562aa)" style="fill:#eed0c0"/>
                <path d="m140.892 179.708 2.693 3.004 1.84 2.005-2.686-2.995z" clip-path="url(#p94a45562aa)" style="fill:#ead4c8"/>
                <path d="m135.488 173.278 2.705 3.285 1.855 2.024-2.697-3.276z" clip-path="url(#p94a45562aa)" style="fill:#f2cbb7"/>
                <path d="m143.585 182.712 2.689 2.865 1.832 1.996-2.681-2.856z" clip-path="url(#p94a45562aa)" style="fill:#e5d8d1"/>
                <path d="m274.097 207.632 2.777-2.412 1.483-.557-2.775 2.408z" clip-path="url(#p94a45562aa)" style="fill:#bbd1f8"/>
                <path d="m252.854 218.32 2.726-1.696 1.52.111-2.723 1.692z" clip-path="url(#p94a45562aa)" style="fill:#9fbfff"/>
                <path d="m146.274 185.577 2.684 2.725 1.824 1.989-2.676-2.718z" clip-path="url(#p94a45562aa)" style="fill:#e1dad6"/>
                <path d="m191.445 217.304 2.651.655 1.683 1.487-2.646-.654z" clip-path="url(#p94a45562aa)" style="fill:#9ebeff"/>
                <path d="m181.786 212.88 2.651 1.067 1.713 1.637-2.645-1.065z" clip-path="url(#p94a45562aa)" style="fill:#a9c6fd"/>
                <path d="m169.428 205.45 2.656 1.616 1.752 1.795-2.65-1.613z" clip-path="url(#p94a45562aa)" style="fill:#bad0f8"/>
                <path d="m148.958 188.302 2.68 2.586 1.816 1.981-2.672-2.578z" clip-path="url(#p94a45562aa)" style="fill:#dcdddd"/>
                <path d="m208.043 222.039 2.66-.031 1.633 1.2-2.654.03z" clip-path="url(#p94a45562aa)" style="fill:#94b6ff"/>
                <path d="m308.495 176.018 2.883-3.47 1.456-1.901-2.882 3.47z" clip-path="url(#p94a45562aa)" style="fill:#f4c5ad"/>
                <path d="m214.996 223.041 2.666-.306 1.614 1.062-2.661.304z" clip-path="url(#p94a45562aa)" style="fill:#92b4fe"/>
                <path d="m292.653 192.975 2.832-3.005 1.463-1.227-2.83 3.003z" clip-path="url(#p94a45562aa)" style="fill:#dcdddd"/>
                <path d="m233.153 222.907 2.69-.997 1.565.65-2.686.994z" clip-path="url(#p94a45562aa)" style="fill:#93b5fe"/>
                <path d="m278.357 204.663 2.79-2.56 1.478-.69-2.789 2.556z" clip-path="url(#p94a45562aa)" style="fill:#c3d5f4"/>
                <path d="m248.607 219.63 2.718-1.555 1.529.245-2.715 1.552z" clip-path="url(#p94a45562aa)" style="fill:#9dbdff"/>
                <path d="m151.638 190.888 2.676 2.447 1.809 1.975-2.669-2.44z" clip-path="url(#p94a45562aa)" style="fill:#d8dce2"/>
                <path d="m201.079 220.345 2.655.244 1.655 1.343-2.65-.245z" clip-path="url(#p94a45562aa)" style="fill:#98b9ff"/>
                <path d="m302.716 182.727 2.865-3.312 1.457-1.63-2.864 3.31z" clip-path="url(#p94a45562aa)" style="fill:#edd1c2"/>
                <path d="m221.943 223.354 2.675-.581 1.595.924-2.67.579z" clip-path="url(#p94a45562aa)" style="fill:#92b4fe"/>
                <path d="m154.314 193.335 2.673 2.309 1.801 1.969-2.665-2.303z" clip-path="url(#p94a45562aa)" style="fill:#d4dbe6"/>
                <path d="m282.625 201.413 2.803-2.707 1.473-.824-2.802 2.704z" clip-path="url(#p94a45562aa)" style="fill:#cbd8ee"/>
                <path d="m172.084 207.066 2.657 1.48 1.745 1.791-2.65-1.476z" clip-path="url(#p94a45562aa)" style="fill:#b7cff9"/>
                <path d="m244.357 220.666 2.71-1.417 1.54.381-2.707 1.413z" clip-path="url(#p94a45562aa)" style="fill:#9bbcff"/>
                <path d="m184.437 213.947 2.653.93 1.706 1.635-2.646-.928z" clip-path="url(#p94a45562aa)" style="fill:#a7c5fe"/>
                <path d="m194.096 217.959 2.654.518 1.677 1.487-2.648-.518z" clip-path="url(#p94a45562aa)" style="fill:#9fbfff"/>
                <path d="m156.987 195.644 2.67 2.17 1.794 1.964-2.663-2.165z" clip-path="url(#p94a45562aa)" style="fill:#cfdaea"/>
                <path d="m296.948 188.743 2.848-3.159 1.46-1.36-2.847 3.156z" clip-path="url(#p94a45562aa)" style="fill:#e5d8d1"/>
                <path d="m228.89 222.978 2.686-.86 1.577.789-2.681.856z" clip-path="url(#p94a45562aa)" style="fill:#94b6ff"/>
                <path d="m286.9 197.882 2.819-2.856 1.468-.958-2.816 2.854z" clip-path="url(#p94a45562aa)" style="fill:#d4dbe6"/>
                <path d="m240.103 221.425 2.704-1.277 1.55.518-2.7 1.273z" clip-path="url(#p94a45562aa)" style="fill:#9abbff"/>
                <path d="m174.74 208.545 2.657 1.343 1.739 1.79-2.65-1.34z" clip-path="url(#p94a45562aa)" style="fill:#b5cdfa"/>
                <path d="m159.657 197.815 2.667 2.033 1.788 1.957-2.661-2.027z" clip-path="url(#p94a45562aa)" style="fill:#ccd9ed"/>
                <path d="m210.703 222.008 2.665-.17 1.628 1.203-2.66.168z" clip-path="url(#p94a45562aa)" style="fill:#97b8ff"/>
                <path d="m307.038 177.784 2.883-3.47 1.457-1.765-2.883 3.469z" clip-path="url(#p94a45562aa)" style="fill:#f3c7b1"/>
                <path d="m217.662 222.735 2.673-.445 1.608 1.064-2.667.443z" clip-path="url(#p94a45562aa)" style="fill:#96b7ff"/>
                <path d="m203.734 220.589 2.66.105 1.65 1.345-2.655-.107z" clip-path="url(#p94a45562aa)" style="fill:#9abbff"/>
                <path d="m264.09 212.895 2.756-2.127 1.5-.153-2.754 2.122z" clip-path="url(#p94a45562aa)" style="fill:#b2ccfb"/>
                <path d="m268.347 210.615 2.768-2.271 1.493-.287-2.765 2.267z" clip-path="url(#p94a45562aa)" style="fill:#b7cff9"/>
                <path d="m259.834 214.898 2.747-1.984 1.508-.02-2.744 1.98z" clip-path="url(#p94a45562aa)" style="fill:#adc9fd"/>
                <path d="m187.09 214.876 2.654.793 1.701 1.635-2.649-.792z" clip-path="url(#p94a45562aa)" style="fill:#a7c5fe"/>
                <path d="m272.608 208.057 2.78-2.416 1.486-.42-2.777 2.411z" clip-path="url(#p94a45562aa)" style="fill:#bed2f6"/>
                <path d="m255.58 216.624 2.737-1.842 1.517.116-2.734 1.837z" clip-path="url(#p94a45562aa)" style="fill:#a9c6fd"/>
                <path d="m162.324 199.848 2.666 1.895 1.78 1.953-2.658-1.89z" clip-path="url(#p94a45562aa)" style="fill:#c9d7f0"/>
                <path d="m291.187 194.068 2.834-3.007 1.464-1.091-2.832 3.005z" clip-path="url(#p94a45562aa)" style="fill:#dcdddd"/>
                <path d="m235.843 221.91 2.698-1.139 1.562.654-2.695 1.135z" clip-path="url(#p94a45562aa)" style="fill:#9abbff"/>
                <path d="m301.257 184.224 2.865-3.314 1.459-1.495-2.865 3.312z" clip-path="url(#p94a45562aa)" style="fill:#edd1c2"/>
                <path d="m224.618 222.773 2.682-.723 1.59.928-2.677.719z" clip-path="url(#p94a45562aa)" style="fill:#97b8ff"/>
                <path d="m196.75 218.477 2.657.38 1.672 1.488-2.652-.38z" clip-path="url(#p94a45562aa)" style="fill:#a1c0ff"/>
                <path d="m177.397 209.888 2.656 1.205 1.733 1.787-2.65-1.203z" clip-path="url(#p94a45562aa)" style="fill:#b3cdfb"/>
                <path d="m276.874 205.22 2.793-2.562 1.48-.554-2.79 2.559z" clip-path="url(#p94a45562aa)" style="fill:#c5d6f2"/>
                <path d="m251.325 218.075 2.729-1.701 1.526.25-2.726 1.696z" clip-path="url(#p94a45562aa)" style="fill:#a5c3fe"/>
                <path d="m164.99 201.743 2.665 1.758 1.773 1.949-2.658-1.754z" clip-path="url(#p94a45562aa)" style="fill:#c6d6f1"/>
                <path d="m281.147 202.104 2.806-2.71 1.475-.688-2.803 2.707z" clip-path="url(#p94a45562aa)" style="fill:#cdd9ec"/>
                <path d="m247.068 219.25 2.721-1.561 1.536.386-2.718 1.555z" clip-path="url(#p94a45562aa)" style="fill:#a3c2fe"/>
                <path d="m295.485 189.97 2.85-3.161 1.461-1.225-2.848 3.159z" clip-path="url(#p94a45562aa)" style="fill:#e5d8d1"/>
                <path d="m231.576 222.118 2.693-1 1.574.792-2.69.997z" clip-path="url(#p94a45562aa)" style="fill:#9bbcff"/>
                <path d="m189.744 215.669 2.657.655 1.695 1.635-2.651-.655z" clip-path="url(#p94a45562aa)" style="fill:#a7c5fe"/>
                <path d="m213.368 221.838 2.67-.308 1.624 1.205-2.666.306z" clip-path="url(#p94a45562aa)" style="fill:#9bbcff"/>
                <path d="m206.394 220.694 2.665-.033 1.644 1.347-2.66.03z" clip-path="url(#p94a45562aa)" style="fill:#9dbdff"/>
                <path d="m180.053 211.093 2.658 1.068 1.726 1.786-2.651-1.067z" clip-path="url(#p94a45562aa)" style="fill:#b2ccfb"/>
                <path d="m167.655 203.5 2.663 1.62 1.766 1.946-2.656-1.616z" clip-path="url(#p94a45562aa)" style="fill:#c4d5f3"/>
                <path d="m285.428 198.706 2.82-2.86 1.471-.82-2.818 2.856z" clip-path="url(#p94a45562aa)" style="fill:#d5dbe5"/>
                <path d="m242.807 220.148 2.715-1.42 1.546.521-2.711 1.417z" clip-path="url(#p94a45562aa)" style="fill:#a2c1ff"/>
                <path d="m305.58 179.415 2.884-3.47 1.457-1.63-2.883 3.47z" clip-path="url(#p94a45562aa)" style="fill:#f3c7b1"/>
                <path d="m220.335 222.29 2.679-.585 1.604 1.068-2.675.581z" clip-path="url(#p94a45562aa)" style="fill:#9abbff"/>
                <path d="m199.407 218.857 2.661.242 1.666 1.49-2.655-.244z" clip-path="url(#p94a45562aa)" style="fill:#a2c1ff"/>
                <path d="m299.796 185.584 2.867-3.315 1.46-1.359-2.866 3.314z" clip-path="url(#p94a45562aa)" style="fill:#edd1c2"/>
                <path d="m227.3 222.05 2.69-.863 1.586.931-2.686.86z" clip-path="url(#p94a45562aa)" style="fill:#9dbdff"/>
                <path d="m170.318 205.12 2.663 1.483 1.76 1.942-2.657-1.48z" clip-path="url(#p94a45562aa)" style="fill:#c1d4f4"/>
                <path d="m289.719 195.026 2.835-3.011 1.467-.954-2.834 3.007z" clip-path="url(#p94a45562aa)" style="fill:#dddcdc"/>
                <path d="m238.541 220.771 2.709-1.282 1.557.66-2.704 1.276z" clip-path="url(#p94a45562aa)" style="fill:#a1c0ff"/>
                <path d="m182.711 212.161 2.659.93 1.72 1.785-2.653-.93z" clip-path="url(#p94a45562aa)" style="fill:#b2ccfb"/>
                <path d="m266.846 210.768 2.771-2.276 1.498-.148-2.768 2.27z" clip-path="url(#p94a45562aa)" style="fill:#bbd1f8"/>
                <path d="m262.581 212.914 2.76-2.132 1.505-.014-2.757 2.127z" clip-path="url(#p94a45562aa)" style="fill:#b6cefa"/>
                <path d="m192.401 216.324 2.66.517 1.689 1.636-2.654-.518z" clip-path="url(#p94a45562aa)" style="fill:#a9c6fd"/>
                <path d="m271.115 208.344 2.782-2.42 1.49-.283-2.779 2.416z" clip-path="url(#p94a45562aa)" style="fill:#c1d4f4"/>
                <path d="m258.317 214.782 2.75-1.988 1.514.12-2.747 1.984z" clip-path="url(#p94a45562aa)" style="fill:#b2ccfb"/>
                <path d="m275.388 205.64 2.794-2.566 1.485-.416-2.793 2.562z" clip-path="url(#p94a45562aa)" style="fill:#c9d7f0"/>
                <path d="m254.054 216.374 2.74-1.847 1.523.255-2.737 1.842z" clip-path="url(#p94a45562aa)" style="fill:#aec9fc"/>
                <path d="m209.06 220.661 2.67-.172 1.638 1.35-2.665.169z" clip-path="url(#p94a45562aa)" style="fill:#a1c0ff"/>
                <path d="m216.039 221.53 2.677-.449 1.619 1.21-2.673.444z" clip-path="url(#p94a45562aa)" style="fill:#9fbfff"/>
                <path d="m172.981 206.603 2.663 1.345 1.753 1.94-2.656-1.343z" clip-path="url(#p94a45562aa)" style="fill:#bfd3f6"/>
                <path d="m294.02 191.06 2.851-3.163 1.464-1.088-2.85 3.16z" clip-path="url(#p94a45562aa)" style="fill:#e6d7cf"/>
                <path d="m202.068 219.1 2.666.103 1.66 1.491-2.66-.105z" clip-path="url(#p94a45562aa)" style="fill:#a3c2fe"/>
                <path d="m234.27 221.118 2.702-1.144 1.57.797-2.7 1.139z" clip-path="url(#p94a45562aa)" style="fill:#a2c1ff"/>
                <path d="m279.667 202.658 2.807-2.715 1.479-.55-2.806 2.71z" clip-path="url(#p94a45562aa)" style="fill:#cfdaea"/>
                <path d="m249.79 217.689 2.732-1.705 1.532.39-2.73 1.7z" clip-path="url(#p94a45562aa)" style="fill:#abc8fd"/>
                <path d="m304.122 180.91 2.884-3.472 1.458-1.494-2.883 3.471z" clip-path="url(#p94a45562aa)" style="fill:#f3c8b2"/>
                <path d="m223.014 221.705 2.686-.726 1.6 1.071-2.682.723z" clip-path="url(#p94a45562aa)" style="fill:#9fbfff"/>
                <path d="m185.37 213.092 2.66.793 1.714 1.784-2.654-.793z" clip-path="url(#p94a45562aa)" style="fill:#b2ccfb"/>
                <path d="m195.061 216.841 2.663.38 1.683 1.636-2.657-.38z" clip-path="url(#p94a45562aa)" style="fill:#aac7fd"/>
                <path d="m283.953 199.393 2.822-2.863 1.473-.684-2.82 2.86z" clip-path="url(#p94a45562aa)" style="fill:#d7dce3"/>
                <path d="m245.522 218.727 2.725-1.565 1.542.527-2.721 1.56z" clip-path="url(#p94a45562aa)" style="fill:#a9c6fd"/>
                <path d="m175.644 207.948 2.663 1.208 1.746 1.937-2.656-1.205z" clip-path="url(#p94a45562aa)" style="fill:#bed2f6"/>
                <path d="m298.335 186.809 2.867-3.318 1.46-1.222-2.866 3.315z" clip-path="url(#p94a45562aa)" style="fill:#edd1c2"/>
                <path d="m229.99 221.187 2.698-1.005 1.581.936-2.693 1z" clip-path="url(#p94a45562aa)" style="fill:#a2c1ff"/>
                <path d="m288.248 195.846 2.837-3.014 1.469-.817-2.835 3.01z" clip-path="url(#p94a45562aa)" style="fill:#dedcdb"/>
                <path d="m241.25 219.49 2.718-1.426 1.554.663-2.715 1.421z" clip-path="url(#p94a45562aa)" style="fill:#a9c6fd"/>
                <path d="m211.73 220.49 2.675-.312 1.634 1.352-2.671.308z" clip-path="url(#p94a45562aa)" style="fill:#a3c2fe"/>
                <path d="m204.734 219.203 2.67-.035 1.655 1.493-2.665.033z" clip-path="url(#p94a45562aa)" style="fill:#a6c4fe"/>
                <path d="m188.03 213.885 2.664.655 1.707 1.784-2.657-.655z" clip-path="url(#p94a45562aa)" style="fill:#b2ccfb"/>
                <path d="m218.716 221.081 2.684-.588 1.614 1.212-2.68.585z" clip-path="url(#p94a45562aa)" style="fill:#a3c2fe"/>
                <path d="m178.307 209.156 2.664 1.07 1.74 1.935-2.658-1.068z" clip-path="url(#p94a45562aa)" style="fill:#bed2f6"/>
                <path d="m265.341 210.782 2.774-2.28 1.502-.01-2.77 2.276z" clip-path="url(#p94a45562aa)" style="fill:#c0d4f5"/>
                <path d="m269.617 208.492 2.785-2.425 1.495-.144-2.782 2.42z" clip-path="url(#p94a45562aa)" style="fill:#c5d6f2"/>
                <path d="m261.067 212.794 2.763-2.137 1.511.125-2.76 2.132z" clip-path="url(#p94a45562aa)" style="fill:#bbd1f8"/>
                <path d="m197.724 217.221 2.667.241 1.677 1.637-2.66-.242z" clip-path="url(#p94a45562aa)" style="fill:#abc8fd"/>
                <path d="m292.554 192.015 2.852-3.167 1.465-.951-2.85 3.164z" clip-path="url(#p94a45562aa)" style="fill:#e7d7ce"/>
                <path d="m236.972 219.974 2.713-1.286 1.565.801-2.709 1.282z" clip-path="url(#p94a45562aa)" style="fill:#a9c6fd"/>
                <path d="m273.897 205.923 2.797-2.571 1.488-.278-2.794 2.567z" clip-path="url(#p94a45562aa)" style="fill:#cbd8ee"/>
                <path d="m256.795 214.527 2.753-1.993 1.52.26-2.75 1.988z" clip-path="url(#p94a45562aa)" style="fill:#b7cff9"/>
                <path d="m302.663 182.269 2.885-3.474 1.458-1.357-2.884 3.472z" clip-path="url(#p94a45562aa)" style="fill:#f3c8b2"/>
                <path d="m225.7 220.98 2.695-.868 1.595 1.075-2.69.863z" clip-path="url(#p94a45562aa)" style="fill:#a5c3fe"/>
                <path d="m278.182 203.074 2.81-2.72 1.482-.41-2.807 2.714z" clip-path="url(#p94a45562aa)" style="fill:#d2dbe8"/>
                <path d="m252.522 215.984 2.744-1.852 1.529.395-2.741 1.847zM190.694 214.54l2.666.517 1.701 1.784-2.66-.517z" clip-path="url(#p94a45562aa)" style="fill:#b3cdfb"/>
                <path d="m180.971 210.225 2.666.932 1.733 1.935-2.659-.93z" clip-path="url(#p94a45562aa)" style="fill:#bcd2f7"/>
                <path d="m282.474 199.943 2.824-2.868 1.477-.545-2.822 2.863z" clip-path="url(#p94a45562aa)" style="fill:#d9dce1"/>
                <path d="m248.247 217.162 2.736-1.71 1.539.532-2.733 1.705z" clip-path="url(#p94a45562aa)" style="fill:#b1cbfc"/>
                <path d="m296.871 187.897 2.869-3.32 1.462-1.086-2.867 3.318z" clip-path="url(#p94a45562aa)" style="fill:#eed0c0"/>
                <path d="m232.688 220.182 2.707-1.148 1.577.94-2.703 1.144z" clip-path="url(#p94a45562aa)" style="fill:#a9c6fd"/>
                <path d="m207.404 219.168 2.676-.174 1.65 1.495-2.67.172z" clip-path="url(#p94a45562aa)" style="fill:#aac7fd"/>
                <path d="m214.405 220.178 2.682-.451 1.629 1.354-2.677.449z" clip-path="url(#p94a45562aa)" style="fill:#a7c5fe"/>
                <path d="m200.391 217.462 2.671.102 1.672 1.64-2.666-.105z" clip-path="url(#p94a45562aa)" style="fill:#aec9fc"/>
                <path d="m221.4 220.493 2.691-.73 1.61 1.216-2.687.726z" clip-path="url(#p94a45562aa)" style="fill:#a9c6fd"/>
                <path d="m286.775 196.53 2.838-3.019 1.472-.68-2.837 3.015z" clip-path="url(#p94a45562aa)" style="fill:#e0dbd8"/>
                <path d="m243.968 218.064 2.729-1.57 1.55.668-2.725 1.565z" clip-path="url(#p94a45562aa)" style="fill:#afcafc"/>
                <path d="m183.637 211.157 2.667.794 1.727 1.934-2.661-.793z" clip-path="url(#p94a45562aa)" style="fill:#bcd2f7"/>
                <path d="m193.36 215.057 2.669.379 1.695 1.785-2.663-.38z" clip-path="url(#p94a45562aa)" style="fill:#b5cdfa"/>
                <path d="m301.202 183.491 2.886-3.477 1.46-1.22-2.885 3.475z" clip-path="url(#p94a45562aa)" style="fill:#f3c8b2"/>
                <path d="m228.395 220.112 2.702-1.01 1.59 1.08-2.697 1.005z" clip-path="url(#p94a45562aa)" style="fill:#abc8fd"/>
                <path d="m291.085 192.832 2.853-3.17 1.468-.814-2.852 3.167z" clip-path="url(#p94a45562aa)" style="fill:#e8d6cc"/>
                <path d="m239.685 218.688 2.722-1.43 1.561.806-2.718 1.425z" clip-path="url(#p94a45562aa)" style="fill:#afcafc"/>
                <path d="m268.115 208.502 2.787-2.43 1.5-.005-2.785 2.425z" clip-path="url(#p94a45562aa)" style="fill:#c9d7f0"/>
                <path d="m263.83 210.657 2.777-2.285 1.508.13-2.774 2.28z" clip-path="url(#p94a45562aa)" style="fill:#c4d5f3"/>
                <path d="m272.402 206.067 2.8-2.577 1.492-.138-2.797 2.571z" clip-path="url(#p94a45562aa)" style="fill:#cedaeb"/>
                <path d="m259.548 212.534 2.766-2.142 1.516.265-2.763 2.137z" clip-path="url(#p94a45562aa)" style="fill:#c0d4f5"/>
                <path d="m210.08 218.994 2.68-.314 1.645 1.498-2.676.311zM217.087 219.727l2.69-.592 1.623 1.358-2.684.588z" clip-path="url(#p94a45562aa)" style="fill:#adc9fd"/>
                <path d="m203.062 217.564 2.676-.037 1.666 1.641-2.67.035z" clip-path="url(#p94a45562aa)" style="fill:#b1cbfc"/>
                <path d="m276.694 203.352 2.812-2.724 1.486-.273-2.81 2.719z" clip-path="url(#p94a45562aa)" style="fill:#d5dbe5"/>
                <path d="m255.266 214.132 2.757-2 1.525.402-2.753 1.993zM186.304 211.95l2.67.656 1.72 1.934-2.663-.655z" clip-path="url(#p94a45562aa)" style="fill:#bcd2f7"/>
                <path d="m295.406 188.848 2.87-3.324 1.464-.948-2.869 3.321z" clip-path="url(#p94a45562aa)" style="fill:#eed0c0"/>
                <path d="m235.395 219.034 2.717-1.29 1.573.944-2.713 1.286z" clip-path="url(#p94a45562aa)" style="fill:#afcafc"/>
                <path d="m280.992 200.355 2.826-2.873 1.48-.407-2.824 2.868z" clip-path="url(#p94a45562aa)" style="fill:#dbdcde"/>
                <path d="m250.983 215.452 2.748-1.858 1.535.538-2.744 1.852z" clip-path="url(#p94a45562aa)" style="fill:#bad0f8"/>
                <path d="m224.091 219.763 2.7-.871 1.604 1.22-2.695.867z" clip-path="url(#p94a45562aa)" style="fill:#aec9fc"/>
                <path d="m196.029 215.436 2.672.24 1.69 1.786-2.667-.241z" clip-path="url(#p94a45562aa)" style="fill:#b6cefa"/>
                <path d="m285.298 197.075 2.84-3.023 1.475-.54-2.838 3.018z" clip-path="url(#p94a45562aa)" style="fill:#e2dad5"/>
                <path d="m246.697 216.494 2.74-1.717 1.546.675-2.736 1.71z" clip-path="url(#p94a45562aa)" style="fill:#b7cff9"/>
                <path d="m299.74 184.576 2.887-3.48 1.461-1.082-2.886 3.477z" clip-path="url(#p94a45562aa)" style="fill:#f3c7b1"/>
                <path d="m231.097 219.103 2.712-1.153 1.586 1.084-2.707 1.148z" clip-path="url(#p94a45562aa)" style="fill:#b1cbfc"/>
                <path d="m188.973 212.606 2.672.517 1.715 1.934-2.666-.517z" clip-path="url(#p94a45562aa)" style="fill:#bed2f6"/>
                <path d="m212.76 218.68 2.688-.455 1.64 1.502-2.683.451z" clip-path="url(#p94a45562aa)" style="fill:#b1cbfc"/>
                <path d="m205.738 217.527 2.68-.177 1.662 1.644-2.676.174z" clip-path="url(#p94a45562aa)" style="fill:#b3cdfb"/>
                <path d="m289.613 193.511 2.855-3.175 1.47-.675-2.853 3.17z" clip-path="url(#p94a45562aa)" style="fill:#e9d5cb"/>
                <path d="m242.407 217.257 2.733-1.575 1.557.812-2.729 1.57z" clip-path="url(#p94a45562aa)" style="fill:#b7cff9"/>
                <path d="m219.776 219.135 2.697-.734 1.618 1.362-2.691.73z" clip-path="url(#p94a45562aa)" style="fill:#b1cbfc"/>
                <path d="m198.701 215.676 2.677.1 1.684 1.788-2.671-.102z" clip-path="url(#p94a45562aa)" style="fill:#b9d0f9"/>
                <path d="m266.607 208.372 2.79-2.437 1.505.136-2.787 2.431z" clip-path="url(#p94a45562aa)" style="fill:#cdd9ec"/>
                <path d="m270.902 206.071 2.802-2.582 1.497.001-2.8 2.577z" clip-path="url(#p94a45562aa)" style="fill:#d2dbe8"/>
                <path d="m262.314 210.392 2.78-2.292 1.513.272-2.777 2.285z" clip-path="url(#p94a45562aa)" style="fill:#c9d7f0"/>
                <path d="m226.79 218.892 2.708-1.015 1.6 1.226-2.703 1.01z" clip-path="url(#p94a45562aa)" style="fill:#b3cdfb"/>
                <path d="m293.938 189.661 2.872-3.328 1.466-.81-2.87 3.325z" clip-path="url(#p94a45562aa)" style="fill:#efcebd"/>
                <path d="m238.112 217.743 2.726-1.436 1.569.95-2.722 1.431z" clip-path="url(#p94a45562aa)" style="fill:#b7cff9"/>
                <path d="m275.201 203.49 2.815-2.73 1.49-.132-2.812 2.724z" clip-path="url(#p94a45562aa)" style="fill:#d8dce2"/>
                <path d="m258.023 212.133 2.769-2.149 1.522.408-2.766 2.142z" clip-path="url(#p94a45562aa)" style="fill:#c5d6f2"/>
                <path d="m191.645 213.123 2.676.377 1.708 1.936-2.67-.379z" clip-path="url(#p94a45562aa)" style="fill:#bfd3f6"/>
                <path d="m279.506 200.628 2.828-2.879 1.484-.267-2.826 2.873z" clip-path="url(#p94a45562aa)" style="fill:#dedcdb"/>
                <path d="m253.73 213.594 2.76-2.005 1.533.544-2.757 1.999z" clip-path="url(#p94a45562aa)" style="fill:#c3d5f4"/>
                <path d="m208.419 217.35 2.686-.317 1.656 1.647-2.681.314z" clip-path="url(#p94a45562aa)" style="fill:#b7cff9"/>
                <path d="m215.448 218.225 2.694-.596 1.634 1.506-2.689.592z" clip-path="url(#p94a45562aa)" style="fill:#b6cefa"/>
                <path d="m283.818 197.482 2.842-3.028 1.478-.402-2.84 3.023z" clip-path="url(#p94a45562aa)" style="fill:#e4d9d2"/>
                <path d="m249.437 214.777 2.752-1.863 1.542.68-2.748 1.858z" clip-path="url(#p94a45562aa)" style="fill:#c0d4f5"/>
                <path d="m298.276 185.524 2.889-3.484 1.462-.944-2.887 3.48z" clip-path="url(#p94a45562aa)" style="fill:#f4c6af"/>
                <path d="m233.809 217.95 2.72-1.297 1.583 1.09-2.717 1.291z" clip-path="url(#p94a45562aa)" style="fill:#b7cff9"/>
                <path d="m201.378 215.776 2.682-.04 1.678 1.791-2.676.037z" clip-path="url(#p94a45562aa)" style="fill:#bbd1f8"/>
                <path d="m222.473 218.4 2.704-.876 1.613 1.368-2.699.871z" clip-path="url(#p94a45562aa)" style="fill:#b6cefa"/>
                <path d="m288.138 194.052 2.857-3.18 1.473-.536-2.855 3.175z" clip-path="url(#p94a45562aa)" style="fill:#ead4c8"/>
                <path d="m245.14 215.682 2.744-1.723 1.553.818-2.74 1.717z" clip-path="url(#p94a45562aa)" style="fill:#bfd3f6"/>
                <path d="m194.32 213.5 2.68.238 1.701 1.938-2.672-.24z" clip-path="url(#p94a45562aa)" style="fill:#c0d4f5"/>
                <path d="m229.498 217.877 2.716-1.158 1.595 1.23-2.712 1.154z" clip-path="url(#p94a45562aa)" style="fill:#bad0f8"/>
                <path d="m292.468 190.336 2.874-3.333 1.468-.67-2.872 3.328z" clip-path="url(#p94a45562aa)" style="fill:#f0cdbb"/>
                <path d="m240.838 216.307 2.737-1.582 1.565.957-2.733 1.575z" clip-path="url(#p94a45562aa)" style="fill:#bed2f6"/>
                <path d="m211.105 217.033 2.693-.459 1.65 1.65-2.687.456z" clip-path="url(#p94a45562aa)" style="fill:#bbd1f8"/>
                <path d="m269.397 205.935 2.805-2.589 1.502.143-2.802 2.582z" clip-path="url(#p94a45562aa)" style="fill:#d6dce4"/>
                <path d="m265.094 208.1 2.793-2.443 1.51.278-2.79 2.437z" clip-path="url(#p94a45562aa)" style="fill:#d2dbe8"/>
                <path d="m204.06 215.737 2.686-.18 1.673 1.793-2.681.177z" clip-path="url(#p94a45562aa)" style="fill:#bed2f6"/>
                <path d="m273.704 203.489 2.817-2.736 1.495.008-2.815 2.73z" clip-path="url(#p94a45562aa)" style="fill:#dbdcde"/>
                <path d="m260.792 209.984 2.783-2.298 1.519.414-2.78 2.292z" clip-path="url(#p94a45562aa)" style="fill:#cdd9ec"/>
                <path d="m218.142 217.629 2.702-.739 1.629 1.51-2.697.735z" clip-path="url(#p94a45562aa)" style="fill:#bad0f8"/>
                <path d="m278.016 200.76 2.83-2.884 1.488-.127-2.828 2.879z" clip-path="url(#p94a45562aa)" style="fill:#e0dbd8"/>
                <path d="m256.49 211.59 2.774-2.155 1.528.55-2.77 2.148z" clip-path="url(#p94a45562aa)" style="fill:#cbd8ee"/>
                <path d="m197 213.738 2.682.099 1.696 1.94-2.677-.101z" clip-path="url(#p94a45562aa)" style="fill:#c3d5f4"/>
                <path d="m296.81 186.333 2.89-3.489 1.465-.804-2.889 3.484z" clip-path="url(#p94a45562aa)" style="fill:#f4c5ad"/>
                <path d="m236.53 216.653 2.73-1.442 1.578 1.096-2.726 1.436z" clip-path="url(#p94a45562aa)" style="fill:#bfd3f6"/>
                <path d="m225.177 217.524 2.712-1.02 1.609 1.373-2.708 1.015z" clip-path="url(#p94a45562aa)" style="fill:#bcd2f7"/>
                <path d="m282.334 197.75 2.845-3.035 1.481-.261-2.842 3.028z" clip-path="url(#p94a45562aa)" style="fill:#e7d7ce"/>
                <path d="m252.189 212.914 2.763-2.012 1.539.687-2.76 2.005z" clip-path="url(#p94a45562aa)" style="fill:#c9d7f0"/>
                <path d="m286.66 194.454 2.86-3.186 1.475-.396-2.857 3.18z" clip-path="url(#p94a45562aa)" style="fill:#edd2c3"/>
                <path d="m247.884 213.96 2.755-1.87 1.55.824-2.752 1.863z" clip-path="url(#p94a45562aa)" style="fill:#c6d6f1"/>
                <path d="m232.214 216.72 2.725-1.303 1.59 1.236-2.72 1.297zM206.746 215.557l2.693-.321 1.666 1.797-2.686.317z" clip-path="url(#p94a45562aa)" style="fill:#c0d4f5"/>
                <path d="m213.798 216.574 2.7-.6 1.644 1.655-2.694.596z" clip-path="url(#p94a45562aa)" style="fill:#bfd3f6"/>
                <path d="m199.682 213.837 2.688-.043 1.69 1.943-2.682.04z" clip-path="url(#p94a45562aa)" style="fill:#c5d6f2"/>
                <path d="m290.995 190.872 2.876-3.338 1.47-.531-2.873 3.333z" clip-path="url(#p94a45562aa)" style="fill:#f1ccb8"/>
                <path d="m243.575 214.725 2.748-1.729 1.56.963-2.743 1.723z" clip-path="url(#p94a45562aa)" style="fill:#c5d6f2"/>
                <path d="m220.844 216.89 2.709-.881 1.624 1.515-2.704.877z" clip-path="url(#p94a45562aa)" style="fill:#bfd3f6"/>
                <path d="m267.887 205.657 2.808-2.595 1.507.284-2.805 2.589z" clip-path="url(#p94a45562aa)" style="fill:#dadce0"/>
                <path d="m227.889 216.505 2.72-1.164 1.605 1.378-2.716 1.158z" clip-path="url(#p94a45562aa)" style="fill:#c1d4f4"/>
                <path d="m272.202 203.346 2.82-2.742 1.5.15-2.818 2.735z" clip-path="url(#p94a45562aa)" style="fill:#dfdbd9"/>
                <path d="m263.575 207.686 2.797-2.45 1.515.421-2.793 2.443z" clip-path="url(#p94a45562aa)" style="fill:#d6dce4"/>
                <path d="m295.342 187.003 2.891-3.493 1.467-.666-2.89 3.489z" clip-path="url(#p94a45562aa)" style="fill:#f5c4ac"/>
                <path d="m239.26 215.211 2.742-1.588 1.573 1.102-2.737 1.582z" clip-path="url(#p94a45562aa)" style="fill:#c5d6f2"/>
                <path d="m276.521 200.753 2.834-2.89 1.491.013-2.83 2.885z" clip-path="url(#p94a45562aa)" style="fill:#e4d9d2"/>
                <path d="m259.264 209.435 2.786-2.305 1.525.556-2.783 2.298z" clip-path="url(#p94a45562aa)" style="fill:#d3dbe7"/>
                <path d="m280.846 197.876 2.847-3.04 1.486-.12-2.845 3.033z" clip-path="url(#p94a45562aa)" style="fill:#ead5c9"/>
                <path d="m254.952 210.902 2.777-2.161 1.535.694-2.773 2.154z" clip-path="url(#p94a45562aa)" style="fill:#d1dae9"/>
                <path d="m209.439 215.236 2.698-.463 1.661 1.801-2.693.459zM216.498 215.973l2.706-.743 1.64 1.66-2.702.739z" clip-path="url(#p94a45562aa)" style="fill:#c4d5f3"/>
                <path d="m202.37 213.794 2.692-.183 1.684 1.946-2.686.18z" clip-path="url(#p94a45562aa)" style="fill:#c7d7f0"/>
                <path d="m234.94 215.417 2.735-1.449 1.586 1.243-2.731 1.442z" clip-path="url(#p94a45562aa)" style="fill:#c6d6f1"/>
                <path d="m285.179 194.715 2.861-3.191 1.48-.256-2.86 3.186z" clip-path="url(#p94a45562aa)" style="fill:#eed0c0"/>
                <path d="m250.64 212.09 2.767-2.02 1.545.832-2.763 2.012z" clip-path="url(#p94a45562aa)" style="fill:#cedaeb"/>
                <path d="m223.553 216.009 2.717-1.025 1.619 1.52-2.712 1.02z" clip-path="url(#p94a45562aa)" style="fill:#c5d6f2"/>
                <path d="m289.52 191.268 2.877-3.344 1.474-.39-2.876 3.338z" clip-path="url(#p94a45562aa)" style="fill:#f2cab5"/>
                <path d="m246.323 212.996 2.76-1.876 1.556.97-2.755 1.87z" clip-path="url(#p94a45562aa)" style="fill:#cdd9ec"/>
                <path d="m230.61 215.341 2.73-1.309 1.6 1.385-2.726 1.302z" clip-path="url(#p94a45562aa)" style="fill:#c7d7f0"/>
                <path d="m293.87 187.534 2.894-3.5 1.47-.524-2.892 3.493z" clip-path="url(#p94a45562aa)" style="fill:#f5c1a9"/>
                <path d="m242.002 213.623 2.752-1.736 1.57 1.11-2.749 1.728z" clip-path="url(#p94a45562aa)" style="fill:#ccd9ed"/>
                <path d="m212.137 214.773 2.705-.605 1.656 1.805-2.7.601z" clip-path="url(#p94a45562aa)" style="fill:#c9d7f0"/>
                <path d="m205.062 213.611 2.698-.324 1.679 1.949-2.693.32z" clip-path="url(#p94a45562aa)" style="fill:#cbd8ee"/>
                <path d="m219.204 215.23 2.715-.886 1.634 1.665-2.71.881z" clip-path="url(#p94a45562aa)" style="fill:#c9d7f0"/>
                <path d="m270.695 203.062 2.823-2.75 1.504.292-2.82 2.742z" clip-path="url(#p94a45562aa)" style="fill:#e3d9d3"/>
                <path d="m266.372 205.237 2.81-2.603 1.513.428-2.808 2.595z" clip-path="url(#p94a45562aa)" style="fill:#dedcdb"/>
                <path d="m275.022 200.604 2.836-2.898 1.497.156-2.834 2.891z" clip-path="url(#p94a45562aa)" style="fill:#e7d7ce"/>
                <path d="m262.05 207.13 2.8-2.457 1.522.564-2.797 2.45z" clip-path="url(#p94a45562aa)" style="fill:#dbdcde"/>
                <path d="m279.355 197.862 2.85-3.047 1.488.021-2.847 3.04z" clip-path="url(#p94a45562aa)" style="fill:#ecd3c5"/>
                <path d="m257.729 208.74 2.79-2.312 1.53.702-2.785 2.305z" clip-path="url(#p94a45562aa)" style="fill:#d8dce2"/>
                <path d="m237.675 213.968 2.746-1.595 1.581 1.25-2.741 1.588z" clip-path="url(#p94a45562aa)" style="fill:#cdd9ec"/>
                <path d="m226.27 214.984 2.726-1.17 1.614 1.527-2.721 1.164z" clip-path="url(#p94a45562aa)" style="fill:#cad8ef"/>
                <path d="m283.693 194.836 2.864-3.198 1.483-.114-2.861 3.191z" clip-path="url(#p94a45562aa)" style="fill:#f0cdbb"/>
                <path d="m253.407 210.07 2.78-2.168 1.542.839-2.777 2.161z" clip-path="url(#p94a45562aa)" style="fill:#d6dce4"/>
                <path d="m288.04 191.524 2.88-3.351 1.477-.25-2.878 3.345z" clip-path="url(#p94a45562aa)" style="fill:#f3c7b1"/>
                <path d="m249.083 211.12 2.771-2.026 1.553.977-2.768 2.018z" clip-path="url(#p94a45562aa)" style="fill:#d5dbe5"/>
                <path d="m207.76 213.287 2.704-.467 1.673 1.953-2.698.463z" clip-path="url(#p94a45562aa)" style="fill:#cedaeb"/>
                <path d="m214.842 214.168 2.712-.749 1.65 1.811-2.706.743z" clip-path="url(#p94a45562aa)" style="fill:#ccd9ed"/>
                <path d="m233.34 214.032 2.74-1.455 1.595 1.391-2.736 1.449z" clip-path="url(#p94a45562aa)" style="fill:#cedaeb"/>
                <path d="m221.919 214.344 2.722-1.031 1.629 1.67-2.717 1.026z" clip-path="url(#p94a45562aa)" style="fill:#cdd9ec"/>
                <path d="m292.397 187.924 2.895-3.505 1.472-.384-2.893 3.499z" clip-path="url(#p94a45562aa)" style="fill:#f6bea4"/>
                <path d="m244.754 211.887 2.764-1.884 1.565 1.117-2.76 1.876z" clip-path="url(#p94a45562aa)" style="fill:#d4dbe6"/>
                <path d="m228.996 213.814 2.735-1.315 1.609 1.533-2.73 1.31z" clip-path="url(#p94a45562aa)" style="fill:#cfdaea"/>
                <path d="m269.183 202.634 2.826-2.757 1.51.435-2.824 2.75z" clip-path="url(#p94a45562aa)" style="fill:#e7d7ce"/>
                <path d="m240.42 212.373 2.757-1.743 1.577 1.257-2.752 1.736z" clip-path="url(#p94a45562aa)" style="fill:#d4dbe6"/>
                <path d="m273.518 200.312 2.84-2.905 1.5.3-2.836 2.897z" clip-path="url(#p94a45562aa)" style="fill:#ead4c8"/>
                <path d="m264.85 204.673 2.815-2.61 1.518.571-2.811 2.603z" clip-path="url(#p94a45562aa)" style="fill:#e3d9d3"/>
                <path d="m277.858 197.706 2.853-3.054 1.493.163-2.85 3.047z" clip-path="url(#p94a45562aa)" style="fill:#efcfbf"/>
                <path d="m260.519 206.428 2.803-2.464 1.528.709-2.8 2.457z" clip-path="url(#p94a45562aa)" style="fill:#e0dbd8"/>
                <path d="m210.464 212.82 2.71-.61 1.668 1.958-2.705.605z" clip-path="url(#p94a45562aa)" style="fill:#d2dbe8"/>
                <path d="m282.204 194.815 2.867-3.205 1.486.028-2.864 3.198z" clip-path="url(#p94a45562aa)" style="fill:#f2cbb7"/>
                <path d="m256.187 207.902 2.794-2.32 1.538.846-2.79 2.313z" clip-path="url(#p94a45562aa)" style="fill:#dddcdc"/>
                <path d="m217.554 213.42 2.72-.893 1.645 1.817-2.715.886z" clip-path="url(#p94a45562aa)" style="fill:#d1dae9"/>
                <path d="m236.08 212.577 2.75-1.602 1.59 1.398-2.745 1.595z" clip-path="url(#p94a45562aa)" style="fill:#d4dbe6"/>
                <path d="m286.557 191.638 2.882-3.358 1.48-.107-2.879 3.35z" clip-path="url(#p94a45562aa)" style="fill:#f5c4ac"/>
                <path d="m251.854 209.094 2.785-2.177 1.548.985-2.78 2.169z" clip-path="url(#p94a45562aa)" style="fill:#dcdddd"/>
                <path d="m224.64 213.313 2.732-1.176 1.624 1.677-2.726 1.17z" clip-path="url(#p94a45562aa)" style="fill:#d3dbe7"/>
                <path d="m290.92 188.173 2.897-3.512 1.475-.242-2.895 3.505z" clip-path="url(#p94a45562aa)" style="fill:#f7bca1"/>
                <path d="m247.518 210.003 2.776-2.034 1.56 1.125-2.771 2.026z" clip-path="url(#p94a45562aa)" style="fill:#dbdcde"/>
                <path d="m231.731 212.499 2.745-1.462 1.604 1.54-2.74 1.455zM213.175 212.21l2.717-.754 1.662 1.963-2.712.749z" clip-path="url(#p94a45562aa)" style="fill:#d6dce4"/>
                <path d="m243.177 210.63 2.768-1.892 1.573 1.265-2.764 1.884z" clip-path="url(#p94a45562aa)" style="fill:#dadce0"/>
                <path d="M220.274 212.527 223 211.49l1.64 1.823-2.722 1.03z" clip-path="url(#p94a45562aa)" style="fill:#d6dce4"/>
                <path d="m272.01 199.877 2.841-2.913 1.506.443-2.839 2.905z" clip-path="url(#p94a45562aa)" style="fill:#eed0c0"/>
                <path d="m267.665 202.062 2.83-2.765 1.514.58-2.826 2.757z" clip-path="url(#p94a45562aa)" style="fill:#ebd3c6"/>
                <path d="m276.357 197.407 2.856-3.063 1.498.308-2.853 3.054z" clip-path="url(#p94a45562aa)" style="fill:#f1ccb8"/>
                <path d="m263.322 203.964 2.819-2.62 1.524.718-2.815 2.61z" clip-path="url(#p94a45562aa)" style="fill:#e8d6cc"/>
                <path d="m280.71 194.652 2.87-3.214 1.49.172-2.866 3.205z" clip-path="url(#p94a45562aa)" style="fill:#f3c7b1"/>
                <path d="m258.98 205.582 2.808-2.473 1.534.855-2.803 2.464z" clip-path="url(#p94a45562aa)" style="fill:#e5d8d1"/>
                <path d="m227.372 212.137 2.74-1.323 1.619 1.685-2.735 1.315z" clip-path="url(#p94a45562aa)" style="fill:#d8dce2"/>
                <path d="m238.83 210.975 2.762-1.75 1.585 1.405-2.756 1.743z" clip-path="url(#p94a45562aa)" style="fill:#dbdcde"/>
                <path d="m285.07 191.61 2.885-3.366 1.484.036-2.882 3.358z" clip-path="url(#p94a45562aa)" style="fill:#f5c0a7"/>
                <path d="m254.639 206.917 2.797-2.329 1.545.994-2.794 2.32z" clip-path="url(#p94a45562aa)" style="fill:#e3d9d3"/>
                <path d="m215.892 211.456 2.726-.899 1.656 1.97-2.72.892z" clip-path="url(#p94a45562aa)" style="fill:#dadce0"/>
                <path d="m289.44 188.28 2.9-3.52 1.477-.099-2.897 3.512z" clip-path="url(#p94a45562aa)" style="fill:#f7b89c"/>
                <path d="m250.294 207.97 2.788-2.186 1.557 1.133-2.785 2.177z" clip-path="url(#p94a45562aa)" style="fill:#e2dad5"/>
                <path d="m234.476 211.037 2.755-1.61 1.6 1.548-2.751 1.602z" clip-path="url(#p94a45562aa)" style="fill:#dcdddd"/>
                <path d="m223.001 211.49 2.737-1.183 1.634 1.83-2.731 1.176z" clip-path="url(#p94a45562aa)" style="fill:#dbdcde"/>
                <path d="m245.945 208.738 2.78-2.042 1.569 1.273-2.776 2.034z" clip-path="url(#p94a45562aa)" style="fill:#e1dad6"/>
                <path d="m230.112 210.814 2.75-1.47 1.614 1.693-2.745 1.462z" clip-path="url(#p94a45562aa)" style="fill:#dddcdc"/>
                <path d="m241.592 209.224 2.772-1.9 1.581 1.414-2.768 1.892z" clip-path="url(#p94a45562aa)" style="fill:#e1dad6"/>
                <path d="m270.494 199.297 2.846-2.922 1.511.589-2.842 2.913z" clip-path="url(#p94a45562aa)" style="fill:#f1cdba"/>
                <path d="m274.851 196.964 2.859-3.072 1.503.452-2.856 3.063z" clip-path="url(#p94a45562aa)" style="fill:#f3c8b2"/>
                <path d="m266.14 201.345 2.834-2.774 1.52.726-2.83 2.765z" clip-path="url(#p94a45562aa)" style="fill:#efcfbf"/>
                <path d="m279.213 194.344 2.872-3.221 1.495.315-2.87 3.214z" clip-path="url(#p94a45562aa)" style="fill:#f5c2aa"/>
                <path d="m261.788 203.109 2.822-2.628 1.53.864-2.818 2.619z" clip-path="url(#p94a45562aa)" style="fill:#ecd3c5"/>
                <path d="m218.618 210.557 2.733-1.043 1.65 1.976-2.727 1.037z" clip-path="url(#p94a45562aa)" style="fill:#dedcdb"/>
                <path d="m283.58 191.438 2.887-3.373 1.488.18-2.884 3.365z" clip-path="url(#p94a45562aa)" style="fill:#f7bca1"/>
                <path d="m257.436 204.588 2.811-2.482 1.541 1.003-2.807 2.473z" clip-path="url(#p94a45562aa)" style="fill:#ead4c8"/>
                <path d="m237.231 209.427 2.766-1.76 1.595 1.557-2.762 1.751z" clip-path="url(#p94a45562aa)" style="fill:#e2dad5"/>
                <path d="m225.738 210.307 2.745-1.33 1.63 1.837-2.741 1.323z" clip-path="url(#p94a45562aa)" style="fill:#e0dbd8"/>
                <path d="m287.955 188.244 2.903-3.527 1.481.044-2.9 3.52z" clip-path="url(#p94a45562aa)" style="fill:#f7b497"/>
                <path d="m253.082 205.784 2.802-2.338 1.552 1.142-2.797 2.329z" clip-path="url(#p94a45562aa)" style="fill:#e9d5cb"/>
                <path d="m248.726 206.696 2.792-2.194 1.564 1.282-2.788 2.185z" clip-path="url(#p94a45562aa)" style="fill:#e8d6cc"/>
                <path d="m232.862 209.345 2.76-1.618 1.61 1.7-2.756 1.61z" clip-path="url(#p94a45562aa)" style="fill:#e3d9d3"/>
                <path d="m244.364 207.324 2.785-2.051 1.577 1.423-2.78 2.042z" clip-path="url(#p94a45562aa)" style="fill:#e8d6cc"/>
                <path d="m221.35 209.514 2.743-1.19 1.645 1.983L223 211.49z" clip-path="url(#p94a45562aa)" style="fill:#e3d9d3"/>
                <path d="m273.34 196.375 2.861-3.08 1.509.597-2.859 3.072z" clip-path="url(#p94a45562aa)" style="fill:#f5c2aa"/>
                <path d="m268.974 198.57 2.849-2.93 1.517.735-2.846 2.922z" clip-path="url(#p94a45562aa)" style="fill:#f3c8b2"/>
                <path d="m277.71 193.892 2.875-3.23 1.5.46-2.872 3.222z" clip-path="url(#p94a45562aa)" style="fill:#f6bea4"/>
                <path d="m264.61 200.48 2.837-2.783 1.527.874-2.833 2.774z" clip-path="url(#p94a45562aa)" style="fill:#f2cbb7"/>
                <path d="m228.483 208.977 2.755-1.477 1.624 1.845-2.75 1.47z" clip-path="url(#p94a45562aa)" style="fill:#e5d8d1"/>
                <path d="m239.997 207.668 2.778-1.91 1.59 1.566-2.773 1.9z" clip-path="url(#p94a45562aa)" style="fill:#e8d6cc"/>
                <path d="m282.085 191.123 2.89-3.383 1.492.325-2.887 3.373z" clip-path="url(#p94a45562aa)" style="fill:#f7b79b"/>
                <path d="m260.247 202.106 2.826-2.637 1.537 1.012-2.822 2.628z" clip-path="url(#p94a45562aa)" style="fill:#f0cdbb"/>
                <path d="m286.467 188.065 2.905-3.536 1.486.188-2.903 3.527z" clip-path="url(#p94a45562aa)" style="fill:#f7af91"/>
                <path d="m255.884 203.446 2.815-2.491 1.548 1.151-2.81 2.482z" clip-path="url(#p94a45562aa)" style="fill:#efcfbf"/>
                <path d="m251.518 204.502 2.806-2.347 1.56 1.291-2.802 2.338z" clip-path="url(#p94a45562aa)" style="fill:#eed0c0"/>
                <path d="m235.622 207.727 2.771-1.768 1.604 1.709-2.766 1.759z" clip-path="url(#p94a45562aa)" style="fill:#e9d5cb"/>
                <path d="m224.093 208.324 2.75-1.338 1.64 1.991-2.745 1.33z" clip-path="url(#p94a45562aa)" style="fill:#e8d6cc"/>
                <path d="m247.15 205.273 2.796-2.203 1.572 1.432-2.792 2.194z" clip-path="url(#p94a45562aa)" style="fill:#edd1c2"/>
                <path d="m231.238 207.5 2.765-1.627 1.62 1.854-2.761 1.618z" clip-path="url(#p94a45562aa)" style="fill:#ead4c8"/>
                <path d="m242.775 205.759 2.789-2.06 1.585 1.574-2.785 2.051z" clip-path="url(#p94a45562aa)" style="fill:#edd1c2"/>
                <path d="m271.823 195.64 2.865-3.09 1.513.744-2.861 3.08z" clip-path="url(#p94a45562aa)" style="fill:#f6bda2"/>
                <path d="m276.201 193.294 2.879-3.24 1.505.608-2.875 3.23z" clip-path="url(#p94a45562aa)" style="fill:#f7b89c"/>
                <path d="m267.447 197.697 2.852-2.94 1.524.882-2.85 2.932z" clip-path="url(#p94a45562aa)" style="fill:#f5c1a9"/>
                <path d="m280.585 190.662 2.893-3.392 1.497.47-2.89 3.383z" clip-path="url(#p94a45562aa)" style="fill:#f7b194"/>
                <path d="m263.073 199.47 2.84-2.794 1.534 1.021-2.837 2.784z" clip-path="url(#p94a45562aa)" style="fill:#f4c5ad"/>
                <path d="m284.975 187.74 2.908-3.545 1.49.334-2.906 3.536z" clip-path="url(#p94a45562aa)" style="fill:#f7a98b"/>
                <path d="m258.7 200.955 2.83-2.647 1.543 1.161-2.826 2.637z" clip-path="url(#p94a45562aa)" style="fill:#f3c8b2"/>
                <path d="m238.393 205.96 2.782-1.92 1.6 1.719-2.778 1.909z" clip-path="url(#p94a45562aa)" style="fill:#edd1c2"/>
                <path d="m226.843 206.986 2.76-1.486 1.635 2-2.755 1.477z" clip-path="url(#p94a45562aa)" style="fill:#ecd3c5"/>
                <path d="m254.324 202.155 2.82-2.501 1.555 1.301-2.815 2.491z" clip-path="url(#p94a45562aa)" style="fill:#f2cab5"/>
                <path d="m249.946 203.07 2.81-2.357 1.568 1.442-2.806 2.347z" clip-path="url(#p94a45562aa)" style="fill:#f2cbb7"/>
                <path d="m234.003 205.873 2.777-1.776 1.613 1.862-2.77 1.768z" clip-path="url(#p94a45562aa)" style="fill:#efcfbf"/>
                <path d="m245.564 203.698 2.802-2.213 1.58 1.585-2.797 2.203z" clip-path="url(#p94a45562aa)" style="fill:#f2cbb7"/>
                <path d="m274.688 192.55 2.882-3.251 1.51.755-2.879 3.24z" clip-path="url(#p94a45562aa)" style="fill:#f7b194"/>
                <path d="m270.3 194.756 2.868-3.1 1.52.893-2.865 3.09z" clip-path="url(#p94a45562aa)" style="fill:#f7b79b"/>
                <path d="m279.08 190.054 2.896-3.402 1.502.618-2.893 3.392z" clip-path="url(#p94a45562aa)" style="fill:#f7aa8c"/>
                <path d="m265.914 196.676 2.856-2.952 1.53 1.032-2.853 2.941z" clip-path="url(#p94a45562aa)" style="fill:#f7bca1"/>
                <path d="m229.604 205.5 2.77-1.635 1.63 2.008-2.766 1.627z" clip-path="url(#p94a45562aa)" style="fill:#f0cdbb"/>
                <path d="m241.175 204.04 2.795-2.07 1.594 1.728-2.79 2.06z" clip-path="url(#p94a45562aa)" style="fill:#f2cbb7"/>
                <path d="m283.478 187.27 2.911-3.555 1.494.48-2.908 3.545z" clip-path="url(#p94a45562aa)" style="fill:#f6a385"/>
                <path d="m261.53 198.308 2.844-2.804 1.54 1.172-2.84 2.793z" clip-path="url(#p94a45562aa)" style="fill:#f6bea4"/>
                <path d="m257.144 199.654 2.834-2.658 1.551 1.312-2.83 2.647z" clip-path="url(#p94a45562aa)" style="fill:#f5c1a9"/>
                <path d="m236.78 204.097 2.787-1.928 1.608 1.872-2.782 1.918z" clip-path="url(#p94a45562aa)" style="fill:#f2cab5"/>
                <path d="m252.757 200.713 2.824-2.512 1.563 1.453-2.82 2.501z" clip-path="url(#p94a45562aa)" style="fill:#f5c2aa"/>
                <path d="m248.366 201.485 2.815-2.367 1.576 1.595-2.81 2.357z" clip-path="url(#p94a45562aa)" style="fill:#f5c4ac"/>
                <path d="m232.374 203.865 2.782-1.787 1.624 2.019-2.777 1.776z" clip-path="url(#p94a45562aa)" style="fill:#f3c8b2"/>
                <path d="m243.97 201.97 2.806-2.223 1.59 1.738-2.802 2.213z" clip-path="url(#p94a45562aa)" style="fill:#f5c4ac"/>
                <path d="m273.168 191.656 2.886-3.261 1.516.904-2.882 3.25z" clip-path="url(#p94a45562aa)" style="fill:#f7aa8c"/>
                <path d="m277.57 189.299 2.9-3.413 1.506.766-2.896 3.402z" clip-path="url(#p94a45562aa)" style="fill:#f6a385"/>
                <path d="m268.77 193.724 2.873-3.111 1.525 1.043-2.869 3.1z" clip-path="url(#p94a45562aa)" style="fill:#f7af91"/>
                <path d="m281.976 186.652 2.914-3.566 1.499.629-2.911 3.555z" clip-path="url(#p94a45562aa)" style="fill:#f59c7d"/>
                <path d="m264.374 195.504 2.86-2.962 1.536 1.182-2.856 2.952z" clip-path="url(#p94a45562aa)" style="fill:#f7b497"/>
                <path d="m259.978 196.996 2.849-2.815 1.547 1.323-2.845 2.804z" clip-path="url(#p94a45562aa)" style="fill:#f7b79b"/>
                <path d="m239.567 202.169 2.8-2.081 1.603 1.883-2.795 2.07z" clip-path="url(#p94a45562aa)" style="fill:#f5c2aa"/>
                <path d="m255.58 198.2 2.839-2.668 1.559 1.464-2.834 2.658z" clip-path="url(#p94a45562aa)" style="fill:#f7b99e"/>
                <path d="m251.18 199.118 2.83-2.523 1.57 1.606-2.823 2.512z" clip-path="url(#p94a45562aa)" style="fill:#f7ba9f"/>
                <path d="m235.156 202.078 2.792-1.938 1.619 2.029-2.787 1.928z" clip-path="url(#p94a45562aa)" style="fill:#f5c1a9"/>
                <path d="m246.776 199.747 2.82-2.378 1.585 1.749-2.815 2.367z" clip-path="url(#p94a45562aa)" style="fill:#f7ba9f"/>
                <path d="m276.054 188.395 2.903-3.425 1.512.916-2.9 3.413z" clip-path="url(#p94a45562aa)" style="fill:#f59c7d"/>
                <path d="m271.643 190.613 2.889-3.273 1.522 1.055-2.886 3.261z" clip-path="url(#p94a45562aa)" style="fill:#f6a283"/>
                <path d="m280.47 185.886 2.917-3.577 1.503.777-2.914 3.566z" clip-path="url(#p94a45562aa)" style="fill:#f39475"/>
                <path d="m267.234 192.542 2.877-3.123 1.532 1.194-2.873 3.111z" clip-path="url(#p94a45562aa)" style="fill:#f7a688"/>
                <path d="m242.366 200.088 2.812-2.235 1.598 1.894-2.806 2.224z" clip-path="url(#p94a45562aa)" style="fill:#f7ba9f"/>
                <path d="m262.827 194.181 2.864-2.974 1.543 1.335-2.86 2.962z" clip-path="url(#p94a45562aa)" style="fill:#f7aa8c"/>
                <path d="m258.419 195.532 2.853-2.826 1.555 1.475-2.85 2.815z" clip-path="url(#p94a45562aa)" style="fill:#f7ad90"/>
                <path d="m237.948 200.14 2.805-2.091 1.613 2.039-2.8 2.08z" clip-path="url(#p94a45562aa)" style="fill:#f7b99e"/>
                <path d="m254.01 196.595 2.842-2.68 1.567 1.617-2.838 2.669zM249.596 197.369l2.833-2.535 1.58 1.76-2.828 2.524z" clip-path="url(#p94a45562aa)" style="fill:#f7b093"/>
                <path d="m245.178 197.853 2.824-2.39 1.594 1.906-2.82 2.378z" clip-path="url(#p94a45562aa)" style="fill:#f7b194"/>
                <path d="m274.532 187.34 2.907-3.437 1.518 1.067-2.903 3.425z" clip-path="url(#p94a45562aa)" style="fill:#f29274"/>
                <path d="m278.957 184.97 2.921-3.589 1.509.928-2.918 3.577z" clip-path="url(#p94a45562aa)" style="fill:#f08b6e"/>
                <path d="m270.11 189.419 2.894-3.286 1.528 1.207-2.89 3.273z" clip-path="url(#p94a45562aa)" style="fill:#f4987a"/>
                <path d="m265.691 191.207 2.88-3.135 1.54 1.347-2.877 3.123z" clip-path="url(#p94a45562aa)" style="fill:#f59d7e"/>
                <path d="m261.272 192.706 2.869-2.987 1.55 1.488-2.864 2.974z" clip-path="url(#p94a45562aa)" style="fill:#f5a081"/>
                <path d="m240.753 198.049 2.817-2.246 1.608 2.05-2.812 2.235z" clip-path="url(#p94a45562aa)" style="fill:#f7b093"/>
                <path d="m256.852 193.915 2.858-2.84 1.562 1.63-2.853 2.827z" clip-path="url(#p94a45562aa)" style="fill:#f6a385"/>
                <path d="m252.43 194.834 2.847-2.693 1.575 1.774-2.843 2.68z" clip-path="url(#p94a45562aa)" style="fill:#f6a586"/>
                <path d="m248.002 195.463 2.839-2.546 1.588 1.917-2.833 2.535z" clip-path="url(#p94a45562aa)" style="fill:#f7a688"/>
                <path d="m277.439 183.903 2.925-3.601 1.514 1.08-2.921 3.588z" clip-path="url(#p94a45562aa)" style="fill:#ec8165"/>
                <path d="m273.004 186.133 2.91-3.449 1.525 1.22-2.907 3.436z" clip-path="url(#p94a45562aa)" style="fill:#ef886b"/>
                <path d="m268.572 188.072 2.897-3.298 1.535 1.36-2.893 3.285z" clip-path="url(#p94a45562aa)" style="fill:#f18f71"/>
                <path d="m243.57 195.803 2.83-2.402 1.602 2.062-2.824 2.39z" clip-path="url(#p94a45562aa)" style="fill:#f6a586"/>
                <path d="m264.14 189.72 2.886-3.149 1.546 1.5-2.88 3.136z" clip-path="url(#p94a45562aa)" style="fill:#f29274"/>
                <path d="m259.71 191.076 2.873-3 1.558 1.643-2.869 2.987z" clip-path="url(#p94a45562aa)" style="fill:#f39577"/>
                <path d="m255.277 192.141 2.862-2.851 1.57 1.786-2.857 2.839z" clip-path="url(#p94a45562aa)" style="fill:#f4987a"/>
                <path d="m250.84 192.917 2.853-2.706 1.584 1.93-2.848 2.693z" clip-path="url(#p94a45562aa)" style="fill:#f49a7b"/>
                <path d="m246.4 193.4 2.843-2.56 1.598 2.077-2.839 2.546z" clip-path="url(#p94a45562aa)" style="fill:#f49a7b"/>
                <path d="m275.915 182.684 2.928-3.615 1.52 1.233-2.924 3.601z" clip-path="url(#p94a45562aa)" style="fill:#e8765c"/>
                <path d="m271.47 184.774 2.914-3.463 1.53 1.373-2.91 3.45z" clip-path="url(#p94a45562aa)" style="fill:#eb7d62"/>
                <path d="m267.026 186.571 2.901-3.312 1.542 1.515-2.897 3.298z" clip-path="url(#p94a45562aa)" style="fill:#ed8366"/>
                <path d="m262.583 188.076 2.89-3.162 1.553 1.657-2.885 3.148z" clip-path="url(#p94a45562aa)" style="fill:#ee8669"/>
                <path d="m258.14 189.29 2.877-3.014 1.566 1.8-2.873 3z" clip-path="url(#p94a45562aa)" style="fill:#f08a6c"/>
                <path d="m253.693 190.211 2.868-2.865 1.578 1.944-2.862 2.851zM249.243 190.84l2.857-2.718 1.593 2.09-2.852 2.705z" clip-path="url(#p94a45562aa)" style="fill:#f08b6e"/>
                <path d="m274.384 181.311 2.933-3.63 1.526 1.388-2.928 3.615z" clip-path="url(#p94a45562aa)" style="fill:#e36b54"/>
                <path d="m269.927 183.26 2.92-3.477 1.537 1.528-2.915 3.463z" clip-path="url(#p94a45562aa)" style="fill:#e57058"/>
                <path d="m265.472 184.914 2.907-3.325 1.548 1.67-2.901 3.312z" clip-path="url(#p94a45562aa)" style="fill:#e8765c"/>
                <path d="m261.017 186.276 2.894-3.175 1.561 1.813-2.889 3.162z" clip-path="url(#p94a45562aa)" style="fill:#e97a5f"/>
                <path d="m256.56 187.346 2.883-3.028 1.574 1.958-2.878 3.014z" clip-path="url(#p94a45562aa)" style="fill:#ea7b60"/>
                <path d="m252.1 188.122 2.873-2.88 1.588 2.104-2.868 2.865z" clip-path="url(#p94a45562aa)" style="fill:#eb7d62"/>
                <path d="m272.847 179.783 2.937-3.645 1.533 1.544-2.933 3.63z" clip-path="url(#p94a45562aa)" style="fill:#dc5d4a"/>
                <path d="m268.379 181.589 2.923-3.492 1.545 1.686-2.92 3.476z" clip-path="url(#p94a45562aa)" style="fill:#df634e"/>
                <path d="m263.911 183.1 2.911-3.34 1.557 1.829-2.907 3.325z" clip-path="url(#p94a45562aa)" style="fill:#e16751"/>
                <path d="m259.443 184.318 2.9-3.19 1.568 1.973-2.894 3.175z" clip-path="url(#p94a45562aa)" style="fill:#e36b54"/>
                <path d="m254.973 185.242 2.888-3.042 1.582 2.118-2.882 3.028z" clip-path="url(#p94a45562aa)" style="fill:#e36c55"/>
                <path d="m271.302 178.097 2.942-3.66 1.54 1.701-2.937 3.645z" clip-path="url(#p94a45562aa)" style="fill:#d44e41"/>
                <path d="m266.822 179.76 2.928-3.508 1.552 1.845-2.923 3.492z" clip-path="url(#p94a45562aa)" style="fill:#d75445"/>
                <path d="m262.342 181.127 2.916-3.356 1.564 1.989-2.91 3.34z" clip-path="url(#p94a45562aa)" style="fill:#d95847"/>
                <path d="m257.86 182.2 2.904-3.207 1.578 2.134-2.899 3.191z" clip-path="url(#p94a45562aa)" style="fill:#da5a49"/>
                <path d="m269.75 176.252 2.946-3.676 1.548 1.861-2.942 3.66z" clip-path="url(#p94a45562aa)" style="fill:#ca3b37"/>
                <path d="m265.258 177.771 2.933-3.524 1.56 2.005-2.929 3.508z" clip-path="url(#p94a45562aa)" style="fill:#cd423b"/>
                <path d="m260.764 178.993 2.921-3.372 1.573 2.15-2.916 3.356z" clip-path="url(#p94a45562aa)" style="fill:#d0473d"/>
                <path d="m268.19 174.247 2.952-3.693 1.554 2.022-2.946 3.676z" clip-path="url(#p94a45562aa)" style="fill:#c0282f"/>
                <path d="m263.685 175.62 2.938-3.54 1.568 2.167-2.933 3.524z" clip-path="url(#p94a45562aa)" style="fill:#c32e31"/>
                <path d="m266.623 172.08 2.956-3.711 1.563 2.185-2.951 3.693z" clip-path="url(#p94a45562aa)" style="fill:#b40426"/>
            </g>
            <g id="text_18" transform="matrix(.12 0 0 -.12 180.89 35.472)">
                <defs>
                    <path id="DejaVuSans-4c" d="M628 4666h631V531h2272V0H628v4666z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-6f" d="M1959 3097q-462 0-731-361t-269-989q0-628 267-989 268-361 733-361 460 0 728 362 269 363 269 988 0 622-269 986-268 364-728 364zm0 487q750 0 1178-488 429-487 429-1349 0-859-429-1349Q2709-91 1959-91q-753 0-1180 489-426 490-426 1349 0 862 426 1349 427 488 1180 488z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-73" d="M2834 3397v-544q-243 125-506 187-262 63-544 63-428 0-642-131t-214-394q0-200 153-314t616-217l197-44q612-131 870-370t258-667q0-488-386-773Q2250-91 1575-91q-281 0-586 55T347 128v594q319-166 628-249 309-82 613-82 406 0 624 139 219 139 219 392 0 234-158 359-157 125-692 241l-200 47q-534 112-772 345-237 233-237 639 0 494 350 762 350 269 994 269 318 0 599-47 282-46 519-140z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-46" d="M628 4666h2681v-532H1259V2759h1850v-531H1259V0H628v4666z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-75" d="M544 1381v2119h575V1403q0-497 193-746 194-248 582-248 465 0 735 297 271 297 271 810v1984h575V0h-575v538q-209-319-486-474-276-155-642-155-603 0-916 375-312 375-312 1097zm1447 2203z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-6e" d="M3513 2113V0h-575v2094q0 497-194 743-194 247-581 247-466 0-735-297-269-296-269-809V0H581v3500h578v-544q207 316 486 472 280 156 646 156 603 0 912-373 310-373 310-1098z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-63" d="M3122 3366v-538q-244 135-489 202t-495 67q-560 0-870-355-309-354-309-995t309-996q310-354 870-354 250 0 495 67t489 202V134Q2881 22 2623-34q-257-57-548-57-791 0-1257 497-465 497-465 1341 0 856 470 1346 471 491 1290 491 265 0 518-55 253-54 491-163z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-74" d="M1172 4494v-994h1184v-447H1172V1153q0-428 117-550t477-122h590V0h-590q-666 0-919 248-253 249-253 905v1900H172v447h422v994h578z" transform="scale(.01563)"/>
                    <path id="DejaVuSans-69" d="M603 3500h575V0H603v3500zm0 1363h575v-729H603v729z" transform="scale(.01563)"/>
                </defs>
                <use xlink:href="#DejaVuSans-4d"/>
                <use xlink:href="#DejaVuSans-53" x="86.279"/>
                <use xlink:href="#DejaVuSans-45" x="149.756"/>
                <use xlink:href="#DejaVuSans-20" x="212.939"/>
                <use xlink:href="#DejaVuSans-4c" x="244.727"/>
                <use xlink:href="#DejaVuSans-6f" x="298.689"/>
                <use xlink:href="#DejaVuSans-73" x="359.871"/>
                <use xlink:href="#DejaVuSans-73" x="411.971"/>
                <use xlink:href="#DejaVuSans-20" x="464.07"/>
                <use xlink:href="#DejaVuSans-46" x="495.857"/>
                <use xlink:href="#DejaVuSans-75" x="547.877"/>
                <use xlink:href="#DejaVuSans-6e" x="611.256"/>
                <use xlink:href="#DejaVuSans-63" x="674.635"/>
                <use xlink:href="#DejaVuSans-74" x="729.615"/>
                <use xlink:href="#DejaVuSans-69" x="768.824"/>
                <use xlink:href="#DejaVuSans-6f" x="796.607"/>
                <use xlink:href="#DejaVuSans-6e" x="857.789"/>
            </g>
        </g>
    </g>
    <defs>
        <clipPath id="p94a45562aa">
            <path d="M103.104 41.472h266.112v266.112H103.104z"/>
        </clipPath>
    </defs>
</svg>

##### 数值微分

**导数** $\frac{df(x)}{dx}=\lim\limits_{h\rightarrow 0}\frac{f(x+h)-f(x)}{h}$

```python
# 这种实现方法有舍入误差, 比如 float32 下 h = 0.0 而不是我们希望的 0.000....01
def numerical_diff(f, x): 
    h = 10e-50
    return (f(x + h) - f(x)) / h
```

为了减小这个误差，我们可以计算函数 $f$ 在 $(x + h)$ 和 $(x − h)$ 之间的差分

因为这种计算方法以 $x$ 为中心，计算它左右两边的差分，所以也称为中心差分（而 $(x + h)$ 和 $x$ 之间的差分称为前向差分）

<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="614.4" height="460.8" viewBox="0 0 460.8 345.6">
    <defs>
        <style>
            *{stroke-linejoin:round;stroke-linecap:butt}
        </style>
    </defs>
    <g id="figure_1">
        <path id="patch_1" d="M0 345.6h460.8V0H0z" style="fill:#fff"/>
        <g id="axes_1">
            <path id="patch_2" d="M57.6 307.584h357.12V41.472H57.6z" style="fill:#fff"/>
            <g id="matplotlib.axis_1">
                <g id="xtick_1">
                    <g id="line2d_1">
                        <defs>
                            <path id="mab8129b54e" d="M0 0v3.5" style="stroke:#000;stroke-width:.8"/>
                        </defs>
                        <use xlink:href="#mab8129b54e" x="73.833" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    </g>
                    <g id="text_1" transform="matrix(.1 0 0 -.1 61.691 322.182)">
                        <defs>
                            <path id="DejaVuSans-2212" d="M678 2272h4006v-531H678v531z" transform="scale(.01563)"/>
                            <path id="DejaVuSans-32" d="M1228 531h2203V0H469v531q359 372 979 998 621 627 780 809 303 340 423 576 121 236 121 464 0 372-261 606-261 235-680 235-297 0-627-103-329-103-704-313v638q381 153 712 231 332 78 607 78 725 0 1156-363 431-362 431-968 0-288-108-546-107-257-392-607-78-91-497-524-418-433-1181-1211z" transform="scale(.01563)"/>
                            <path id="DejaVuSans-2e" d="M684 794h660V0H684v794z" transform="scale(.01563)"/>
                            <path id="DejaVuSans-30" d="M2034 4250q-487 0-733-480-245-479-245-1442 0-959 245-1439 246-480 733-480 491 0 736 480 246 480 246 1439 0 963-246 1442-245 480-736 480zm0 500q785 0 1199-621 414-620 414-1801 0-1178-414-1799Q2819-91 2034-91q-784 0-1198 620-414 621-414 1799 0 1181 414 1801 414 621 1198 621z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-2212"/>
                        <use xlink:href="#DejaVuSans-32" x="83.789"/>
                        <use xlink:href="#DejaVuSans-2e" x="147.412"/>
                        <use xlink:href="#DejaVuSans-30" x="179.199"/>
                    </g>
                </g>
                <g id="xtick_2">
                    <use xlink:href="#mab8129b54e" id="line2d_2" x="114.415" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_2" transform="matrix(.1 0 0 -.1 102.273 322.182)">
                        <defs>
                            <path id="DejaVuSans-31" d="M794 531h1031v3560L703 3866v575l1116 225h631V531h1031V0H794v531z" transform="scale(.01563)"/>
                            <path id="DejaVuSans-35" d="M691 4666h2478v-532H1269V2991q137 47 274 70 138 23 276 23 781 0 1237-428 457-428 457-1159 0-753-469-1171Q2575-91 1722-91q-294 0-599 50Q819 9 494 109v635q281-153 581-228t634-75q541 0 856 284 316 284 316 772 0 487-316 771-315 285-856 285-253 0-505-56-251-56-513-175v2344z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-2212"/>
                        <use xlink:href="#DejaVuSans-31" x="83.789"/>
                        <use xlink:href="#DejaVuSans-2e" x="147.412"/>
                        <use xlink:href="#DejaVuSans-35" x="179.199"/>
                    </g>
                </g>
                <g id="xtick_3">
                    <use xlink:href="#mab8129b54e" id="line2d_3" x="154.996" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_3" transform="matrix(.1 0 0 -.1 142.855 322.182)">
                        <use xlink:href="#DejaVuSans-2212"/>
                        <use xlink:href="#DejaVuSans-31" x="83.789"/>
                        <use xlink:href="#DejaVuSans-2e" x="147.412"/>
                        <use xlink:href="#DejaVuSans-30" x="179.199"/>
                    </g>
                </g>
                <g id="xtick_4">
                    <use xlink:href="#mab8129b54e" id="line2d_4" x="195.578" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_4" transform="matrix(.1 0 0 -.1 183.437 322.182)">
                        <use xlink:href="#DejaVuSans-2212"/>
                        <use xlink:href="#DejaVuSans-30" x="83.789"/>
                        <use xlink:href="#DejaVuSans-2e" x="147.412"/>
                        <use xlink:href="#DejaVuSans-35" x="179.199"/>
                    </g>
                </g>
                <g id="xtick_5">
                    <use xlink:href="#mab8129b54e" id="line2d_5" x="236.16" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_5" transform="matrix(.1 0 0 -.1 228.208 322.182)">
                        <use xlink:href="#DejaVuSans-30"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-30" x="95.41"/>
                    </g>
                </g>
                <g id="xtick_6">
                    <use xlink:href="#mab8129b54e" id="line2d_6" x="276.742" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_6" transform="matrix(.1 0 0 -.1 268.79 322.182)">
                        <use xlink:href="#DejaVuSans-30"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-35" x="95.41"/>
                    </g>
                </g>
                <g id="xtick_7">
                    <use xlink:href="#mab8129b54e" id="line2d_7" x="317.324" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_7" transform="matrix(.1 0 0 -.1 309.372 322.182)">
                        <use xlink:href="#DejaVuSans-31"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-30" x="95.41"/>
                    </g>
                </g>
                <g id="xtick_8">
                    <use xlink:href="#mab8129b54e" id="line2d_8" x="357.905" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_8" transform="matrix(.1 0 0 -.1 349.954 322.182)">
                        <use xlink:href="#DejaVuSans-31"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-35" x="95.41"/>
                    </g>
                </g>
                <g id="xtick_9">
                    <use xlink:href="#mab8129b54e" id="line2d_9" x="398.487" y="307.584" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_9" transform="matrix(.1 0 0 -.1 390.536 322.182)">
                        <use xlink:href="#DejaVuSans-32"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-30" x="95.41"/>
                    </g>
                </g>
            </g>
            <g id="matplotlib.axis_2">
                <g id="ytick_1">
                    <g id="line2d_10">
                        <defs>
                            <path id="m3829eca154" d="M0 0h-3.5" style="stroke:#000;stroke-width:.8"/>
                        </defs>
                        <use xlink:href="#m3829eca154" x="57.6" y="294.011" style="stroke:#000;stroke-width:.8"/>
                    </g>
                    <g id="text_10" transform="matrix(.1 0 0 -.1 19.955 297.81)">
                        <use xlink:href="#DejaVuSans-2212"/>
                        <use xlink:href="#DejaVuSans-30" x="83.789"/>
                        <use xlink:href="#DejaVuSans-2e" x="147.412"/>
                        <use xlink:href="#DejaVuSans-32" x="179.199"/>
                        <use xlink:href="#DejaVuSans-35" x="242.822"/>
                    </g>
                </g>
                <g id="ytick_2">
                    <use xlink:href="#m3829eca154" id="line2d_11" x="57.6" y="262.131" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_11" transform="matrix(.1 0 0 -.1 28.334 265.93)">
                        <use xlink:href="#DejaVuSans-30"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-30" x="95.41"/>
                        <use xlink:href="#DejaVuSans-30" x="159.033"/>
                    </g>
                </g>
                <g id="ytick_3">
                    <use xlink:href="#m3829eca154" id="line2d_12" x="57.6" y="230.252" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_12" transform="matrix(.1 0 0 -.1 28.334 234.052)">
                        <use xlink:href="#DejaVuSans-30"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-32" x="95.41"/>
                        <use xlink:href="#DejaVuSans-35" x="159.033"/>
                    </g>
                </g>
                <g id="ytick_4">
                    <use xlink:href="#m3829eca154" id="line2d_13" x="57.6" y="198.373" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_13" transform="matrix(.1 0 0 -.1 28.334 202.172)">
                        <use xlink:href="#DejaVuSans-30"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-35" x="95.41"/>
                        <use xlink:href="#DejaVuSans-30" x="159.033"/>
                    </g>
                </g>
                <g id="ytick_5">
                    <use xlink:href="#m3829eca154" id="line2d_14" x="57.6" y="166.494" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_14" transform="matrix(.1 0 0 -.1 28.334 170.293)">
                        <defs>
                            <path id="DejaVuSans-37" d="M525 4666h3000v-269L1831 0h-659l1594 4134H525v532z" transform="scale(.01563)"/>
                        </defs>
                        <use xlink:href="#DejaVuSans-30"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-37" x="95.41"/>
                        <use xlink:href="#DejaVuSans-35" x="159.033"/>
                    </g>
                </g>
                <g id="ytick_6">
                    <use xlink:href="#m3829eca154" id="line2d_15" x="57.6" y="134.615" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_15" transform="matrix(.1 0 0 -.1 28.334 138.414)">
                        <use xlink:href="#DejaVuSans-31"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-30" x="95.41"/>
                        <use xlink:href="#DejaVuSans-30" x="159.033"/>
                    </g>
                </g>
                <g id="ytick_7">
                    <use xlink:href="#m3829eca154" id="line2d_16" x="57.6" y="102.736" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_16" transform="matrix(.1 0 0 -.1 28.334 106.535)">
                        <use xlink:href="#DejaVuSans-31"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-32" x="95.41"/>
                        <use xlink:href="#DejaVuSans-35" x="159.033"/>
                    </g>
                </g>
                <g id="ytick_8">
                    <use xlink:href="#m3829eca154" id="line2d_17" x="57.6" y="70.857" style="stroke:#000;stroke-width:.8"/>
                    <g id="text_17" transform="matrix(.1 0 0 -.1 28.334 74.656)">
                        <use xlink:href="#DejaVuSans-31"/>
                        <use xlink:href="#DejaVuSans-2e" x="63.623"/>
                        <use xlink:href="#DejaVuSans-35" x="95.41"/>
                        <use xlink:href="#DejaVuSans-30" x="159.033"/>
                    </g>
                </g>
            </g>
            <path id="line2d_18" d="m73.833 262.126 36.072-.048 36.073-.438 36.073-3.9 36.073-27.19 36.072-64.354 36.073-27.189 36.073-3.901 36.073-.438 36.072-.047" clip-path="url(#p9bb4692542)" style="fill:none;stroke:#1f77b4;stroke-width:1.5;stroke-linecap:square"/>
            <path id="line2d_19" d="m73.833 256.03 36.072-19.865 36.073-19.865 36.073-19.866 36.073-19.865 36.072-19.865 36.073-19.865 36.073-19.866 36.073-19.865 36.072-19.865" clip-path="url(#p9bb4692542)" style="fill:none;stroke:#ff7f0e;stroke-width:1.5;stroke-linecap:square"/>
            <path id="line2d_20" d="m73.833 295.488 36.072-26.88 36.073-26.88 36.073-26.88 36.073-26.88 36.072-26.88 36.073-26.88 36.073-26.88 36.073-26.88 36.072-26.88" clip-path="url(#p9bb4692542)" style="fill:none;stroke:#2ca02c;stroke-width:1.5;stroke-linecap:square"/>
            <path id="patch_3" d="M57.6 307.584V41.472" style="fill:none;stroke:#000;stroke-width:.8;stroke-linejoin:miter;stroke-linecap:square"/>
            <path id="patch_4" d="M414.72 307.584V41.472" style="fill:none;stroke:#000;stroke-width:.8;stroke-linejoin:miter;stroke-linecap:square"/>
            <path id="patch_5" d="M57.6 307.584h357.12" style="fill:none;stroke:#000;stroke-width:.8;stroke-linejoin:miter;stroke-linecap:square"/>
            <path id="patch_6" d="M57.6 41.472h357.12" style="fill:none;stroke:#000;stroke-width:.8;stroke-linejoin:miter;stroke-linecap:square"/>
            <g id="legend_1">
                <path id="patch_7" d="M64.6 93.506h125.33q2 0 2-2V48.472q0-2-2-2H64.6q-2 0-2 2v43.034q0 2 2 2z" style="fill:#fff;opacity:.8;stroke:#ccc;stroke-linejoin:miter"/>
                <path id="line2d_21" d="M66.6 54.57h20" style="fill:none;stroke:#1f77b4;stroke-width:1.5;stroke-linecap:square"/>
                <g id="text_18" transform="matrix(.1 0 0 -.1 94.6 58.07)">
                    <defs>
                        <path id="DejaVuSans-4f" d="M2522 4238q-688 0-1093-513-404-512-404-1397 0-881 404-1394 405-512 1093-512 687 0 1089 512 402 513 402 1394 0 885-402 1397-402 513-1089 513zm0 512q981 0 1568-658 588-658 588-1764 0-1103-588-1761Q3503-91 2522-91q-984 0-1574 656-589 657-589 1763t589 1764q590 658 1574 658z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-72" d="M2631 2963q-97 56-211 82-114 27-251 27-488 0-749-317t-261-911V0H581v3500h578v-544q182 319 472 473 291 155 707 155 59 0 131-8 72-7 159-23l3-590z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-69" d="M603 3500h575V0H603v3500zm0 1363h575v-729H603v729z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-67" d="M2906 1791q0 625-258 968-257 344-723 344-462 0-720-344-258-343-258-968 0-622 258-966t720-344q466 0 723 344 258 344 258 966zm575-1357q0-893-397-1329-396-436-1215-436-303 0-572 45t-522 139v559q253-137 500-202 247-66 503-66 566 0 847 295t281 892v285q-178-310-456-463T1784 0Q1141 0 747 490 353 981 353 1791q0 812 394 1302 394 491 1037 491 388 0 666-153t456-462v531h575V434z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-6e" d="M3513 2113V0h-575v2094q0 497-194 743-194 247-581 247-466 0-735-297-269-296-269-809V0H581v3500h578v-544q207 316 486 472 280 156 646 156 603 0 912-373 310-373 310-1098z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-61" d="M2194 1759q-697 0-966-159t-269-544q0-306 202-486 202-179 548-179 479 0 768 339t289 901v128h-572zm1147 238V0h-575v531q-197-318-491-470T1556-91q-537 0-855 302-317 302-317 808 0 590 395 890 396 300 1180 300h807v57q0 397-261 614t-733 217q-300 0-585-72-284-72-546-216v532q315 122 612 182 297 61 578 61 760 0 1135-394 375-393 375-1193z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-6c" d="M603 4863h575V0H603v4863z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-66" d="M2375 4863v-479h-550q-309 0-430-125-120-125-120-450v-309h947v-447h-947V0H697v3053H147v447h550v244q0 584 272 851 272 268 862 268h544z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-75" d="M544 1381v2119h575V1403q0-497 193-746 194-248 582-248 465 0 735 297 271 297 271 810v1984h575V0h-575v538q-209-319-486-474-276-155-642-155-603 0-916 375-312 375-312 1097zm1447 2203z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-63" d="M3122 3366v-538q-244 135-489 202t-495 67q-560 0-870-355-309-354-309-995t309-996q310-354 870-354 250 0 495 67t489 202V134Q2881 22 2623-34q-257-57-548-57-791 0-1257 497-465 497-465 1341 0 856 470 1346 471 491 1290 491 265 0 518-55 253-54 491-163z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-74" d="M1172 4494v-994h1184v-447H1172V1153q0-428 117-550t477-122h590V0h-590q-666 0-919 248-253 249-253 905v1900H172v447h422v994h578z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-6f" d="M1959 3097q-462 0-731-361t-269-989q0-628 267-989 268-361 733-361 460 0 728 362 269 363 269 988 0 622-269 986-268 364-728 364zm0 487q750 0 1178-488 429-487 429-1349 0-859-429-1349Q2709-91 1959-91q-753 0-1180 489-426 490-426 1349 0 862 426 1349 427 488 1180 488z" transform="scale(.01563)"/>
                    </defs>
                    <use xlink:href="#DejaVuSans-4f"/>
                    <use xlink:href="#DejaVuSans-72" x="78.711"/>
                    <use xlink:href="#DejaVuSans-69" x="119.824"/>
                    <use xlink:href="#DejaVuSans-67" x="147.607"/>
                    <use xlink:href="#DejaVuSans-69" x="211.084"/>
                    <use xlink:href="#DejaVuSans-6e" x="238.867"/>
                    <use xlink:href="#DejaVuSans-61" x="302.246"/>
                    <use xlink:href="#DejaVuSans-6c" x="363.525"/>
                    <use xlink:href="#DejaVuSans-20" x="391.309"/>
                    <use xlink:href="#DejaVuSans-66" x="423.096"/>
                    <use xlink:href="#DejaVuSans-75" x="458.301"/>
                    <use xlink:href="#DejaVuSans-6e" x="521.68"/>
                    <use xlink:href="#DejaVuSans-63" x="585.059"/>
                    <use xlink:href="#DejaVuSans-74" x="640.039"/>
                    <use xlink:href="#DejaVuSans-69" x="679.248"/>
                    <use xlink:href="#DejaVuSans-6f" x="707.031"/>
                    <use xlink:href="#DejaVuSans-6e" x="768.213"/>
                </g>
                <path id="line2d_22" d="M66.6 69.249h20" style="fill:none;stroke:#ff7f0e;stroke-width:1.5;stroke-linecap:square"/>
                <g id="text_19" transform="matrix(.1 0 0 -.1 94.6 72.749)">
                    <defs>
                        <path id="DejaVuSans-41" d="m2188 4044-857-2322h1716l-859 2322zm-357 622h716L4325 0h-656l-425 1197H1141L716 0H50l1781 4666z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-79" d="M2059-325q-243-625-475-815-231-191-618-191H506v481h338q237 0 368 113 132 112 291 531l103 262L191 3500h609L1894 763l1094 2737h609L2059-325z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-65" d="M3597 1894v-281H953q38-594 358-905t892-311q331 0 642 81t618 244V178Q3153 47 2828-22t-659-69q-838 0-1327 487-489 488-489 1320 0 859 464 1363 464 505 1252 505 706 0 1117-455 411-454 411-1235zm-575 169q-6 471-264 752-258 282-683 282-481 0-770-272t-333-766l2050 4z" transform="scale(.01563)"/>
                    </defs>
                    <use xlink:href="#DejaVuSans-41"/>
                    <use xlink:href="#DejaVuSans-6e" x="68.408"/>
                    <use xlink:href="#DejaVuSans-61" x="131.787"/>
                    <use xlink:href="#DejaVuSans-6c" x="193.066"/>
                    <use xlink:href="#DejaVuSans-79" x="220.85"/>
                    <use xlink:href="#DejaVuSans-74" x="280.029"/>
                    <use xlink:href="#DejaVuSans-69" x="319.238"/>
                    <use xlink:href="#DejaVuSans-63" x="347.021"/>
                    <use xlink:href="#DejaVuSans-61" x="402.002"/>
                    <use xlink:href="#DejaVuSans-6c" x="463.281"/>
                    <use xlink:href="#DejaVuSans-20" x="491.064"/>
                    <use xlink:href="#DejaVuSans-74" x="522.852"/>
                    <use xlink:href="#DejaVuSans-61" x="562.061"/>
                    <use xlink:href="#DejaVuSans-6e" x="623.34"/>
                    <use xlink:href="#DejaVuSans-67" x="686.719"/>
                    <use xlink:href="#DejaVuSans-65" x="750.195"/>
                    <use xlink:href="#DejaVuSans-6e" x="811.719"/>
                    <use xlink:href="#DejaVuSans-74" x="875.098"/>
                </g>
                <path id="line2d_23" d="M66.6 83.927h20" style="fill:none;stroke:#2ca02c;stroke-width:1.5;stroke-linecap:square"/>
                <g id="text_20" transform="matrix(.1 0 0 -.1 94.6 87.427)">
                    <defs>
                        <path id="DejaVuSans-4e" d="M628 4666h850L3547 763v3903h612V0h-850L1241 3903V0H628v4666z" transform="scale(.01563)"/>
                        <path id="DejaVuSans-6d" d="M3328 2828q216 388 516 572t706 184q547 0 844-383 297-382 297-1088V0h-578v2094q0 503-179 746-178 244-543 244-447 0-707-297-259-296-259-809V0h-578v2094q0 506-178 748t-550 242q-441 0-701-298-259-298-259-808V0H581v3500h578v-544q197 322 472 475t653 153q382 0 649-194 267-193 395-562z" transform="scale(.01563)"/>
                    </defs>
                    <use xlink:href="#DejaVuSans-4e"/>
                    <use xlink:href="#DejaVuSans-75" x="74.805"/>
                    <use xlink:href="#DejaVuSans-6d" x="138.184"/>
                    <use xlink:href="#DejaVuSans-65" x="235.596"/>
                    <use xlink:href="#DejaVuSans-72" x="297.119"/>
                    <use xlink:href="#DejaVuSans-69" x="338.232"/>
                    <use xlink:href="#DejaVuSans-63" x="366.016"/>
                    <use xlink:href="#DejaVuSans-61" x="420.996"/>
                    <use xlink:href="#DejaVuSans-6c" x="482.275"/>
                    <use xlink:href="#DejaVuSans-20" x="510.059"/>
                    <use xlink:href="#DejaVuSans-74" x="541.846"/>
                    <use xlink:href="#DejaVuSans-61" x="581.055"/>
                    <use xlink:href="#DejaVuSans-6e" x="642.334"/>
                    <use xlink:href="#DejaVuSans-67" x="705.713"/>
                    <use xlink:href="#DejaVuSans-65" x="769.189"/>
                    <use xlink:href="#DejaVuSans-6e" x="830.713"/>
                    <use xlink:href="#DejaVuSans-74" x="894.092"/>
                </g>
            </g>
        </g>
    </g>
    <defs>
        <clipPath id="p9bb4692542">
            <path d="M57.6 41.472h357.12v266.112H57.6z"/>
        </clipPath>
    </defs>
</svg>



我们可以看到数值微分也更符合我们对于线性回归的定义

我们发现虽然严格意义上它们并不一致，但误差非常小

实际上，误差小到基本上可以认为它们是相等的



**偏导数**

```python
def fun(x):
    return x[0] ** 2 + x[1] ** 2
```

$f(x_0,x_1)=x_0^2+x_1^2$ 的两个偏导 $\frac{\partial f}{x_0}$ $\frac{\partial f}{x_1}$

问题当 $x_0=3~~~x_1=4$ 时，求偏导数  $\frac{\partial f}{x_0}$ $\frac{\partial f}{x_1}$



##### 梯度推导

全部变量的偏导数汇总而成的向量称为梯度 (**矢量**)

$\frac{d}{dx}(f+g)=\frac{d}{dx}f+\frac{d}{dx}g$

$\frac{d}{dx}(af)=a\frac{d}{dx}f~(a\in C)$

$\frac{d}{dx}(fg)=\frac{d}{dx}f\times g+\frac{d}{dx}g\times f~(a\in C)$

$\frac{d}{dx}(f/g)=((\frac{d}{dx}f)g-(\frac{d}{dx}g)f)/g^2$ （由乘法公式和 $1/x$ 的导数推导）



$\frac{d}{dx}f(g(x))=\frac{d}{dx}f(g(x))\times \frac{d}{dx}g(x)$ （内偏导乘外偏导）

$L=(y_{_{10}}-y_{predict_{10}})^2=(y_{_0}-y_{predict_0})^2+(y_{_1}-y_{predict_1})^2......=L_0+L_1+...+L_9$ 

$L_0=(y-A_2(\vec{l_0}\times w+\vec{b_1}))^2=(y-A_2(A_1(x+\vec{b_0})*w_1+\vec{b_1}))^2$

上面的公式 $y$ 和 $x$ 是定值分辨代表了理想模型的 $y$ 和代表了输入数据的 $x$

所以梯度向量为 $(\frac{\partial L}{\partial b_{1_i}}~~~\frac{\partial L}{\partial b_{0_i}}~~~\frac{\partial L}{\partial w_{1_{i,j}}})$ $(注意 i 代表第 i 个节点的偏置)$

其中 $\frac{\partial L}{\partial b_{1_i}}=\frac{\partial L_0}{\partial b_{1_i}}+\frac{\partial L_1}{\partial b_{1_i}}+...$

分别计算 $\frac{\partial L_0}{b_{1_i}}=(y-y_{pred_0})^2$ 复合函数求导

$=2(y_0-y_{pred_0})\frac{\partial}{b_{1_i}}(y_0-y_{pred_0})$

$=-2(y_0-y_{pred_0})\frac{\partial}{b_{1_i}}(y_{pred_0})$

$=-2(y_0-y_{pred_0})\frac{\partial}{b_{1_i}}(A_{2,0}(\vec{l_0}\times w_1+\vec{b_1}))$

将向量的所有分量的偏导数求和，其实是将每个分量的偏导数都考虑进来，得到一个总的偏导数，它描述了向量在所有维度上的变化率。这个总的偏导数的大小和方向可以帮助我们决定向量在哪个方向上变化最快，以及变化的速率有多快

$=-2(y_0-y_{pred_0})\sum\limits_{m=1}^{1\to m}((\partial_m (A_{2,0}(\vec{l_0}\times w_{1}+\vec{b_{1}}))\frac{\partial}{b_{1_i}}(\vec{l_0}\times w_{1_m}+\vec{b_{1_m}}))$

为什么下面这步的 $m\to i$ 因为如果 $m\neq i$ 那么后面的方程都是变量 $=0$ 没有意义

$=-2(y_0-y_{pred_0})\partial_i A_{2,0}(\vec{l_0}\times w_{1}+\vec{b_{1}})$

为什么是 $b_1$ 的原因是因为这部分不求导所以不用分开，复合函数的内导数作为定值

$\frac{\partial L_1}{b_{1_i}}=-2(y_0-y_{pred_1})\partial_i A_{2,1}(\vec{l_0}\times w_{1}+\vec{b_{1}})$

$\frac{\partial L}{b_{1_i}}=-2(\vec y_0-\vec y_{pred})\partial_i A_{2}(\vec{l_0}\times w_{1}+\vec{b_{1}})$



$\frac{\partial L_0}{w_{1_{i,j}}}=-2(y_0- y_{pred_0})\sum\limits_{m=1}^{1\to m}\partial_m A_{2,0}(\vec{l_0}\times w_{1}+\vec{b_{1}})\frac{\partial}{w_{1_{i,j}}}(\vec{l_0}\times w_{1_m}+\vec{b_{1_m}})$

$=-2(y_0-y_{pred_0})(\partial_j (A_{2,0}(\vec{l_{0_i}}))$

$\frac{\partial L}{w_{1_{i,j}}}=-2(\vec y_0-\vec y_{pred})(\partial_j (A_{2}(\vec{l_{0_i}}))$



$\frac{\partial L_0}{b_{0_i}}=-2(y_0-y_{pred_0})\sum\limits_{m=1}^{1\to m}\partial_m A_{2,0}(\vec{l_0}\times w_{1}+\vec{b_{1}})\frac{\partial}{b_{0_i}}(\vec{l_0}\times w_{1_m}+\vec{b_{1_m}})$

$=-2(y_0-y_{pred_0})\sum\limits_{m=1}^{1\to m}\partial_m A_{2,0}(\frac{\partial}{\partial b_{0_i}} l_0 w_1)_m$

$=-2(y_0-y_{pred_0})\sum\limits_{m=1}^{1\to m}\partial_m A_{2,0}(w_1\sum\limits_{n=1}^{1\to n}\frac{\partial}{\partial b_{0_i}} (x+b_{0_n}))_m$

$=-2(y_0-y_{pred_0})\sum\limits_{m=1}^{1\to m}\partial_m A_{2,0}(w_1\partial_iA_1)_m$

$=-2(y_0-y_{pred_0})(\partial_iA_1~w_1~\vec {\partial}A_{2,0})$

$\frac{\partial L}{b_{0_i}}=-2(\vec y_0-\vec y_{pred})(\partial_iA_1~w_1~\vec {\partial}A_{2,0})$



##### 函数的推导

$tanh'=\frac{1}{cosh^2x}$

$\partial_iA_1=$ $$\begin{pmatrix}
\frac{1}{cosh^2x_1}&...&0\\
...&\frac{1}{cosh^2x_i}&...\\
0&0&\frac{1}{cosh^2x_n}\\
\end{pmatrix}$$

**于其他数计算时矩阵恰好会乘起来**

$softmax_i(x)={e^{x_i}}/{\sum\limits_{m}e^{x_m}}$

$\partial /\partial x_i(softmax_j(x))=(\delta_{i,j}e^{x_i}(\sum\limits_{m}e^{m}))-e^{x_i}e^{x_j}/(\sum\limits_{m}e^{m})^2$
$$
\delta_{i,j}e^{x_i}=\left\{
\begin{array}{ll}
    e^{x_i}~(i=j)&\\
	0~~~~(i\neq j)&
\end{array}\right.\\
$$
$=\delta_{i,j}softmax_i(x)-softmax_i(x)softmax_j(x)$



##### 结论

$\frac{\partial L}{b_{0_i}}=-2(\vec y_0-\vec y_{pred})(\partial_iA_1~w_1~\vec {\partial}A_{2,0})$

$\frac{\partial L}{b_{1_i}}=-2(\vec y_0-\vec y_{pred})\partial_i A_{2}(\vec{l_0}\times w_{1}\times\vec{b_{1}})$

$\frac{\partial L}{w_{1_{i,j}}}=-2(\vec y_0-\vec y_{pred})(\partial_j (A_{2}(\vec{l_{0_i}}))$



