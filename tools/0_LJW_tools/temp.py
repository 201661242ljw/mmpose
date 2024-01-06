import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def see__or_oks():
    # 计算Z值
    S = 256
    sigma = 15
    w1 = w2 = h1 = h2 = sigma * 3
    x1 = 1000
    y1 = 1000

    thresh = 0.5
    # 定义横纵坐标范围和步长
    x = np.arange(2 * x1)
    y = np.arange(2 * y1)

    # 生成网格
    X, Y = np.meshgrid(x, y)

    scale = 100

    X = X[x1 - scale:x1 + scale, y1 - scale:y1 + scale]
    Y = Y[x1 - scale:x1 + scale, y1 - scale:y1 + scale]

    left = np.maximum(x1, X)
    top = np.maximum(y1, Y)
    right = np.minimum(x1 + w1, X + w2)
    bottom = np.minimum(y1 + h1, Y + h2)

    c1 = right > left
    c2 = bottom > top

    c = c1.astype(float) * c2.astype(float)
    intersection = (right - left) * (bottom - top)
    union = w1 * h1 + w2 * h2 - intersection

    c = c * intersection / union

    dd = (x1 - X) * (x1 - X) + (y1 - Y) * (y1 - Y)
    oks = np.exp(-(dd / 1 / sigma / sigma))

    c = ((c > thresh).astype(int) * 255).astype(np.uint8)

    oks = ((oks > thresh).astype(int) * 255).astype(np.uint8)

    print(np.sum(c))
    print(np.sum(oks))

    img = cv2.merge([c * 0, c, oks])

    cv2.imwrite(r"oks_iou.jpg", img)

    # 创建图形对象
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    scale = 10
    # 绘制3D图
    # ax.plot_surface(X[x1 - scale:x1 + scale, y1 - scale:y1 + scale], Y[x1 - scale:x1 + scale, y1 - scale:y1 + scale],
    #                 c[x1 - scale:x1 + scale, y1 - scale:y1 + scale], cmap='viridis')  # 绘制c的三维图
    # ax.plot_surface(X[x1 - scale:x1 + scale, y1 - scale:y1 + scale], Y[x1 - scale:x1 + scale, y1 - scale:y1 + scale],
    #                 oks[x1 - scale:x1 + scale, y1 - scale:y1 + scale], cmap='plasma')  # 绘制oks的三维图
    #
    # # 设置坐标轴标签
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    #
    # # 显示图形
    # plt.show()


if __name__ == '__main__':
    path = os.getcwd()
    print(path)