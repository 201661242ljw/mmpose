import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
# 模拟一个混淆矩阵数据（这里是一个示例数据）
confusion_matrix_data = np.array([[151, 13, 0],
                                  [4, 52, 0],
                                  [0, 0, 6]])

# 标签（类别）名称
labels = ['0', '1', '2']

# 创建图像和子图
plt.figure(figsize=(3, 3))
plt.imshow(confusion_matrix_data, interpolation='nearest',cmap=plt.get_cmap('Blues'))

# 添加颜色标尺
plt.colorbar()

# 设置坐标轴标签
plt.xticks(np.arange(len(labels)), labels)
plt.yticks(np.arange(len(labels)), labels)
plt.xlabel('Predicted Risk Level')
plt.ylabel('True Risk Level')

# 在方块中显示混淆矩阵的数值
for i in range(len(labels)):
    for j in range(len(labels)):
        plt.text(j, i, str(confusion_matrix_data[i, j]), horizontalalignment='center', color='red')
# plt.title('Confusion Matrix')
plt.tight_layout()

plt.savefig(r"C:\Users\concrete\Desktop\_0_毕业论文\图\cf.png", dpi=330)