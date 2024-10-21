import scipy.io 
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt
import seaborn as sns 
import pandas as pd
from tensorflow.keras.models import load_model

# 加载模型
loaded_model = load_model('cnn_1d_model.h5')
print("模型已加载")

# 导入数据
def ImportData():
    X99_normal = scipy.io.loadmat('99.mat')['X099_DE_time']              
    X108_InnerRace_007 = scipy.io.loadmat('108.mat')['X108_DE_time']
    X121_Ball_007 = scipy.io.loadmat('121.mat')['X121_DE_time']
    X133_Outer_007 = scipy.io.loadmat('133.mat')['X133_DE_time']
    X172_InnerRace_014 = scipy.io.loadmat('172.mat')['X172_DE_time']
    X188_Ball_014 = scipy.io.loadmat('188.mat')['X188_DE_time']
    X200_Outer_014 = scipy.io.loadmat('200.mat')['X200_DE_time']
    X212_InnerRace_021 = scipy.io.loadmat('212.mat')['X212_DE_time']
    X225_Ball_021 = scipy.io.loadmat('225.mat')['X225_DE_time']
    X237_Outer_021 = scipy.io.loadmat('237.mat')['X237_DE_time']
    return [X99_normal, X108_InnerRace_007, X121_Ball_007, X133_Outer_007, X172_InnerRace_014, X188_Ball_014, X200_Outer_014, X212_InnerRace_021, X225_Ball_021, X237_Outer_021]

# 获取数据
data = ImportData()

# 数据前处理
def DataPreparation(Data, interval_length, samples_per_block):
    for count, i in enumerate(Data):
        SplitData = Sampling(i, interval_length, samples_per_block)
        y = np.zeros([len(SplitData), 10])
        y[:, count] = 1
        y1 = np.zeros([len(SplitData), 1])
        y1[:, 0] = count
        # 堆叠并标记数据
        if count == 0:
            X = SplitData
            LabelPositional = y
            Label = y1
        else:
            X = np.append(X, SplitData, axis=0)
            LabelPositional = np.append(LabelPositional, y, axis=0)
            Label = np.append(Label, y1, axis=0)
    return X, LabelPositional, Label

# 采样
def Sampling(Data, interval_length, samples_per_block):
    No_of_blocks = (round(len(Data) / interval_length) - round(samples_per_block / interval_length) - 1)
    SplitData = np.zeros([No_of_blocks, samples_per_block])
    for i in range(No_of_blocks):
        SplitData[i, :] = (Data[i * interval_length:(i * interval_length) + samples_per_block]).T
    return SplitData

interval_length = 200  # 信号间隔长度
samples_per_block = 1681  # 每块样本点数

# 数据前处理
X, Y_CNN, Y = DataPreparation(data, interval_length, samples_per_block) 

# 重塑数据
Input_1D = X.reshape([-1, 1681, 1])

# 数据集划分
X_1D_train, X_1D_test, y_1D_train, y_1D_test = train_test_split(Input_1D, Y_CNN, train_size=0.75, test_size=0.25, random_state=101)

# 使用加载的模型进行评估
loaded_model_test_loss, loaded_model_test_accuracy = loaded_model.evaluate(X_1D_test, y_1D_test)
loaded_model_test_accuracy *= 100
print('加载模型的测试准确率 =', loaded_model_test_accuracy)

# 使用加载的模型进行预测
y_pred = np.argmax(loaded_model.predict(X_1D_test), axis=1)

# 定义混淆矩阵
def ConfusionMatrix(Model, X, y):
    y_pred = np.argmax(Model.predict(X), axis=1)
    ConfusionMat = confusion_matrix(np.argmax(y, axis=1), y_pred)
    return ConfusionMat

# 绘制1D-CNN的结果
plt.figure(1)
plt.title('Confusion Matrix - CNN 1D Test') 
sns.heatmap(ConfusionMatrix(loaded_model, X_1D_test, y_1D_test), annot=True, fmt='d', annot_kws={"fontsize": 8}, cmap="YlGnBu")
plt.show()