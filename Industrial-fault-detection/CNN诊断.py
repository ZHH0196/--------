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
    return [X99_normal,X108_InnerRace_007,X121_Ball_007,X133_Outer_007, X172_InnerRace_014,X188_Ball_014,X200_Outer_014,X212_InnerRace_021,X225_Ball_021,X237_Outer_021]

# 获取数据
data = ImportData()

# 假设 X_1D_test 和 y_1D_test 是从 data 中获取的
# 这里需要根据你的具体数据处理逻辑来定义 X_1D_test 和 y_1D_test
# 例如：
X_1D_test = np.array(data)  # 这只是一个示例，你需要根据实际情况进行处理
y_1D_test = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])  # 这只是一个示例标签



# 使用加载的模型进行评估
loaded_model_test_loss, loaded_model_test_accuracy = loaded_model.evaluate(X_1D_test, y_1D_test)
loaded_model_test_accuracy *= 100
print('加载模型的测试准确率 =', loaded_model_test_accuracy)

# 使用加载的模型进行预测
y_pred = np.argmax(loaded_model.predict(X_1D_test), axis=1)

# 评估模型在测试集上的准确性
CNN_1D_test_loss, CNN_1D_test_accuracy = Classification_1D.model.evaluate(X_1D_test, y_1D_test)
CNN_1D_test_accuracy *= 100
print('CNN 1D test accuracy =', CNN_1D_test_accuracy)

# 定义混淆矩阵
def ConfusionMatrix(Model, X, y):
  y_pred = np.argmax(Model.model.predict(X), axis=1)
  ConfusionMat = confusion_matrix(np.argmax(y, axis=1), y_pred)
  return ConfusionMat

# 绘制1D-CNN的结果
plt.figure(1)
plt.title('Confusion Matrix - CNN 1D Train') 
sns.heatmap(ConfusionMatrix(Classification_1D, X_1D_train, y_1D_train) , annot=True, fmt='d',annot_kws={"fontsize":8},cmap="YlGnBu")
plt.show()

plt.figure(2)
plt.title('Confusion Matrix - CNN 1D Test') 
sns.heatmap(ConfusionMatrix(Classification_1D, X_1D_test, y_1D_test) , annot=True, fmt='d',annot_kws={"fontsize":8},cmap="YlGnBu")
plt.show()

plt.figure(3)
plt.title('Train - Accuracy - CNN 1D')
plt.bar(np.arange(1,kSplits+1),[i*100 for i in accuracy_1D])
plt.ylabel('accuracy')
plt.xlabel('folds')
plt.ylim([70,100])
plt.show()

plt.figure(4)
plt.title('Train vs Test Accuracy - CNN 1D')
plt.bar([1,2],[CNN_1D_train_accuracy,CNN_1D_test_accuracy])
plt.ylabel('accuracy')
plt.xlabel('folds')
plt.xticks([1,2],['Train', 'Test'])
plt.ylim([70,100])
plt.show()