import random
import numpy as np

def sigmoid(inX):
    return 1.0 / (1+np.exp(-inX))

# 梯度上升算法
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()     # 转换成numpy的mat并进行转置
    m, n = np.shape(dataMatrix)                    # 返回dataMatrix矩阵的大小，m是行数，n是列数
    alpha = 0.01                                   # 移动步长，也就是学习速率
    maxCycles = 500                                # 最大迭代次数
    weights = np.ones((n,1))
    for k in range(maxCycles):                     # 梯度上升矢量化公式
        h = sigmoid(dataMatrix*weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
    return weights.getA()                      # 将矩阵转化为数组，并返回权重数组

# 随机梯度上升算法
def stocGradAscentl(dataMatrix, classLabels, numIter=150):
    m, n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0 + j + i) + 0.01          # 降低alpha的大小，每次减小1/(j+1)
            randIndex = int(random.uniform(0,len(dataIndex)))   # 随机选取样本
            h = sigmoid(sum(dataMatrix[randIndex] * weights))   # 选择随机选取的一个样本，记作h
            error = classLabels[randIndex] - h                  # 计算误差
            weights = weights + alpha * error * dataMatrix[randIndex]  # 更新回归系数
            del(dataIndex[randIndex])                           # 删除已经使用的样本
    return weights

"""
说明：分类函数
基本思想：把数据集的每个特征向量诚意最优化方法得来的最优回归系数，
然后在将该乘积结果求和，最后输入到sigmoid函数中即可
"""
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else: return 0.0

# 分类器做测试
def colicTest():
    frTrain = open('horseColicTraining.txt')        # 打开训练集数据
    frTest = open('horseColicTest.txt')             # 打开测试集数据
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():                # 训练数据
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine)-1):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)                 # 相当于将数据的内容按每行一个list(lineArr)的方式存放到一个总的list(trainSet)中
        trainingLabels.append(float(currLine[-1]))
    trainWeights = stocGradAscentl(np.array(trainingSet), trainingLabels, 500)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():                 # 测试数据
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(len(currLine) - 1):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr),trainWeights)) != int(currLine[-1]):   # 见classifyVector函数说明
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec) * 100
    print("测试集错误率为: %.2f%%" % errorRate)

if __name__ == '__main__':
    colicTest()
