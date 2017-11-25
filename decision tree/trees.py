from math import log
import operator
import pickle


def createDataSet():
    dataSet = [[0, 0, 0, 0, 'no'],  # 数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作', '有自己的房子', '信贷情况']  # 特征标签
    return dataSet, labels


# 计算信息熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)  # 记录数据集中实例的总数
    labelCounts = {}  # 保存每个标签出现次数的字典
    for featVec in dataSet:
        currentLabel = featVec[-1]  # 保存数据最后一列所表示表示的类别（标签的信息）
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0  # 如果标签中没有放入统计次数的字典，就添加进去
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0  # 信息熵
    for key in labelCounts:  # 计算信息熵
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


# 按照给定的特征划分数据集
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:  # 遍历数据集
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]  # 去掉axis特征
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)  # 将符合条件的添加到返回的数据集
    return retDataSet


# 选择最优特征
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1  # 特征数量
    baseEntropy = calcShannonEnt(dataSet)  # 计算数据集的特征熵
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # 获取dataSet的第i个所有特征
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)                  # 创建set集合{}，元素不可重复
        newEntropy = 0.0                            # 经验条件熵
        for value in uniqueVals:                    # 通过该特征划分的三种属性，计算信息增益(相加再减)
            subDataSet = splitDataSet(dataSet, i, value)     # subDataSet划分后的子集
            prob = len(subDataSet) / float(len(dataSet))     # 计算子集的概率
            newEntropy += prob * calcShannonEnt(subDataSet)  # 根据公式计算经验条件熵
        infoGain = baseEntropy - newEntropy         # 计算信息增益
        print("第%d个特征的增益为%.3f" % (i, infoGain))
        if (infoGain > bestInfoGain):               # 找出最大的信息增益及索引
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature                              # 返回最大的信息增益的特征索引


# 统计classList中出现次数最多的元素(类标签)
def majorityCnt(classList):
    classCount = {}
    for vote in classList:  # 统计每个元素出现的次数
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 根据字典的值降序排列
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


"""
函数说明：递归构建决策树
Parameters:
    dataSet - 训练数据集
    labels  - 分类训练属性
    featLabels - 存储选择的最优特征标签
"""


def createTree(dataSet, labels, featLabels):
    classList = [example[-1] for example in dataSet]        # 取分类标签(最后一个属性：是否放贷)
    if classList.count(classList[0]) == len(classList):     # 类别完全相同则停止继续划分
        return classList[0]
    if len(dataSet[0]) == 1:                                # 遍历完所有特征,返回出现次数最多的类标签
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)            # 根据信息增益，算出最优特征索引
    bestFeatLabel = labels[bestFeat]                        # 取出最优标签索引
    featLabels.append(bestFeatLabel)
    myTree = {bestFeatLabel: {}}                            # 建立以该特征为结点的枝干（根）
    del (labels[bestFeat])                                  # 删除该标签
    featValues = [example[bestFeat] for example in dataSet]
    uniquevalues = set(featValues)                          # 取出该标签下的特征数量（去重复）
    for value in uniquevalues:                              # 遍历所有特征进行递归，创建决策树
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), labels, featLabels)
    return myTree


"""
说明：根据给出的决策树进行分类
Parameters：
    inputTree:已经生成的决策树
    featLabels:存储选择的最优特征标签（决策树中存在的标签）
    testVec:测试数据
Returns:
    classLabels:分类结果
"""
def classify(inputTree,featLabels,testVec):
    firstStr = next(iter(inputTree))          # 取字典中的键为先判断的条件
    secondDict = inputTree[firstStr]          # 取以该决策树的键值为字典
    featIndex = featLabels.index(firstStr)    # 取该判断条件在最优特征标签的位置
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':  # 当该分支下还有分支时，继续向下递归，否则就取该值
                classLabel = classify(secondDict[key],featLabels,testVec)
            else:
                classLabel = secondDict[key]
    return classLabel

# 存储决策树
def storeTree(inputTree,filename):
    with open(filename,'wb') as fw:
        pickle.dump(inputTree,fw)

# 读取决策树
def grabTree(filename):
    fr = open(filename,'rb')
    return pickle.load(fr)

if __name__ == '__main__':
    dataSet, features = createDataSet()
    # print("最优特征索引值：" + str(chooseBestFeatureToSplit(dataSet)))
    featLabels = []
    # myTree = createTree(dataSet, features, featLabels)
    # storeTree(myTree,'classifierStorage.txt')
    myTree = grabTree('classifierStorage.txt')
    print(myTree)
    # print(myTree)
    # testVec = [0,1]                                        #测试数据
    # result = classify(myTree, featLabels, testVec)
    # if result == 'yes':
    #     print('放贷')
    # if result == 'no':
    #     print('不放贷')