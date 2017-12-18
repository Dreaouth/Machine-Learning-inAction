import numpy as np
from functools import reduce

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],          # 切分的词条
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]       # 1代表侮辱性文字，0代表正常言论
    return postingList, classVec

# 去除重复词汇
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)     # 进行或操作，相当于删除重复词汇
    return list(vocabSet)

# 定义包含整个词汇表的向量，判断切分的词条的向量的位置，即如果该单词在词汇表里就为1，不在就为0
# 输入参数：vocabList：词汇表，inputSet：整个文档
# 输出参数：文档向量
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0]*len(vocabList)          # 创建一个和词汇表等长的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word：%s is not in my vocavulary!" % word)
    return returnVec

"""
朴素贝叶斯分类器训练函数
输入：文档矩阵trainMatrix，文档类别标签构成的向量trainCategory
改进：（1）拉普拉斯平滑：将词的出现数初始化为1，并将分母初始化为2
      （2）对数似然：对乘积结果去自然对数
"""
def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)         # 计算训练的文档数量
    numWords = len(trainMatrix[0])          # 计算每篇文档的单词数
    pAbusive = sum(trainCategory)/float(numTrainDocs)   # 文档属于侮辱性的概率
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0            # 贝叶斯公式的分母
    for i in range(numTrainDocs):           # 遍历文档，判断属于侮辱性和非侮辱性词汇
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

"""
说明：朴素贝叶斯分类器分类函数
Parameters:
    vec2classify - 待分类的词条数组
    p0vec - 侮辱类的条件概率数组
    p1vec - 非侮辱类的条件概率数组
    pClass1 - 文档属于侮辱类的概率
"""
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # p1 = reduce(lambda x, y: x*y, vec2Classify * p1Vec) * pClass1      # 对应元素相乘
    # p0 = reduce(lambda x, y: x*y, vec2Classify * p0Vec) * (1.0 - pClass1)
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)    # 对应元素相乘。logA * B = logA + logB，所以这里加上log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    print('p0:', p0)
    print('p1:', p1)
    if p1 > p0:
        return 1
    else:
        return 0

# 测试朴素贝叶斯分类器
def testingNB():
    listOPosts, listClasses = loadDataSet()         # 加载数据
    myVocabList = createVocabList(listOPosts)       # 创建词汇表
    trainMat = []
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))     # 将实验文档向量化存入trainMat
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listClasses))  # 训练朴素贝叶斯分类器
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))      # 测试文档向量化
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry,'属于侮辱类')
    else:
        print(testEntry, '属于非侮辱类')

    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    if classifyNB(thisDoc, p0V, p1V, pAb):
        print(testEntry,'属于侮辱类')
    else:
        print(testEntry, '属于非侮辱类')

if __name__ == '__main__':
    # postingList, classVec = loadDataSet()           # 加载数据
    # myVocabList = createVocabList(postingList)      # 去除掉list中的重复词汇
    # print('myVocabList:\n',myVocabList)
    # trainMat = []
    # for postinDoc in postingList:
    #     trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # print("trainMat\n", np.mat(trainMat))
    # p0V, p1V, pAb = trainNB0(trainMat, classVec)
    # print('p0V:\n', p0V)
    # print('p1V:\n', p1V)
    # print('classVec:\n', classVec)
    # print('pAb:\n', pAb)
    testingNB()
