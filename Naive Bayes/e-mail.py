import re
import random
import numpy as np

# 将大字符串解析为字符串列表
def textParse(bigString):
    listOfTokens = re.split(r'\W*', bigString)                      # 提取非特殊字符的字符串
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]    # 返回大于等于两个字符的字符串

# 将切分的实验样本词条整理成不重复都的词条列表，也就是词汇表
def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)     # 取并集
    return list(vocabSet)

# 将词条向量化
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)    # 创建一个其中所含元素都为0的向量
    for word in inputSet:               # 遍历每个词条
        if word in vocabList:           # 如果词条存在于词汇表中，则置1
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary!" % word)
    return returnVec                    # 返回文档向量

# 根据vocabList词汇表，构建词袋模型
def bagOfWord2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:           # 如果词条存在于词汇表中，则计数加一
            returnVec[vocabList.index(word)] += 1
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
    # print('p0:', p0)
    # print('p1:', p1)
    if p1 > p0:
        return 1
    else:
        return 0

# 测试朴素贝叶斯分类器
def spanTest():
    docList = []; classList = []; fullText = []
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i, 'r').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)                 # 标记垃圾邮件
        wordList = textParse(open('email/ham/%d.txt' % i, 'r').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)                 # 标记非垃圾邮件
    vocabList = createVocabList(docList)    # 创建不重复的词汇表
    trainingSet = list(range(50)); testSet = []   # 训练集和测试集的列表
    for i in range(10):                     # 随机选择10个作为测试集
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])         # 删除测试集数据
    trainMat = []; trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))    # 将生成的词集模型添加到训练矩阵中
        trainClasses.append(classList[docIndex])                         # 将类别添加到训练集类别标签向量中
    p0V,p1V,pSpam = trainNB0(np.array(trainMat), np.array(trainClasses)) # 训练模型
    errorCount = 0      # 错误分类个数
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])        # 测试集的词集模型
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
            print("分类错误的测试集：",docList[docIndex])
    print('错误率：%.2f%%' % (float(errorCount) / len(testSet) * 100))

if __name__ == '__main__':
    # docList = []; classList = []
    # for i in range(1, 26):                                                  #遍历25个txt文件
    #     wordList = textParse(open('email/spam/%d.txt' % i, 'r').read())     #读取每个垃圾邮件，并字符串转换成字符串列表
    #     docList.append(wordList)
    #     classList.append(1)                                                 #标记垃圾邮件，1表示垃圾文件
    #     wordList = textParse(open('email/ham/%d.txt' % i, 'r').read())      #读取每个非垃圾邮件，并字符串转换成字符串列表
    #     docList.append(wordList)
    #     classList.append(0)                                                 #标记非垃圾邮件，1表示垃圾文件
    # vocabList = createVocabList(docList)                                    #创建词汇表，不重复
    # print(vocabList)
    spanTest()