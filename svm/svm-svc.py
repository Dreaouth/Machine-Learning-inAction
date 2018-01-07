import numpy as np
import operator
from os import listdir
from sklearn.svm import SVC

# 将32*32 的二进制图像转换为1*1024向量
def img2vector(filename):
    returnVec = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVec[0, 32*i + j] = int(lineStr[j])
    return returnVec

# 手写数字分类测试
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('../kNN/trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumber)
        trainingMat[i,:] = img2vector('../kNN/trainingDigits/%s' % (fileNameStr))
    clf = SVC(C=200, kernel='rbf')
    clf.fit(trainingMat, hwLabels)
    testFileList = listdir('../kNN/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)       # 测试数据的数量
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        vectorunderTest = img2vector('../kNN/testDigits/%s' % (fileNameStr))
        classifierResult = clf.predict(vectorunderTest)
        print('分类返回结果为%d\t真实结果为%d' % (classifierResult, classNumber))
        if(classifierResult != classNumber):
            errorCount += 1.0
    print('总共错了%d个数据\n错误率为%f%%' % (errorCount, errorCount/mTest * 100))

if __name__ == '__main__':
    handwritingClassTest()

