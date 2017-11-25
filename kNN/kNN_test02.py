from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import operator
import numpy as np
from numpy import *

def classify(inx,dataSet,labels,k):
    # 相当于取矩阵第二维的长度，例如createDataSet()就是2
    datasetSize = dataSet.shape[0]

    # 把inx纵向赋值datasetSize行（把inX二维数组化），然后与dataSet相减，
    # 相当于前一个二维数组的矩阵的每一个元素减后一个数组对应的元素值，这样就实现了矩阵的减法
    diffMat = tile(inx,(datasetSize,1))-dataSet
    # 对上一布的结果取平方
    sqDiffMat = diffMat**2
    # 取每行的和作为sqDistances
    sqDistances = sum(sqDiffMat,axis=1)
    # 取根号
    distances = sqDistances**0.5
    # 以上过程为计算样本输入集之间的欧氏距离，返回一个关于距离的集合

    # 对计算得得到的距离进行从小到大排序，返回distances中元素排序后的索引值
    sortedDistIndicies = argsort(distances)
    # classCount字典类
    classCount = {}
    for i in range(k):
        # 取出前k个元素的类别
        voteIlabel = labels[sortedDistIndicies[i]]
        # 计算类别的次数
        classCount[voteIlabel] = classCount.get(voteIlabel,0)+1
    # 按照classCount转化为数组的第二个属性进行（即字典的‘值’）排序，排序方式为降序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1),reverse=True)
    # 返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename,'r')
    # 得到文件的所有内容
    array_Lines=fr.readlines()
    # 得到文件行数
    number_Lines=len(array_Lines)
    # 返回的numpy矩阵，解析完成的数据：number_Lines行，3列
    returnMat=np.zeros((number_Lines,3))
    # 返回的分类标签向量
    classLabelVector=[]
    # 行的索引值
    index = 0
    for line in array_Lines:
        # s.strip(rm)，当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        line=line.strip()
        # 使用s.split(str="",num=string,cout(str))将字符串根据'\t'分隔符进行切片。
        listFromLine=line.split('\t')
        # 将数据前三列举出来，存放到returnMat的NumPy矩阵中，也就是特征矩阵
        returnMat[index,:] = listFromLine[0:3]
        # 根据文本中标记的喜欢的程度进行分类，1代表不喜欢，2代表魅力一般，3代表极具魅力
        if listFromLine[-1] == "didntLike":
            classLabelVector.append(1)
        elif listFromLine[-1] == "smallDoses":
            classLabelVector.append(2)
        elif listFromLine[-1] == "largeDoses":
            classLabelVector.append(3)
        index += 1
    return  returnMat,classLabelVector

def showdatas(datingDataMat, datingLabels):
    #设置汉字格式
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    #将fig画布分隔成1行1列,不共享x轴和y轴,fig画布的大小为(13,8)
    #当nrow=2,nclos=2时,代表fig画布被分为四个区域,axs[0][0]表示第一行第一个区域
    fig, axs = plt.subplots(nrows=2, ncols=2,sharex=False, sharey=False, figsize=(13,8))

    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')
    #画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第二列(玩游戏)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][0].scatter(x=datingDataMat[:,0], y=datingDataMat[:,1], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs0_title_text = axs[0][0].set_title(u'每年获得的飞行常客里程数与玩视频游戏所消耗时间占比',FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'玩视频游戏所消耗时间占',FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    #画出散点图,以datingDataMat矩阵的第一(飞行常客例程)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[0][1].scatter(x=datingDataMat[:,0], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs1_title_text = axs[0][1].set_title(u'每年获得的飞行常客里程数与每周消费的冰激淋公升数',FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'每年获得的飞行常客里程数',FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'每周消费的冰激淋公升数',FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    #画出散点图,以datingDataMat矩阵的第二(玩游戏)、第三列(冰激凌)数据画散点数据,散点大小为15,透明度为0.5
    axs[1][0].scatter(x=datingDataMat[:,1], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
    #设置标题,x轴label,y轴label
    axs2_title_text = axs[1][0].set_title(u'玩视频游戏所消耗时间占比与每周消费的冰激淋公升数',FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'玩视频游戏所消耗时间占比',FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'每周消费的冰激淋公升数',FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')
    #设置图例
    didntLike = mlines.Line2D([], [], color='black', marker='.',
                      markersize=6, label='didntLike')
    smallDoses = mlines.Line2D([], [], color='orange', marker='.',
                      markersize=6, label='smallDoses')
    largeDoses = mlines.Line2D([], [], color='red', marker='.',
                      markersize=6, label='largeDoses')
    #添加图例
    axs[0][0].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[0][1].legend(handles=[didntLike,smallDoses,largeDoses])
    axs[1][0].legend(handles=[didntLike,smallDoses,largeDoses])
    #显示图片
    plt.show()

def autoNorm(dataSet):
    # 获得数据的最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    # 最大值和最小值的范围
    ranges = maxVals - minVals
    # shape(dataSet)返回dataSet的矩阵行列数
    normDataSet = np.zeros(np.shape(dataSet))
    # 返回dataSet的行数
    m = dataSet.shape[0]
    # 原始值-最小值
    normDataSet = dataSet - np.tile(minVals,(m,1))
    # 除以最大值和最小值的差，得到归一化数据
    normDataSet = normDataSet / np.tile(ranges,(m,1))
    # 返回归一化数据结果，数据范围，最小值
    return normDataSet,ranges,minVals

def classifyPerson():
    #输出结果
    resultList = ['讨厌','有些喜欢','非常喜欢']
    #三位特征用户输入
    percentTats = float(input('玩游戏所占时间百分比：'))
    ffMiles = float(input("每年获得的飞行常客里程数:"))
    iceCream = float(input("每周消费的冰激淋公升数:"))
    filename = 'date.txt'
    # 打开并处理数据
    datingDataMat,datingLabels = file2matrix(filename)
    # 训练集归一化
    normMat,ranges,minvals = autoNorm(datingDataMat)
    # 生辰numPy数组，测试集
    inArr = np.array([percentTats,ffMiles,iceCream])
    # 测试集归一化
    norminArr = (inArr - minvals)/ranges
    # 返回分类结果
    classifierResult = classify(norminArr,normMat,datingLabels,3)
    print(classifierResult)
    print("你可能%s这个人" % (resultList[classifierResult]))
    # showdatas(datingDataMat, datingLabels)

def datingClassTest():
    #打开的文件名
    filename = "date.txt"
    #将返回的特征矩阵和分类向量分别存储到datingDataMat和datingLabels中
    datingDataMat, datingLabels = file2matrix(filename)
    #取所有数据的百分之十
    hoRatio = 0.10
    #数据归一化,返回归一化后的矩阵,数据范围,数据最小值
    normMat, ranges, minVals = autoNorm(datingDataMat)
    #获得normMat的行数
    m = normMat.shape[0]
    #百分之十的测试数据的个数
    numTestVecs = int(m * hoRatio)
    #分类错误计数
    errorCount = 0.0

    for i in range(numTestVecs):
        #前numTestVecs个数据作为测试集,后m-numTestVecs个数据作为训练集
        classifierResult = classify(normMat[i,:], normMat[numTestVecs:m,:],
            datingLabels[numTestVecs:m], 4)
        print("分类结果:%d\t真实类别:%d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("错误率:%f%%" %(errorCount/float(numTestVecs)*100))


if __name__ == '__main__':
    # filename="date.txt"
    # datingDataMat,datingLabels=file2matrix(filename)
    # normDataSet,ranges,minvals = autoNorm(datingDataMat)
    # print(datingDataMat)
    # print(datingDataMat.shape)
    # print(datingLabels)
    # print(normDataSet)
    # print(ranges)
    # print(minvals)
    # showdatas(datingDataMat, datingLabels)
    classifyPerson()
    # datingClassTest()