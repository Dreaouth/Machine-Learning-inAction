from numpy import *
import operator


# 建立测试数据
def createDataSet():
    group = array([[1, 101], [5, 89], [108, 5], [115, 8]])
    labels = ['爱情片', '爱情片', '动作片', '动作片']
    return group, labels


# group,labels=createDataSet()
# print(group)

# k-近邻算法
# 本函数有四个输入参数：inX是用于分类的输入向量
#                     dataSet是输入的训练样本输入集（矩阵）
#                     lables是标签向量
#                     k表示用于选择最近邻局的数目
# 其中标签向量的元素数目和矩阵dataSet的行数相同
def classify(inx, dataSet, labels, k):
    # 相当于取矩阵第二维的长度，例如createDataSet()就是2
    datasetSize = dataSet.shape[0]

    # 把inx纵向赋值datasetSize行（把inX二维数组化），然后与dataSet相减，
    # 相当于前一个二维数组的矩阵的每一个元素减后一个数组对应的元素值，这样就实现了矩阵的减法
    diffMat = tile(inx, (datasetSize, 1)) - dataSet
    # 对上一布的结果取平方
    sqDiffMat = diffMat ** 2
    # 取每行的和作为sqDistances
    sqDistances = sum(sqDiffMat, axis=1)
    # 取根号
    distances = sqDistances ** 0.5
    # 以上过程为计算样本输入集之间的欧氏距离，返回一个关于距离的集合

    # 对计算得得到的距离进行从小到大排序，返回distances中元素排序后的索引值
    sortedDistIndicies = argsort(distances)
    # classCount字典类
    classCount = {}
    for i in range(k):
        # 取出前k个元素的类别
        voteIlabel = labels[sortedDistIndicies[i]]
        # 计算类别的次数
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 按照classCount转化为数组的第二个属性进行（即字典的‘值’）排序，排序方式为降序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]


if __name__ == '__main__':
    group, labels = createDataSet()
    # 测试点
    test = [101, 20]
    # KNN分类
    test_class = classify(test, group, labels, 3)
    print(test_class)

# print(classify([0,0],group,labels,3))
