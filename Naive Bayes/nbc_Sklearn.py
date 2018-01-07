import os
import jieba
import random
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

"""
函数说明:中文文本处理

Parameters:
    folder_path - 文本存放的路径
    test_size - 测试集占比，默认占所有数据集的百分之20
Returns:
    all_words_list - 按词频降序排序的训练集列表
    train_data_list - 训练集列表
    test_data_list - 测试集列表
    train_class_list - 训练集标签列表
    test_class_list - 测试集标签列表
"""
def TextProcessing(folder_path, test_size = 0.2):
    folder_list = os.listdir(folder_path)   # 查看folder_path下的文件
    data_list = []
    class_list = []
    for folder in folder_list:              # 遍历每个子文件夹
        new_folder_path = os.path.join(folder_path, folder)     # 根据子文件夹，生成新的路径
        files = os.listdir(new_folder_path)                     # 存放子文件夹下的txt文件的列表
        j = 1
        for file in files:                  # 遍历每个txt文件
            if j > 100:                     # 每类txt样本数最多输出100个
                break
            with open(os.path.join(new_folder_path, file), 'r', encoding='utf-8') as f:
                raw = f.read()
            word_cut = jieba.cut(raw,cut_all=False)     # 返回一个可迭代的generator
            word_list = list(word_cut)                  # 将其转化为list
            data_list.append(word_list)
            class_list.append(folder)
            j += 1

    data_class_list = list(zip(data_list, class_list))  # zip压缩合并，将数据与标签对应压缩
    random.shuffle(data_class_list)                     # 将data_class_list乱序
    index = int(len(data_class_list) * test_size) + 1   # 训练集和测试结切分的索引值
    train_list = data_class_list[index:]
    test_list = data_class_list[:index]
    train_data_list, train_class_list = zip(*train_list)
    test_data_list, test_class_list = zip(*test_list)
    all_words_dict = {}
    for word_list in train_data_list:
        for word in word_list:
            if word in all_words_dict.keys():
                all_words_dict[word] += 1
            else:
                all_words_dict[word] = 1
    # 提取键的值倒序排序
    all_words_tuple_list = sorted(all_words_dict.items(), key= lambda f:f[1], reverse=True)
    all_words_list, all_words_nums = zip(*all_words_tuple_list)     # 解压缩
    all_words_list = list(all_words_list)
    return all_words_list, train_data_list, test_data_list, train_class_list, test_class_list

# 读取文件里的内容，并去重
def MakeWordsSet(words_file):
    words_set = set()
    with open(words_file, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            word = line.strip()         # 去回车
            if len(word) > 0:           # 有文本，则添加到words_set中
                words_set.add(word)
    return words_set

# 文本特征提取
def words_dict(all_words_list, deleteN, stopwords_set = set()):
    feature_words = []      # 特征列表
    n = 1
    for t in range(deleteN, len(all_words_list), 1):        # 从过滤的字符位置开始，到词条向量总长度，开始遍历
        if n > 1000:                                        # 选取1000个
            break
        # 如果这个词不是数字，并且不是指定的结束语，并且单词当都大于1小于5，那么这个词可以作为特征词
        if not all_words_list[t].isdigit() and all_words_list[t] not in stopwords_set and 1 < len(all_words_list[t]) < 5:
            feature_words.append(all_words_list[t])
        n += 1
    return feature_words

# 根据feature_words将文本向量化
def TextFeatures(train_data_list, test_data_list, feature_words):
    def text_features(text, feature_words):         # 如果出现在特征集中，则置1
        text_words = set(text)
        features = [1 if word in text_words else 0 for word in feature_words]
        return features
    train_feature_list = [text_features(text, feature_words) for text in train_data_list]
    test_feature_list = [text_features(text, feature_words) for text in test_data_list]
    return train_feature_list, test_feature_list

"""
函数说明:新闻分类器
Parameters:
    train_feature_list - 训练集向量化的特征文本
    test_feature_list - 测试集向量化的特征文本
    train_class_list - 训练集分类标签
    test_class_list - 测试集分类标签
Returns:
    test_accuracy - 分类器精度
"""
def TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list):
    classifier = MultinomialNB().fit(train_feature_list,train_class_list)
    test_accuracy = classifier.score(test_feature_list,test_class_list)
    return test_accuracy

# 画出过滤点数与分类精确度关系的图像
def draw_deleteN():
    folder_path = './SogouC/Sample'
    # TextProcessing(folder_path)
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path, test_size=0.2)
    stopwords_file = './stopwords_cn.txt'           # 生成stopwords_set
    stopwords_set = MakeWordsSet(stopwords_file)
    test_accuracy_list = []
    deleteNs = range(0,1000,20)
    for deleteN in deleteNs:
        feature_words = words_dict(all_words_list, deleteN, stopwords_set)
        train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words)
        test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
        test_accuracy_list.append(test_accuracy)
    plt.figure()
    plt.plot(deleteNs, test_accuracy_list)
    plt.title('Relationship of deleteNs and test_accuracy')
    plt.xlabel('deleteNs')
    plt.ylabel('test_accuracy')
    plt.show()

if __name__ == '__main__':
    # TextProcessing(folder_path)
    draw_deleteN()
    folder_path = './SogouC/Sample'
    all_words_list, train_data_list, test_data_list, train_class_list, test_class_list = TextProcessing(folder_path, test_size=0.2)
    stopwords_file = './stopwords_cn.txt'       # 生成stopwords_set
    stopwords_set = MakeWordsSet(stopwords_file)

    # 这里我直接采用了过滤450个词，也可以采取循环遍历然后取平均值的方法
    test_accuracy_list = []
    feature_words = words_dict(all_words_list, 450, stopwords_set)
    train_feature_list, test_feature_list = TextFeatures(train_data_list, test_data_list, feature_words)
    test_accuracy = TextClassifier(train_feature_list, test_feature_list, train_class_list, test_class_list)
    test_accuracy_list.append(test_accuracy)
    print(train_class_list)
    print(train_feature_list)
    print(test_accuracy_list)
    ave = lambda c: sum(c) / len(c)
    print(ave(test_accuracy_list))
