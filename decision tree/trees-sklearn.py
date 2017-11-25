from sklearn import tree
import pandas
import pydotplus
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.externals.six import StringIO

"""
说明：使用sk-learn算法预测隐形眼镜类型
    在使用fit函数之前，我们需要先对数据集进行编码
        LabelEncoder ：将字符串转换为增量值
        OneHotEncoder：使用One-of-K算法将字符串转换为整数
"""
if __name__ == '__main__':
    with open('lenses.txt','r') as fr:
        lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lenses_target = []              # 获取最终的类别
    for each in lenses:
        lenses_target.append(each[-1])

    lensesLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    lenses_list = []
    lenses_dict = {}
    for each_labels in lensesLabels:  # 遍历标签中的每个类别
        for each in lenses:
            lenses_list.append(each[lensesLabels.index(each_labels)])  # 分别提取该标签下的特征，建立字典
        lenses_dict[each_labels] = lenses_list
        lenses_list = []
    print(lenses_dict)
    lenses_pd = pandas.DataFrame(lenses_dict)   # 生成pandas数据，方便序列化工作
    print(lenses_pd)

    le = LabelEncoder()              # 创建LabelEncoder对象，用于序列化
    for col in lenses_pd.columns:    # 为每一列序列化
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    print(lenses_pd)

    clf = tree.DecisionTreeClassifier(max_depth=4)          # 创建DecisionTreeClassifier()类
    clf = clf.fit(lenses_pd.values.tolist(),lenses_target)  # 构建决策树
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data, feature_names=lenses_pd.keys(),
                          class_names=clf.classes_, filled=True, rounded=True,
                         special_characters=True)           # 绘制决策树
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("tree.pdf")
    print(clf.predict([[1, 1, 1, 0]]))