#-*- coding:utf-8 _*-  
'''


'''

# import os
# 通过分析某网站上的纪录片播放量，来分析什么类型的纪录片受观众是欢迎。本案例共获取11186条数据，数据集名称为：纪录片播放量，其中各维度变量如下：
# ①目标变量Y ------- 播放量：连续变量，取值范围[10,205000]
# ②自变量X---------
# 片名：文本数据，介绍纪录片具体情况；
# 上传时间：从2016-12-12 16:40到2017-01-05 20:19；
# 弹幕数：连续数据，取值范围[0，16000]；
# 硬币数：连续数据，取值范围[0，2604]，硬币数为观众给作者的打赏；
# 收藏数：连续数据，取值范围[0，17000]；
# 上传者投稿数：连续数据，取值范围[1，6994]；
# 上传者粉丝数：连续数据，取值范围[0，1534689]；
# 评论数：连续数据，取值范围[0，1505]；
# 分享数：连续数据，取值范围[0，2189]；
# 简介：文本数据，视频内容介绍。
# 题目要求：（100分）
# 建立回归模型，解释影响纪录片播放量因素，试图找出哪些因素影响着纪录片受欢迎程度，为观众观看提供参考依据，具体要求如下：
# 1.	加载数据集（10分）
# 2.	删除列，分别为标签、上传日期。（10分）
# 3.	使用自定义函数f，若输入值为’1.2万‘，返回值为12000。（10分）
# 4.	使用自定义函数f，对播放数、弹幕数、收藏数特征进行处理。（10分）
# 5.	对播放数进行频数统计。（10分）
# 6.	统计播放数特征缺失值个数。（10分）
# 7.	切片播放数作为Y标签，其余作为特征X。（10分）
# 8.	对Y进行对数转换。（10分）
# 9.	对X进行数据标准化转换。（10分）
# 10.	划分数据集，建立线性回归方程。（5分）
# 11.	打印评估指标。（5分）

# os.chdir(r'')
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy import stats
from sklearn.linear_model import LinearRegression,Ridge
from statsmodels.formula.api import ols      #
#   statsmodels中的线性模型大致分为两种：
# 基于数组的（array-based），和基于公式的（formula-based）。调用的模块为：   # 线性回归
from sklearn import metrics
import matplotlib.pyplot as plt
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor

# 1.	加载数据集（10分）
data = pd.read_excel('纪录片播放量.xls',index_col='片名')
# print(data.head())
# print(data.info())

# 2.	删除列，分别为标签、上传日期。（10分）
del data['标签']
del data['上传日期']

# 3.	使用自定义函数f，若输入值为’1.2万‘，返回值为12000。（10分）
def f(s):
    str = '万'
    if str in s:
        return float(s[0:-1])*10000 #切片 0：-1 包前不包后
    else:
        return float(s)

# 4.	使用自定义函数f，对播放数、弹幕数、收藏数特征进行处理。（10分）
col = ['播放数','弹幕数','收藏数']
for i in col:
    data[i] = data[i].map(f)

# 5.	对播放数进行频数统计。（10分）
print(data.info())
print(data['播放数'].value_counts())   # 频数统计

# 6.	统计播放数特征缺失值个数。（10分）
print(sum(data['播放数'].isnull()))    # 缺失值判断

# 7.	切片播放数作为Y标签，其余作为特征X。（10分）
data['简介'] = data['简介'].map(lambda x:jieba.lcut(x))  #分词后放在列表
data['简介'] = data['简介'].map(lambda x:[i for i in x if len(i) > 1]) # 返回的是一个列表， 词个数大于1的留下来
print(data['简介'].head())
# 停用词处理
stop = pd.read_csv('stopwords.txt',sep='\n\t',engine='python',encoding='utf-8')  # 分隔符\n\t
print(stop.head())
data['简介'] = data['简介'].map(lambda x:[i for i in x if i not in stop])
data['简介'] = data['简介'].map(lambda x:' '.join(i for i in x));print(data['简介'])  #转换成字符串里面是一句话，用空格分开的形式
list = data['简介'].tolist() # 首先转化为一个列表  变成list结构 每个元素是一句话
x1 = CountVectorizer().fit_transform(list)  #向量化  把一句话转变成词向量稀疏矩阵
print(x1.shape)
# print(x1,toarray())

# 降维处理
pc = TruncatedSVD(n_components=10)   # 设置降为10个维度
x1 = pc.fit_transform(x1)
print(x1.shape)

del data['简介']

# 提取 x和 y
target = ['播放数']
feature = [x for x in data.columns if x not in target]

# 8.	对Y进行对数转换。（10分）
# 对数转换
data[target] = np.log(data[target])

# 9.	对X进行数据标准化转换。（10分）
# 数据标准化
data[feature] = StandardScaler().fit_transform(data[feature]) # data切片多列

# 10.	划分数据集，建立线性回归方程。（5分）
# 切片x y
y = data.loc[:,target]
x2 = data.loc[:,feature]  #
print(x2.shape)
x = np.concatenate((x1,x2),axis=1) # 合并数据 x1就是 简介 向量化
print(x.shape)

# 划分数据集
trainx,testx,trainy,testy = train_test_split(x,y,test_size=0.3,random_state=123)
# 训练模型
model = Ridge(alpha=1000).fit(trainx,trainy)

# 11. 打印评估指标。（5分）
# 打印均方误差和R平方
print(metrics.mean_squared_error(testy,model.predict(testx)),
metrics.r2_score(testy,model.predict(testx)))
# 回归方程系数
print(model.coef_)

# print(data.corr(method='pearson'))

# model = ols(u'播放数 ~  收藏数 ',data).fit()
# print(model.params)
# print(model.summary())

# model = ols('播放数 ~  收藏数+弹幕数+硬币数+上传者投稿数+上传者粉丝数+评论数+分享数 ',data).fit()
# print(model.params)
# print(model.summary())

# model = ols('播放数 ~  弹幕数+上传者粉丝数+评论数+分享数 ',data).fit()
# print(model.params)
# print(model.summary())

