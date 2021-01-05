import sys
import pandas as pd
import numpy as np
#数据集与代码地址：https://www.kaggle.com/c/ml2020spring-hw1/notebooks

path = "ml2020spring-hw1/"
# data = pd.read_csv('gdrive/My Drive/hw1-regression/train.csv', header = None, encoding = 'big5')
train_data = pd.read_csv(path + 'train.csv', encoding = 'big5')
# 1.预处理
train_data[train_data == 'NR'] = 0 #降雨量NR置为0
train_data_v2 = train_data.copy()
train_data_v2 = train_data_v2.drop(columns = '測站')#删掉了测站
for num in range(0,23):
    num = str(num)
    train_data[num] = train_data[num].astype(float)

# 输入前9个小时的18个特征，输出预测第10个小时的PM2.5值
# 原数据 -->> x轴:18个特征，y轴:按照时间顺序排列的数据
train_data_new = pd.DataFrame()
for x in range(24):
    train_data_first = train_data_v2[['日期','測項',str(x)]].copy()
    train_data_first['日期'] = pd.to_datetime(train_data_first['日期']+' '+str(x)+':00:00')
    train_data_first = train_data_first.pivot(index = '日期',columns = '測項', values = str(x))
    train_data_new = pd.concat([train_data_new,train_data_first])
train_data_new = train_data_new.astype('float64').sort_index().reset_index().drop(['日期'], axis = 1)

# 数据标准化：0均值标准化，将原始数据集归一化为均值为0，方差为1的数据集 公式：z=(x-期望)/方差
train_mean = train_data_new.mean().copy()
train_std = train_data_new.std().copy()
for liecolumn in train_data_new:
    train_data_new[liecolumn] = (train_data_new[liecolumn] - train_mean[liecolumn]) / train_std[liecolumn]
# 最终结果：每行是前9天的特征+第10天的PM2.5值
tx = train_data_new.copy()
tx.columns = tx.columns + '_0'
for i in range(1,10):
    ty = train_data_new.copy()
    if i == 9:
        ty = ty[['PM2.5']]
        # 结果列不需要标准化，需要放大回去
        ty = ty * train_std['PM2.5'] + train_mean['PM2.5']
    ty.columns = ty.columns + '_' + str(i)
    for j in range(i):
        ty = ty.drop([j])
    tx = pd.concat([tx, ty.reset_index().drop(['index'], axis=1)], axis=1)
# 每个月删除最后9行，删了108行数据：x轴不对的数据 第一个月(471-479)
# 注：删除行号时，不需要从后往前删，删除哪行，哪行的x轴索引消失
for i in range(12):
    for j in range(9):
        tx = tx.drop([480*(i+1)-9+j])
train_data = tx
train_data.describe()
# x:前9个小时的所有数据(0-8)，y:第10个小时的PM2.5值
train_x = train_data.drop(['PM2.5_9'],axis = 1)
train_y = train_data[['PM2.5_9']]
x = np.hstack((train_x.values,np.ones((np.size(train_x.values,0),1),'double'))) #x的x轴加1,是因为有常数项
y = train_y.values

# np.size(train_x.values,0) train_x有多少行 = 5652

data = np.random.random((np.size(x,1),1)) #162个特征参数+1个常参数

#学习速率
learning_rate = 0.00000006

#把训练数据分成四份 按照3:1的比例设置为 训练集和验证集
train_X = x[:4320]
train_Y = y[:4320]
vari_X = x[4320:]
vari_Y = y[4320:]

train_x_x = train_X.T @ train_X
train_x_y = train_X.T @ train_Y

# 预测方法
# LOSS function 均方根误差
def loss(x,y,w):
    return np.sum((y- x @ w)**2)/np.size(y,0)

# 梯度下降
def gradientDescent(w):
    return  train_x_x @ w - train_x_y#等价于x.T @ (x @ w - y)

for i in range(200001):
    data = data - learning_rate * gradientDescent(data)
    if i %50000 == 0 :
        #输出训练集误差和验证集误差
        print(i,loss(train_X,train_Y,data), loss(vari_X,vari_Y,data))
np.save('weight.npy', data)
data = np.load('weight.npy')
#预测
test_data = pd.read_csv(path + 'test.csv',encoding='big5',names=['id', '测项', '0', '1', '2', '3', '4', '5', '6', '7', '8'])
test_data['id'] = test_data['id'].str.split('_',expand = True)[1].astype('int')

test_data_new = pd.DataFrame()
for i in range(9):
    test_data_slice = test_data[['id', '测项', str(i)]].copy()
    test_data_slice = test_data_slice.pivot(index='id', columns='测项', values=str(i))
    test_data_slice.columns = test_data_slice.columns + '_' + str(i)
    for j in range(18):
        test_data_slice.iloc[:,[j]] = (test_data_slice.iloc[:,[j]].replace('NR', '0').astype('float64') - train_mean[j]) / train_std[j]
    test_data_new = pd.concat([test_data_new, test_data_slice], axis=1)

test_data_new = test_data_new.replace('NR', '0').astype('float64').reset_index().drop(['id'], axis=1)

test_x = np.hstack((test_data_new.values, np.ones((np.size(test_data_new.values,0), 1), 'double')))
print(np.size(test_x,0), np.size(test_x,1))

test_y = test_x @ (data)

test_data_id = test_data['id']

submission = pd.DataFrame({
        "id": test_data_id.unique(),
        "value": test_y.T[0]
    })

submission.to_csv('submission.csv', index=False)
if __name__ == "__main__":
    pass