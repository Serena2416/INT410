import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt

#calculate distance function，训练样本和测试样本的距离
def cal_distance(test,cal):
    label_distance_list=[]#list of label and distance
    for i in range(len(cal)):
        label_distance=[]
        label_distance.append(cal[i, 4])#append corresponding label
        d=test-cal[i,:] #test sample value-training sample value
        label_distance.append(np.linalg.norm(d)) #calculate distance
        label_distance_list.append(label_distance)
    return label_distance_list

if __name__=='__main__':
    iris_data=pd.read_csv("iris.data",header=None)
    labels_codes=pd.Categorical(iris_data[4]).codes
    for i in range(150):
        iris_data.loc[i,4]=labels_codes[i]
    datalist=iris_data.values.tolist()
    random.seed(17)
    random.shuffle(datalist)
    data_set=np.mat(datalist)

    average_acc=[]
    k_list=[]
    #  test different K values:
    for K in range(1, 120):
        if K%3!=0: #排除k=3的倍数，避免一些偏差导致过拟合或者欠拟合
            # for visualization
            k_list.append(K)
            accuracy = []
            # implementing 5-fold training process:
            for i in range(5): #模型在每个折上测试，在其余折上训练，循环5次
                fold_size = len(data_set) // 5 # 计算每个折的大小
                start = i * fold_size # 计算当前折的开始索引
                end = (i + 1) * fold_size if i < 4 else len(data_set) # 计算当前折的结束索引
                test_data = data_set[start:end] # 当前折的测试数据
                cal_data = np.vstack((data_set[:start], data_set[end:])) # 将训练数据集分为训练集和验证集
                # doing KNN
                right = 0 # 初始化正确分类的样本数
                for x in range(len(test_data)):
                    d_set = np.mat(cal_distance(test_data[x], cal_data))#list of distance，计算测试样本与训练样本的距离
                    d_set = (d_set[np.lexsort(d_set.T)])[0, :, :]#sort distance list，对距离进行排序
                    p_wk = [0, 0, 0] #be used to record P(wk|x)，用于记录每个类别的概率 
                    #calculate P(wk|x) for each test sample，计算每个测试样本属于每个类别的概率
                    for y in range(K):
                        if d_set[y, 0] == 0: # 如果距离为0，则表示该样本属于该类别
                            p_wk[0] = p_wk[0] + 1 / K # 计算每个类别的概率
                        elif d_set[y, 0] == 1:
                            p_wk[1] = p_wk[1] + 1 / K
                        else:
                            p_wk[2] = p_wk[2] + 1 / K
                    #calculate accuracy
                    if p_wk.index(max(p_wk)) == test_data[x, 4]:
                        right = right + 1
                accuracy.append(right / len(test_data))
            accuracy=np.mat(accuracy)
            print(accuracy)
            average_acc.append(np.mean(accuracy))
    plt.scatter(k_list,average_acc)
    plt.title('KNN')
    plt.xlabel('hyperparameter K')
    plt.ylabel('average accuracy')
    plt.show()
    k_max_list=[]
    for z in range(len(average_acc)):
        if average_acc[z]==max(average_acc):
            k_max_list.append(k_list[z])
    print("highest average accuracy:",round(max(average_acc),3))
    print("corresponding K:",k_max_list)
