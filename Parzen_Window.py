##
# @file     MQDF.py
#
# @date     2023-04
#
# @brief    Python code for INT301 Lab. Discriminant Functions & Non-parametric Classifiers
#           This code will implement the MQDF algorithm for iris.data classification
#           without using any third-party algorithm library.

# ----------------------------------------------------------------------------------------------------------- #
###############################################################################################################
#                             You need to fill the missing part of the code                                   #
#                        detailed instructions have been given in each function                              #
###############################################################################################################
# ----------------------------------------------------------------------------------------------------------- #
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import timeit
from math import *

###############################################################################################################
#                                        Self-defined functions                                               #
###############################################################################################################

def twoD_plot(filename): # To check the general properties of the dataset in 2D (Additional task)
    data = pd.read_csv(filename, names=["sepal length", "sepal width", "petal length", "petal width", "class"])
    data.head(5)
    data.describe()
    data.groupby('class').size()
    sns.pairplot(data, hue="class", height=2, palette='colorblind');
    plt.show()

def fourD_plot(p1, p2, p3, p4): # To check the general properties of the dataset in 4D (Additional task)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = p1.astype(float) # sepal length
    y = p2.astype(float) # sepal width
    z = p3.astype(float) # petal length
    c = p4.astype(float) # petal width
    img = ax.scatter(x, y, z, c=c, cmap=plt.hot()) # The 4D datasets will be shown in the 3D coordinate with color gradient
    fig.colorbar(img)
    # Add axis
    ax.set_xlabel('sepal length', fontweight='bold')
    ax.set_ylabel('sepal width', fontweight='bold')
    ax.set_zlabel('petal length', fontweight='bold')
    plt.show()

def multivariate_normal_kernel(x, xi, h, cov): # Kernel function for datasets higher than 1D，多元正态分布核函数
    det_cov = np.linalg.det(cov) # Determinant of the covariance matrix. Tip: use np.linalg.det() 协方差矩阵行列式
    inv_cov = np.linalg.inv(cov) # Inverse of the covariance matrix. Tip: np.linalg.inv() 协方差矩阵的逆
    u = (x - xi) / h # Compute the distance, math is (x - xi)/h. You may need to turn it into matrix
    numer = np.exp(-0.5 * np.dot(np.dot(u.T, inv_cov), u)) # 多元高斯分布的分子Numerator of the multivariate gaussian distribution, math is pow(e, ( -0.5 * u * inv_cov * u.T))
    denom = np.sqrt(np.power(2 * np.pi, len(x)) * det_cov) # 多元高斯分布的分母Denominator of the multivariate gaussian distribution, math is pow(pow(2*pi, len(x)) * det_cov, 1/2)
    kernel = numer / denom # 多元高斯分布的核函数numer / denom
    return kernel

def normal_kernel(x, xi, h): # Kernal function for 1D datasets，一维正态分布核函数
    u = (x - xi) / h # Compute the distance, math is (x - xi) / h
    kernel = np.exp(-(u ** 2) / 2) / np.sqrt(2 * np.pi) # math is :exp(-(abs(u) ** 2) / 2) / (sqrt(2 * pi)), use numpy methods will be helpful
    return kernel

def parzen_window(test, train, h, d): # Parzen Window function to output the conditional probability，Parzen窗函数
    cov = np.identity(d) # 创建一个与数据集维度相同的单位矩阵，用于协方差矩阵。Tip: use np.identity()
    if d == 1: # For 1D, apply the normal distribution
        p = normal_kernel(test, train, h) / h # 一维正态分布核函数
    else: # For higher dimension, apply the multivariate normal distribution
        p = multivariate_normal_kernel(test, train, h, cov) / (h ** d) # 多元正态分布核函数
    return p

def pz_predict(x_len, np_array): # To find the predicted class of the test set，预测类别
    x_pred = []
    for i in range(x_len):
        max = np.max(np_array[:,i]) # Get the maximum probability in the ith column (ith data of x)
        if max == np_array[0][i]: # If 'max' is equal to the ith value in setosa array
            pred = 'Iris-setosa'
        elif max == np_array[1][i]: # If 'max' is equal to the ith value in versicolor array
            pred = 'Iris-versicolor'
        else: #  If 'max' is equal to the ith value in virginica array
            pred = 'Iris-virginica'
        x_pred.append(pred) # Store the predicted class in the order of the test datasets
    return x_pred


def pz_accuracy(pred_class, class_x): # To obtain the accuracy of the predicted result
    acc = 0  # Initialize the accuracy
    for ind, pred in enumerate(pred_class):
        if pred == class_x[ind]: # Compare the predicted classes with the actual classes of the test set
            acc += 1 # Increase the accuracy parameter if it is correct
        else:
            pass # If not correct, pass
    return (acc / len(pred_class) * 100)

###############################################################################################################
#                                   Class for Data pre-processing                                             #
###############################################################################################################

class Data_process: # Class for data pre-processing
    def __init__(self):
        self.filename = "iris.data" # Dataset folder name
        # Predefined parameters
        self.line_comp = []
        self.iris_list = []

    def load_data(self): # Method to load the dataset and store them in a list
        with open(self.filename) as f:
            for line in f:
                text_lines = line.strip()
                line_comp = text_lines.split(',')
                self.iris_list.append(line_comp)
        del self.iris_list[-1] # Remove the empty element of the list
        return self.iris_list

    def shuffle(self): # Method to shuffle the stored dataset
        random.seed(97) # Define the seed value first to keep the shuffled data same
        random.shuffle(self.iris_list) # Shuffle the list
        return self.iris_list

    def separate_data(self): # Method to separate the dataset into five parts for 5-fold cross validation，将数据集分为五部分，用于五折交叉验证
        length = int(len(self.iris_list) / 5) # Cutting length of the list
        data1 = self.iris_list[:length]
        data2 = self.iris_list[length:length * 2]
        data3 = self.iris_list[length * 2:length * 3]
        data4 = self.iris_list[length * 3:length * 4]
        data5 = self.iris_list[length * 4:length * 5]
        return data1, data2, data3, data4, data5

    def combine_train(self, ind, total_data): # Method to separate combined train sets and a test set
        train = []
        for i in range(len(total_data)): # According to the index, the test set will be chosen among the five subsets
            if ind == i:
                test = total_data[i]
            else:
                train += total_data[i]
        return train, test

    def separate_class(self, dataset): # Method to separate dataset into three given classes，将数据集分为三类
        setosa = []
        versicolor = []
        virginica = []
        for info in dataset:
            if info[4] == 'Iris-setosa': # 如果类别为setosa，则将其添加到setosa列表中
                setosa.append(info)
            elif info[4] == 'Iris-versicolor':
                versicolor.append(info)
            else:
                virginica.append(info)
        return setosa, versicolor, virginica

    def numeric_n_name(self, nested_list): # Method to separate the numeric data and class_names，将数据集分为数值数据和类别名称
        num_list = []
        class_list = []
        for instance in nested_list:
            num_data = instance[:4] # Extract the numeric data
            class_name = instance[4:] # Extract the class names of the data sets
            num_list.append(num_data)
            class_list += class_name
        return num_list, class_list # Numeric data can be converted into numpy array

    def data_analyzer(self, info): # Method to plot the 2D and 4D figures of the given dataset to analyze the properties
        np_info = np.array(info)
        sepal_length = np_info[:,0]
        sepal_width = np_info[:,1]
        petal_length = np_info[:,2]
        petal_width = np_info[:, 3]

        fourD_plot(sepal_length, sepal_width, petal_length, petal_width)
        twoD_plot(self.filename)

    def prior_prob(self, dataset): # Method to calculate the prior probabilities of each class，计算每个类别的先验概率
        prior_prob_se = len(dataset[0]) / (len(dataset[0]) + len(dataset[1]) + len(dataset[2])) # Setosa
        prior_prob_ve = len(dataset[1]) / (len(dataset[0]) + len(dataset[1]) + len(dataset[2])) # Versicolor
        prior_prob_vi = len(dataset[2]) / (len(dataset[0]) + len(dataset[1]) + len(dataset[2])) # Virginica
        return prior_prob_se, prior_prob_ve, prior_prob_vi

###############################################################################################################
#                                              Main Part                                                      #
###############################################################################################################

if __name__ == '__main__':
    iris = Data_process() # Define Class Data_process()
    irist_data = iris.load_data() # Load the iris dataset
    div_data = iris.numeric_n_name(irist_data) # Separate numeric dataset and class names
    iris.data_analyzer(div_data[0]) # Check the general properties of the dataset
    init_data = iris.shuffle() # Shuffle the dataset
    five_data = iris.separate_data()

    # Initialize lists for checking the results
    h_list = []
    Lh_list = []
    acc_list = []

    max_acc = 0 #记录最高准确率
    optimal_h = 0 #记录最高准确率对应的h

    start = timeit.default_timer() # Start timer to count the running time of Parzen Window Method
    # for h in range(1, 4): # 'for loop' condition to compare the program time of Pazen Window and MQDF
    for h in np.arange(0.2, 3, 0.1): # 通过改变h值找到最优核
        opt_size = 0  # 找到最优核大小
        sum_avg_acc = 0 # 计算五折交叉验证的平均准确率
        for index in range(len(five_data)): # 5-fold Cross-Validation
            total_subset = iris.combine_train(index, five_data) # Index denotes the array for testing 
            sep_dataset = iris.separate_class(total_subset[0]) # 将训练数据集分为三类
            sep_data = [sep_dataset[0], sep_dataset[1], sep_dataset[2]]
            prior_prob = iris.prior_prob(sep_dataset) # 计算三个类别的先验概率
            # Convert the three train datasets into numpy array
            np_se = np.array(iris.numeric_n_name(sep_data[0])[0])
            np_ver = np.array(iris.numeric_n_name(sep_data[1])[0])
            np_vir = np.array(iris.numeric_n_name(sep_data[2])[0])
            # Prepare the train dataset in 'float' type
            train = [np_se.astype(float), np_ver.astype(float), np_vir.astype(float)]

            d = len(np_se[0]) # Dimension of the dataset

            x = np.array(iris.numeric_n_name(total_subset[1])[0]) # Extract the numeric data of test set 提取测试集的数值数据
            np_x = x.astype(float)
            x_len = len(np_x)
            class_x = iris.numeric_n_name(total_subset[1])[1] # 测试集的类别名称
            # 存储每个测试数据的条件概率
            p_se = []
            p_ver = []
            p_vir = []
            cn = 0 # 计数器，检查for循环中的训练数据集的类别

            # 开始Parzen Window算法
            for name in train: # 对于三个类名
                for x in np_x: # 对于测试集的每个数据列表
                    p_x = 0 # define the initial probability of x
                    for x_i in name: # For each data list of the train set
                        con_prob = parzen_window(x, x_i, h, d) # 计算PZ的核函数
                        p_x += con_prob # 将每个训练数据列表的核函数输出相加
                    p_xw = p_x / len(name) # 计算测试数据的条件概率
                    opt_size += np.sum(np.log10(p_xw + 1e-10))  # Add a small value to avoid log(0)

                    # Add the probability into its category
                    if cn == 0:
                        p_se.append(p_xw * prior_prob[0]) # Posterior probability of x in Setosa
                    elif cn == 1:
                        p_ver.append(p_xw * prior_prob[1]) # Posterior probability of x in Versicolor
                    else:
                        p_vir.append(p_xw * prior_prob[2]) # Posterior probability of x in Virginica
                cn += 1 # Count when the loop of one category is finished

            prob_array = np.array([p_se, p_ver, p_vir]) # 将三个类别的后验概率组合在一起

            pred_class = pz_predict(x_len, prob_array) # 获取Parzen Window方法的预测结果

            pz_acc = pz_accuracy(pred_class, class_x) # 计算分类准确率
            sum_avg_acc += pz_acc
            # print("Accuracy:", pz_acc)

        avg_acc = sum_avg_acc / len(five_data)  # Average accuracy
        if avg_acc > max_acc:#更新最高准确率
            max_acc = avg_acc
            optimal_h = h#更新最高准确率对应的h
        Lh = opt_size / len(five_data) # 计算所选h的平均对数似然值

        # Store the results to plot on graphs
        h_list.append(h)
        Lh_list.append(Lh)
        acc_list.append(avg_acc)

    stop = timeit.default_timer() # Stop timer for the running time of Parzen Window algortihm
    print('Running time of Parzen Window with 5-fold cross validation:', stop - start)

    # Plot the result of Maximum likelihood estimation of h & average classification accuracy in terms of h
    x = np.array(h_list)
    y = np.array(acc_list)
    y2 = np.array(Lh_list)

    plot1 = plt.figure(1)
    plt.plot(x, y, label="Accuracy")
    plt.xlabel("h")
    plt.ylabel("Average accuracy")

    plot2 = plt.figure(2)
    plt.plot(x, y2, label="L(h)")
    plt.xlabel("h")
    plt.ylabel("L(h)")

    ymax = max(y2)
    xpos = np.where(y2 == ymax)
    xmax = x[xpos]
    opt_avg_acc = float(y[xpos])

    plt.annotate('Optimal Kernel Size',
                 xy=(xmax, ymax),
                 xycoords='data',
                 xytext=(-30, 30),
                 textcoords='offset points',
                 arrowprops=dict(arrowstyle="->"))
    plt.legend()
    plt.show()

    print('Optimal Kernel Size:', optimal_h)
    print('Highest average accuracy:', max_acc, '%') # 最高准确率
    print('Average accuracy when h =', xmax, ':', opt_avg_acc, '%')