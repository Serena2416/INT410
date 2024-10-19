# @file     MQDF.py
#
# @date     2023-09
#
# @brief    Python code for INT410 Lab. Discriminant Functions & Non-parametric Classifiers
#           This code will implement the MQDF algorithm for iris.data classification
#           without using any third-party algorithm library.

# ----------------------------------------------------------------------------------------------------------- #
###############################################################################################################
#                             You need to fill the missing part of the code                                   #
#                        detailed instructions have been given in each function                              #
###############################################################################################################
# ----------------------------------------------------------------------------------------------------------- #


import math
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

def twoD_plot(filename):  # To check the general properties of the dataset in 2D (Additional task)
    data = pd.read_csv(filename, names=["sepal length", "sepal width", "petal length", "petal width", "class"])
    data.head(5)
    data.describe()
    data.groupby('class').size()
    sns.pairplot(data, hue="class", height=2, palette='colorblind');
    plt.show()


def fourD_plot(p1, p2, p3, p4):  # To check the general properties of the dataset in 4D (Additional task)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = p1.astype(float)  # sepal length
    y = p2.astype(float)  # sepal width
    z = p3.astype(float)  # petal length
    c = p4.astype(float)  # petal width
    img = ax.scatter(x, y, z, c=c,
                     cmap=plt.hot())  # The 4D datasets will be shown in the 3D coordinate with color gradient
    fig.colorbar(img)
    # Add axis
    ax.set_xlabel('sepal length', fontweight='bold')
    ax.set_ylabel('sepal width', fontweight='bold')
    ax.set_zlabel('petal length', fontweight='bold')
    plt.show()



def pz_predict(x_len, np_array):  # To find the predicted class of the test set
    x_pred = []
    for i in range(x_len):
        max = np.max(np_array[:, i])  # Get the maximum probability in the ith column (ith data of x)
        if max == np_array[0][i]:  # If 'max' is equal to the ith value in setosa array
            pred = 'Iris-setosa'
        elif max == np_array[1][i]:  # If 'max' is equal to the ith value in versicolor array
            pred = 'Iris-versicolor'
        else:  # If 'max' is equal to the ith value in virginica array
            pred = 'Iris-virginica'
        x_pred.append(pred)  # Store the predicted class in the order of the test datasets
    return x_pred


def pz_accuracy(pred_class, class_x):  # To obtain the accuracy of the predicted result
    acc = 0  # Initialize the accuracy
    for ind, pred in enumerate(pred_class):
        if pred == class_x[ind]:  # Compare the predicted classes with the actual classes of the test set
            acc += 1  # Increase the accuracy parameter if it is correct
        else:
            pass  # If not correct, pass
    return (acc / len(pred_class) * 100)


###############################################################################################################
#                                   Class for Data pre-processing                                             #
###############################################################################################################

class Data_process:  # Class for data pre-processing
    def __init__(self):
        self.filename = "iris.data"  # Dataset folder name
        # Predefined parameters
        self.line_comp = []
        self.iris_list = []

    def load_data(self):  # Method to load the dataset and store them in a list
        with open(self.filename) as f:
            for line in f:
                text_lines = line.strip() #移除每一行前后空白字符，包括换行符
                line_comp = text_lines.split(',')
                self.iris_list.append(line_comp)
        del self.iris_list[-1]  # Remove the empty element of the list
        return self.iris_list

    def shuffle(self):  # Method to shuffle the stored dataset
        random.seed(97)  # Define the seed value first to keep the shuffled data same 随机种子用于确保每次打乱后的顺序一致
        random.shuffle(self.iris_list)  # Shuffle the list 打乱存储在 self.iris_list 中的数据集
        return self.iris_list

    def separate_data(self):  # Method to separate the dataset into five parts for 5-fold cross validation
        length = int(len(self.iris_list) / 5)  # Cutting length of the list
        data1 = self.iris_list[:length]
        data2 = self.iris_list[length:length * 2]
        data3 = self.iris_list[length * 2:length * 3]
        data4 = self.iris_list[length * 3:length * 4]
        data5 = self.iris_list[length * 4:length * 5]
        return data1, data2, data3, data4, data5

    def combine_train(self, ind, total_data):  # Method to separate combined train sets and a test set
        train = []
        for i in range(len(total_data)):  # According to the index, the test set will be chosen among the five subsets
            if ind == i:
                test = total_data[i]
            else:
                train += total_data[i]
        return train, test

    def separate_class(self, dataset):  # Method to separate dataset into three given classes
        setosa = []
        versicolor = []
        virginica = []
        for info in dataset:
            if info[4] == 'Iris-setosa':
                setosa.append(info)
            elif info[4] == 'Iris-versicolor':
                versicolor.append(info)
            else:
                virginica.append(info)
        return setosa, versicolor, virginica

    def numeric_n_name(self, nested_list):  # Method to separate the numeric data and class_names
        num_list = []
        class_list = []
        for instance in nested_list:
            num_data = instance[:4]  # Extract the numeric data
            class_name = instance[4:]  # Extract the class names of the data sets
            num_list.append(num_data)
            class_list += class_name
        return num_list, class_list  # Numeric data can be converted into numpy array

    def data_analyzer(self,
                      info):  # Method to plot the 2D and 4D figures of the given dataset to analyze the properties
        np_info = np.array(info)
        sepal_length = np_info[:, 0]
        sepal_width = np_info[:, 1]
        petal_length = np_info[:, 2]
        petal_width = np_info[:, 3]

        fourD_plot(sepal_length, sepal_width, petal_length, petal_width)
        twoD_plot(self.filename)

    def prior_prob(self, dataset):  # Method to calculate the prior probabilities of each class  先验概率计算
        prior_prob_se = len(dataset[0]) / (len(dataset[0]) + len(dataset[1]) + len(dataset[2]))  # Setosa
        prior_prob_ve = len(dataset[1]) / (len(dataset[0]) + len(dataset[1]) + len(dataset[2]))  # Versicolor
        prior_prob_vi = len(dataset[2]) / (len(dataset[0]) + len(dataset[1]) + len(dataset[2]))  # Virginica
        return prior_prob_se, prior_prob_ve, prior_prob_vi



###############################################################################################################
#                                   Self-defined functions for MQDF                                           #
###############################################################################################################

def QDF_model(train_data, class_num):  # Modified function from QDF to obtain the required parameters for MQDF
    mean = []  # Initialize a list to store the mean
    cov_matrix = []  # Initialize a list to store covariance matrices
    for i in range(class_num):  # For all classes
        train_data[i] = np.array(train_data[i], dtype=np.float64)  # Convert to numpy array with 64-bit precision
        mean.append(np.mean(train_data[i], axis=0))  # Calculate mean for all features in ith class
        cov_matrix.append(np.cov(train_data[i].T))  # Calculate covariance matrix of ith class
    return mean, cov_matrix


def MQDF2_model(cov, d, k, class_num):  # Function to obtain the trained parameters of MQDF2
    eigenvalue = []  # Initialize a list to store eigenvalues  特征值
    eigenvector = []  # Initialize a list to store eigenvectors 特征向量
    delta = [0] * class_num  # Each delta value of classes will be stored here 初始化delta参数
    for i in range(class_num):  # For all classes
        covMat = cov[i] #取出第 i 类的协方差矩阵
        eigvals, eigvecs = np.linalg.eig(covMat) #计算该矩阵的特征值和特征向量
        id = eigvals.argsort() #对特征值进行排序，得到其索引
        id = id[::-1]  # Convert to descending order 将排序后的索引翻转为降序，以便最大特征值排在前面
        eigvals = eigvals[id] #根据降序索引重新排列特征值
        eigvecs = eigvecs[:, id]
        eigenvector.append(eigvecs[:, :k])  # Store the eigenvectors from j=1 to k ，将前 k 个特征向量添加到 eigenvector 列表中，只保留k个降维提高泛化能力
        eigenvalue.append(eigvals[:k])  # Store the eigenvalues from j=1 to k
        #delta[i] = sum(eigvals[int(k):]) / (d - k)  # Compute delta as the mean of minor values 
        #将较小的特征值（即第 k+1 到最后一个特征值）的和除以 d - k，得到 delta 参数。这是 MQDF2 模型中的一个重要参数，用于对高维度数据中的较小特征值进行平滑处理
        if d - k > 0:
            delta[i] = sum(eigvals[k:]) / (d - k) # Ensure delta is positive
        else:
            delta[i] = 1e-6  # Set a small positive value if d - k <= 0
    return eigenvalue, eigenvector, delta


def predict_MQDF2(d, np_x, class_num, k, mean, eigenvalue, eigenvector, delta): # Function to perform classification based on the MQDF2 trained parameters
    # assert (k < d and k > 0)  # Assertion error when k greater or equal to d and negative k
    pred_label = [] # Initialize a list to store the predicted classes
    for sample in np_x: # For the number of test samples
        test_x = np.matrix(sample, np.float64).T # Convert a sample data to a matrix 将样本数据转置并转换为 float64 类型的矩阵
        max_g2 = -float('inf') # The initial value of max_g2 is set to the negative infinity
        for i in range(class_num): # For all classes
            dis = np.linalg.norm(test_x.reshape((d,)) - mean[i].reshape((d,))) ** 2 # Compute the distance between the sample data and the mean, and then square it
            # Second term of the residual of subspace projection
            euc_dis = [0] * int(k) # Initialization
            ma_dis = [0] * int(k)  # Initialization
            for j in range(int(k)): # For the range of k
                euc_dis[j] = ((eigenvector[i][:, j].T * (test_x - mean[i].reshape(-1,1)))[0, 0]) ** 2 #计算样本在第 j 个特征向量上的投影残差
                eigenvector_col = eigenvector[i][:, j].reshape(-1, 1)
                ma_dis[j] = (((test_x - mean[i].reshape(-1,1)).T * eigenvector_col)[0, 0]) ** 2 #计算均值在第 j 个特征向量上的投影残差

            g2 = 0  # Initialize the MQDF2，分类的得分，特征值越大，表示该特征在分类中越重要，因此其对 g2 的贡献也越小
            for j in range(int(k)): # For the range of k
                # Firstly, compute the terms including j and add them to g2
                g2 += (euc_dis[j] * 1.0 / eigenvalue[i][j])+ math.log(eigenvalue[i][j])
                #用每个特征向量的投影残差除以对应的特征值，再加上特征值的对数

            # Secondly, compute the terms only including i and add them to g2
            g2 += ((dis - sum(ma_dis)) / delta[i]) + ((d - int(k)) * math.log(delta[i]))
            g2 = -g2 # Convert the g2 values to minus to find the maximum value

            if g2 > max_g2: # If the current g2 > previous max g2
                max_g2 = g2 # Replace the g2 value
                prediction = i # Store the class id of current g2
            elif g2 == max_g2:
                print(i, "==", prediction) # Error if two g2 values are equal
            else:
                pass # Ignore current g2 if it's smaller than max_g2
        pred_label.append(prediction) # After for loop, append the current max g2 class id
    return pred_label


def MQDF1_model(cov, d, class_num, h):  # Function to obtain the trained parameters of MQDF1
    eigenvalue = []  # Initialize a list to store eigenvalues
    eigenvector = []  # Initialize a list to store eigenvectors
    delta = [0] * class_num  # Each delta value of classes will be stored here
    for i in range(class_num):  # For all classes
        covMat = cov[i]
        eigvals, eigvecs = np.linalg.eig(covMat)
        eigenvector.append(eigvecs)  # Append eigvecs to eigenvector
        eigenvalue.append(eigvals + h**2)  # Add h^2 to eigvals, append to eigenvalue
        delta[i] = np.mean(eigvals)  # Compute delta as the mean of eigenvalues
    return eigenvalue, eigenvector, delta


def predict_MQDF1(d, np_x, class_num, mean, eigenvalue, eigenvector,
                  delta):  # Function to perform classification based on the MQDF1 trained parameters
    # assert (k < d and k > 0)  # Assertion error when k greater or equal to d and negative k
    pred_label = []  # Initialize a list to store the predicted classes
    for sample in np_x:  # For the number of test samples
        test_x = np.matrix(sample, np.float64).T  # Convert a sample data to a matrix
        max_g = -float('inf')  # The initial value of max_g2 is set to the negative infinity
        for i in range(class_num):  # For all classes
            dis = np.linalg.norm(test_x.reshape((d,)) - mean[i].reshape(
                (d,))) ** 2  # Compute the distance between the sample data and the mean, and then square it
            # Second term of the residual of subspace projection
            euc_dis = [0] * int(d)  # Initialization
            ma_dis = [0] * int(d)  # Initialization
            # difference = test_x - mean[i]
            for j in range(int(d)):  # For the range of d
                euc_dis[j] = ((eigenvector[i][:, j].T * (test_x - mean[i].reshape(-1,1)))[0, 0]) ** 2 
                #样本和类别均值之间在特征向量方向上的欧氏距离
                eigenvector_col = eigenvector[i][:, j].reshape(-1, 1)
                ma_dis[j] = (((test_x - mean[i].reshape(-1,1)).T * eigenvector_col)[0, 0]) ** 2 #马氏距离
                # ma_dis[j] = (np.dot(difference.T, eigenvector[i][:, j]) ** 2)[0, 0]

            g = 0  # Initialize the MQDF1
            for j in range(int(d)):  # For the range of d
                # Firstly, compute the terms including j and add them to g
                g += (euc_dis[j] * 1.0 /eigenvalue[i][j]) +math.log(eigenvalue[i][j])


            # Secondly, compute the terms only including i and add them to g
            #g += ((dis - sum(ma_dis)) / max(delta[i], 1e-6))
            g = -g  # Convert the g2 values to minus to find the maximum value

            if g > max_g:  # If the current g > previous max g
                max_g = g  # Replace the g value
                prediction = i  # Store the class id of current g
            elif g == max_g:
                print(i, "==", prediction)  # Error if two g values are equal
            else:
                pass  # Ignore current g if it's smaller than max_g
        pred_label.append(prediction)  # After for loop, append the current max g class id
    return pred_label


def mqdf_accuracy(predic, class_x): # Function to calculate the prediction accuracy of MQDF2
    conv_pred = []
    for pred in predic: # For the predicted id numbers of test dataset
        if pred == 0: # If the predicted value is '0'
            conv_pred.append('Iris-setosa') # It is setosa
        elif pred == 1: # If it is '1'
            conv_pred.append('Iris-versicolor') # It is versicolor
        elif pred == 2: # If it is '2'
            conv_pred.append('Iris-virginica') # It is virginica
        else: # Out of range, there is an error
            print('Wrong prediction')  # Error when index out of the range

    accuracy = pz_accuracy(conv_pred, class_x) # Accuracy can be calculated using 'pz_accuracy()'
    return accuracy





###############################################################################################################
#                                              Main Part                                                      #
###############################################################################################################


if __name__ == '__main__':
    print('starting...')
    import os
    os.environ['MKL_NUM_THREADS'] = '1' #设置线程为1，减少并发处理带来的问题
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" #允许在KMP中存在重复的库，解决库冲突的问题
    iris = Data_process() # Define class
    irist_data = iris.load_data() # Load the iris dataset

    # 调用 twoD_plot 函数
    twoD_plot("iris.data")
    # 准备数据并调用 fourD_plot 函数
    np_data = np.array(iris.numeric_n_name(irist_data)[0], dtype=float)

    div_data = iris.numeric_n_name(irist_data) # Separate numeric dataset and class names
    init_data = iris.shuffle() # Shuffle the dataset
    five_data = iris.separate_data() # Divide the dataset for 5-fold cross validation
    # Predefined parameters
    classNum = 3  # Three classes
    d = len(div_data[0][0])  # Feature numbers = 4

    k_list = []
    mqdf_acc_list = []

    print('loaded training set')
    start = timeit.default_timer()  # To measure the running time of MQDF1, start timer

    for h in range(1, 101):  # Test h from 0.01 to 1.00 后面h会*0.01
        mqdf_sum_avg_acc = 0
        cnt = 1 #计数器，跟踪折数
        for index in range(len(five_data)): # 5-fold cross validation
            total_subset = iris.combine_train(index, five_data)  # Index denotes the array for testing
            sep_dataset = iris.separate_class(total_subset[0])  # Return separated train datasets by three classes
            sep_data = [sep_dataset[0], sep_dataset[1], sep_dataset[2]] # Nested list of the three datasets
            # Only extract the numeric data from the datasets
            np_se = np.array(iris.numeric_n_name(sep_data[0])[0])
            np_ver = np.array(iris.numeric_n_name(sep_data[1])[0])
            np_vir = np.array(iris.numeric_n_name(sep_data[2])[0])
            # Prepare the train dataset by converting the numbers in 'str' to 'float
            train = [np_se.astype(float), np_ver.astype(float), np_vir.astype(float)]

            mean, cov = QDF_model(train, classNum) # Obtain the mean and covariance matrices
            eigval, eigvec, delta = MQDF1_model(cov, d, classNum, h= h*.01) # Obtain the eigenvalues, eigenvectors and delta

            # mean, eigenvalues, eigenvectors, k and delta will be the trained parameters of MQDF2 for prediction
            # print(f'Training process of the MQDF1 {index} model finished.')

            # Prepare the test dataset
            x = np.array(iris.numeric_n_name(total_subset[1])[0])  # numeric data of test set
            np_x = x.astype(float)
            x_len = len(np_x)

            class_x = iris.numeric_n_name(total_subset[1])[1]  # Real class names of each test set
            # predict_MQDF1(d, np_x, class_num, mean, eigenvalue, eigenvector, delta):
            predic = predict_MQDF1(d, np_x, classNum, mean, eigval, eigvec, delta) # Input the trained parameters to predict

            MQDF_accuracy = mqdf_accuracy(predic, class_x) # Based on the prediction result, compute the classification accuracy
            mqdf_sum_avg_acc += MQDF_accuracy
            # print(cnt, 'th accuracy:', MQDF_accuracy)
            cnt += 1

        MQDF_avg_acc  = mqdf_sum_avg_acc / len(five_data) # Calculate the average accuracy of 5-fold cross validation
        print('Averay accuracy of 5-fold cross-validation when h =', h*.01, ':', MQDF_avg_acc)
        print("________________________________________________________________________________")

        k_list.append(h)
        mqdf_acc_list.append(MQDF_avg_acc)
    stop = timeit.default_timer() # Stop timer
    print('Running time of MQDF1 with 5-fold cross validation:', stop - start) # Running time of MQDF2

    print()


    for k in range(1, d+1):  # Test k from 1 to d (4)
        mqdf_sum_avg_acc = 0
        cnt = 1
        for index in range(len(five_data)): # 5-fold cross validation
            total_subset = iris.combine_train(index, five_data)  # Index denotes the array for testing
            sep_dataset = iris.separate_class(total_subset[0])  # Return separated train datasets by three classes
            sep_data = [sep_dataset[0], sep_dataset[1], sep_dataset[2]] # Nested list of the three datasets
            # Only extract the numeric data from the datasets
            np_se = np.array(iris.numeric_n_name(sep_data[0])[0])
            np_ver = np.array(iris.numeric_n_name(sep_data[1])[0])
            np_vir = np.array(iris.numeric_n_name(sep_data[2])[0])
            # Prepare the train dataset by converting the numbers in 'str' to 'float
            train = [np_se.astype(float), np_ver.astype(float), np_vir.astype(float)]

            mean, cov = QDF_model(train, classNum) # Obtain the mean and covariance matrices
            eigval, eigvec, delta = MQDF2_model(cov, d, k, classNum) # Obtain the eigenvalues, eigenvectors and delta

            # mean, eigenvalues, eigenvectors, k and delta will be the trained parameters of MQDF2 for prediction
            # print(f'Training process of the MQDF2 {index} model finished.')

            # Prepare the test dataset
            x = np.array(iris.numeric_n_name(total_subset[1])[0])  # numeric data of test set
            np_x = x.astype(float)
            x_len = len(np_x)

            class_x = iris.numeric_n_name(total_subset[1])[1]  # Real class names of each test set

            predic = predict_MQDF2(d, np_x, classNum, k, mean, eigval, eigvec, delta) # Input the trained parameters to predict

            MQDF_accuracy = mqdf_accuracy(predic, class_x) # Based on the prediction result, compute the classification accuracy
            mqdf_sum_avg_acc += MQDF_accuracy
            # print(cnt, 'th accuracy:', MQDF_accuracy)
            cnt += 1

        MQDF_avg_acc  = mqdf_sum_avg_acc / len(five_data) # Calculate the average accuracy of 5-fold cross validation
        print('Averay accuracy of 5-fold cross-validation when k =', k, ':', MQDF_avg_acc)
        print("________________________________________________________________________________")

        k_list.append(k)
        mqdf_acc_list.append(MQDF_avg_acc)
    stop = timeit.default_timer() # Stop timer
    print('Running time of MQDF2 with 5-fold cross validation:', stop - start) # Running time of MQDF2
