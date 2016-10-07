import os
import numpy as np
import math

_author_ = 'Zhu Peihao'


# The function of training the model parameters and calculating  correct rate.
def learning():
    """

    :return: LDF anf QDF correct rate
    """
    # Load array information
    train_images = np.load('train_images.npy')
    train_label = np.load('train_label.npy')
    test_images = np.load('test_images.npy')
    test_label = np.load('test_label.npy')
    train_images = np.mat(train_images, float).T
    m, n = np.shape(train_images)
    train_temp = train_images - np.mean(train_images, axis=1) * np.mat(np.ones(n))
    U, Sigma = pca(train_temp)
    K = 0

    # Calculate the trace of a matrix, to select the dimension to be reduced
    for i in range(m):
        if np.trace(Sigma[0:i+1, 0:i+1]) / np.trace(Sigma) >= 0.85:
            K = i + 1
            break

    U_reduce = U[:, 0:K]        # reducing dimension transformation
    Qdf_param = QdfModel(train_images, train_label, U_reduce, K)        # QDFparam
    Qdf_param.cal_parameter()
    Ldf_param = LdfModel(train_images, train_label, U_reduce, K)        # LDFparam
    Ldf_param.cal_parameter()

    # Dimension reduction of test set
    test_images = np.mat(test_images, float).T
    test_dim = U_reduce.T * test_images

    QDF_accuracy = Qdf_func(Qdf_param, test_dim, test_label)
    LDF_accuracy = Ldf_func(Ldf_param, test_dim, test_label)

    return LDF_accuracy, QDF_accuracy


# PCA dimension reduction function
def pca(train_temp):
    m, n = np.shape(train_temp)
    cov_mat = (train_temp * train_temp.T) / n
    U, Sigma, VT = np.linalg.svd(cov_mat)       # singular value decomposition SVD
    Sigma =  np.diag(Sigma)
    return U, Sigma


# Function of calculating QDF parameters
def Qdf_func(QdfModel, test_dim, test_label):
    g = np.zeros(10, float)
    G = np.zeros(len(test_label),float)
    for i in range(len(test_label)):
        for j in range(10):
            g[j] = test_dim[:, i].T * QdfModel.big_W[j] * test_dim[:, i] + QdfModel.w[j] * test_dim[:, i] + QdfModel.w0[j]
        temp = np.argmax(g)
        if temp == test_label[i]:
            G[i] = 1
    num = np.where(G == 1)
    Qdf_accuracy = len(num[0]) / float(len(test_label))
    return Qdf_accuracy


# Function of calculating LDF parameters
def Ldf_func(LdfModel, test_dim, test_label):
    g = np.zeros(10, float)
    G = np.zeros(len(test_label), float)
    for i in range(len(test_label)):
        for j in range(10):
            g[j] = LdfModel.w[j] * test_dim[:, i] + LdfModel.w0[j]
        temp = np.argmax(g)
        if temp == test_label[i]:
            G[i] = 1
    num = np.where(G == 1)
    Ldf_accuracy = len(num[0]) / float(len(test_label))
    return Ldf_accuracy


# Class of QDF model
class QdfModel:
    def __init__(self, train_images, train_labels, U_reduce, K):
        self.train_images = train_images
        self.train_labels = train_labels
        self.U_reduce = U_reduce
        self.train_dim = self.U_reduce.T * self.train_images
        self.mu = np.mat(np.zeros((10, K), float))
        self.cov = [np.mat(np.zeros((K, K), float))] * 10
        self.big_W = [np.mat(np.zeros((K, K), float))] * 10
        self.w = np.mat(np.zeros((10, K), float))
        self.w0 = np.zeros(10, float)
        self.P = np.zeros(10, float)
        self.K = K

    def cal_parameter(self):
        for i in range(10):
            num = np.where(self.train_labels == i)
            self.P[i] = len(num[0])/float(len(self.train_labels))
            self.mu[i] = np.mean(self.train_dim[:, self.train_labels == i], axis=1).T
            train_temp = self.train_dim[:, num[0]] - self.mu[i].T * np.mat(np.ones(len(num[0])))
            self.cov[i] = (train_temp * train_temp.T) / len(num[0])
            self.big_W[i] = -0.5 * np.linalg.inv(self.cov[i])
            self.w[i] = (np.linalg.inv(self.cov[i]) * self.mu[i].T).T
            self.w0[i] = -0.5 * self.mu[i] * np.linalg.inv(self.cov[i]) * self.mu[i].T \
                         - 0.5 * math.log(np.linalg.det(self.cov[i])) + math.log(self.P[i])


# Class of LDF model
class LdfModel:
    def __init__(self, train_images, train_labels, U_reduce, K):
        self.train_images = train_images
        self.train_labels = train_labels
        self.U_reduce = U_reduce
        self.train_dim = self.U_reduce.T * self.train_images
        self.mu = np.mat(np.zeros((10, K), float))
        self.cov = np.mat(np.zeros((K, K), float))
        self.w = np.mat(np.zeros((10, K), float))
        self.w0 = np.mat(np.zeros((10, 1), float))
        self.P = np.zeros(10, float)
        self.K = K

    def cal_parameter(self):
        train_temp = self.train_dim - np.mean(self.train_dim, axis=1) * np.mat(np.ones(len(self.train_labels)))
        self.cov = (train_temp * train_temp.T) / len(self.train_labels)

        for i in range(10):
            num = np.where(self.train_labels == i)
            self.P[i] = len(num[0]) / float(len(self.train_labels))
            self.mu[i] = np.mean(self.train_dim[:, self.train_labels == i], axis=1).T
            self.w[i] = (np.linalg.inv(self.cov) * self.mu[i].T).T
            self.w0[i] = -0.5 * self.mu[i] * np.linalg.inv(self.cov) * self.mu[i].T + math.log(self.P[i])


# Test call
if __name__ == '__main__':
    LDF_accuracy, QDF_accuracy = learning()
    print 'LDF accuracy', LDF_accuracy
    print 'QDF accuracy', QDF_accuracy
