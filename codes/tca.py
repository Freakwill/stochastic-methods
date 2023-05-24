#!/usr/bin/env python

"""
Transfer Component Analysis

*ref.*
王晋东 迁移学习导论
"""

import numpy as np
import numpy.linalg as LA
import scipy.linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


class TCA(BaseEstimator, TransformerMixin):
    def __init__(self, kernel_type='primal', repr_ndim=30, lamb=1, gamma=1):
        '''
        Init func
        param kernel_type: kernel, values: 'primal'|linear'| 'rbf'
        :param repr_ndim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma:kernel bandwidth for rbf kernel
        '''
        self.kernel_type = kernel_type
        self.repr_ndim = repr_ndim
        self.lamb = lamb
        self.gamma = gamma

    def fit_transform(self, Xs, Xt, normalized=False):
        '''
        Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return:Xs_new and Xt_new after TCA
        '''
        X = np.hstack((Xs.T, Xt.T))
        X /= LA.norm(X, axis=0)
        m, n = X.shape
        ns, nt=len(Xs), len(Xt)
        e = np.vstack((1/ns * np.ones((ns,1)), -1/nt * np.ones((nt,1))))
        M = e * e.T
        M = M / LA.norm(M,'fro')
        H = np.eye(n) - 1/n * np.ones((n, n))
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        a, b = (LA.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye),
               LA.multi_dot([K, H, K.T]))
        w, V = scipy.linalg.eig(a, b)
        ind = np.argsort(w)
        A = V[:, ind[:self.repr_ndim]]
        self.W = A
        Z = np.dot(A.T, K)
        if normalized:
            Z /= LA.norm(Z,axis=0)
        Xs_new, Xt_new = Z[:,:ns].T, Z[:,ns:].T
        return Xs_new, Xt_new

    def transform(self, X, normalized=False):
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        Z = np.dot(self.W.T, K)
        if normalized:
            Z /= LA.norm(Z,axis=0)
        return X_new

    def fit(self):
        self.fit_transform()
        return self


    def fit_predict(self, Xs, Ys, Xt):
        '''
        Torredictions Fe,then make predictions on targer nsing ll
        cparam Xs:ns*n_feature,source feature
        :param Ys:ns*1,source label
        param Xt:nt*n_feature,target feature
        '''
        Xs_new, Xt_new = self.fit_transform(Xs, Xt)
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(Xs_new, Ys.ravel())
        return clf.predict(Xt_new)



if __name__ == '__main__':
    
    from sklearn.datasets import load_digits

    X, y = load_digits(return_X_y=True)
    Xs = X[(y<5)]
    Xt = X[(y>=5)]
    size = 8, 8

    # from datasets import load_hanzi
    # Xs_Xt = load_hanzi(channel=0, ravel=True)
    # Xs, Xt = Xs_Xt[:100], Xs_Xt[100:]

    tca = TCA(repr_ndim=2)
    Xs_new, Xt_new = tca.fit_transform(Xs, Xt)
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title("Demo of Transfer Component Analysis")
    ax.set_xlim((-0.5, 0.5))
    ax.set_ylim((-0.5, 0.5))
    import random
    for xs, xt, zs, zt in zip(Xs, Xt, Xs_new, Xt_new):
        if random.random() < 0.06:
            c1, c2 = zs
            ax.imshow(xs.reshape(size), extent=(c1-0.03, c1+0.03, c2-0.03, c2+0.03), alpha=0.5, cmap='GnBu')
            c1, c2 = zt
            ax.imshow(xt.reshape(size), extent=(c1-0.03, c1+0.03, c2-0.03, c2+0.03), alpha=0.5, cmap='RdPu')

    plt.show()

