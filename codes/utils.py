
import numpy as np
import scipy.special as ss
import pandas as pd
from sklearn.preprocessing import FunctionTransformer


def expit(x, lb=0, ub=255):
    # inverse of logit
    return np.uint8(np.round((ub-lb+1) * ss.expit(x) + lb - 0.5))

def logit(x, lb=0, ub=255):
    # [lb, ub] -> (-oo, +oo)
    return ss.logit((x-lb+0.5) / (ub-lb+1))


def normalize(X, axis=1):
    # normalize a matrix
    if X.ndim == 1:
        return X / X.sum()
    else:
        if axis == 1:
            return X / X.sum(axis=1)[:,None]
        elif axis == 0:
            return X / X.sum(axis=0)
        else:
            raise ValueError('`axis` must be 0 or 1!')

def exp_normalize(X, axis=1):
    return normalize(np.exp(X), axis=axis)


class LogitTransformer(FunctionTransformer):
    def __init__(self, lb=0, ub=256, *args, **kwargs):
        super().__init__(func=_logit, inverse_func=_expit, *args, **kwargs)


def compare(models:dict, X_train, X_test, y_train, y_test, timeit=False, predict=False):
    """Compare the performance of models
    models: machine learning models
    """
    
    from collections import defaultdict
    import time

    _result = defaultdict(list)

    for name, model in models.items():
        time1 = time.perf_counter()
        model.fit(X_train, y_train)
        _result['models'].append(name)
        if timeit:
            time2 = time.perf_counter()
            _result['times'].append(time2 - time1)
        _result['train scores'].append(model.score(X_train, y_train))
        _result['test scores'].append(model.score(X_test, y_test))
        if predict:
            _result[f'predict={y_test[0]}'].append(model.predict(X_test[:1])[0])
    result = pd.DataFrame(_result)
    print(result.round(decimals=4).to_markdown())


def visualize(axes, model, X, y=None, x1lim=None, x2lim=None, N1=200, N2=200,
    label1='$x_1$', label2='$x_2$',
    markers = ('o', '+', 'x', 's', 'v', '>', '<', '^', 'd'), 
    colors = ('r', 'b', 'g', 'y', 'm', 'c', 'k'),
    boundary = False,
    boundary_kw={},
    backgroud = True,
    backgroud_kw={'alpha': 0.1, 'marker':'x'}, scatter_kw={}):
    """To visulize the result of classification

    make sure that the number of classes <= 7

    model: classifier
    """

    X = np.asarray(X)
    if y is None:
        model.fit(X)
        y = model.predict(X)
        labels = np.unique(y)
    else:
        model.fit(X, y)
        labels = model.classes_

    # draw the background/show the decision boundary
    if backgroud:
        if x1lim is None:
            l, u = X[:,0].min(), X[:,0].max()
            d = (u-l)*0.02
            x1lim = l-d, u+d
        if x2lim is None:
            l, u = X[:,1].min(), X[:,1].max()
            d = (u-l)*0.02
            x2lim = l-d, u+d
        x1 = np.linspace(*x1lim, N1)
        x2 = np.linspace(*x2lim, N2)
        x2_, x1_ = np.meshgrid(x2, x1)
        X_ = np.column_stack((x1_.ravel(), x2_.ravel()))
        y_ = model.predict(X_)
        for k, c in zip(labels, colors):
            axes.scatter(X_[y_==k, 0], X_[y_==k, 1], c=c, **backgroud_kw)
        Y_ = y_.reshape(N1, N2)

    # draw boundary
    if boundary:
        # x_ = np.array([(x1i, x2j) for i, x1i in enumerate(x1) for j, x2j in enumerate(x2)
        #      if (i < N1-1 and Y_[i,j] != Y_[i+1,j]) or (j<N2-1 and Y_[i,j] != Y_[i, j+1])])
        x_ = []
        for i, x1i in enumerate(x1):
            for j, x2j in enumerate(x2):
                if i < N1-1 and Y_[i,j] != Y_[i+1,j]:
                    if (j<N2-1 and Y_[i,j] != Y_[i, j+1]):
                        x_.append(((x1i+x1[i+1])/2, (x2j+x2[j+1])/2))
                    else:
                        x_.append(((x1i+x1[i+1])/2, x2j))
                else:
                    if j<N2-1 and Y_[i,j] != Y_[i, j+1]:
                        x_.append((x1i, (x2j+x2[j+1])/2))
        x_ = np.asarray(x_)

        axes.scatter(x_[:,0], x_[:,1], marker='.', c='k', **boundary_kw)

    # draw data points
    handles = [axes.scatter(X[y==k, 0], X[y==k, 1], c=c, marker=m, **scatter_kw)
            for k, m, c in zip(labels, markers, colors)]
    axes.set_xlabel(label1)
    axes.set_ylabel(label2)
    return handles


def show_on_plain(ax, X=None, Z=None, y=None, n=20, alpha=0.4, color='black', *args, **kwargs):
    """Embedding the images X to the 2D coordinates Z
    
    Args:
        ax: axes
        X (2d-array, 3d-array): the images
        Z (array): 2 X n_samples, the coordinates of X in the 2D latent sp.
        n (int, optional): number of images to be shown in the axes
        y: labels
    
    Returns:
        TYPE: Description
    """
    
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox

    if n > 0:
        N = len(X)
        ind = np.random.choice(N, n)
        X1 = X[ind]
        Z1 = Z[ind]
        for x, z in zip(X1, Z1):
            imagebox = OffsetImage(np.uint8(x), zoom=0.9)
            ab = AnnotationBbox(imagebox, z, xybox=(0, 0), xycoords='data', boxcoords="offset points", frameon=False)
            ax.add_artist(ab)

    # ax.set_xlabel(xlabel)
    # ax.set_ylabel(ylabel)
    if y is None:
        ax.scatter(Z[:,0], Z[:,1], alpha=alpha, color=color, *args, **kwargs)
    else:
        show_crd(ax, Z, y)
    return ax


def show_crd(ax, Z, y):
    """plot the crds
    
    Args:
        ax: axex
        Z (array): coords
        y_ (array): labels of the coords
    """
    
    labels = np.unique(y)
    for k, c, m in zip(labels, 'rbgymkc', 'o+x*sd^v<>'):
        ax.scatter(Z[y==k][:,0], Z[y==k][:,1], color=c, marker=m)


def show_crd_cluster(ax, Z, y, centers=None, *args, **kwargs):
    # plot the crds after clustering
    labels = np.unique(y)
    for k, c in zip(labels, 'rbgymkc'):
        ax.scatter(Z[y==k][:,0], Z[y==k][:,1], color=c, marker='o', *args, **kwargs)
    if centers is None: return
    for k, c, m in zip(centers, 'rbgymkc', '+x*sd^v<>'):
        ax.scatter(k[0], k[1], color=c, marker=m)


def show_pc(ax, X, Z, components_, center=np.zeros(2), scale=0.1, scatter_kw={},
    arrow_kw={}, text1='$v_1$', text2='$v_2$', line=True):
    # show PC1 and PC2 with two arrows
    default_scatter_kw = {'alpha':0.7, 'color':'grey', 's':8}
    scatter_kw = default_scatter_kw | scatter_kw
    default_arrow_kw = {'color':'k', 'head_length':0.02, 'alpha':0.5, 'width':0.002}
    arrow_kw = default_arrow_kw | arrow_kw

    if line:
        t = np.linspace(Z[:,0].min(),Z[:,0].max(),100)
        ax.plot(components_[0,0]*t+center[0], components_[0,1]*t+center[1], '-', alpha=0.9)
        t = np.linspace(Z[:,1].min(),Z[:,1].max(),100)
        ax.plot(components_[1,0]*t+center[0], components_[1,1]*t+center[1], '--', alpha=0.9)
    ax.scatter(X[:,0], X[:,1], **scatter_kw)
    ax.arrow(*center, *(components_[0] * scale), **arrow_kw)
    ax.arrow(*center, *(components_[1] * scale), **arrow_kw)
    ax.text(*(components_[0] * scale + center), text1, va='bottom', ha='right')
    ax.text(*(components_[1] * scale + center), text2, va='bottom', ha='left')


from sklearn.base import BaseEstimator, TransformerMixin

class BaseEncoder(TransformerMixin, BaseEstimator):
    def encode(self, X):
        return self.transform(X)

    def decode(self, X):
        return self.inverse_transform(X)

from sklearn.neural_network import MLPRegressor
from sklearn.utils.extmath import safe_sparse_dot

def inplace_relu(X):
    np.maximum(X, 0, out=X)

class MLPEncoder(BaseEncoder, MLPRegressor):
    def __init__(n_components=2, *args, **kwargs):
        super().__init__(hidden_layer_sizes=(n_components,), *args, **kwargs)

    def transform(self, X):
        hidden_activation = inplace_relu
        activation = X
        activation = safe_sparse_dot(activation, self.coefs_[0]) + intercepts_[0]
        hidden_activation(activation)
        return activation

    def inverse_transform(self, Y):
        activation = Y
        activation = safe_sparse_dot(activation, self.coefs_[1]) + intercepts_[1]
        # output_activation(activation)
        return activation

    def fit(self, X):
        return super().fit(X, X)


def make_grid(X, h, w, r, c, v=0):
    for k in range(1, r):
        X = np.insert(X, h*k, v, axis=0)
    for k in range(1, c):
        X = np.insert(X, w*k, v, axis=1)
    return X


def show_crd(ax, Z, y, colors='rbgymkc', markers='o+x*sd^v<>', centers=None, *args, **kwargs):
    """plot the crds
    
    Args:
        ax: axex
        Z (array): coords
        y_ (array): labels of the coords
    """
    
    labels = np.unique(y)
    for k, c, m in zip(labels, colors, markers):
        ax.scatter(Z[y==k][:,0], Z[y==k][:,1], color=c, marker=m, *args, **kwargs)

    if centers is None: return
    for cc, c, m in zip(centers, colors, markers):
        ax.scatter(cc[0], cc[1], color='k', marker=m)


def minimize_matrix(f, X0, *args, **kwargs):
    from scipy.optimize import minimize
    size = X0.shape
    def _f(x):
        return f(x.reshape(size))
    return minimize(_f, np.ravel(X0), *args, **kwargs).x.reshape(size)
