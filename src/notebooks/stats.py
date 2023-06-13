from scipy.stats import f,friedmanchisquare, rankdata
from math import sqrt
import numpy as np
import pandas as pd

def X_F_sqr(k,N,R):
    return ((12*N)/(k*(k+1)))*(np.sum(R**2)-(k*(k+1)**2)/4)

def F_F(k,N,X_F):
    return ((N-1)*X_F)/(N*(k-1)-X_F)

def critical_value(k, N, a=0.05):
    d1 = k - 1
    d2 = (k-1)*(N-1)
    return f.isf(a, d1, d2)

def cd(k,N,q_a):
    return q_a * sqrt((k*(k+1))/(6*N))

def main(df, a=0.01):
    # df = pd.read_csv('Results - friedman table.csv')
    df['classifier'] = df['classifier'] + ' ' +  df['features']
    df.drop(['features',], axis=1, inplace=True)
    df = df.T
    classifiers = df.loc['classifier'].values
    df.columns = classifiers
    df.drop('classifier', axis=0, inplace=True)
    df['Dataset'] = df.index.values
    df.reset_index(level=0, inplace=True)
    df.drop(['index',], axis=1, inplace=True)
    df = df[['Dataset',] + list(classifiers)]
    scores = df

    classifiers = list(set(scores.columns) - set(['Dataset']))
    scores_data = scores[list(scores.columns)[1:]].values

    # parameters
    k = scores_data.shape[1]
    N=scores_data.shape[0]
    #a = 0.01

    ranks = np.zeros(scores_data.shape)
    for i,scores_ in enumerate(scores_data):
        ranks[i] = len(scores_)+1 - rankdata(scores_)

    R = np.average(ranks, axis=0)

    X_F = X_F_sqr(k=k,N=N,R=R)
    print('k:', k, ' '*5, 'N:', N, ' '*5, 'a:', a)
    print('chi2: ', X_F)
    print("Friedman's F: ", F_F(k=k,N=N,X_F=X_F))
    print('F({},{})|{}: '.format(k-1,(k-1)*(N-1),a), critical_value(k=k,N=N, a=a))

    t = pd.DataFrame(columns=list(scores.columns)[1:], index=[0])
    t.loc[0] = R
    return t, t.T.sort_values(0)