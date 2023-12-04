#import warnings
#warnings.filterwarnings(action='ignore', category=UserWarning)
import random
random.seed(42)

import numpy as np
import pandas as pd


def get_data(data):
    data.dropna(inplace=True)
    X = data.iloc[:, :-1]
    X.columns = range(X.shape[1])
    y = data.iloc[:, -1]
    y = pd.DataFrame(y)
    y.columns = [0]
    return X, y

from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from lightgbm import LGBMRegressor
from sklearn.svm import SVR

class Evaluate:

    def __init__(self, X, y, estimator):
        
        
        self.X = X
        self.y = y
        self.estimator = estimator


        # split the data, creating a group of training/validation sets to be used in the k-fold validation process:
        #self.kfold = KFold(n_splits=3)

        pipeline = Pipeline([("scaler", StandardScaler()),
                             ("model", self.estimator)])
        self.model = TransformedTargetRegressor(regressor=pipeline, transformer=StandardScaler())
        self.RKFold = RepeatedKFold(n_splits=5, n_repeats=2, random_state=42)

    def __len__(self):
        """
        :return: the total number of features used in this Regression problem
        """
        return self.X.shape[1]
    
    def ant_accuracy(self, ant):
        currentX = self.X.iloc[:, ant.X]
        cv_results = cross_val_score(self.model, currentX, self.y, cv=self.RKFold, scoring='r2', n_jobs=-1)
        return np.mean(cv_results)
    def colony_accuracy(self, colony):
        currentX = self.X.iloc[:, colony]
        cv_results = cross_val_score(self.model, currentX, self.y, cv=self.RKFold, scoring='r2', n_jobs=-1, error_score="raise")
        return np.mean(cv_results)
    
    


class Ant():
    def __init__(self):
        self.X = list()
        self.features = list()
    
    def add(self, feature):
        if feature % 2 == 1:
            self.X.append((feature - 1) / 2) 
        self.features.append(feature)
        
class Colony:
    
    def __init__(self):
        self.colony = list()
        self.ant_features = list()
    def colony_add(self, ant):
        self.colony.append(ant.X)
        self.ant_features.append(ant.features)
        
    def best(self, colony, eval_class, hof):
        cv_result = list()
        for i in range(len(colony.colony)):
            cv_result.append(eval_class.colony_accuracy(colony.colony[i]))
        indices = sorted(range(len(cv_result)), key=lambda i: cv_result[i])[-hof:]
        indices.reverse()
        r2_indices = np.array(cv_result)[indices]
        return indices, r2_indices, cv_result
    
    def best_add(self, ant):
        self.colony.append(ant)
        list_of_features = list()
        for i in ant:
            feature = i * 2 + 1
            list_of_features.append(int(feature))
        self.ant_features.append(list_of_features)
    
def map_features(n_f):
    a = np.ones(shape=(n_f*2, n_f*2))
    for i in range(len(a)):
        a[i][i] = 0
        if i % 2 == 0 :
            a[i][i+1] = 0
        if i % 2 == 1 :
            a[i][i-1] = 0
    return pd.DataFrame(a)

def corr(n_f, data):
    corr = abs(data.corr()).values
    a = map_features(n_f)
    for i in range(len(a)):
        if i % 2 == 0:
            for j in range(a.shape[1]):
                if j % 2 == 0:
                    a.values[i][j] = corr[int(i/2)][int(j/2)]
                if j % 2 == 1:
                    a.values[i][j] = 1 - corr[int(i/2)][int((j-1)/2)]
        if i % 2 == 1:
            for j in range(a.shape[1]):
                if j % 2 == 0:
                    a.values[i][j] = 1 - corr[int((i-1)/2)][int(j/2)]
                if j % 2 == 1:
                    a.values[i][j] = corr[int((i-1)/2)][int((j-1)/2)]
    for i in range(len(a)):
        a[i][i] = 0
        if i % 2 == 0 :
            a.values[i][i+1] = 0
        if i % 2 == 1 :
            a.values[i][i-1] = 0
    return a

def pheromone(n_f, n_iter=0, pher_before=None, pher_change=None, ro=0.5):
    if n_iter == 0:
        a = map_features(n_f)
        return a
    else:
        return (pher_before * (1 - ro) + pher_change)
def pher_change(n_f, ant_features_best, r2_best):
    a = np.zeros(shape=(n_f*2, n_f*2))
    for i in ant_features_best:
        features = ant_features_best
        features.remove(i)
        a[i, [features]] = r2_best
        a[[features], i] = r2_best
    return pd.DataFrame(a)
def pher_update(pher, low=0.1, High=50):
    pher[pher < low] = 0
    return pher
def pher_empty(n_f):
    a = np.zeros(shape=(n_f*2, n_f*2))
    return a


def select_for_initial_point(ant, features, initial_point, n_f, eta, pher, alpha, beta):
    available_nodes = [i for i in range(features.shape[1])]
    if initial_point % 2 == 0:
        #b = features.iloc[initial_point: initial_point + 2, :]
        b_pher = pher.iloc[initial_point: initial_point + 2, :]
        b_eta = eta.iloc[initial_point: initial_point + 2, :]
        sigma = np.sum(np.sum(np.multiply(b_pher, b_eta)))
        probability = list((np.multiply(b_pher, b_eta) / sigma).values.flatten())
        c = np.random.choice([i for i, j in enumerate(probability)], 1, p=probability)[0]
        available_nodes.remove(initial_point)
        available_nodes.remove(initial_point+1)
        features.iloc[initial_point: initial_point + 2, :] = 0
        features.iloc[:, initial_point: initial_point + 2] = 0
        if c < n_f * 2:
            ant.add(initial_point)
            ant.add(c)
        if c >= n_f * 2:
            c = c - (2 * n_f)
            ant.add(initial_point+1)
            ant.add(c)
        return c, available_nodes, features
    if initial_point % 2 == 1:
        #b = features.iloc[initial_point - 1 : initial_point + 1, :]
        b_pher = pher.iloc[initial_point - 1 :initial_point + 1, :]
        b_eta = eta.iloc[initial_point - 1 : initial_point + 1, :]
        sigma = np.sum(np.sum(np.multiply(b_pher, b_eta)))
        probability = list((np.multiply(b_pher, b_eta) / sigma).values.flatten())
        c = np.random.choice([i for i, j in enumerate(probability)], 1, p=probability)[0]
        available_nodes.remove(initial_point)
        available_nodes.remove(initial_point-1)
        features.iloc[initial_point - 1 : initial_point + 1, :] = 0
        features.iloc[:, initial_point - 1 : initial_point + 1] = 0
        if c < n_f * 2:
            ant.add(initial_point-1)
            ant.add(c)
        if c >= n_f * 2 :
            c = c - (2 * n_f)
            ant.add(initial_point)
            ant.add(c)
        return c, available_nodes, features
    
    
def select_next_feature(features, n_f, ant=None, pher=None, eta=None, alpha=0.5, beta=0.5):
    initial_point = np.random.randint(low=0, high=n_f * 2)
    a = features.copy()
    pher = np.power(pher, alpha)
    eta = np.power(eta, beta)
    next_point, available_nodes, a = select_for_initial_point(ant, a, initial_point, n_f, eta, pher, alpha, beta)
    counter = 1
    while counter < n_f / 2  : # n_f -1
        b = a.iloc[next_point: next_point + 1, :]
        b_pher = pher.iloc[next_point: next_point + 1, :]
        b_eta = eta.iloc[next_point: next_point + 1, :]
        b_eta = np.multiply(b_eta, b)
        sigma = np.sum(np.sum(np.multiply(b_pher, b_eta)))
        probability = list((np.multiply(b_pher, b_eta) / sigma).values.flatten())
        c = np.random.choice([i for i, j in enumerate(probability)], 1, p=probability)[0]
        if next_point % 2 == 0:
#             available_nodes.remove(next_point)
#             available_nodes.remove(next_point+1)
            a.iloc[next_point: next_point + 2, :] = 0
            a.iloc[:, next_point: next_point + 2] = 0
        if next_point % 2 == 1:
#             available_nodes.remove(next_point)
#             available_nodes.remove(next_point-1)
            a.iloc[next_point - 1 : next_point + 1, :] = 0
            a.iloc[:, next_point - 1 : next_point + 1] = 0
        next_point = c
        ant.add(next_point)
        counter += 1
        
def stats(eval_class, colony):
    scores = list()
    for i in range(len(colony.colony)):
        scores.append(eval_class.colony_accuracy(colony.colony[i]))
    return scores

