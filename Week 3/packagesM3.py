# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 07:38:45 2017

@author: belle
"""
from numpy import linalg as LA
import pandas as pd
import sympy as sym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import sklearn
import scipy as sp
from scipy.optimize import fsolve #Check solution 
import math
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
from scipy.fftpack import fft,fftn,rfft 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor

def graph(formula, x_min,x_max,x_step):  
    x = np.linspace(x_min,x_max,x_step)
    #^ use x as input variable
    y = formula(x)
    #^          ^call the lambda expression with x
    #| use y as function result
    plt.plot(x,y)  
    plt.show() 
    
    
    
def graph3(formula, x_min,x_max, y_min,y_max,step):  
    x = np.outer(np.linspace(x_min, x_max, step), np.ones(step))
    y = (np.outer(np.linspace(y_min, y_max, step), np.ones(step))).T
    #^ use x,y  as input variable
    z = formula(x,y)
    #^           ^call the lambda expression with x,y
    #| use z as function result
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    ax.plot_surface(x,y,z, rstride=1, cstride=1, linewidth=0)
    plt.show() 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)   # not shown in the book
    plt.xlabel("Training set size", fontsize=14) # not shown
    plt.ylabel("RMSE", fontsize=14)              # not shown