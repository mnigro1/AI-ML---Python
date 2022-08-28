#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import seaborn as sns
start_time = time.time()

from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.nn import functional as F
import pandas as pd
import scipy
import sys
from datetime import datetime
from dateutil import parser

df = pd.read_csv('/Users/matthewnigro/Documents/Fall 2020/MAE 600/Midterm Proj/MidtermProjectData.csv')

df = df.dropna()

df_workhours = df
df_offhours = df
df_weekdays = df
df_weekends = df

df_weekends['Hour'] = np.nan

for i in range(len(df_workhours) - 1, -1, -1):
    hour = parser.parse(df_workhours.iloc[i, 0])
    df_workhours.iloc[i, 10] = hour.strftime('%H')
    if 17 <= int(df_workhours['Hour'][i]) <= 8:
        df_workhours = df_workhours.drop(df_weekends.index[i])
    if int(df_workhours['Hour'][i]) == 0:
        df_workhours = df_workhours.drop(df_weekends.index[i])

for i in range(len(df_offhours) - 1, -1, -1):
    hour = parser.parse(df_offhours.iloc[i, 0])
    df_offhours.iloc[i, 10] = hour.strftime('%H')
    if 8 <= int(df_offhours['Hour'][i]) <= 17:
        df_offhours = df_offhours.drop(df_offhours.index[i])

df_workhours = df_workhours.drop(labels='Hour', axis=1)
df_offhours = df_offhours.drop(labels='Hour', axis=1)
df_weekends = df_weekends.drop(labels='Hour', axis=1)
df = df.drop(labels='Hour', axis=1)

# %% Splitting data into weekdays and weekends
df_offhours = df_offhours.reset_index()
df_workhours = df_workhours.reset_index()

df_workhours = df_workhours.drop(labels='index', axis=1)
df_offhours = df_offhours.drop(labels='index', axis=1)

for i in range(len(df_weekends) - 1, -1, -1):
    date = parser.parse(df_weekends.iloc[i, 0])
    df_weekends.iloc[i, 0] = date.strftime('%A')
    if df_weekends['Date/Time'][i] == 'Monday':
        df_weekends = df_weekends.drop(df_weekends.index[i])
    elif df_weekends['Date/Time'][i] == 'Tuesday':
        df_weekends = df_weekends.drop(df_weekends.index[i])
    elif df_weekends['Date/Time'][i] == 'Wednesday':
        df_weekends = df_weekends.drop(df_weekends.index[i])
    elif df_weekends['Date/Time'][i] == 'Thursday':
        df_weekends = df_weekends.drop(df_weekends.index[i])
    elif df_weekends['Date/Time'][i] == 'Friday':
        df_weekends = df_weekends.drop(df_weekends.index[i])

for i in range(len(df_workhours) - 1, -1, -1):
    date = parser.parse(df_workhours.iloc[i, 0])
    df_workhours.iloc[i, 0] = date.strftime('%A')
    if df_workhours['Date/Time'][i] == 'Saturday':
        df_workhours = df_workhours.drop(df_workhours.index[i])
    elif df_workhours['Date/Time'][i] == 'Sunday':
        df_workhours = df_workhours.drop(df_workhours.index[i])

for i in range(len(df_offhours) - 1, -1, -1):
    date = parser.parse(df_offhours.iloc[i, 0])
    df_offhours.iloc[i, 0] = date.strftime('%A')
    if df_offhours['Date/Time'][i] == 'Saturday':
        df_offhours = df_offhours.drop(df_offhours.index[i])
    elif df_offhours['Date/Time'][i] == 'Sunday':
        df_offhours = df_offhours.drop(df_offhours.index[i])

for i in range(len(df_weekdays) - 1, -1, -1):
    date = parser.parse(df_weekdays.iloc[i, 0])
    df_weekdays.iloc[i, 0] = date.strftime('%A')
    if df_weekdays['Date/Time'][i] == 'Saturday':
        df_weekdays = df_weekdays.drop(df_weekdays.index[i])
    elif df_weekdays['Date/Time'][i] == 'Sunday':
        df_weekdays = df_weekdays.drop(df_weekdays.index[i])

df_offhours = df_offhours.reset_index()
df_workhours = df_workhours.reset_index()
df_weekdays = df_weekdays.reset_index()

df_workhours = df_workhours.drop(labels='index', axis=1)
df_offhours = df_offhours.drop(labels='index', axis=1)
df_weekdays = df_weekdays.drop(labels='index', axis=1)

#%%Corellation matrices
plt.clf()
corrM_weekdays=abs(df_weekdays.corr())
ax = plt.axes()
figure=sns.heatmap(corrM_weekdays, annot=True, cmap="BuPu")
ax.set_title('Weekdays Correlation')
plt.show()

plt.clf()
ax = plt.axes()
corrM_weekends=abs(df_weekends.corr())
figure=sns.heatmap(corrM_weekends, annot=True, cmap="BuPu")
ax.set_title('Weekends Correlation')
plt.show()

plt.clf()
ax = plt.axes()
corrM_workhours=abs(df_workhours.corr())
figure=sns.heatmap(corrM_workhours, annot=True, cmap="BuPu")
ax.set_title('Workhours Correlation')
plt.show()

plt.clf()
ax = plt.axes()
corrM_workhours=abs(df.corr())
figure=sns.heatmap(corrM_workhours, annot=True, cmap="BuPu")
ax.set_title('Original DataFrame Correlation')
plt.show()


# %% Removing columns with low correlation
correlation_array = []
for i in range(1, len(np.transpose(df)) - 1):
    x = np.corrcoef(df.iloc[:, 9], df.iloc[:, i])
    correlation_array.append(x)

corr_values = []
for i in range(0, len(correlation_array)):
    x = abs(correlation_array[i][1][0])
    corr_values.append(x)

max_coer = (max(corr_values))
min_coer = (min(corr_values))
print(
    "The highest correlation is with the variable Solar Radiation, W/m^2, SATC Rooftop with a value of " + str(max_coer) \
    + "\nThe lowest correlation is with Temperature, *C, SATC Rooftop"     'with a value of ' + str(min_coer))




df_weekdays = df_weekdays.drop(labels='Date/Time', axis=1)
df_weekdays = df_weekdays.drop(labels='Temperature, *C, SATC Rooftop', axis=1)
df_weekdays = df_weekdays.drop(labels='Dew Point, *C, SATC Rooftop', axis=1)
df_weekdays = df_weekdays.drop(labels='Wind Direction, *, SATC Rooftop', axis=1)
df_weekdays = df_weekdays.drop(labels='Pressure, mbar, SATC Rooftop', axis=1)

df_weekends = df_weekends.drop(labels='Date/Time', axis=1)
df_weekends = df_weekends.drop(labels='Temperature, *C, SATC Rooftop', axis=1)
df_weekends = df_weekends.drop(labels='Dew Point, *C, SATC Rooftop', axis=1)
df_weekends = df_weekends.drop(labels='Wind Direction, *, SATC Rooftop', axis=1)
df_weekends = df_weekends.drop(labels='Pressure, mbar, SATC Rooftop', axis=1)

df_workhours = df_workhours.drop(labels='Date/Time', axis=1)
df_workhours = df_workhours.drop(labels='Temperature, *C, SATC Rooftop', axis=1)
df_workhours = df_workhours.drop(labels='Dew Point, *C, SATC Rooftop', axis=1)
df_workhours = df_workhours.drop(labels='Wind Direction, *, SATC Rooftop', axis=1)
df_workhours = df_workhours.drop(labels='Pressure, mbar, SATC Rooftop', axis=1)

df_offhours = df_offhours.drop(labels='Date/Time', axis=1)
df_offhours = df_offhours.drop(labels='Temperature, *C, SATC Rooftop', axis=1)
df_offhours = df_offhours.drop(labels='Dew Point, *C, SATC Rooftop', axis=1)
df_offhours = df_offhours.drop(labels='Wind Direction, *, SATC Rooftop', axis=1)
df_offhours = df_offhours.drop(labels='Pressure, mbar, SATC Rooftop', axis=1)



#%%Corellation matrices
plt.clf()
corrM_weekdays=abs(df_weekdays.corr())
ax = plt.axes()
figure=sns.heatmap(corrM_weekdays, annot=True, cmap="BuPu")
ax.set_title('Weekdays Correlation After Dropping Columns')
plt.show()

plt.clf()
ax = plt.axes()
corrM_weekends=abs(df_weekends.corr())
figure=sns.heatmap(corrM_weekends, annot=True, cmap="BuPu")
ax.set_title('Weekends Correlation After Dropping Columns')
plt.show()

plt.clf()
ax = plt.axes()
corrM_workhours=abs(df_workhours.corr())
figure=sns.heatmap(corrM_workhours, annot=True, cmap="BuPu")
ax.set_title('Workhours Correlation After Dropping Columns')
plt.show()


# %%Averaging every four points - Hourly Time Series
df_weekdays = df_weekdays.groupby(np.arange(len(df_weekdays)) // 4).mean()
df_weekends = df_weekends.groupby(np.arange(len(df_weekends)) // 4).mean()
df_workhours = df_workhours.groupby(np.arange(len(df_workhours)) // 4).mean()
df_offhours = df_offhours.groupby(np.arange(len(df_offhours)) // 4).mean()




# %% Identifying outliers
# Removing Outliers
def outliers(dataframe):
    for i in range(0, len(np.transpose(dataframe))):
        abs_z_scores = abs(scipy.stats.zscore(dataframe.iloc[:, i]))
        abs_z_scores = abs_z_scores.reshape(-1, 1)
        filtered_entries = (abs_z_scores < 3).all(axis=1)
        dataframe = dataframe[filtered_entries]

    outliers_weekdays = []
    for i in range(0, len(np.transpose(dataframe))):
        abs_z_scores = abs(scipy.stats.zscore(dataframe.iloc[:, i]))
        abs_z_scores = abs_z_scores.reshape(-1, 1)
        filtered_entries = (abs_z_scores < 4).all(axis=1)
        num_of_true = np.count_nonzero(filtered_entries)
        num_of_outliers = len(filtered_entries) - num_of_true
        outliers_weekdays.append(num_of_outliers)
    return dataframe


df_weekdays = outliers((df_weekdays))
df_weekends = outliers(df_weekends)
df_workhours = outliers(df_workhours)
df_offhours = outliers(df_offhours)


#%%
def correlation(dataframe, frame):
    correlation_array = []
    for i in range(0, len(np.transpose(dataframe)) - 1):
        x = np.corrcoef(dataframe.iloc[:, 5], dataframe.iloc[:, i])
        correlation_array.append(x)

    corr_values = []
    for i in range(1, len(correlation_array)):
        x = abs(correlation_array[i][1][0])
        corr_values.append(x)

    max_coer = (max(corr_values))
    min_coer = (min(corr_values))
    print(
        frame + " The highest correlation is with the variable Solar Radiation, W/m^2, SATC Rooftop with a value of " + str(
            max_coer) \
        + "\nThe lowest correlation is with Temperature, *C, SATC Rooftop"     'with a value of ' + str(min_coer))
    return corr_values


off_hours_correlation = correlation(df_offhours, 'off hours')
workhours_correlation = correlation(df_workhours, 'work hours')
weekends_correlation = correlation(df_weekends, 'weekends')
weekdays_correlation = correlation(df_weekdays, 'weekdays')







#%%Corellation matrices
plt.clf()
corrM_weekdays=abs(df_weekdays.corr())
ax = plt.axes()
figure=sns.heatmap(corrM_weekdays, annot=True, cmap="BuPu")
ax.set_title('Weekdays Final Correlation Matrix')
plt.show()

plt.clf()
ax = plt.axes()
corrM_weekends=abs(df_weekends.corr())
figure=sns.heatmap(corrM_weekends, annot=True, cmap="BuPu")
ax.set_title('Weekends Final Correlation Matrix')
plt.show()

plt.clf()
ax = plt.axes()
corrM_workhours=abs(df_workhours.corr())
figure=sns.heatmap(corrM_workhours, annot=True, cmap="BuPu")
ax.set_title('Workhours Correlation With Outliers Removed')
plt.show()




########################################################################
######################################################################
###############################################################
##########################################################
####           Week days         ###
X=df_weekdays.values[:,:5]
Y=df_weekdays.values[:,5:]
X_train, X_test = X[:6696, :5], X[6696:, :5]
Y_train, Y_test = Y[:6696], Y[6696:]

#X_train, X_test, y_train, y_test = train_test_split(
#...     X, y, test_size=0.33, random_state=42)

def plot_results(Y_test, y_predicted):
    figx = plt.figure()
    plt.plot(range(len(Y_test)), Y_test, 'go', label='True data', alpha=0.5)
    plt.plot(range(len(Y_test)), y_predicted, '-*', label='Predictions', alpha=0.7)
    plt.legend(loc='best')
    return figx
def error_metrics(Y_test, y_predicted):
    # RMSE, MAPE, MAE, MBE, R2
    rmse = np.sqrt(np.mean((y_predicted - Y_test) ** 2))
    mape = np.mean(np.abs(y_predicted - Y_test) / Y_test)
    mae = np.mean(np.abs(y_predicted - Y_test))
    mbe = np.mean(y_predicted - Y_test)
    r2 = np.corrcoef(y_predicted.squeeze(), Y_test.squeeze())[0, 1]**2
    return rmse, mape, mae, mbe, r2

#########Models
#%%
# K-fold cross-validation
from sklearn.model_selection import KFold
def kfold_ridge_lasso(X, Y, K, epochs,ridge,lasso):
    
    e = epochs
    r = ridge
    l = lasso
    
    kf = KFold(n_splits=K, shuffle=False)
    rmse_trn_cv, rmse_tst_cv = np.empty(0), np.empty(0)
    r2_trn_cv, r2_tst_cv = np.empty(0), np.empty(0)
    i=0
    for trn_idx, tst_idx in kf.split(X):
        X_train, X_test = X[trn_idx, :], X[tst_idx, :]
        Y_train, Y_test = Y[trn_idx], Y[tst_idx]

        modelK = train(X_train=X_train, Y_train=Y_train, epochs = e, ridge = r, lasso = l)
        with torch.no_grad():
            yhat_trn = modelK(torch.from_numpy(X_train).float()).numpy()
            yhat_tst = modelK(torch.from_numpy(X_test).float()).numpy()

        rmse_trn = np.sqrt(np.mean((yhat_trn - Y_train) ** 2))
        rmse_tst = np.sqrt(np.mean((yhat_tst - Y_test) ** 2))

        rmse_trn_cv = np.append(rmse_trn_cv, rmse_trn)
        rmse_tst_cv = np.append(rmse_tst_cv, rmse_tst)

        r2_trn = np.corrcoef(yhat_trn.squeeze(), Y_train.squeeze())[0, 1]**2
        r2_tst = np.corrcoef(yhat_tst.squeeze(), Y_test.squeeze())[0, 1]**2

        r2_trn_cv = np.append(r2_trn_cv, r2_trn)
        r2_trn_cv = r2_trn_cv[~np.isnan(r2_trn_cv)]
        
        r2_tst_cv = np.append(r2_tst_cv, r2_tst)
        r2_tst_cv = r2_tst_cv[~np.isnan(r2_tst_cv)]
        
        fig_train = plot_results(Y_train, yhat_trn)
        i+=1
        fig_train.suptitle('Training plot fold ' + str(i))
        fig_test = plot_results(Y_test, yhat_tst)
        fig_test.suptitle('Test plot fold ' + str(i))
        
    return rmse_trn_cv.mean(), rmse_tst_cv.mean(),r2_trn_cv.mean(), r2_tst_cv.mean(), fig_train, fig_test


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


def train(X_train, Y_train, epochs, ridge=None, lasso=None):
    model = linearRegression(X_train.shape[1], Y_train.shape[1])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    if ridge:
        optimizer.param_groups[0]['weight_decay'] = 5

    for epoch in range(epochs):
        inputs = torch.from_numpy(X_train).float()
        labels = torch.from_numpy(Y_train).float()

        optimizer.zero_grad()
        outputs = model(inputs)

        if lasso:
            l1_norm = torch.norm(model.linear.weight, p=1)
        else:
            l1_norm = 0
        loss = criterion(outputs, labels)
        loss += l1_norm
        loss.backward()

        optimizer.step()
        print('epoch {}: loss = {}'.format(epoch, loss.item()))

    return model
#%%

model1 = train(X_train=X_train, Y_train=Y_train, epochs=10000)
with torch.no_grad():
    y_predicted = model1(torch.from_numpy(X_test).float())
    y_train_predicted = model1(torch.from_numpy(X_train).float())

def plot_results(Y_test, y_predicted):
        figx = plt.figure()
        plt.plot(range(len(Y_test)), Y_test, 'go', label='True data', alpha=0.5)
        plt.plot(range(len(Y_test)), y_predicted, '-*', label='Predictions', alpha=0.7)
        plt.legend(loc='best')
        return figx
fig_LR = plot_results(Y_test, y_predicted)
rmse1, mape1, mae1, mbe1, r21 = error_metrics(Y_test, y_predicted.numpy())

rmse_trn_val, rmse_tst_val, r2_trn_val, r2_tst_val, fig_train, fig_test = kfold_ridge_lasso(X_train, Y_train, 5, 10000,None,None)

#%%Ridge
model2 = train(X_train=X_train, Y_train=Y_train, epochs=10000, ridge=True)
with torch.no_grad():
    y_predicted2 = model2(torch.from_numpy(X_test).float())

print(model2.linear.weight.detach())
print(model2.linear.bias.detach())

fig_ridge = plot_results(Y_test, y_predicted2)
rmse2, mape2, mae2, mbe2, r22 = error_metrics(Y_test, y_predicted2.numpy())


rmse_trn_val, rmse_tst_val, r2_trn_val, r2_tst_val, fig_train, fig_test = kfold_ridge_lasso(X_train, Y_train, 5, 10000,True,None)


sys.exit()

#%%


#Lasso

model3 = train(X_train=X_train, Y_train=Y_train, epochs=80000, lasso=True)
with torch.no_grad():
    y_predicted3 = model3(torch.from_numpy(X_test).float())

print(model3.linear.weight.detach())
print(model3.linear.bias.detach())

fig_lasso = plot_results(Y_test, y_predicted3)
rmse3, mape3, mae3, mbe3, r23 = error_metrics(Y_test, y_predicted3.numpy())

rmse_trn_val, rmse_tst_val, r2_trn_val, r2_tst_val, fig_train, fig_test = kfold_ridge_lasso(X_train, Y_train, 5, 10000,True,True)

#%%Model: Elastic Net
from sklearn.linear_model import ElasticNet

en = ElasticNet(alpha=1.0, l1_ratio=0.8)
# fit model
model=en.fit(X_train, Y_train)
y_predicted5=torch.from_numpy(model.predict(X_test))

rmse5, mape5, mae5, mbe5, r25 = error_metrics(Y_test, y_predicted5.numpy())
fig_el=plot_results(Y_test,y_predicted5)


def kfold_elastic_net(X, Y, K):
    
    kf = KFold(n_splits=K, shuffle=False)
    rmse_trn_cv, rmse_tst_cv = np.empty(0), np.empty(0)
    r2_trn_cv, r2_tst_cv = np.empty(0), np.empty(0)
    i=0
    for trn_idx, tst_idx in kf.split(X):
        X_train, X_test = X[trn_idx, :], X[tst_idx, :]
        Y_train, Y_test = Y[trn_idx], Y[tst_idx]

        modelK = en.fit(X_train, Y_train)
        with torch.no_grad():
            yhat_trn = torch.from_numpy(modelK.predict(X_train)).numpy()
            yhat_tst = torch.from_numpy(modelK.predict(X_test)).numpy()
           # yhat_tst = modelK(torch.from_numpy(X_test).float()).numpy()

        rmse_trn = np.sqrt(np.mean((yhat_trn - Y_train) ** 2))
        rmse_tst = np.sqrt(np.mean((yhat_tst - Y_test) ** 2))

        rmse_trn_cv = np.append(rmse_trn_cv, rmse_trn)
        rmse_tst_cv = np.append(rmse_tst_cv, rmse_tst)

        r2_trn = np.corrcoef(yhat_trn.squeeze(), Y_train.squeeze())[0, 1]**2
        r2_tst = np.corrcoef(yhat_tst.squeeze(), Y_test.squeeze())[0, 1]**2

        r2_trn_cv = np.append(r2_trn_cv, r2_trn)
        r2_trn_cv = r2_trn_cv[~np.isnan(r2_trn_cv)]
        
        r2_tst_cv = np.append(r2_tst_cv, r2_tst)
        r2_tst_cv = r2_tst_cv[~np.isnan(r2_tst_cv)]
        
        fig_train = plot_results(Y_train, yhat_trn)
        i+=1
        fig_train.suptitle('Training plot fold ' + str(i))
        fig_test = plot_results(Y_test, yhat_tst)
        fig_test.suptitle('Test plot fold ' + str(i))
        
    return rmse_trn_cv.mean(), rmse_tst_cv.mean(),r2_trn_cv.mean(), r2_tst_cv.mean(), fig_train, fig_test

rmse_trn_val, rmse_tst_val, r2_trn_val, r2_tst_val, fig_train, fig_test = kfold_elastic_net(X_train, Y_train, 5)
#%%
########################################################################
######################################################################
###############################################################
##########################################################
####           Weekends         ###
X=df_weekends.values[:,:5]
Y=df_weekends.values[:,5:]
X_train, X_test = X[:2685, :5], X[2685:, :5]
Y_train, Y_test = Y[:2685], Y[2685:]

#X_train, X_test, y_train, y_test = train_test_split(
#...     X, y, test_size=0.33, random_state=42)

def plot_results(Y_test, y_predicted):
    figx = plt.figure()
    plt.plot(range(len(Y_test)), Y_test, 'go', label='True data', alpha=0.5)
    plt.plot(range(len(Y_test)), y_predicted, '-*', label='Predictions', alpha=0.7)
    plt.legend(loc='best')
    return figx
def error_metrics(Y_test, y_predicted):
    # RMSE, MAPE, MAE, MBE, R2
    rmse = np.sqrt(np.mean((y_predicted - Y_test) ** 2))
    mape = np.mean(np.abs(y_predicted - Y_test) / Y_test)
    mae = np.mean(np.abs(y_predicted - Y_test))
    mbe = np.mean(y_predicted - Y_test)
    r2 = np.corrcoef(y_predicted.squeeze(), Y_test.squeeze())[0, 1]**2
    return rmse, mape, mae, mbe, r2

#########Models



class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize):
        super(linearRegression, self).__init__()
        self.linear = torch.nn.Linear(inputSize, outputSize)

    def forward(self, x):
        out = self.linear(x)
        return out


def train(X_train, Y_train, epochs, ridge=None, lasso=None):
    model = linearRegression(X_train.shape[1], Y_train.shape[1])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    if ridge:
        optimizer.param_groups[0]['weight_decay'] = 5

    for epoch in range(epochs):
        inputs = torch.from_numpy(X_train).float()
        labels = torch.from_numpy(Y_train).float()

        optimizer.zero_grad()
        outputs = model(inputs)

        if lasso:
            l1_norm = torch.norm(model.linear.weight, p=1)
        else:
            l1_norm = 0
        loss = criterion(outputs, labels)
        loss += l1_norm
        loss.backward()

        optimizer.step()
        print('epoch {}: loss = {}'.format(epoch, loss.item()))

    return model


model1 = train(X_train=X_train, Y_train=Y_train, epochs=80000)
with torch.no_grad():
    y_predicted = model1(torch.from_numpy(X_test).float())
    y_train_predicted = model1(torch.from_numpy(X_train).float())

def plot_results(Y_test, y_predicted):
        figx = plt.figure()
        plt.plot(range(len(Y_test)), Y_test, 'go', label='True data', alpha=0.5)
        plt.plot(range(len(Y_test)), y_predicted, '-*', label='Predictions', alpha=0.7)
        plt.legend(loc='best')
        return figx
fig_LR = plot_results(Y_test, y_predicted)
rmse1, mape1, mae1, mbe1, r21 = error_metrics(Y_test, y_predicted.numpy())



#Ridge
model2 = train(X_train=X_train, Y_train=Y_train, epochs=80000, ridge=True)
with torch.no_grad():
    y_predicted2 = model2(torch.from_numpy(X_test).float())

print(model2.linear.weight.detach())
print(model2.linear.bias.detach())

fig_ridge = plot_results(Y_test, y_predicted2)
rmse2, mape2, mae2, mbe2, r22 = error_metrics(Y_test, y_predicted2.numpy())



#Lasso

model3 = train(X_train=X_train, Y_train=Y_train, epochs=80000, lasso=True)
with torch.no_grad():
    y_predicted3 = model3(torch.from_numpy(X_test).float())

print(model3.linear.weight.detach())
print(model3.linear.bias.detach())

fig_lasso = plot_results(Y_test, y_predicted3)
rmse3, mape3, mae3, mbe3, r23 = error_metrics(Y_test, y_predicted3.numpy())



##Model: Elastic Net
from sklearn.linear_model import ElasticNet

en = ElasticNet(alpha=1.0, l1_ratio=0.8)
# fit model
model=en.fit(X_train, Y_train)
y_predicted5=torch.from_numpy(model.predict(X_test))

rmse5, mape5, mae5, mbe5, r25 = error_metrics(Y_test, y_predicted5.numpy())
fig_el=plot_results(Y_test,y_predicted5)



#%% Neural Network
# Define a network
data = df_weekdays.drop(labels='Meter Reading, W', axis=1)
target = df_weekdays.iloc[:,5]

data = np.asarray(data)
target = np.asarray(target)

data = data.reshape(-1,5)
target = target.reshape(-1,1)

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H,H)
        self.linear3 = torch.nn.Linear(H, D_out)
        
    def forward(self, x):
        h_relu = self.linear1(x).clamp(min=0)
        h_relu = self.linear2(h_relu)
        y_pred = self.linear3(h_relu)
        return y_pred

# Training function
def train(X_train, Y_train, H, learning_rate, epochs=20000):
    model = TwoLayerNet(X_train.shape[1], H, Y_train.shape[1])
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        inputs = torch.from_numpy(X_train).float()
        labels = torch.from_numpy(Y_train).float()

        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

    return model


# K-fold cross-validation
from sklearn.model_selection import KFold
def kfold_CV(X, Y, K, H, learning_rate):
    hidden = H
    lr = learning_rate

    kf = KFold(n_splits=K, shuffle=True)
    rmse_trn_cv, rmse_tst_cv = np.empty(0), np.empty(0)
    r2_trn_cv, r2_tst_cv = np.empty(0), np.empty(0)
    i=0
    for trn_idx, tst_idx in kf.split(X):
        X_train, X_test = X[trn_idx, :], X[tst_idx, :]
        Y_train, Y_test = Y[trn_idx], Y[tst_idx]

        modelK = train(X_train=X_train, Y_train=Y_train, H=hidden, learning_rate=lr)
        with torch.no_grad():
            yhat_trn = modelK(torch.from_numpy(X_train).float()).numpy()
            yhat_tst = modelK(torch.from_numpy(X_test).float()).numpy()

        rmse_trn = np.sqrt(np.mean((yhat_trn - Y_train) ** 2))
        rmse_tst = np.sqrt(np.mean((yhat_tst - Y_test) ** 2))

        rmse_trn_cv = np.append(rmse_trn_cv, rmse_trn)
        rmse_tst_cv = np.append(rmse_tst_cv, rmse_tst)

        r2_trn = np.corrcoef(yhat_trn.squeeze(), Y_train.squeeze())[0, 1]**2
        r2_tst = np.corrcoef(yhat_tst.squeeze(), Y_test.squeeze())[0, 1]**2

        r2_trn_cv = np.append(r2_trn_cv, r2_trn)
        r2_trn_cv = r2_trn_cv[~np.isnan(r2_trn_cv)]
        
        r2_tst_cv = np.append(r2_tst_cv, r2_tst)
        r2_tst_cv = r2_tst_cv[~np.isnan(r2_tst_cv)]
        
        fig_train = plot_results(Y_train, yhat_trn)
        i+=1
        fig_train.suptitle('Training plot fold ' + str(i))
        fig_test = plot_results(Y_test, yhat_tst)
        fig_test.suptitle('Test plot fold ' + str(i))
        
    return rmse_trn_cv.mean(), rmse_tst_cv.mean(),r2_trn_cv.mean(), r2_tst_cv.mean(), fig_train, fig_test, X_train, Y_train, r2_trn_cv

def kfold_CV_noplot(X, Y, K, H, learning_rate):
    hidden = H
    lr = learning_rate

    kf = KFold(n_splits=K, shuffle=True)
    rmse_trn_cv, rmse_tst_cv = np.empty(0), np.empty(0)
    r2_trn_cv, r2_tst_cv = np.empty(0), np.empty(0)
    #i=0
    for trn_idx, tst_idx in kf.split(X):
        X_train, X_test = X[trn_idx, :], X[tst_idx, :]
        Y_train, Y_test = Y[trn_idx], Y[tst_idx]

        modelK = train(X_train=X_train, Y_train=Y_train, H=hidden, learning_rate=lr)
        with torch.no_grad():
            yhat_trn = modelK(torch.from_numpy(X_train).float()).numpy()
            yhat_tst = modelK(torch.from_numpy(X_test).float()).numpy()

        rmse_trn = np.sqrt(np.mean((yhat_trn - Y_train) ** 2))
        rmse_tst = np.sqrt(np.mean((yhat_tst - Y_test) ** 2))

        rmse_trn_cv = np.append(rmse_trn_cv, rmse_trn)
        rmse_tst_cv = np.append(rmse_tst_cv, rmse_tst)

        r2_trn = np.corrcoef(yhat_trn.squeeze(), Y_train.squeeze())[0, 1]**2
        r2_tst = np.corrcoef(yhat_tst.squeeze(), Y_test.squeeze())[0, 1]**2

        r2_trn_cv = np.append(r2_trn_cv, r2_trn)
        r2_trn_cv = r2_trn_cv[~np.isnan(r2_trn_cv)]
        
        r2_tst_cv = np.append(r2_tst_cv, r2_tst)
        r2_tst_cv = r2_tst_cv[~np.isnan(r2_tst_cv)]
        #fig_train = plot_results(Y_train, yhat_trn)
        #i+=1
        #fig_train.suptitle('Training plot fold ' + str(i))
        #fig_test = plot_results(Y_test, yhat_tst)
        #fig_test.suptitle('Test plot fold ' + str(i))
        
    return rmse_trn_cv.mean(), rmse_tst_cv.mean(),r2_trn_cv.mean(), r2_tst_cv.mean()


#Plot results 
def plot_results(Y_test, y_predicted):
    figx = plt.figure()
    plt.plot(range(len(Y_test)), Y_test, 'go', label='True data', alpha=0.5)
    plt.plot(range(len(Y_test)), y_predicted, '--', label='Predictions', alpha=0.5)
    plt.legend(loc='best')
    return figx



H = 11
lr = 0.001
rmse_trn_val, rmse_tst_val, r2_trn_val, r2_tst_val, fig_train, fig_test,X_train,Y_train, r2_trn_cv = kfold_CV(data, target, 5, H, lr)
print('H = {}, lr = {}: Training RMSE = {}, Testing RMSE = {}'.format(H, lr, rmse_trn_val, rmse_tst_val))
print('H = {}, lr = {}: Training R2 = {}, Testing R2 = {}'.format(H, lr, r2_trn_val, r2_tst_val))


# Hyperparameter tuning: Grid Search
H_list = list(range(1,20))
lr_list = [1e-4, 1e-3, 1e-2, 1e-1]

rmse_trn = np.zeros((len(H_list), len(lr_list)))
rmse_tst = np.zeros_like(rmse_trn)
r2_trn = np.zeros((len(H_list), len(lr_list)))
r2_tst = np.zeros_like(rmse_trn)
for h, H in enumerate(H_list):
    for l, lr in enumerate(lr_list):
        rmse_trn_val, rmse_tst_val, r2_trn_val, r2_trn_val = kfold_CV_noplot(data, target, 5, H, lr)
        rmse_trn[h, l] = rmse_trn_val
        rmse_tst[h, l] = rmse_tst_val
        r2_trn[h,l] = r2_trn_val
        r2_tst[h,l] = r2_tst_val
        
        print('H = {}, lr = {}: Training RMSE = {}, Testing RMSE = {}'.format(H, lr, rmse_trn_val, rmse_tst_val))
       # print('H = {}, lr = {}: Training R2 = {}, Testing R2 = {}'.format(H, lr, r2_trn_val, r2_tst_val))
        
i, j = np.argwhere(rmse_tst == np.min(rmse_tst))[0]
h_best, lr_best = H_list[i], lr_list[j]



print('H best = {}, lr best = {}'.format(h_best, lr_best))


#%% RNN & LSTM Analysis


import time
start_time = time.time()
from sklearn import datasets
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.nn import functional as F
import pandas as pd
import scipy
import sys
from datetime import datetime
from dateutil import parser
import numpy as np
from matplotlib import pyplot as plt


# Generate data

data = df_workhours.iloc[:,5]
data = np.asarray(data)
data = data[0:1000]
# Split into test/train
test_size = 40

data_train = data[:-test_size]
data_test = data[-test_size:]

plt.figure()
plt.plot(range(len(data)-test_size), data_train, label='train')
plt.plot(range(len(data)-test_size-1, len(data)-1), data_test, label='test')
plt.legend()

# Normalize data & convert to tensor
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
data_train_norm = scaler.fit_transform(data_train.reshape(-1, 1))
plt.figure()
plt.plot(data_train_norm)

data_train_norm = torch.FloatTensor(data_train_norm) #.view(-1)


# Create sequences of (input_data, output_data) tuples
def make_sequences(input_data, sl):
    inout_seq = []
    L = len(input_data)
    for i in range(L-sl):
        train_seq = input_data[i:i+sl]
        train_label = input_data[i+sl:i+sl+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq


# Set a training sequence length
seq_len = 40

training_sequences = make_sequences(data_train_norm, seq_len)
len(training_sequences)
#training_sequences[:5]  # What does a "training sequence" look like?


# RNN model
class RNN(torch.nn.Module):
    def __init__(self, D_in, D_hidden, D_out, nLayers=1):
        super().__init__()
        self.D_hidden = D_hidden
        self.nLayers = nLayers
        self.rnn = torch.nn.RNN(D_in, D_hidden, num_layers=nLayers)
        self.linear = torch.nn.Linear(D_hidden, D_out)
        self.hidden = torch.zeros(nLayers, 1, D_hidden)

    def forward(self, input_seq):
        rnn_out, self.hidden = self.rnn(input_seq.view(len(input_seq), 1, -1), self.hidden)
        y_pred = self.linear(rnn_out.view(len(input_seq), -1))
        return y_pred[-1]


# GRU model
class GRU(torch.nn.Module):
    def __init__(self, D_in, D_hidden, D_out, nLayers=1):
        super().__init__()
        self.D_hidden = D_hidden
        self.nLayers = nLayers
        self.gru = torch.nn.GRU(D_in, D_hidden, num_layers=nLayers)
        self.linear = torch.nn.Linear(D_hidden, D_out)
        self.hidden = torch.zeros(nLayers, 1, D_hidden)

    def forward(self, input_seq):
        gru_out, self.hidden = self.gru(input_seq.view(len(input_seq), 1, -1), self.hidden)
        y_pred = self.linear(gru_out.view(len(input_seq), -1))
        return y_pred[-1]


# LSTM model
class LSTM(torch.nn.Module):
    def __init__(self, D_in, D_hidden, D_out, nLayers=1):
        super().__init__()
        self.D_hidden = D_hidden
        self.nLayers = nLayers
        self.lstm = torch.nn.LSTM(D_in, D_hidden, num_layers=nLayers)
        self.linear = torch.nn.Linear(D_hidden, D_out)
        self.hidden = (torch.zeros(nLayers, 1, D_hidden),
                       torch.zeros(nLayers, 1, D_hidden))

    def forward(self, input_seq):
        lstm_out, self.hidden = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden)
        y_pred = self.linear(lstm_out.view(len(input_seq), -1))
        return y_pred[-1]


# Training function
def train(training_sequences, rnn_type, num_layers, learning_rate, epochs=9000):
    D_in = training_sequences[0][0].shape[1]
    D_out = training_sequences[0][1].shape[1]
    D_hidden = 50

    if rnn_type.upper() == 'RNN':
        model = RNN(D_in, D_hidden, D_out, nLayers=num_layers)
    elif rnn_type.upper() == 'GRU':
        model = GRU(D_in, D_hidden, D_out, nLayers=num_layers)
    elif rnn_type.upper() == 'LSTM':
        model = LSTM(D_in, D_hidden, D_out, nLayers=num_layers)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epochs-1,epochs):
        for seq, labels in training_sequences:

            optimizer.zero_grad()

            if rnn_type.upper() == 'LSTM':
                model.hidden = (torch.zeros(model.nLayers, 1, model.D_hidden),
                                torch.zeros(model.nLayers, 1, model.D_hidden))
            else:
                model.hidden = torch.zeros(model.nLayers, 1, model.D_hidden)

            outputs = model(seq)
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

        #if epoch % 1 == 0:
            #print('epoch {}: loss = {}'.format(epoch, loss.item()))

    return model


# Testing function
def test(model, rnn_type, test_inputs, pred_len):
    model.eval()
    for i in range(pred_len):
        test_seq = torch.FloatTensor(test_inputs[-seq_len:])
        with torch.no_grad():
            if rnn_type.upper() == 'LSTM':
                model.hidden = (torch.zeros(model.nLayers, 1, model.D_hidden),
                                torch.zeros(model.nLayers, 1, model.D_hidden))
            else:
                model.hidden = torch.zeros(model.nLayers, 1, model.D_hidden)

            test_inputs.append(model(test_seq))

    # print(np.array(test_inputs[pred_len:]))
    return test_inputs[pred_len:]


# Train, test model
rnn_type = 'rnn'
number_layers = 4
lr = 0.0001
model = train(training_sequences, rnn_type, number_layers,lr)

pred_len = test_size
test_inputs = data_train_norm[-seq_len:].tolist()
y_pred_unscaled = test(model, rnn_type, test_inputs, pred_len)

# Inverse scaling
y_pred = scaler.inverse_transform(np.array(y_pred_unscaled).reshape(-1, 1))
print(y_pred)

# RMSE, MAPE
sys.exit()
data_test = data_test.reshape(-1, 1)

rmse = np.sqrt(np.mean((y_pred - data_test) ** 2))
mape = np.mean(np.abs((y_pred - data_test) / data_test))
r2 = np.corrcoef(y_pred.squeeze(), data_test.squeeze())[0, 1]**2
print(rmse)
# Plotting
plt.figure()
plt.plot(data, label='actual', linewidth=2)
plt.plot(range(len(data)-test_size, len(data)), y_pred, '-o', markersize=7, label='predictions')
plt.legend(loc='best', fontsize=14)
plt.xlabel('Time step', fontsize=14)
plt.ylabel('Y-value', fontsize=14)
plt.xlim(0,1080)
plt.title('{} Method: RMSE = {}, MAPE = {}, R2 = {}'.format(rnn_type.upper(), round(rmse, 3), round(100*mape, 1), round(r2,3)), fontsize=14)
plt.show()


#
rnn_type = 'lstm'
number_layers = 1
lr = 0.005
model = train(training_sequences, rnn_type, number_layers,lr)

pred_len = test_size
test_inputs = data_train_norm[-seq_len:].tolist()
y_pred_unscaled = test(model, rnn_type, test_inputs, pred_len)

# Inverse scaling
y_pred = scaler.inverse_transform(np.array(y_pred_unscaled).reshape(-1, 1))
print(y_pred)

# RMSE, MAPE

data_test = data_test.reshape(-1, 1)

rmse = np.sqrt(np.mean((y_pred - data_test) ** 2))
mape = np.mean(np.abs((y_pred - data_test) / data_test))
r2 = np.corrcoef(y_pred.squeeze(), data_test.squeeze())[0, 1]**2
print(rmse)
# Plotting
plt.figure()
plt.plot(data, label='actual', linewidth=2)
plt.plot(range(len(data)-test_size, len(data)), y_pred, '-o', markersize=7, label='predictions')
plt.legend(loc='best', fontsize=14)
plt.xlabel('Time step', fontsize=14)
plt.ylabel('Y-value', fontsize=14)
plt.xlim(0,1080)
plt.title('{} Method: RMSE = {}, MAPE = {}%, R2 = {}'.format(rnn_type.upper(), round(rmse, 3), round(100*mape, 1), round(r2,3)), fontsize=14)
plt.show()

sys.exit()
# Hyperparameter tuning: Grid Search
num_layers_list = list(range(1,5))
lr_list = [1e-4, 1e-3, 1e-2, 1e-1]

rmse_trn = np.zeros((len(num_layers_list), len(num_layers_list)))
rmse_tst = np.zeros_like(rmse_trn)
r2_trn = np.zeros((len(num_layers_list), len(num_layers_list)))
r2_tst = np.zeros_like(rmse_trn)
for h, H in enumerate(num_layers_list):
    for l, lr in enumerate(lr_list):
        model = train(training_sequences, rnn_type, H, lr)
        pred_len = test_size
        test_inputs = data_train_norm[-seq_len:].tolist()
        y_pred_unscaled = test(model, rnn_type, test_inputs, pred_len)

        # Inverse scaling
        y_pred = scaler.inverse_transform(np.array(y_pred_unscaled).reshape(-1, 1))


        # RMSE, MAPE
        data_test = data_test.reshape(-1, 1)
        rmse = np.sqrt(np.mean((y_pred - data_test) ** 2))
        mape = np.mean(np.abs((y_pred - data_test) / data_test))

        
        print('num_layers = {}, lr = {}: RMSE ={}'.format(H, lr, rmse))
       # print('H = {}, lr = {}: Training R2 = {}, Testing R2 = {}'.format(H, lr, r2_trn_val, r2_tst_val))
        
i, j = np.argwhere(rmse_tst == np.min(rmse_tst))[0]
h_best, lr_best = num_layers_list[i], lr_list[j]



print('H best = {}, lr best = {}'.format(h_best, lr_best))








print("--- %s seconds ---" % (time.time() - start_time))


