# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 11:17:09 2020

@author: Giovanni
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm # barra de progresso 
from sklearn.utils import shuffle
from keras.utils import to_categorical
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
#from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn import preprocessing 
# mlutil só uns módulos com funções que usa no treino, só pra poder chamar 
# lê, dropa as colunas, transforma tudo em float, normaliza 
# transforma para dados de teste para depois poder rodar só o teste se precisar, separadamente 

#df = dataframe, size = numero de instancias de cada classe
def split(data, size):
	d1 = data.loc[lambda data: data['STATUS2'] == 1]
	data = data.loc[lambda data: data['STATUS2'] == 0]

	data = shuffle(data, random_state = 7)
	df = data[:size]
	data = data[size:]
	y_train1 = df['STATUS2']
	x_train1 = df.drop(['STATUS2'], axis = 1)
	y_test1 = data['STATUS2']
	x_test1 = data.drop(['STATUS2'], axis = 1)

	d1 = shuffle(d1, random_state = 7)
	d2 = d1[size:]
	d1 = d1[:size]
	y_train2 = d1['STATUS2']
	x_train2 = d1.drop(['STATUS2'], axis = 1)
	y_test2 = d2['STATUS2']
	x_test2 = d2.drop(['STATUS2'], axis = 1)

	x_train1 = x_train1.values
	x_train2 = x_train2.values
	y_train1 = y_train1.values
	y_train2 = y_train2.values
	x_test1 = x_test1.values
	x_test2 = x_test2.values
	y_test1 = y_test1.values
	y_test2 = y_test2.values

	x_train = np.concatenate((x_train1, x_train2))
	x_test = np.concatenate((x_test1, x_test2))
	y_train = np.concatenate((y_train1, y_train2))
	y_test = np.concatenate((y_test1, y_test2))

	y_train = y_train.reshape(-1, 1)
	y_test = y_test.reshape(-1, 1)

	x_train, y_train = shuffle(x_train, y_train, random_state = 1)
	x_test, y_test = shuffle(x_test, y_test, random_state = 1)

	return x_train, y_train, x_test, y_test

def split2x(data, size):
	d1 = data.loc[lambda data: data['STATUS2'] == 0]
	data = data.loc[lambda data: data['STATUS2'] == 1]

	if len(d1) > len(data):
		d2 = d1
		d1 = data
		data = d2

	data = shuffle(data, random_state = 10)
	df = data[:size*2]
	data = data[size*2:]
	y_train1 = df['STATUS2']
	x_train1 = df.drop(['STATUS2'], axis = 1)
	y_test1 = data['STATUS2']
	x_test1 = data.drop(['STATUS2'], axis = 1)

	d1 = shuffle(d1, random_state = 10)
	d2 = d1[size:]
	d1 = d1[:size]
	y_train2 = d1['STATUS2']
	x_train2 = d1.drop(['STATUS2'], axis = 1)
	y_test2 = d2['STATUS2']
	x_test2 = d2.drop(['STATUS2'], axis = 1)

	x_train1 = x_train1.values
	x_train2 = x_train2.values
	y_train1 = y_train1.values
	y_train2 = y_train2.values
	x_test1 = x_test1.values
	x_test2 = x_test2.values
	y_test1 = y_test1.values
	y_test2 = y_test2.values

	x_train = np.concatenate((x_train1, x_train2))
	x_test = np.concatenate((x_test1, x_test2))
	y_train = np.concatenate((y_train1, y_train2))
	y_test = np.concatenate((y_test1, y_test2))

	y_train = y_train.reshape(-1, 1)
	y_test = y_test.reshape(-1, 1)

	x_train, y_train = shuffle(x_train, y_train, random_state = 1)
	x_test, y_test = shuffle(x_test, y_test, random_state = 1)

	return x_train, y_train, x_test, y_test

#model = modelo salvo em .h5 ou .hdf5
def modelTest(model, x_train, y_train, x_test, y_test):
	if len(y_train[0]) < 2: # Separa os dados em treinamento e teste 
		y_train = to_categorical(y_train, num_classes = 2)
		y_test = to_categorical(y_test, num_classes = 2)
	print(y_train[0], y_test[0])

	print('No train:')
	counter1 = 0
	for i in tqdm(range(len(y_train))):
	  if np.argmax(y_train[i]) == 0:
	    counter1 = counter1 + 1
	    
	print("aprovados: ", counter1)
	print("negados: ", len(x_train) - counter1)
		# Passa da minoritária. Você vai fazer um eda para saber quantos são negados e quantos são aprovados 
		# Normalmente, negado é sempre minoria, então é preciso balancear isso de alguma forma 
	print('\nNo teste:')
	counter2 = 0
	for i in tqdm(range(len(y_test))):
	  if np.argmax(y_test[i]) == 0:
	    counter2 = counter2 + 1
	    
	print("aprovados: ", counter2)
	print("negados: ", len(x_test) - counter2)

	aprovado_cnt, negado_cnt, aprovado_correct, negado_correct = 0, 0, 0, 0
	print('\nTeste no x_train:')
	print('Numero de aprovados:', counter1, '\tNumero de negados:', len(x_train) - counter1)

	predict = model.predict(x_train)

	aprovado_cnt, negado_cnt, aprovado_correct, negado_correct = 0, 0, 0, 0
	for x in range(len(predict)):
	  if predict[x] == np.argmax(y_train[x]):
	    if np.argmax(y_train[x]) == 1:
	      negado_correct += 1
	    else:
	      aprovado_correct += 1
	       
	  if np.argmax(y_train[x]) == 1:
	    negado_cnt += 1
	  else:
	    aprovado_cnt += 1

	#loss_train, acc_train = model.evaluate(x_train, y_train) ###
	print('\nAcuracia total no treino:', acc_train) #, '\tloss:', loss_train)
	print("aprovado_acc", aprovado_correct/aprovado_cnt*100, "%", '\tAcertos:', aprovado_correct, ' de ', aprovado_cnt)
	print("negado_acc", negado_correct/negado_cnt*100, "%", '\tAcertos:', negado_correct, ' de ', negado_cnt)


	print('\n\nTeste no x_test:')
	print('Numero de aprovados:', counter2, '\tNumero de negados:', len(x_test) - counter2)

	predict = model.predict(x_test)

	aprovado_cnt, negado_cnt, aprovado_correct, negado_correct = 0, 0, 0, 0 
	for x in range(len(predict)):
	  if predict[x] == np.argmax(y_test[x]):
	    if np.argmax(y_test[x]) == 1:
	      negado_correct += 1
	    else:
	      aprovado_correct += 1
	       
	  if np.argmax(y_test[x]) == 1:
	    negado_cnt += 1
	  else:
	    aprovado_cnt += 1

	#loss_test, acc_test = model.evaluate(x_test, y_test) ### 
	print('\nAcuracia total no teste:', acc_test) #, '\tloss:', loss_test)
	print("aprovado_acc", aprovado_correct/aprovado_cnt*100, "%", '\tAcertos:', aprovado_correct, ' de ', aprovado_cnt)
	print("negado_acc", negado_correct/negado_cnt*100, "%", '\tAcertos:', negado_correct, ' de ', negado_cnt, '\n\n')

def split3x(data, size):
	d1 = data.loc[lambda data: data['STATUS2'] == 0]
	data = data.loc[lambda data: data['STATUS2'] == 1]

	if len(d1) > len(data):
		d2 = d1
		d1 = data
		data = d2

	data = shuffle(data, random_state = 1)
	df = data[:size*3]
	data = data[size*3:]
	y_train1 = df['STATUS2']
	x_train1 = df.drop(['STATUS2'], axis = 1)
	y_test1 = data['STATUS2']
	x_test1 = data.drop(['STATUS2'], axis = 1)

	d1 = shuffle(d1, random_state = 1)
	d2 = d1[size:]
	d1 = d1[:size]
	y_train2 = d1['STATUS2']
	x_train2 = d1.drop(['STATUS2'], axis = 1)
	y_test2 = d2['STATUS2']
	x_test2 = d2.drop(['STATUS2'], axis = 1)

	x_train1 = x_train1.values
	x_train2 = x_train2.values
	y_train1 = y_train1.values
	y_train2 = y_train2.values
	x_test1 = x_test1.values
	x_test2 = x_test2.values
	y_test1 = y_test1.values
	y_test2 = y_test2.values

	x_train = np.concatenate((x_train1, x_train2))
	x_test = np.concatenate((x_test1, x_test2))
	y_train = np.concatenate((y_train1, y_train2))
	y_test = np.concatenate((y_test1, y_test2))

	y_train = y_train.reshape(-1, 1)
	y_test = y_test.reshape(-1, 1)

	x_train, y_train = shuffle(x_train, y_train, random_state = 1)
	x_test, y_test = shuffle(x_test, y_test, random_state = 1)

	return x_train, y_train, x_test, y_test

def split4x(data, size):
	d1 = data.loc[lambda data: data['STATUS2'] == 0]
	data = data.loc[lambda data: data['STATUS2'] == 1]

	if len(d1) > len(data):
		d2 = d1
		d1 = data
		data = d2

	data = shuffle(data, random_state = 1)
	df = data[:size*4]
	data = data[size*4:]
    
    #print(df['STATUS2'].values)
    
	y_train1 = df['STATUS2']
	x_train1 = df.drop(['STATUS2'], axis = 1)
	y_test1 = data['STATUS2']
	x_test1 = data.drop(['STATUS2'], axis = 1)

	d1 = shuffle(d1, random_state = 7)
	d2 = d1[size:]
	d1 = d1[:size]
	y_train2 = d1['STATUS2']
	x_train2 = d1.drop(['STATUS2'], axis = 1)
	y_test2 = d2['STATUS2']
	x_test2 = d2.drop(['STATUS2'], axis = 1)

	x_train1 = x_train1.values
	x_train2 = x_train2.values
	y_train1 = y_train1.values
	y_train2 = y_train2.values
	x_test1 = x_test1.values
	x_test2 = x_test2.values
	y_test1 = y_test1.values
	y_test2 = y_test2.values

	x_train = np.concatenate((x_train1, x_train2))
	x_test = np.concatenate((x_test1, x_test2))
	y_train = np.concatenate((y_train1, y_train2))
	y_test = np.concatenate((y_test1, y_test2))

	y_train = y_train.reshape(-1, 1)
	y_test = y_test.reshape(-1, 1)

	x_train, y_train = shuffle(x_train, y_train, random_state = 1)
	x_test, y_test = shuffle(x_test, y_test, random_state = 1)

	return x_train, y_train, x_test, y_test


def modelTestAllData(model, x_test, y_test):

	if len(y_test[0]) < 2:
		#y_train = to_categorical(y_train, num_classes = 2)
		y_test = to_categorical(y_test, num_classes = 2)
	#print(y_train[0], y_test[0])

	print('\nNo teste:')
	counter2 = 0
	for i in tqdm(range(len(y_test))):
	  if np.argmax(y_test[i]) == 0:
	    counter2 = counter2 + 1
	    
	print("aprovados: ", counter2)
	print("negados: ", len(x_test) - counter2)

	predict = model.predict(x_test)

	aprovado_cnt, negado_cnt, aprovado_correct, negado_correct = 0, 0, 0, 0
	for x in range(len(predict)):
	  if predict[x] == np.argmax(y_test[x]):
	    if np.argmax(y_test[x]) == 1:
	      negado_correct += 1
	    else:
	      aprovado_correct += 1
	       
	  if np.argmax(y_test[x]) == 1:
	    negado_cnt += 1
	  else:
	    aprovado_cnt += 1

	#loss_test, acc_test = model.evaluate(x_test, y_test)
	#print('\nAcuracia total no teste:', acc_test) #, '\tloss:', loss_test)
	print("aprovado_acc", aprovado_correct/aprovado_cnt*100, "%", '\tAcertos:', aprovado_correct, ' de ', aprovado_cnt)
	print("negado_acc", negado_correct/negado_cnt*100, "%", '\tAcertos:', negado_correct, ' de ', negado_cnt, '\n\n')

def normalize_data(data, attribute):
	attribute_list = data[attribute].tolist()

	min_value = data[attribute].min()
	max_value = data[attribute].max()

	neg_max = data[attribute].min()

	if min_value < 0:
		min_value = 0
	
	for i in range(len(data)):
		if data[attribute][i] >= 0:
			data.at[i, attribute] = (data[attribute].iloc[i] - min_value) / (max_value - min_value)
		else:
			data.at[i, attribute] = (-(abs(data[attribute].iloc[i])) / (abs(neg_max)))

	data.loc[:, attribute] = data.loc[:, attribute].apply(lambda x: "%.5f" % x)

	return data#, max_value, min_value

def pred_prob(data):
    if data[0,0] > data[0,1]:
        maior = 0
    else: 
        maior = 1
    
    return maior 

def gridSearch(x_test, x_train, y_test, y_train):
    param_grid = {'C': [1, 5, 10, 20, 30, 40, 50],
             'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]}
    grid = GridSearchCV(estimator=SVC(kernel='rbf'), param_grid=param_grid, cv=5)
    
    grid.fit(x_train, y_train)
    print(grid.best_params_)

    y_out = grid.predict(x_test) 
    y_in = grid.predict(x_train)
    
    print(classification_report(y_train, y_in))
    print(classification_report(y_test, y_out))
    

def trainModel(x_test, x_train, y_test, y_train, C_, gamma_):
    
    model = SVC(kernel='rbf', C=C_, gamma=gamma_)
    model.fit(x_train, y_train)

    model.get_params()
    y_pred = model.predict(x_test) 
    y_predtrain = model.predict(x_train)

    print(classification_report(y_test, y_pred))

    acc_test = accuracy_score(y_test,y_pred)
    acc_train = accuracy_score(y_train,y_predtrain)
    
    print("acuracia treino: " + str(acc_train))
    print("acuracia teste: " + str(acc_test))
    
    print("Numero de VS: " + str(model.n_support_))
    #print(model.support_)
    nVS = sum(model.n_support_)
    print(nVS)
    
    N = len(x_test) + len(x_train)
    
    E_out = (nVS / N) * 100
    print("[E_out] =" + str(E_out) + "%")
    
    
def K_Means(X_, Y):
    X = np.array(X_)
    #Y = np.array(Y_)
    
    #normalizeX(X)
    
    kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    Y_clustered = kmeans.fit_predict(X)
    
    print(confusion_matrix(Y, Y_clustered))
    
    
    i = 0
    not_outliers = []
    for i in range(len(X)):
        if Y_clustered[i] == Y[i]: 
            not_outliers.append(i)

    return not_outliers

def split_(x, y):
    
    x0 = []
    x1 = []
    for i in range(len(y)):
        if y[i] == 1:
            x1.append(x[i])
        else: 
            x0.append(x[i])
            
    prop = len(x0) / len(x1)
    
    size0 = int(len(x0) * ((prop - 1) / prop))
    size1 = int(len(x1) * ((prop - 1) / prop))
    
    x0 = shuffle(x0, random_state = 1)
    x_train0 = x0[:size0]
    x_test0 = x0[size0:]
    y_train0 = [0] * len(x_train0)
    y_test0 = [0] * len(x_test0)
    
    x1 = shuffle(x1, random_state=1)
    x_train1 = x1[:size1]
    x_test1 = x1[size1:]
    y_train1 = [1] * len(x_train1)
    y_test1 = [1] * len(x_test1)

    x_train = np.concatenate((x_train0, x_train1))
    x_test = np.concatenate((x_test0, x_test1))
    y_train = np.concatenate((y_train0, y_train1))
    y_test = np.concatenate((y_test0, y_test1))


    return x_train, y_train, x_test, y_test

def normalizeX(X):
    
    col = [0,1,2,4,19,20,24,25,26]
    
    #MinMax = MinMaxScaler() 
    
    for c in col:
        attribute_list = [[]]
        
        for i in range(len(X)):
            attribute_list[0].append(X[i][c])
            
        #print(attribute_list)
        #scaled = MinMax.fit_transform(attribute_list)[0]
        scaled = preprocessing.normalize(attribute_list)[0]
        #print(scaled)
        for i in range(len(X)):
            X[i][c] = scaled[i]
    

    
    