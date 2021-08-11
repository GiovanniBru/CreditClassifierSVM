"""
Created on Fri Dec 18 20:01:21 2020

@author: Giovanni
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from keras.utils import to_categorical
from mlutil2 import split, split2x, modelTest, split3x, split4x, normalize_data, modelTestAllData
import tensorflow.compat.v1 as tf
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import pickle 
tf.disable_v2_behavior()
#from sklearn.model_selection import GridSearchCV
#from keras import optimizers
#from keras.models import load_model
#from sklearn.utils import shuffle
#from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt

# Carregando dados: 
data_path = "data_train_final1.csv"
data = pd.read_csv(data_path, encoding = "utf-8")
print(len(data), data.head())

# Colunas com correlação mais baixa que atrapalham a performance do modelo
dropcolumns = [ 'NEGADO ANTERIORMENTE',
 'ATRASO 60 DIAS',
 'CLIENTE EM ATRASO',
 'REGRA - NEGADO ANTERIORMENTE',
 'REGRA - RENOV. COMP. RESIDEN',
 'REGRA - FINANCIAMENTO P/ AUT',
 'REGRA - CLIENTE EM ATRASO',
 'REGRA - REGRA MOVEIS',
 'RESIDENCIA CEDIDA',
 'STATUS COMPROVANTE RESIDENCIA']
#dropcolumns = ['COD_FILIAL']
try:
  data = data.drop(dropcolumns, axis = 1)
except Exception as e:
    print(e)

#Movendo as colunas: 
data = data[["STATUS2", "COD_FILIAL", "VALOR", "EXP. PROFISSIONAL","PRIMEIRA COMPRA", "CAPACIDADE FINANCIA", "CLI. RESTR. MERCADO", 
             "REGRA - COD_FILIAL", "REGRA - IDADE PERMITIDA", "REGRA - EXP. PROFISSIONAL", 
             "REGRA - TEMPO DE RESIDENCIA", "REGRA - RENOV. COM. RENDA", "REGRA - COMPRA S/ ENTRADA", 
             "REGRA - CLIENTE SCORE X", "REGRA - CAPACIDADE FINANCIA", "REGRA - ATRASO 60 DIAS", 
             "REGRA - VERIFICACAO FIADOR", "REGRA - CLI. RESTR. MERCADO", "REGRA - CLIENTE ALTO RISCO", 
             "REGRA - COMPRA S/ ENTRADA", "NUMERO DE PARCELAS", "NEGADO A N DIAS", "NEGADO HOJE", 
             "RESIDENCIA ALUGADA", "RESIDENCIA PROPRIA", "TIPO RENDA", "STATUS COMPROVANTE RENDA", 
             "CLIENTE CLASSIFICACAO"]]

# Transformando dados pro tipo necessário:
data.loc[:] = data.loc[:].apply(lambda x: x.astype(float))


# split4x = 4 pois o número de Aprovados é 4 vezes maior que o de Negados
# Essa função faz com que os dados sejam distribuidos corretamente em treino e teste.
x_train, y_train, x_test, y_test = split4x(data, 3300)
x_train = pd.DataFrame(data=x_train)
x_test = pd.DataFrame(data=x_test)
#columns = ['VALOR', 'EXP. PROFISSIONAL', 'NUMERO DE PARCELAS', 'NEGADO A N DIAS']
#columns = ['1', '2', '19', '20']

# Normalizando valores float: 
x_train = normalize_data(x_train, 1)
x_train = normalize_data(x_train, 2)
x_train = normalize_data(x_train, 19)
x_train = normalize_data(x_train, 20)
x_test = normalize_data(x_test, 1)
x_test = normalize_data(x_test, 2)
x_test = normalize_data(x_test, 19)
x_test = normalize_data(x_test, 20)


#class_weight = {0: 1.,
#                1: 4.}  Pode-se usar isso ao invés do 'class_weight'

# Como descobri os melhores parâmetros pro problema: 
#param_grid = {'C': [1, 5, 10, 50],
#              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 100]}
#grid = GridSearchCV(estimator=SVC(kernel='rbf'), param_grid=param_grid, cv=5)
#grid.fit(x_train, y_train)
#print(grid.best_params_)

grid = SVC(kernel='rbf', C=5, gamma=0.05, class_weight='balanced', probability=True)
grid.fit(x_train, y_train)

grid.get_params()

print(grid.n_support_)
print(grid.support_)

vs = 6906 + 1930

e = vs / 18613

y_pred = grid.predict(x_test) 
y_pred2 = grid.predict(x_train)
#print(y_pred)

print(classification_report(y_test, y_pred))

#print(accuracy_score(y_test,y_pred))
acc_test = accuracy_score(y_test,y_pred)
#print(accuracy_score(y_train,y_pred2))
acc_train = accuracy_score(y_train,y_pred2)

#modelTest(grid, x_train, y_train, x_test, y_test)

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
predict = grid.predict(x_train)
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
#loss_train, acc_train = grid.evaluate(x_train, y_train) ###
#print('\nAcuracia total no treino:', acc_train, '\tloss:', loss_train)
print('\nAcuracia total no treino:',acc_train )
print("aprovado_acc", aprovado_correct/aprovado_cnt*100, "%", '\tAcertos:', aprovado_correct, ' de ', aprovado_cnt)
print("negado_acc", negado_correct/negado_cnt*100, "%", '\tAcertos:', negado_correct, ' de ', negado_cnt)
print('\n\nTeste no x_test:')
print('Numero de aprovados:', counter2, '\tNumero de negados:', len(x_test) - counter2)
predict = grid.predict(x_test)
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
#loss_test, acc_test = grid.evaluate(x_test, y_test) ### 
#print('\nAcuracia total no teste:', acc_test, '\tloss:', loss_test)
print('\nAcuracia total no teste:',acc_test )
print("aprovado_acc", aprovado_correct/aprovado_cnt*100, "%", '\tAcertos:', aprovado_correct, ' de ', aprovado_cnt)
print("negado_acc", negado_correct/negado_cnt*100, "%", '\tAcertos:', negado_correct, ' de ', negado_cnt, '\n\n')

# Salvando e Carregando o modelo para testar com todos os dados: 
pickle.dump(grid, open('modeloNOVO.sav', 'wb'))
best_model = pickle.load(open('modeloNOVO.sav','rb'))
modelTestAllData(best_model, x_test, y_test)

# Testando com entrada de única instância nova: 
new = {
      "VALOR" : 0.65811,
      "NEGADO ANTERIORMENTE" : 0.0,
      "IDADE PERMITIDA" : 0.5977,
      "EXP. PROFISSIONAL" : 0.01,
      "PRIMEIRA COMPRA" : 0.0,
      "CAPACIDADE FINANCEIRA" : 0.0,
      "ATRASO 60 DIAS" : 0.0,
      "CLI. REST. MERCADO" : 0.0,
      "CLIENTE EM ATRASO" : 1.0,
      "REGRA - NEGADO ANTERIORMENTE" : 0.0,
      "REGRA - IDADE PERMITIDA" : 0.0,
      "REGRA - EXP. PROFISSIONAL" : 0.0,
      "REGRA - TEMPO DE RESIDENCIA" : 0.0,
      "REGRA - RENOV. COMP. RESIDEN" : 1.0,
      "REGRA - RENOV. COM. RENDA" : 1.0,
      "REGRA - FINANCIAMENTO P/ AUT" : 1.0,
      "REGRA - COMPRA S/ ENTRADA" : 0.0,
      "REGRA - CLIENTE SCORE X" : 0.0,
      "REGRA - CAPACIDADE FINANCIA" : 1.0,
      "REGRA - ATRASO 60 DIAS" : 0.0,
      "REGRA - VERIFICACAO FIADOR" : 0.0,
      "REGRA - CLI. RESTR. MERCADO" : 0.0,
      "REGRA - CLIENTE EM ATRASO" : 1.0,
      "REGRA - CLIENTE ALTO RISCO" : 0.0,
      "REGRA - REGRA MOVEIS" : 1.0,
      "COMPRA COM ENTRADA" : 1.0,
      "NUMERO DE PARCELAS" : 1,
      "NEGADO A N DIAS" : 0,
      "NEGADO HOJE" : 0.0,
      "RESIDENCIA ALUGADA" : 0.0,
      "RESIDENCIA CEDIDA" : 0.0,
      "RESIDENCIA PROPRIA" : 1.0,
      "TIPO RENDA" : 0.0,
      "STATUS COMPROVANTE RENDA" : 0.0,
      "STATUS COMPROVANTE RESIDENCIA" : 0.0,
      "CLIENTE CLASSIFICACAO" : 2.0
      }
df = pd.DataFrame([new])
dropcolumns2 = [ 'NEGADO ANTERIORMENTE',
 'ATRASO 60 DIAS',
 'CLIENTE EM ATRASO',
 'REGRA - NEGADO ANTERIORMENTE',
 'REGRA - RENOV. COMP. RESIDEN',
 'REGRA - FINANCIAMENTO P/ AUT',
 'REGRA - CLIENTE EM ATRASO',
 'REGRA - REGRA MOVEIS',
 'RESIDENCIA CEDIDA',
 'STATUS COMPROVANTE RESIDENCIA']
try:
  df = df.drop(dropcolumns2, axis = 1)
except Exception as e:
    print(e)

pred2 = grid.predict_proba(df)
print(pred2)

new2 = {
      "VALOR" : 0.03521,
      "NEGADO ANTERIORMENTE" : 0.0,
      "IDADE PERMITIDA" : 0,
      "EXP. PROFISSIONAL" : 0.52861,
      "PRIMEIRA COMPRA" : 0.0,
      "CAPACIDADE FINANCEIRA" : 1,
      "ATRASO 60 DIAS" : 0.0,
      "CLI. REST. MERCADO" : 0.0,
      "CLIENTE EM ATRASO" : 0.0,
      "REGRA - NEGADO ANTERIORMENTE" : 0.0,
      "REGRA - IDADE PERMITIDA" : 0.0,
      "REGRA - EXP. PROFISSIONAL" : 0.0,
      "REGRA - TEMPO DE RESIDENCIA" : 0.0,
      "REGRA - RENOV. COMP. RESIDEN" : 0.0,
      "REGRA - RENOV. COM. RENDA" : 0.0,
      "REGRA - FINANCIAMENTO P/ AUT" : 1.0,
      "REGRA - COMPRA S/ ENTRADA" : 0.0,
      "REGRA - CLIENTE SCORE X" : 0.0,
      "REGRA - CAPACIDADE FINANCIA" : 1.0,
      "REGRA - ATRASO 60 DIAS" : 0.0,
      "REGRA - VERIFICACAO FIADOR" : 0.0,
      "REGRA - CLI. RESTR. MERCADO" : 0.0,
      "REGRA - CLIENTE EM ATRASO" : 0.0,
      "REGRA - CLIENTE ALTO RISCO" : 0.0,
      "REGRA - REGRA MOVEIS" : 0.0,
      "COMPRA COM ENTRADA" : 1.0,
      "NUMERO DE PARCELAS" : 0.91667,
      "NEGADO A N DIAS" : 0.0,
      "NEGADO HOJE" : -1,
      "RESIDENCIA ALUGADA" : 0.0,
      "RESIDENCIA CEDIDA" : 0.0,
      "RESIDENCIA PROPRIA" : 1.0,
      "TIPO RENDA" : 1,
      "STATUS COMPROVANTE RENDA" : 0.0,
      "STATUS COMPROVANTE RESIDENCIA" : 2,
      "CLIENTE CLASSIFICACAO" : 2.0
      }
df2 = pd.DataFrame([new2])
try:
  df2 = df2.drop(dropcolumns2, axis = 1)
except Exception as e:
    print(e)
    
pred2 = grid.predict_proba(df2)
print(pred2)

new2 = {
      "COD_FILIAL" : 7,
      "VALOR" : 0.03521,
      "EXP. PROFISSIONAL" : 0.52861,
      "PRIMEIRA COMPRA" : 0.0,
      "CAPACIDADE FINANCEIRA" : 1,
      "CLI. REST. MERCADO" : 0.0,
      "REGRA - COD_FILIAL": 1,
      "REGRA - IDADE PERMITIDA" : 0.0,
      "REGRA - EXP. PROFISSIONAL" : 0.0,
      "REGRA - TEMPO DE RESIDENCIA" : 0.0,
      "REGRA - RENOV. COM. RENDA" : 0.0,
      "REGRA - COMPRA S/ ENTRADA" : 0.0,
      "REGRA - CLIENTE SCORE X" : 0.0,
      "REGRA - CAPACIDADE FINANCIA" : 1.0,
      "REGRA - ATRASO 60 DIAS" : 0.0,
      "REGRA - VERIFICACAO FIADOR" : 0.0,
      "REGRA - CLI. RESTR. MERCADO" : 0.0,
      "REGRA - CLIENTE ALTO RISCO" : 0.0,
      "COMPRA COM ENTRADA" : 1.0,
      "NUMERO DE PARCELAS" : 0.91667,
      "NEGADO A N DIAS" : 0.0,
      "NEGADO HOJE" : -1,
      "RESIDENCIA ALUGADA" : 0.0,
      "RESIDENCIA PROPRIA" : 1.0,
      "TIPO RENDA" : 1,
      "STATUS COMPROVANTE RENDA" : 0.0,
      "CLIENTE CLASSIFICACAO" : 2.0
      }
df2 = pd.DataFrame([new2])
pred2 = grid.predict_proba(df2)
print(pred2)