from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib

import tensorflow as tf

#import tensorflow.compat.v2.feature_colum as fc

#Caricamento set di dati
dftrain = pd.read_csv(r"C:\Users\david\OneDrive\Desktop\esercizi python\TensorFlow\dati CSV\train.csv")# dati di addestramento
dfeval = pd.read_csv(r'C:\Users\david\OneDrive\Desktop\esercizi python\TensorFlow\dati CSV\eval.csv') # test di dati

#Estrazione colonna
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

#Troviamo una riga specifica --> usiamo il .loc e posizione
print(dftrain.loc[0], y_train.loc[0])

print("x------\n")

#Troviamo i diversi valori di eta
print(dftrain["age"])


print("x------\n")

#Visualizazzione primi 5 elementi del nostro datafarame
print(dftrain.head())

print("x------\n")

#Analisi più statistica dei nostri dati
print(dftrain.describe())

print("x------\n")

#Controllo forma della nostra variabile
print(dftrain.shape)

#Informazioni sulla sopravivenza
print(y_train.head())

'''Generiamo dei graifici perché sono molto prezioni per la lettura'''
#dftrain.age.hist(bins=20)
#dftrain.sex.value_counts().plot(kind='barh')
#dftrain['class'].value_counts().plot(kind='barh')
pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('% survive')

plt.show()

'''Analisi dei dati'''

CATEGORICAL_COLUMS = ['sex', 'n_siblings_spouses', 'parch', 'class','deck',
                      'embark_town','alone']

NUMERIC_COLUMS = ['age','fare']

feature_colums = []
for feature_name in CATEGORICAL_COLUMS:
    vocabulary =  dftrain[feature_name].unique()
    feature_colums.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))
    
for feature_name in NUMERIC_COLUMS:
    feature_colums.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
    
print(feature_colums)


'''Convere i dati in pandas in dati batch che tensorFlow può leggere'''
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
    def input_function(): #Funzione inerna, verrà restituita questa
        ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df)) #Crea l'oggetto tf.data.Dataset con i dati  e la sua etichetta
        if shuffle:
            ds = ds.shuffle(1000) #randomizza l'ordine dei dati
        ds = ds.batch(batch_size).repeat(num_epochs) #divide il set di dati in batch da 32 e ripete il processo per un certo numero di epoche
        return ds #ritorna batch di dati
    return input_function

train_input_fn = make_input_fn(dftrain, y_train)
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)



'''Creazione del modello'''
#Assegnameto colonne che abbiamo creato a riga 62 in giù
linear_est = tf.estimato.LinearClassifier(feature_columns=feature_colums)

'''Addestramento del modello'''
linear_est.train(train_input_fn)
result = linear_est.evaluate(eval_input_fn)

clear_output()
print(result['accuracy'])
print(result)

pred_dicts = list(linear_est.predict(eval_input_fn))
probs = pd.Series([pred['probabilities'][1] for pred in pred_dicts])

probs.plot(kind='hist', bins=20, title='predicted probabilities')

# il risultato avrà una precisione 73,8%
