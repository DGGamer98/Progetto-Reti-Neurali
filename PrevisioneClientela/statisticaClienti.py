'''Import Data'''
import pandas as pd
from sklearn.model_selection import train_test_split

'''Import Dipendenze'''
from tensorflow.keras.models import Sequential, load_model # classe di modello principale ricaricare il modello dalla memoria in un secondo momento
from tensorflow.keras.layers import Dense # Livello connesso alla nostra rete neurale
from sklearn.metrics import accuracy_score # punteggio di accuratezza

df = pd.read_csv(r'C:\Users\david\OneDrive\Desktop\esercizi python\TensorFlow\PrevisioneClientela\Churn.csv')

X = pd.get_dummies(df.drop(['Churn', 'Customer ID'], axis=1))
y = df['Churn'].apply(lambda x: 1 if x=='Yes' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)

print(X_train.head())
print(y_train.head())

'''Build and Compile Model'''
model = Sequential() #L'istanza della nostra classe sequenziale

#Gruppo livelli della rete neurale
model = Sequential()
model.add(Dense(units=32, activation='relu', input_dim=len(X_train.columns))) #units=32 --> avremmo 32 neuroni in questo livello Dense
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=1, activation='sigmoid')) #Output --> range 0 e 1 --> 0 tasso di abbandono, 1 tasso di non abbandono

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy']) # modello compilato --> e quali metriche vogliamo concentrarci

'''Fit, Predict and Evaluate'''
model.fit(X_train, y_train, epochs=200, batch_size=32) #Questo è l'addestramento del modello --> epoche 200 quante volte deve controllare --> batch quanto grande deve essere il batch da mandare a tensorFlow

#previsione
y_hat = model.predict(X_test) #model.predict --> ci permette di fare previsioni --> gli passiamo come parametro X_test di training
y_hat = [0 if val < 0.5 else 1 for val in y_hat] #Il risulatato di tensorFlow sarà di 0 e 1 --> li convertiamo
print(y_hat)


acc = accuracy_score(y_test, y_hat)
print(f"Accuracy: {acc:.4f}")

'''Saving and Reload'''
#model.save(r'C:\Users\david\OneDrive\Desktop\modelloIA\tfmodel.keras')
#del model

model = load_model(r'C:\Users\david\OneDrive\Desktop\modelloIA\tfmodel.keras')




