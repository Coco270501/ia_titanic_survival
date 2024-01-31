import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split

# from sklearn.model_selection import train_test_split
import numpy as np

from tensorflow import keras

data = pd.read_csv('titanic.csv')

#Modifier les valeurs null
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Embarked'].fillna('S', inplace=True)
data['Cabin'].fillna('C00', inplace=True)

#Rajoutes des column selon les valeurs des colonnes pour les valeurs indiquées
data = pd.get_dummies(data, columns=['Sex', 'Embarked'])
features = data[['Pclass','Age','SibSp','Parch','Fare','Sex_female','Embarked_Q', 'Embarked_S']]
labels = data['Survived']

train_features, test_features, train_labels, test_labels = train_test_split(features, labels,test_size=0.2)

test_labels = np.asarray(test_labels).astype('float32')
train_features = np.asarray(train_features).astype('float32')
test_features = np.asarray(test_features).astype('float32')
#Construction
#Dense toutes les couches sont liés entre elles
#relu -> somme des poids des noeuds
#train features 80% des donnees 1 ligne features (1 row et shape -1 dimension du vecter nb de colonnes) 
#entree nb de colonne des featurs -> sortie nb de colonne des label
model = keras.Sequential(
    [
        keras.layers.Dense(10, activation='relu', input_shape=(train_features.shape[-1],)),
        #Ajout d'une couche avec 10 noeuronnes et connexion dense entre la couche 2 et la nouvelle couche
        keras.layers.Dense(10, activation='relu'),
        #Un seul neuronne -> la proba de savoir si survécu -> 99% -> 99% de chance de survit
        keras.layers.Dense(1, activation='sigmoid')
    ]
)
#adam exploration de données et modification des poids
#accuracy 0.99 - 1 = 0.01 proche de 0 donc bonne accuracy pas de modification de poids avec l'optimizer adam
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#epochs durée de l'entrainement -> plus poches long plus entrainé mais peut perdre en précision si trop long
history = model.fit(
    train_features,
    train_labels,
    validation_split=0.2, epochs=30, batch_size = 10) #validation split -> on prend que 20% des données

test_loss, test_accuracy = model.evaluate(train_features, train_labels)
print('Test accuracy {test_accuracy}')

predictions = model.predict(test_features[:5])
print("prediction on the first five test sample :")
for i, prediction in enumerate(predictions):
    print(f'{i+1}: probability of survival: {prediction[0]:.2f} ')
    Actual: {test_labels[i]}
