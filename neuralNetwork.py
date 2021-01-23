import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import  classification_report,confusion_matrix,plot_roc_curve
import  matplotlib.pyplot as plt

data=pd.read_csv('hd.csv')
print(data.head())
print(data.size)
X=data.iloc[:,0:13]
Y=data['target']


# Podaci su vec svi numericki nema potrebe za kodiranjem vrednosti

# deljenje podataka u test i trenin set
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2)

#skaliranje podataka
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

#po difoltu je relu aktivaciona funkcija i adam optimizator
mlp3Layer15Tanh=MLPClassifier(hidden_layer_sizes=(15,15,15),max_iter=1000,activation='tanh')
mlp3Layer15Tanh.fit(X_train,Y_train)

predictions=mlp3Layer15Tanh.predict(X_test)

#evaluacija

print("Mlp 3 Layer 15 Each Tanh activation solver adam")
print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))
plot_roc_curve(mlp3Layer15Tanh,X_test,Y_test)
plt.show()



mlp=MLPClassifier(hidden_layer_sizes=(15,15,15),max_iter=1000,activation='tanh',solver='lbfgs')
mlp.fit(X_train,Y_train)

predictions=mlp.predict(X_test)

#evaluacija

print("Mlp 3 Layer 15 Each Tanh activation solver lbfgs")
print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))
plot_roc_curve(mlp,X_test,Y_test)
plt.show()




mlp=MLPClassifier(hidden_layer_sizes=(15,15,15),max_iter=1000,activation='tanh',solver='sgd')
mlp.fit(X_train,Y_train)

predictions=mlp.predict(X_test)

#evaluacija

print("Mlp 3 Layer 15 Each Tanh activation solver sgd")
print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))
plot_roc_curve(mlp,X_test,Y_test)
plt.show()

#Pokusacemo da napravimo malo vecu neuronsku mrezu

mlp=MLPClassifier(hidden_layer_sizes=(15,15,15,15,15),max_iter=1000,activation='tanh',solver='sgd')
mlp.fit(X_train,Y_train)

predictions=mlp.predict(X_test)

#evaluacija

print("Mlp 6 Layer 15 Each Tanh activation solver lbfgs")
print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))
plot_roc_curve(mlp,X_test,Y_test)
plt.show()

#Hajmo da se vratimo na prethodnu strukturu i probamo relu aktivacionu funkciju

mlp=MLPClassifier(hidden_layer_sizes=(15,15,15),max_iter=1000,activation='relu',solver='sgd')
mlp.fit(X_train,Y_train)

predictions=mlp.predict(X_test)

#evaluacija

print("Mlp 3 Layer 15 Each relu activation solver sgd")
print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))
plot_roc_curve(mlp,X_test,Y_test)
plt.show()

mlp=MLPClassifier(hidden_layer_sizes=(15,15,15),max_iter=1000,activation='relu',solver='lbfgs')
mlp.fit(X_train,Y_train)

predictions=mlp.predict(X_test)

#evaluacija

print("Mlp 3 Layer 15 Each relu activation solver lbfgs")
print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))
plot_roc_curve(mlp,X_test,Y_test)
plt.show()


mlp=MLPClassifier(hidden_layer_sizes=(15,15,15),max_iter=1000,activation='relu',solver='adam')
mlp.fit(X_train,Y_train)

predictions=mlp.predict(X_test)

#evaluacija

print("Mlp 3 Layer 15 Each relu activation solver adam")
print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))
plot_roc_curve(mlp,X_test,Y_test)
plt.show()

#Najbolji rezultati su dobijeni sa relu aktivacionom funkcijom i lbfgs solverom
#Hajde da napravimo tu strukturu sa vise skrivenih slojeva

mlp=MLPClassifier(hidden_layer_sizes=(15,15,15,15,15,15),max_iter=1000,activation='relu',solver='lbfgs')
mlp.fit(X_train,Y_train)

predictions=mlp.predict(X_test)

#evaluacija

print("Mlp 6 Layer 15 Each relu activation solver lbfgs")
print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))
plot_roc_curve(mlp,X_test,Y_test)
plt.show()

#Nismo dobili ocekivano poboljsanje mozda manji broj slojeva

mlp=MLPClassifier(hidden_layer_sizes=(10,10),max_iter=1000,activation='relu',solver='lbfgs')
mlp.fit(X_train,Y_train)

predictions=mlp.predict(X_test)

#evaluacija

print("Mlp 3 Layer 15 Each relu activation solver lbfgs")
print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))
plot_roc_curve(mlp,X_test,Y_test)
plt.show()

#Katastrofalni reyultati

mlp=MLPClassifier(hidden_layer_sizes=(20,20,15),max_iter=1000,activation='relu',solver='lbfgs')
mlp.fit(X_train,Y_train)

predictions=mlp.predict(X_test)

#evaluacija

print("Mlp 3 Layer 15 Each relu activation solver lbfgs")
print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))
plot_roc_curve(mlp,X_test,Y_test)
plt.show()