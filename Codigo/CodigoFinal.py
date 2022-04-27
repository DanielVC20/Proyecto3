import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df_graduados = pd.read_csv("Graduados.csv", delimiter=";")
years = np.array(df_graduados.keys()[3:-2])
df_graduados = np.array(df_graduados)

df_admi = pd.read_csv("Admitidos.csv", delimiter=";")
df_admi = np.array(df_admi)

df_pobl = pd.read_csv("Poblacion.csv", delimiter=";")
df_pobl = np.array(df_pobl)

df_transf_exter = pd.read_csv("TransfExternas.csv", delimiter=";")
df_transf_exter = np.array(df_transf_exter)

df_quiero_est = pd.read_csv("QuieroEstudiar.csv", delimiter=";")
df_quiero_est = np.array(df_quiero_est)

df_PIB = pd.read_csv("PIBPerCapita.csv", delimiter=";")
df_PIB = np.array(df_PIB)
df_PIB = np.flip(df_PIB, 0)

df_prof = pd.read_csv("Doctorados.csv", delimiter=";")
df_prof = np.array(df_prof)

carrera = "Total Uniandes"

ii_grad = df_graduados[:,0] == carrera
ii_admi = df_admi[:,0] == carrera
ii_pobl = df_pobl[:,0] == carrera
ii_transf_exter = df_transf_exter[:,0] == carrera
ii_quiero_est = df_quiero_est[:,0] == "Activo"
ii_prof = df_prof[:,0] == "Porcentaje"


carrera_grad = df_graduados[ii_grad]
carrera_admi = df_admi[ii_admi]
carrera_pobl = df_pobl[ii_pobl]
carrera_transf_exter = df_transf_exter[ii_transf_exter]
quiero_est = df_quiero_est[ii_quiero_est]
prof = df_prof[ii_prof]

#Variables

inscritos = carrera_admi[0, 1::5]
admitidos = carrera_admi[0, 2::5]
matriculados = carrera_admi[0, 5::5]
graduados = carrera_grad[0, 3:40]
poblacion = carrera_pobl[0, 11:]
transf_externas = carrera_transf_exter[0, 3:40]
quiero_est = quiero_est[0, 1:]
porcent_prof = np.float_(prof[0, 2:])/100


for i in range(len(inscritos) - len(quiero_est)):
    quiero_est = np.hstack((0, quiero_est))

l = len(inscritos)

frac_admi_insc = np.zeros(l)
frac_matr_admi = np.zeros(l)
frac_matr_insc = np.zeros(l)

PIB = np.zeros(l)
porcentaje_profesores = np.zeros(l)


for i in range(int(l/2)):
    PIB[2*i] = df_PIB[i,1]
    PIB[2*i + 1] = df_PIB[i,1]
    
    porcentaje_profesores[2*i] = porcent_prof[i]
    porcentaje_profesores[2*i + 1] = porcent_prof[i]

PIB[-1] = df_PIB[-1,1]
porcentaje_profesores[-1] = porcent_prof[-1]

predictors = np.array(["Poblacion", "Inscritos", "Admitidos", "Graduados", "Porc. prof. doctorado"])

X = np.array([poblacion, inscritos, admitidos, graduados, porcentaje_profesores]).transpose()
Y = matriculados

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

years1 = years[0::2]
years2 = years[1::2]

X1 = X[0::2]
X2 = X[1::2]

Y1 = Y[0::2]
Y2 = Y[1::2]

corte = 13

X1_train, X1_test = X1[0:corte], X1[corte:]
Y1_train, Y1_test = Y1[0:corte], Y1[corte:]

X2_train, X2_test = X2[0:corte], X2[corte:]
Y2_train, Y2_test = Y2[0:corte], Y2[corte:]

GB1 = GradientBoostingRegressor(n_estimators=620, learning_rate=0.001)
GB1.fit(X1_train, Y1_train)

GB2 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.01)
GB2.fit(X2_train, Y2_train)

r2_sem1 = r2_score(Y1_test, GB1.predict(X1_test))
r2_sem2 = r2_score(Y2_test, GB2.predict(X2_test))

print(r2_sem1)
print(r2_sem2)


for i in range(len(predictors)):
    plt.figure()
    plt.scatter(X1_train[:,i], Y1_train, label="Reales")
    plt.scatter(X1_train[:,i], GB1.predict(X1_train), c="r", label="Train")
    plt.xlabel(predictors[i])
    plt.ylabel("Matriculados")
    plt.title("Train " + predictors[i] + " Sem 1")
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.scatter(X1_test[:,i], Y1_test)
    plt.scatter(X1_test[:,i], GB1.predict(X1_test), c="r")
    plt.xlabel(predictors[i])
    plt.ylabel("Matriculados")
    plt.title("Test " + predictors[i] + " Sem 1")
    plt.show()
    

for i in range(len(predictors)):
    plt.figure()
    plt.scatter(X2_train[:,i], Y2_train, label="Reales")
    plt.scatter(X2_train[:,i], GB2.predict(X2_train), c="r", label="Train")
    plt.xlabel(predictors[i])
    plt.ylabel("Matriculados")
    plt.title("Train " + predictors[i] + " Sem 2")
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.scatter(X2_test[:,i], Y2_test)
    plt.scatter(X2_test[:,i], GB2.predict(X2_test), c="r")
    plt.xlabel(predictors[i])
    plt.ylabel("Matriculados")
    plt.title("Test " + predictors[i] + " Sem 2")
    plt.show()
    

ii1 = np.argsort(GB1.feature_importances_)
importances1 = GB1.feature_importances_[ii1]
predictors1 = predictors[ii1]

a = pd.Series(importances1, index=predictors1)
a.plot(kind='barh')
plt.xlabel('Feature Importances')
plt.show()

ii2 = np.argsort(GB2.feature_importances_)
importances2 = GB2.feature_importances_[ii2]
predictors2 = predictors[ii2]

a = pd.Series(importances2, index=predictors2)
a.plot(kind='barh')
plt.xlabel('Feature Importances')
plt.show()


plt.figure()
plt.xticks(rotation = 90, size=11)
plt.scatter(years, Y, label="Val Real")
plt.scatter(years1[corte:], GB1.predict(X1_test), label="Test sems 1")
plt.scatter(years2[corte:], GB2.predict(X2_test), label="Test sems 2")
plt.ylabel("Cant Estudiantes", size=13)
plt.title("Matriculados vs Semestre", size=14)
plt.legend()
plt.show()


plt.figure()
plt.xticks(rotation = 90, size=11)

for i in range(len(years2)):
    plt.scatter(years1[i], Y1[i], c="r")
    plt.scatter(years2[i], Y2[i], c="blue")

plt.scatter(years1[-1], Y1[-1], c="r", label="Semestre 1")
plt.scatter(years2[-1], Y2[-1], c="blue", label="Semestre 2")
plt.legend()
plt.title("Matriculados vs Semestre", size=14)
plt.ylabel("Cant Estudiantes", size=13)
plt.show()
