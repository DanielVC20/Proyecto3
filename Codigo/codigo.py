import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df_graduados = pd.read_csv("Graduados.csv", delimiter=";")
years = df_graduados.keys()[1:-2]
df_graduados = np.array(df_graduados)

df_admi = pd.read_csv("Admitidos.csv", delimiter=";")
df_admi = np.array(df_admi)

carrera = "Total uniandes"

ii_grad = df_graduados[:,0] == carrera
ii_admi = df_admi[:,0] == carrera

carrera_grad = df_graduados[ii_grad]
carrera_admi = df_admi[ii_admi]
carrera_admi = carrera_admi[0, 1:]

graduados = carrera_grad[0, 1:-2]
inscritos = carrera_admi[0::5]
matriculados = carrera_admi[4::5]

for i in range(len(graduados) - len(inscritos)):
    inscritos = np.hstack((0 , inscritos))
    matriculados = np.hstack((0 , matriculados))

l = len(graduados) - 2

frac_matr_insc = np.zeros(l)
frac_grad_matr = np.zeros(l)

for i in range(l):
    frac_matr_insc[i] = matriculados[i + 2]/inscritos[i + 2]
    frac_grad_matr[i] = graduados[i + 2]/matriculados[i + 2]


plt.figure()

plt.xticks(rotation = 90)
plt.scatter(years, inscritos, label="Inscritos")
plt.scatter(years, matriculados, label="Matriculados")
plt.scatter(years, graduados, label="Graduados")
plt.legend()
plt.title(carrera)

plt.show()

plt.figure()

plt.xticks(rotation = 90)
plt.scatter(years[2:], frac_matr_insc)
plt.title(carrera + " fracción matriculados/inscritos")

plt.show()

plt.figure()

plt.xticks(rotation = 90)
plt.scatter(years[2:], frac_grad_matr)
plt.title(carrera + " fracción graduados/matriculados")

plt.show()
