#CALCULOS Y PROCESAMIENTO DE DATOS
import pandas as pd
import numpy as np
#GRAFICOS 
import seaborn as sns
import matplotlib.pyplot as plt
#MODELO DE APRENDIZAJE   
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = fetch_california_housing()

x = data.data 
y = data.target

print("Forma de los datos:")
print(data.data.shape) #forma de los datos
print("\nNombres de las variables:")
print(data.feature_names) #etiquetas
print("\nCantidad de variables:")
print(len(data.feature_names)) #cantidad de categorías

df = pd.DataFrame(data.data, columns=data.feature_names)
df["MedHouseVal"] = data.target
print("\nDataFrame:")
df


#PREVISUALIZACION DE LA TABLA
 #Primeras Filas
df.head()
 #Ultimas filas
df.tail()
#DIMENSIONES           
print("Número de filas y columnas: ", df.shape)

#COLUMNAS 
print("\nColumnas: ")
print(df.columns.tolist())

#INFORMACION GENERAL DE LA TABLA(DEL DATASET)
print("\nInformación del dataset: ")
df.info()

#Media
print("Media:")
print(df.mean())
#Mediana
print("\nMediana:")
print(df.median())



#PROCESAMIENT DE DATOS
# Valores Nulos 
print("Valores nulos por columna:")
print(df.isnull().sum())

porcentaje_nulos = df.isnull().mean() * 100

print("Porcentaje de valores nulos:")
print(porcentaje_nulos)

for columna in df.columns:
    if df[columna].isnull().sum() > 0:
        df[columna] = df[columna].fillna(df[columna].median())

#Valores Duplicados
duplicados = df.duplicated().sum()
print("Cantidad de filas duplicadas:", duplicados)
df = df.drop_duplicates()

# VARIANZA y DESVIACION ESTANDAR
#varianza
def calcular_varianza(datos):
    media = sum(datos) / len(datos)
    suma = 0

    for valor in datos:
        suma += (valor - media) ** 2
    varianza = suma / len(datos)
    return varianza

datos = df["MedInc"].values
varianza_resultado = calcular_varianza(datos)
print("Varianza: ", varianza_resultado)

#desviacion estandar
def desviacion_estandar(datos):
    varianza = calcular_varianza(datos)
    desviacion = varianza ** 0.5
    return desviacion

desviacion_resultado = desviacion_estandar(datos)

print("\nDesviación estándar de MedInc:", desviacion_resultado)

#VISUALIZACION 
#histograma
plt.figure(figsize=(10, 6))

sns.scatterplot(data=df,x="MedInc",y="MedHouseVal",alpha=0.3)
plt.title("Distribución del valor medio de las viviendas")
plt.xlabel("Valor medio de la vivienda")
plt.ylabel("Frecuencia")

plt.show()

#grafico de dispersion
plt.figure(figsize=(10, 6))

sns.scatterplot(data=df, x="MedInc", y="MedHouseVal", alpha=0.3)

plt.title("Relación entre ingreso y valor de la vivienda")
plt.xlabel("Ingreso medio")
plt.ylabel("Valor medio de la vivienda")

plt.show()

# CONSTRUIMOS LA ESTRUCTURA DE ML
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2 , random_state=42) #el valor 0.2 indica 80 % para entrenamiento 20% para la prueba
modelo = LinearRegression()

print("\nTamaño de entrenamiento:")
print(x_train.shape)

print("\nTamaño de prueba:")
print(x_test.shape)

#NORMALIZACION
modelo = LinearRegression()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#ENTRENAMIENTO DEL MODELO
modelo.fit(x_train, y_train)

#PREDICCION
y_pred = modelo.predict(x_test)

print("Primeras predicciones:")
print(y_pred[:10]) #Predicciones desde el primero valor hasta el 9no

# EVALUACIÓN
mae = mean_absolute_error(y_test, y_pred) #Representa el error absoluto promedio de las predicciones.
mse = mean_squared_error(y_test, y_pred) #
rmse = np.sqrt(mse) #Penaliza más fuertemente los errores grandes.
r2 = r2_score(y_test, y_pred) #Indica qué proporción de la variabilidad de la variable objetivo puede explicar el modelo.

print("\n========== RESULTADOS ==========")
print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)

#VISUALIZACION DE RESULTADOS
plt.figure(figsize=(8, 6))

plt.scatter(y_test,y_pred,alpha=0.4,color="darkgreen")

plt.xlabel("Valores reales")
plt.ylabel("Valores predichos")
plt.title("Valores reales vs. valores predichos")

#Línea ideal
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()],color="red",linestyle="--")
plt.show()