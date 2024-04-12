import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import json
import joblib

# Cargar los datos desde el archivo Excel
df = pd.read_excel("datos_productos_reales.xlsx")

# Convertir las medidas de pulgadas a centímetros
df['Medidas(PULGADAS)'] = df['Medidas(PULGADAS)'] * 2.54

# Eliminar la columna 'Descripcion' ya que no se utilizará en el entrenamiento
df = df.drop(columns=['Descripcion'])

# Dividir los datos en características (X) y etiquetas (y)
X = df[['Largo(CM)', 'Ancho(CM)', 'Medidas(PULGADAS)']]
y = df['Peso_kg']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Seleccionar un modelo de regresión (en este caso, Linear Regression)
modelo = LinearRegression()

# Entrenar el modelo
modelo.fit(X_train, y_train)

# Hacer predicciones utilizando el conjunto de prueba
predicciones = modelo.predict(X_test)

# Evaluar el rendimiento del modelo
mse = mean_squared_error(y_test, predicciones)
print("Error cuadrático medio (MSE):", mse)

# Calcular el RMSE
rmse = np.sqrt(mse)
print("Raíz del error cuadrático medio (RMSE):", rmse)

# Calcular el coeficiente de determinación (R cuadrado)
r2 = r2_score(y_test, predicciones)
print("Coeficiente de determinación (R cuadrado):", r2)

# Unidad de distancia (en este caso, será "CM" para centímetros)
distancia_unidad = "CM"

# Imprimir el MSE, RMSE, R cuadrado y la unidad de distancia en el formato requerido por SkyDropx
result = {
    "MSE": mse,
    "RMSE": rmse,
    "R_cuadrado": r2,
    "distance_unit": distancia_unidad
}

print(json.dumps(result, indent=4))

# Exportar el modelo entrenado a un archivo
joblib.dump(modelo, 'modelo_DatosReales.pkl')