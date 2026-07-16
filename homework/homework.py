#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#
import pandas as pd
import json
import os
import gzip
import pickle

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
)


# Paso 1
def cargar_datos():
    df_entrenamiento = pd.read_csv(
        "./files/input/train_data.csv.zip", compression="zip"
    )
    df_prueba = pd.read_csv(
        "./files/input/test_data.csv.zip", compression="zip"
    )
    return df_entrenamiento, df_prueba


def preprocesar_tabla(tabla):
    datos = tabla.copy()
    datos["Age"] = 2021 - datos["Year"]
    datos = datos.drop(columns=["Year", "Car_Name"])
    return datos





# Paso 3
def construir_pipeline():
    columnas_categoricas = ["Selling_type", "Fuel_Type", "Transmission"]
    columnas_numericas = [
        col for col in X_train.columns if col not in columnas_categoricas
    ]

    transformador = ColumnTransformer(
        transformers=[
            ("num", MinMaxScaler(), columnas_numericas),
            ("cat", OneHotEncoder(handle_unknown="ignore"), columnas_categoricas),
        ]
    )

    flujo = Pipeline(
        steps=[
            ("preprocessing", transformador),
            ("select_k_best", SelectKBest(score_func=f_regression)),
            ("linear_model", LinearRegression()),
        ]
    )

    return flujo


# Paso 4
def optimizar_hiperparametros(flujo):
    n_features = len(X_train.columns)
    grilla = {
        "select_k_best__k": list(range(1, n_features + 1)),
        "select_k_best__score_func": [f_regression, mutual_info_regression],
        "preprocessing__num__feature_range": [(0, 1), (-1, 1)],
        "linear_model__fit_intercept": [True, False],
    }

    busqueda = GridSearchCV(
        estimator=flujo,
        param_grid=grilla,
        cv=10,
        scoring="neg_mean_absolute_error",
    )

    busqueda.fit(X_train, y_train)
    return busqueda



# Paso 5
def guardar_modelo(est):
    os.makedirs("./files/models", exist_ok=True)
    with gzip.open("./files/models/model.pkl.gz", "wb") as archivo:
        pickle.dump(est, archivo)



# Paso 6
def calcular_y_guardar_metricas(est):
    os.makedirs("./files/output", exist_ok=True)

    y_train_pred = est.predict(X_train)
    y_test_pred = est.predict(X_test)

    met_train = {
        "type": "metrics",
        "dataset": "train",
        "r2": r2_score(y_train, y_train_pred),
        "mse": mean_squared_error(y_train, y_train_pred),
        "mad": mean_absolute_error(y_train, y_train_pred),
    }

    met_test = {
        "type": "metrics",
        "dataset": "test",
        "r2": r2_score(y_test, y_test_pred),
        "mse": mean_squared_error(y_test, y_test_pred),
        "mad": mean_absolute_error(y_test, y_test_pred),
    }

    with open("./files/output/metrics.json", "w") as salida:
        salida.write(json.dumps(met_train) + "\n")
        salida.write(json.dumps(met_test) + "\n")



train_data, test_data = cargar_datos()

train_data = preprocesar_tabla(train_data)
test_data = preprocesar_tabla(test_data)

X_train = train_data.drop(columns=["Present_Price"])
y_train = train_data["Selling_Price"]

X_test = test_data.drop(columns=["Present_Price"])
y_test = test_data["Selling_Price"]

pipeline = construir_pipeline()
modelo_entrenado = optimizar_hiperparametros(pipeline)


guardar_modelo(modelo_entrenado)

calcular_y_guardar_metricas(modelo_entrenado)
