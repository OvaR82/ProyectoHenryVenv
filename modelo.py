# Importación de librerías necesarias
import json
import ast
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from fastapi import FastAPI
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Union

# Recuperación de datos desde el archivo .csv  
data_steam = pd.read_csv('dataset/clean_steam_data.csv')

# Evaluamos la variable 'release_date'
data_steam['release_date'] = pd.to_datetime(data_steam['release_date'], errors='coerce')
data_steam = data_steam.dropna(subset=['release_date'])

# Armado del modelo predictivo
# Adecuación de los datos de la variable 'genres'
data_steam_model = data_steam.copy()
data_steam_model['genres'] = data_steam_model['genres'].replace('', np.nan)
data_steam_model = data_steam_model.dropna(subset=['genres'])

# Conversión de 'release_date' a año
data_steam_model['release_year'] = data_steam_model['release_date'].dt.year

# Conversión de 'early_access' a valores numéricos 
data_steam_model['early_access'] = data_steam_model['early_access'].astype(int)

# Conversión de 'genres' a valores numéricos 
steam_dummies = pd.get_dummies(data_steam_model, columns=['genres'], prefix='', prefix_sep='')

# División del dataframe en sets de entrenamiento y prueba
X = steam_dummies[['release_year', 'metascore', 'early_access'] + list(steam_dummies.columns[steam_dummies.columns.str.contains('genres')])]
y = steam_dummies['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definición de la características polinomiales para el modelo predictivo
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Entrenamiento del modelo
model = BaggingRegressor(n_estimators=200, random_state=42)  # puedes ajustar los parámetros como mejor te parezca
model.fit(X_train_poly, y_train)

# Evaluación del modelo
y_pred = model.predict(X_test_poly)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)

# Guardar el modelo en un archivo .pkl
with open('dataset/model.pkl', 'wb') as file:
    pickle.dump(model, file)

# También necesitas guardar el transformador polinomial 'poly' para reutilizarlo más tarde
with open('dataset/poly_transform.pkl', 'wb') as file:
    pickle.dump(poly, file)

# Guardamos también la variable steam_dummies
with open('dataset/steam_dummies.pkl', 'wb') as file:
    pickle.dump(steam_dummies, file)