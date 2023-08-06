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


# Implementación de FastAPI
app = FastAPI()

# Recuperación de datos desde el archivo .csv  
data_steam = pd.read_csv('dataset/clean_steam_data.csv')

# Evaluamos la variable 'release_date'
data_steam['release_date'] = pd.to_datetime(data_steam['release_date'], errors='coerce')
data_steam = data_steam.dropna(subset=['release_date'])

# Definición de la API con información de los juegos según año de lanzamiento
min_year = data_steam['release_date'].dt.year.min()
max_year = data_steam['release_date'].dt.year.max()

def check_year(year: int):
    if year < min_year or year > max_year:
        raise HTTPException(status_code=400, detail=f"Año sin registro en la base de datos. Elija un año entre {min_year} y {max_year}.")

# Función que retorna los 5 géneros más vendidos
@app.get('/genero/')
def genero(año: int):
    check_year(año)
    filtered_data_steam = data_steam[data_steam['release_date'].dt.year == año]
    exploded_genres_data_steam = filtered_data_steam.explode('genres')
    top_genres = exploded_genres_data_steam['genres'].value_counts().nlargest(5).to_dict()
    return top_genres

# Función que retorna los juegos lanzados
@app.get('/juegos/')
def juegos(año: int):
    check_year(año)
    filtered_data_steam = data_steam[data_steam['release_date'].dt.year == año]
    released_games = filtered_data_steam['app_name'].to_list()
    return {"juegos": released_games}

# Función que retorna el top 5 de especificaciones
@app.get('/especificaciones/')
def especificaciones(año: int) -> Dict[str, Union[int, str]]:
    check_year(año)
    filtered_data_steam = data_steam[data_steam['release_date'].dt.year == año]
    exploded_specs_data_steam = filtered_data_steam.explode('specs')
    top_specs = exploded_specs_data_steam['specs'].value_counts().nlargest(5).to_dict()
    if not top_specs: 
        return {"message": "No se encontraron especificaciones de juegos en este año"}
    return top_specs

# Función que retorna la cantidad de juegos con acceso temprano 
@app.get('/acceso_temprano/')
def acceso_temprano(año: int):
    check_year(año)
    filtered_data_steam = data_steam[data_steam['release_date'].dt.year == año]
    count_early_access = len(filtered_data_steam[filtered_data_steam['early_access'] == True])
    return {"early_access_games": count_early_access}

# Función que retorna el tipo y cantidad de opiniones registradas
@app.get('/opiniones/')
def opiniones(año: int) -> Dict[str, Union[int, str]]:
    check_year(año)
    sentiments = ['Very Positive', 'Mixed', 'Mostly Positive', 'Positive', 
                  'Overwhelmingly Positive', 'Mostly Negative', 'Negative', 
                  'Very Negative', 'Overwhelmingly Negative']
    filtered_data_steam = data_steam[(data_steam['release_date'].dt.year == año) 
                                     & (data_steam['sentiment'].isin(sentiments))]
    sentiment_counts = filtered_data_steam['sentiment'].value_counts().to_dict()
    if not sentiment_counts: 
        return {"message": "No se encontraron opiniones relevantes en este año"}
    return sentiment_counts

# Función que retorna el top 5 de juegos según su puntuación
@app.get('/metascore/')
def metascore(año: int):
    check_year(año)
    filtered_data_steam = data_steam[data_steam['release_date'].dt.year == año]
    top_metascore_games = filtered_data_steam.nlargest(5, 'metascore')[['app_name', 'metascore']].set_index('app_name').to_dict()['metascore']
    return top_metascore_games

'''
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
# print("RMSE:", rmse)
'''

# Cargar el modelo desde un archivo .pkl
with open('dataset/model.pkl', 'rb') as file:
    model = pickle.load(file)

# Cargar el transformador polinomial desde un archivo .pkl
with open('dataset/poly_transform.pkl', 'rb') as file:
    poly = pickle.load(file)

# Cargar el valor de RMSE desde un archivo .pkl
with open('dataset/rmse.pkl', 'rb') as file:
    rmse = pickle.load(file)

# Cargar steam_dummies desde un archivo .pkl (asegúrate de haberlo guardado previamente)
with open('dataset/steam_dummies.pkl', 'rb') as file:
    steam_dummies = pickle.load(file)

print(rmse)

# Obtención de todos los géneros únicos
all_genres = steam_dummies.columns[steam_dummies.columns.str.contains('genres')].tolist()

@app.get("/prediccion/")
async def get_prediccion(
    genero: str = Query(
        ...,  # Parámetro requerido
        description="Elija el género del juego entre los siguientes: " + ', '.join(all_genres),
    ),
    año: int = Query(
        ...,  # Parámetro requerido
        description=f"Elija el año de lanzamiento del juego, entre {min_year} y {max_year}.",
    ),
    metascore: int = Query(
        ...,  # Parámetro requerido
        description="Elija el Metascore del juego.",
    ),
    early_access: str = Query(
        ...,  # Parámetro requerido
        description="Indica si el juego está en acceso anticipado. Elija 'si' o 'no'.",
    )
):
    if genero not in all_genres:
        raise HTTPException(status_code=400, detail="Género no válido. Por favor use un género de la lista de géneros disponibles.")
    if early_access.lower() not in ['si', 'no']:
        raise HTTPException(status_code=400, detail="Valor no válido para 'early_access'. Elija 'si' o 'no'.")
    genre_data = [1 if genre == genero else 0 for genre in all_genres]
    early_access_data = 1 if early_access.lower() == 'si' else 0
    data = np.array([año, metascore, early_access_data] + genre_data).reshape(1, -1)
    
    # Aplicar la transformación polinomial
    data_poly = poly.transform(data)
    price = model.predict(data_poly)[0]
    
    return {'price': price, 'rmse': rmse}

