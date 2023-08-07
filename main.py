'''
Implementación de API con modelo predictivo
----------------------------------------------------------------------------------------------------------------------------------------------------
Se debe empezar desde 0, haciendo un trabajo rápido de Data Engineer y tener un MVP (Minimum Viable Product) para el cierre del proyecto.
Transformaciones: Para este MVP no se necesita transformar los datos dentro del dataset pero si trabajar en leer el dataset con el formato correcto.
Desarrollo API: Se propone disponibilizar los datos de la empresa usando el framework FastAPI. Las consultas serán las siguientes:
Se crearan 6 funciones para los endpoints que se consumirán en la API, debiendo tener un decorador por cada una (@app.get(‘/’)).
    def genero( Año: str ): Se ingresa un año y devuelve una lista con los 5 géneros más ofrecidos en el orden correspondiente.
    def juegos( Año: str ): Se ingresa un año y devuelve una lista con los juegos lanzados en el año.
    def specs( Año: str ): Se ingresa un año y devuelve una lista con los 5 specs que más se repiten en el mismo en el orden correspondiente.
    def earlyacces( Año: str ): Cantidad de juegos lanzados en un año con early access.
    def sentiment( Año: str ): Según el año de lanzamiento, se devuelve una lista con la cantidad de registros que se encuentren categorizados con un análisis de sentimiento.
                        Ejemplo de retorno: {Mixed = 182, Very Positive = 120, Positive = 278}
    def metascore( Año: str ): Top 5 juegos según año con mayor metascore.
    Importante
    El MVP tiene que ser una API que pueda ser consumida segun los criterios de API REST o RESTful. 
    Algunas herramientas, como por ejemplo Streamlit, si bien pueden brindar una interfaz de consulta, 
    no cumplen con las condiciones para ser consideradas una API, sin workarounds.
Deployment: Utilizar Render, Railway o cualquier otro servicio que permita que la API pueda ser consumida desde la web.
'''
# Importación de librerías necesarias
import json
import ast
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from fastapi import FastAPI
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Dict, Union

# Implementación de FastAPI
app = FastAPI()

# Recuperación de datos desde un archivo .json 
dataset = []
with open('dataset/steam_games.json') as f: 
    dataset.extend(ast.literal_eval(line) for line in f)

# Creación del dataframe a partir del dataset obtenido
data_steam = pd.DataFrame(dataset)

# Adecuación y limpieza del dataframe
data_steam['release_date'] = pd.to_datetime(data_steam['release_date'], errors='coerce')
data_steam['metascore'] = pd.to_numeric(data_steam['metascore'], errors='coerce')
data_steam['price'] = pd.to_numeric(data_steam['price'], errors='coerce')
reemplazar_valores = {'publisher': '', 'genres': '', 'tags': '', 'discount_price': 0,
                      'specs': '', 'reviews_url': '', 'app_name': '', 'title': '',
                       'id': '', 'sentiment': '', 'developer': ''}
data_steam.fillna(value=reemplazar_valores, inplace=True)
data_steam = data_steam.dropna(subset=['price'])
data_steam = data_steam.dropna(subset=['release_date'])
data_steam = data_steam.dropna(subset=['metascore'])

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

# Armado del modelo predictivo
# Extracción datos desde listas anidadas en 'genres'
steam_unnested = data_steam.explode('genres').explode('tags').explode('specs')

# Conversión de 'release_date' a año
steam_unnested['release_year'] = steam_unnested['release_date'].dt.year

# Conversión de 'early_access' a valores numéricos 
steam_unnested['early_access'] = steam_unnested['early_access'].astype(int)

# Conversión de 'genres' a valores numéricos 
steam_dummies = pd.get_dummies(steam_unnested, columns=['genres'], prefix=[''], prefix_sep=[''])

# División del dataframe en sets de entrenamiento y prueba
X = steam_dummies[['release_year', 'metascore', 'early_access'] + list(steam_dummies.columns[steam_dummies.columns.str.contains('genres')])]
y = steam_dummies['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definición de la características polinomiales para el modelo predictivo
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Entrenamiento del modelo
model = BaggingRegressor() 
model.fit(X_train_poly, y_train)

# Evaluación del modelo
y_pred = model.predict(X_test_poly)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# r2 = r2_score(y_test, y_pred)
# print('RMSE:', rmse,'R2:', r2)

# Definición de la API que muestra la predicción de precios, RMSE y R2
# Obtención de todos los géneros únicos
all_genres = steam_unnested['genres'].unique().tolist()

# Para la API, también necesitas incluir 'early_access' como un parámetro en la función get_prediccion.
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
    # Usar dummies de 'genres'
    genres = list(steam_dummies.columns[steam_dummies.columns.str.contains('genres')])
    if genero not in all_genres:
        raise HTTPException(status_code=400, detail="Género no válido. Por favor use un género de la lista de géneros disponibles.")
    if early_access.lower() not in ['si', 'no']:
        raise HTTPException(status_code=400, detail="Valor no válido para 'early_access'. Elija 'si' o 'no'.")
    genre_data = [1 if genre == genero else 0 for genre in genres]
    early_access_data = 1 if early_access.lower() == 'si' else 0
    data = np.array([año, metascore, early_access_data] + genre_data).reshape(1, -1)
    
    # Aplicar la transformación polinomial
    data_poly = poly.transform(data)
    price = model.predict(data_poly)[0]
    return {'Precio': price, 'Error Cuadrático Medio': rmse}


