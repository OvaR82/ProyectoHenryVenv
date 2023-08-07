# Análisis y predicción de precios de juegos ofrecidos en la plataforma Steam
## Proyecto ML Ops de Henry: ETL y EDA sobre el dataset de steam_games.json

### Introducción
Este proyecto tiene como objetivo desarrollar un modelo predictivo para predecir el precio de un videojuego utilizando el dataset de steam_games. Para lograr este objetivo, se realizará un proceso de Extracción, Transformación y Carga (ETL) para limpiar y preparar los datos, así como un Análisis Exploratorio de Datos (EDA) para entender las relaciones entre las variables y detectar patrones interesantes.

### Problema de negocio
El problema de negocio consiste en predecir el precio de los videojuegos en la plataforma Steam, utilizando información como género, año de lanzamiento, puntaje en Metacritic (metascore), entre otros. 

### Objetivos
1. Realizar la limpieza inicial de los datos y asegurar que el dataframe contenga la información adecuada para el análisis.
2. Explorar las relaciones entre las variables y verificar si existen outliers o anomalías en los datos.
3. Realizar un Análisis de Componentes Principales (PCA) para reducir la dimensionalidad del dataset.
4. Visualizar la distribución de los datos y detectar patrones interesantes que puedan ser útiles para el modelo predictivo.

### Descripción de las variables del dataframe
* publisher: Empresa publicadora del contenido
* genres: Género del contenido
* app_name: Nombre del contenido
* title: Título del contenido
* url: URL de publicación del contenido
* release_date: Fecha de lanzamiento
* tags: Etiquetas de contenido
* discount_price: Precio de descuento
* reviews_url: Reviews de contenido
* specs: Especificaciones
* price: Precio del contenido (variable objetivo)
* early_access: Acceso temprano
* id: Identificador único de contenido
* developer: Desarrollador
* sentiment: Análisis de sentimientos
* metascore: Score por Metacritic

### Instrucciones para el proceso de ETL y EDA
1. Importar las librerías necesarias para el análisis: pandas, numpy, ast, matplotlib, seaborn y wordcloud.
2. Cargar los datos desde el archivo 'steam_games.json' en un dataframe.
3. Realizar la limpieza inicial de los datos para eliminar valores nulos y duplicados en la variable 'id'.
4. Realizar un Data Wrangling para ajustar el tipo de datos adecuado en las variables y extraer los datos de las variables que contienen listas (genres, tags, specs).
5. Realizar un Análisis de Componentes Principales (PCA) para reducir la dimensionalidad del dataset y calcular el porcentaje de varianza explicada.
6. Realizar visualizaciones y gráficos para explorar la distribución de los datos, como histogramas, violinplots, scatterplots y boxplots.
7. Identificar patrones interesantes y posibles relaciones entre las variables, prestando especial atención a las relaciones entre el precio (variable objetivo) y las demás características del videojuego.

### Conclusiones
El proceso de Extracción, Transformación y Carga (ETL) junto con el Análisis Exploratorio de Datos (EDA) realizado sobre el dataset de steam_games ha sido fundamental para preparar los datos y obtener una visión clara de la información disponible. A través de este proceso, hemos logrado entender la estructura y distribución de los datos, identificar posibles relaciones entre las variables y detectar patrones interesantes que pueden ser útiles para la creación de un modelo predictivo de precios de videojuegos.

En la fase de ETL, se llevaron a cabo diversas acciones para asegurar la calidad y coherencia de los datos. Se realizó una limpieza inicial para eliminar filas duplicadas y valores nulos en la variable 'id', lo que nos permitió tener un dataframe consistente y libre de inconsistencias. Además, se ajustaron los tipos de datos adecuados para cada columna y se extrajeron los datos contenidos en listas, como géneros, etiquetas y especificaciones, para facilitar su análisis y visualización.

En el Análisis Exploratorio de Datos, se utilizaron diversas técnicas de visualización para entender mejor la distribución de los datos y las relaciones entre las variables. Se crearon histogramas para visualizar la distribución de precios y puntajes de metacritic, boxplots para detectar outliers y entender la variabilidad de las variables, violinplots para comprender la distribución de precios y puntajes por género y acceso temprano, y scatterplots para explorar la relación entre el puntaje de metacritic y el precio de los videojuegos.

A través de estas visualizaciones, se pudo observar que los géneros "Indie" y "Action" son los más predominantes en el dataset, mientras que los géneros "Free to Play" y "Early Access" son los menos comunes. También se encontró que la mayoría de los precios de los videojuegos se encuentran en un rango entre 10 y 25 unidades, con algunos valores atípicos por encima de ese rango. Además, se descubrió que los videojuegos con puntajes de metacritic por encima de 60 tienden a tener precios más altos, lo que sugiere una relación positiva entre la calidad del juego y su precio.

El Análisis de Componentes Principales (PCA) fue una herramienta poderosa para reducir la dimensionalidad del dataset y permitir una mejor visualización de los datos. Mediante el PCA, se pudo proyectar el dataset en un espacio de menor dimensión, manteniendo una cantidad significativa de la varianza original. Esto facilita el análisis y la construcción de modelos predictivos, ya que se trabaja con un número reducido de componentes principales que explican la mayor parte de la variabilidad de los datos.

Este proceso de análisis y preparación de datos sienta las bases para un exitoso desarrollo de modelos de Machine Learning y permite tomar decisiones informadas que nos ayudarán a comprender mejor los factores que influyen en los precios de los videojuegos en la plataforma Steam. El siguiente paso será utilizar estos conocimientos para desarrollar un modelo predictivo adecuado y eficiente que sea útil para los objetivos planteados.

## Creación de la API y el modelo predictivo
1. **Implementación de FastAPI**: Se importa el módulo FastAPI y se crea una instancia de la aplicación llamada 'app', que se utilizará para definir las rutas y funciones de la API.
2. **Recuperación de datos desde un archivo .json**: Se lee el archivo 'steam_games.json' que contiene información sobre los juegos de Steam y se almacenan en una lista llamada 'dataset'. Los datos del archivo se convierten en objetos Python utilizando la función 'ast.literal_eval' y se extienden en la lista 'dataset'.
3. **Creación del DataFrame a partir del dataset obtenido**: Se crea un DataFrame llamado 'data_steam' a partir de la lista 'dataset', lo que facilita el procesamiento y análisis de los datos.
4. **Adecuación y limpieza del DataFrame**: Se realizan varias operaciones de limpieza y adecuación en el DataFrame 'data_steam':
   - Las columnas 'release_date', 'metascore' y 'price' se convierten al tipo de datos adecuado (fecha, numérico) utilizando las funciones 'pd.to_datetime' y 'pd.to_numeric'.
   - Se reemplazan los valores faltantes (NaN) en algunas columnas con valores específicos utilizando el diccionario 'reemplazar_valores'.
   - Se eliminan las filas que contienen valores faltantes en las columnas 'price', 'release_date' y 'metascore'.
5. **Definición de la API con información de los juegos según año de lanzamiento**: Se definen varias funciones utilizando el decorador '@app.get' para establecer las rutas de la API y sus correspondientes funciones. Estas funciones toman un parámetro 'año' que representa el año de lanzamiento de los juegos.
6. **Función que retorna los 5 géneros más vendidos**: La función 'genero' filtra el DataFrame 'data_steam' por el año especificado y luego explota la columna 'genres' para obtener cada género como una fila independiente. Luego, cuenta la cantidad de juegos en cada género y devuelve un diccionario con los 5 géneros más vendidos y la cantidad de juegos en cada uno.
7. **Función que retorna los juegos lanzados**: La función 'juegos' filtra el DataFrame 'data_steam' por el año especificado y devuelve una lista con los nombres de los juegos lanzados en ese año.
8. **Función que retorna el top 5 de especificaciones**: La función 'especificaciones' filtra el DataFrame 'data_steam' por el año especificado y explota la columna 'specs' para obtener cada especificación como una fila independiente. Luego, cuenta la cantidad de juegos asociados con cada especificación y devuelve un diccionario con las 5 especificaciones más comunes y la cantidad de juegos asociados a cada una.
9. **Función que retorna la cantidad de juegos con acceso temprano**: La función 'acceso_temprano' filtra el DataFrame 'data_steam' por el año especificado y cuenta la cantidad de juegos que tienen acceso temprano ('early_access' == True) en ese año.
10. **Función que retorna el tipo y cantidad de opiniones registradas**: La función 'opiniones' filtra el DataFrame 'data_steam' por el año especificado y selecciona las filas que tienen opiniones relevantes ('sentiment' se encuentra en la lista 'sentiments'). Luego, cuenta la cantidad de juegos asociados con cada tipo de opinión ('sentiment') y devuelve un diccionario con el tipo de opinión y la cantidad de juegos asociados a cada uno.
11. **Función que retorna el top 5 de juegos según su puntuación (metascore)**: La función 'metascore' filtra el DataFrame 'data_steam' por el año especificado y selecciona los 5 juegos con las puntuaciones más altas de metascore. Luego, devuelve un diccionario con el nombre de cada juego como clave y su puntuación de metascore como valor.
12. **Armado del modelo predictivo**: Se realiza el procesamiento necesario para entrenar un modelo predictivo de precios basado en el DataFrame 'data_steam'. El modelo predictivo se entrena utilizando el algoritmo de regresión de Bagging (BaggingRegressor) y se evalúa utilizando el error cuadrático medio (RMSE).
13. **Definición de la API para la predicción de precios**: Se define una función llamada 'get_prediccion' que toma varios parámetros relacionados con un juego (género, año de lanzamiento, metascore y disponibilidad de acceso temprano). La función utiliza el modelo predictivo previamente entrenado para predecir el precio del juego en función de los valores de entrada proporcionados.

## Resultado final
El archivo 'main.py' contiene una API implementada con FastAPI que proporciona información detallada sobre los juegos de Steam según su año de lanzamiento. La API ofrece diversas rutas que permiten obtener datos relevantes sobre géneros más vendidos, juegos lanzados, especificaciones populares, cantidad de juegos con acceso temprano, tipos de opiniones registradas y los juegos mejor calificados según su metascore. 

La API cuenta con las siguientes rutas y funciones:
1. `/genero/`: Devuelve los 5 géneros más vendidos en un año específico, junto con la cantidad de juegos asociados a cada género.
2. `/juegos/`: Muestra los nombres de los juegos lanzados en un año específico.
3. `/especificaciones/`: Retorna las 5 especificaciones más comunes de los juegos en un año específico, junto con la cantidad de juegos asociados a cada especificación.
4. `/acceso_temprano/`: Muestra la cantidad de juegos que tienen acceso temprano en un año específico.
5. `/opiniones/`: Devuelve la cantidad de juegos registrados para cada tipo de opinión relevante en un año específico.
6. `/metascore/`: Muestra los 5 juegos mejor calificados según su metascore en un año específico.

Sumado a esto, la API incorpora un modelo predictivo que estima el precio de un juego basándose en características como el año de lanzamiento, el metascore y la disponibilidad de acceso temprano. Para obtener una predicción de precio, se debe realizar una solicitud a la ruta `/prediccion/` y proporcionar valores válidos para los parámetros 'genero', 'año', 'metascore' y 'early_access'.
Cabe destacar que el modelo predictivo fue entrenado previamente utilizando el algoritmo de regresión de Bagging (BaggingRegressor) -luego de armar y probar sendos otros algoritmos similares- y se evaluó utilizando el error cuadrático medio (RMSE) para medir su precisión. La API proporciona una función de predicción ('get_prediccion') que utiliza el modelo entrenado para predecir el precio de un juego con base en las características proporcionadas por el usuario.

En resumen, el archivo 'main.py' ofrece una API completa y funcional que permite a los usuarios obtener información detallada sobre juegos de Steam según su año de lanzamiento, así como también realizar predicciones de precios basadas en un modelo predictivo entrenado. La API es una herramienta útil para explorar datos y obtener información valiosa sobre la industria de los videojuegos en Steam.
* Nota: Para garantizar el correcto funcionamiento de la API, es necesario ejecutar el código completo en el archivo 'main.py', asegurándose de incluir todas las importaciones y definiciones necesarias. 