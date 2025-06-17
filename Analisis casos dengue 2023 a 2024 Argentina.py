import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np # Para calcular la raíz cuadrada del MSE (RMSE)

# =======================================================================
# 1. IMPORTACION DE DATASET
# =======================================================================
#### Dataset 2024 a 2025 ####
url_csv_Dengue2023a2024 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQk8BKTUwND6uZ6RpFRBwxv5SMr8xVoK3cleG_AUotr2zK-FrRI6-2qZ1GlJkJzusvvq-7MNjzBbmbK/pub?gid=1660622204&single=true&output=csv"
df_2023a2024 = pd.read_csv(url_csv_Dengue2023a2024, encoding='latin1')
print(df_2023a2024.head())

#### Dataset 2024 ####
url_csv_Dengue2024 = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRXenkq0jmn-p81nMoSMlFQS8bc5aRrrZOlP2KxOR_-cd8bwrTW_0BbCVZNBqpDk-ACnfKGLfPnJ897/pub?gid=1879732170&single=true&output=csv"
df_2024 = pd.read_csv(url_csv_Dengue2024, encoding='latin1')
print(df_2024.head())

#### Dataset 2024 Actualizado ####
url_csv_Dengue2024_Actualizado = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTuwb-1C9-igPPuo8gVzQj7c-Q0jLwBYreAczDPvMzTZDARt1TvPtNy3tW8-9JKKvsSPE8TrpGoCH7N/pub?gid=1178908227&single=true&output=csv"
df_2024Actualizado = pd.read_csv(url_csv_Dengue2024_Actualizado, encoding='latin1')
print(df_2024Actualizado.head())

#### Dataset 2024 a 2025 ####
url_csv_Dengue2024a2025= "https://docs.google.com/spreadsheets/d/1GoBDSlfFRiSxv81Bj-8akBJwUg9jkO0SNlyKiUjt-Do/export?format=csv&gid=2035242007"
df_2024a2025 = pd.read_csv(url_csv_Dengue2024a2025)
print(df_2024a2025.head())

#### Dataset 2025 ####
url_csv_Dengue2025= 'https://docs.google.com/spreadsheets/d/1buUjhC880hEblsZv1W3FdSl-vLFQpdiYHUz1XeOKlJU/export?format=csv&gid=1202678828'
df_2025 = pd.read_csv(url_csv_Dengue2025)
print(df_2025.head())

# =======================================================================
# 2. PRE-PROCESAMIENTO DE DATOS
# =======================================================================
# Corregir nombres de columnas en df_2025 para que coincidan con df_2024
df_2025.rename(columns={
    'ANIO_MIN': 'anio_min',
    'EVENTO': 'evento',
    'ID_GRUPO_ETARIO': 'id_grupo_etario',
    'GRUPO_ETARIO': 'grupo_etario',
    'SEPI_MIN': 'sepi_min'
}, inplace=True)

# Unir los dos DataFrames en uno solo
df_total = pd.concat([df_2023a2024, df_2024, df_2024Actualizado, df_2024a2025, df_2025], ignore_index=True)
print("¡Datasets de 2024 y 2025 combinados!")
print(f"El dataset total tiene {df_total.shape[0]} filas.")
print("-" * 50)

# =======================================================================
# LIMPIEZA Y PREPROCESAMIENTO
# =======================================================================
print("Iniciando limpieza de datos...")

# Eliminar columnas que no usaremos
columnas_a_eliminar = ['evento', 'departamento_residencia', 'provincia_residencia', 'grupo_etario', 'id_depto_indec_residencia']
df_limpio = df_total.drop(columns=columnas_a_eliminar)

# Eliminar las filas con valores nulos
df_limpio.dropna(inplace=True)

# Asegurar que las columnas tengan el tipo de dato correcto (números enteros)
df_limpio['id_prov_indec_residencia'] = df_limpio['id_prov_indec_residencia'].astype(int)
df_limpio['anio_min'] = df_limpio['anio_min'].astype(int)
df_limpio['id_grupo_etario'] = df_limpio['id_grupo_etario'].astype(int)
df_limpio['sepi_min'] = df_limpio['sepi_min'].astype(int)
df_limpio['cantidad'] = df_limpio['cantidad'].astype(int)

print("¡Limpieza completa!")
print(f"Dimensiones finales del dataframe limpio: {df_limpio.shape}")
print(df_limpio.head())
print("-" * 50)

# =======================================================================
# 3. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# =======================================================================
import seaborn as sns
import matplotlib.pyplot as plt

print("Generando visualizaciones del Análisis Exploratorio...")

# --- Gráfico de Casos por Provincia ---
provincia_map = {
    2: 'CABA', 6: 'Buenos Aires', 10: 'Catamarca', 14: 'Córdoba', 18: 'Corrientes',
    22: 'Chaco', 26: 'Chubut', 30: 'Entre Ríos', 34: 'Formosa', 38: 'Jujuy',
    42: 'La Pampa', 46: 'La Rioja', 50: 'Mendoza', 54: 'Misiones', 58: 'Neuquén',
    62: 'Río Negro', 66: 'Salta', 70: 'San Juan', 74: 'San Luis', 78: 'Santa Cruz',
    82: 'Santa Fe', 86: 'S. del Estero', 90: 'Tucumán', 94: 'Tierra del Fuego'
}
df_limpio_vis = df_limpio.copy()
df_limpio_vis['provincia_nombre'] = df_limpio_vis['id_prov_indec_residencia'].map(provincia_map)
df_limpio_vis.dropna(subset=['provincia_nombre'], inplace=True)

casos_por_provincia = df_limpio_vis.groupby("provincia_nombre")["cantidad"].sum().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=casos_por_provincia.head(10).values, y=casos_por_provincia.head(10).index)
plt.title("Top 10 Provincias con Mayor Cantidad de Casos")
plt.xlabel("Total de Casos de Dengue")
plt.ylabel("Provincia")
plt.show()

# --- Gráfico de Estacionalidad por Semana ---
casos_por_semana = df_limpio.groupby('sepi_min')['cantidad'].sum()
plt.figure(figsize=(15, 7))
sns.lineplot(x=casos_por_semana.index, y=casos_por_semana.values, marker='o')
plt.title('Estacionalidad del Dengue: Casos por Semana Epidemiológica')
plt.xlabel('Semana del Año')
plt.ylabel('Cantidad Total de Casos')
plt.grid(True)
plt.show()
print("-" * 50)

# =======================================================================
# =======================================================================
# Para obtener todos los valores únicos de id_depto_indec_residencia
print("Valores únicos de id_depto_indec_residencia:")
#print(df_2024['id_depto_indec_residencia'].unique())

# Para obtener todos los valores únicos de id_prov_indec_residencia
print("\nValores únicos de id_prov_indec_residencia:")
#print(df_2024['id_prov_indec_residencia'].unique())

# Para obtener todos los valores únicos de id_grupo_etario
print("\nValores únicos de id_grupo_etario:")
#print(df_2024['id_grupo_etario'].unique())
# =======================================================================
# =======================================================================

# =======================================================================
# =======================================================================
df_casos_por_provincia = df_2024.groupby("provincia_residencia")["cantidad"].sum().sort_values(ascending=False)
plt.figure(figsize=(12,6))
sns.barplot(x=df_casos_por_provincia.values, y=df_casos_por_provincia.index)
plt.title("Cantidad total de casos de dengue por provincia")
plt.xlabel("Casos de dengue")
plt.ylabel("Provincia")
plt.show()
# =======================================================================
# =======================================================================

# =======================================================================
# =======================================================================
plt.figure(figsize=(12,6))
sns.barplot(x=df_casos_por_provincia.values, y=df_casos_por_provincia.index)
plt.xscale("log")  # Escala logarítmica
plt.title("Cantidad total de casos de dengue por provincia (escala log)")
plt.xlabel("Casos de dengue (log)")
plt.ylabel("Provincia")
plt.show()
# =======================================================================
# =======================================================================

# =======================================================================
# =======================================================================
# Agrupa los datos por grupo etario y suma la cantidad de casos en cada grupo
df_casos_por_edad = df_2024.groupby('id_grupo_etario')['cantidad'].sum().reset_index()

# Ordena los grupos etarios de menor a mayor (si es que no lo están ya)
df_casos_por_edad.sort_values('id_grupo_etario', ascending=True, inplace=True)

# Visualiza el total de casos por grupo etario usando un gráfico de barras
plt.figure(figsize=(10,6))
sns.barplot(x='id_grupo_etario', y='cantidad', data=df_casos_por_edad)
plt.title("Cantidad total de casos de dengue por grupo etario")
plt.xlabel("Grupo Etario (ID)")
plt.ylabel("Casos de dengue")
plt.show()
# =======================================================================
# =======================================================================

# =======================================================================
# 4. MODELO PREDICTIVO
# =======================================================================
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

print("Entrenando el Modelo Predictivo (Random Forest)...")

# Seleccionar las variables para el modelo (features) y el objetivo (target)
features = ['anio_min', 'id_prov_indec_residencia', 'id_grupo_etario', 'sepi_min']
target = 'cantidad'

X = df_limpio[features]
y = df_limpio[target]

# Dividir datos en entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar el modelo
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print("¡Modelo entrenado!")

# Evaluar el modelo con los datos de prueba
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Resultados de la Evaluación del Modelo ---")
print(f"Error Absoluto Medio (MAE): {mae:.2f}")
print(f"Coeficiente de Determinación (R²): {r2:.2f}")

# =======================================================================
# =======================================================================
# --- GRÁFICO 1: CASOS POR GRUPO DE EDAD ---
import seaborn as sns
import matplotlib.pyplot as plt

# Mapa de IDs de grupos etarios para hacer el gráfico más claro
edad_map = {
    1: 'Neonato (<28d)', 2: 'Posneonato (<1a)', 3: '1-2 años', 4: '2-4 años',
    5: '5-9 años', 6: '10-14 años', 7: '15-19 años', 8: '20-24 años',
    9: '25-34 años', 10: '35-44 años', 11: '45-65 años', 12: '>65 años'
}
# Usamos .get para que si un ID no está, no de error
df_limpio_vis = df_limpio.copy()
df_limpio_vis['grupo_etario_nombre'] = df_limpio_vis['id_grupo_etario'].map(edad_map)

# Agrupar por nombre del grupo etario y sumar casos
casos_por_edad = df_limpio_vis.groupby("grupo_etario_nombre")["cantidad"].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=casos_por_edad.values, y=casos_por_edad.index, palette='viridis')
plt.title("Cantidad Total de Casos de Dengue por Grupo Etario")
plt.xlabel("Total de Casos de Dengue")
plt.ylabel("Grupo Etario")
plt.show()
# =======================================================================
# =======================================================================

# =======================================================================
# =======================================================================
# --- GRÁFICO 2: ESTACIONALIDAD EN TOP 3 PROVINCIAS ---
provincia_map = {
    2: 'CABA', 6: 'Buenos Aires', 10: 'Catamarca', 14: 'Córdoba', 18: 'Corrientes',
    22: 'Chaco', 26: 'Chubut', 30: 'Entre Ríos', 34: 'Formosa', 38: 'Jujuy',
    42: 'La Pampa', 46: 'La Rioja', 50: 'Mendoza', 54: 'Misiones', 58: 'Neuquén',
    62: 'Río Negro', 66: 'Salta', 70: 'San Juan', 74: 'San Luis', 78: 'Santa Cruz',
    82: 'Santa Fe', 86: 'S. del Estero', 90: 'Tucumán', 94: 'Tierra del Fuego'
}
df_limpio_vis['provincia_nombre'] = df_limpio_vis['id_prov_indec_residencia'].map(provincia_map)


# Identificar las 3 provincias con más casos
casos_por_provincia = df_limpio_vis.groupby("provincia_nombre")["cantidad"].sum().sort_values(ascending=False)
top_3_provincias = casos_por_provincia.head(3).index

# Filtrar el dataframe para quedarnos solo con esas provincias
df_top_provincias = df_limpio_vis[df_limpio_vis['provincia_nombre'].isin(top_3_provincias)]

# Agrupar por semana y provincia
casos_semanales_top_prov = df_top_provincias.groupby(['sepi_min', 'provincia_nombre'])['cantidad'].sum().reset_index()

# Graficar
plt.figure(figsize=(15, 7))
sns.lineplot(data=casos_semanales_top_prov, x='sepi_min', y='cantidad', hue='provincia_nombre', marker='o')
plt.title('Comparación de la Estacionalidad en las Provincias más Afectadas')
plt.xlabel('Semana Epidemiológica')
plt.ylabel('Cantidad de Casos')
plt.legend(title='Provincia')
plt.grid(True)
plt.show()
# =======================================================================
# =======================================================================
# --- ANÁLISIS 2: COMPARACIÓN DE BROTES AÑO A AÑO ---

print("\nAnálisis 2: Generando comparación de la magnitud de los brotes anuales...")

# Agrupamos por semana y año para ver la curva de cada año
casos_anuales = df_limpio.groupby(['anio_min', 'sepi_min'])['cantidad'].sum().reset_index()

# Graficamos cada año como una línea de color diferente
plt.figure(figsize=(18, 8))
sns.lineplot(data=casos_anuales, x='sepi_min', y='cantidad', hue='anio_min', palette='viridis', lw=2)
plt.title('Magnitud de los Brotes de Dengue Año a Año', fontsize=16)
plt.xlabel('Semana Epidemiológica')
plt.ylabel('Cantidad Total de Casos')
plt.legend(title='Año')
plt.grid(True)
plt.show()
# =======================================================================
# =======================================================================
# --- PREDICCIÓN DE UN NUEVO ESCENARIO ---

# 1. Definimos los datos del escenario que queremos predecir
datos_a_predecir = {
    'anio_min': [2025],
    'id_prov_indec_residencia': [22], # 22: Chaco
    'id_grupo_etario': [9],          # 9: 25 a 34 años
    'sepi_min': [15]                   # 15: Semana 15
}

# 2. Convertimos el diccionario a un DataFrame de pandas, porque el modelo espera este formato
nuevo_escenario_df = pd.DataFrame(datos_a_predecir)

# 3. Usamos el modelo ya entrenado (.predict()) para hacer la estimación
prediccion_casos = model.predict(nuevo_escenario_df)

# 4. Mostramos el resultado de una forma clara
print(f"--- Predicción para el escenario planteado ---")
print(f"Año: 2025, Provincia: Chaco, Grupo de Edad: 25-34 años, Semana: 15")
# Usamos int() para redondear, ya que no puede haber "medio caso"
print(f"El modelo predice una cantidad aproximada de: {int(prediccion_casos[0])} casos.")
# =======================================================================
# =======================================================================
# --- CASO DE USO 1: RIESGO ESTACIONAL ---

# Escenario A: PICO de la temporada (Semana 16)
escenario_A = {
    'anio_min': [2025],
    'id_prov_indec_residencia': [34], # 34: Formosa
    'id_grupo_etario': [9],          # 9: 25 a 34 años
    'sepi_min': [16]                   # 16: Semana de alto riesgo
}
df_escenario_A = pd.DataFrame(escenario_A)
prediccion_A = model.predict(df_escenario_A)

# Escenario B: TEMPORADA BAJA (Semana 40)
escenario_B = {
    'anio_min': [2025],
    'id_prov_indec_residencia': [34], # 34: Formosa
    'id_grupo_etario': [9],          # 9: 25 a 34 años
    'sepi_min': [40]                   # 40: Semana de bajo riesgo
}
df_escenario_B = pd.DataFrame(escenario_B)
prediccion_B = model.predict(df_escenario_B)

print("--- Comparación de Riesgo Estacional en Formosa ---")
print(f"Predicción para la SEMANA PICO (16): {int(prediccion_A[0])} casos.")
print(f"Predicción para la SEMANA BAJA (40): {int(prediccion_B[0])} casos.")
print("\nUtilidad: El modelo cuantifica el aumento drástico del riesgo durante el otoño, justificando la concentración de esfuerzos en esa época.")
# =======================================================================
# =======================================================================
# --- CASO DE USO 2: RIESGO GEOGRÁFICO ---

# Escenario A: Zona Caliente (Salta) en semana pico
escenario_A_geo = {
    'anio_min': [2025],
    'id_prov_indec_residencia': [66], # 66: Salta
    'id_grupo_etario': [9],
    'sepi_min': [16]
}
df_escenario_A_geo = pd.DataFrame(escenario_A_geo)
prediccion_A_geo = model.predict(df_escenario_A_geo)

# Escenario B: Zona Fría (Chubut) en la misma semana
escenario_B_geo = {
    'anio_min': [2025],
    'id_prov_indec_residencia': [26], # 26: Chubut
    'id_grupo_etario': [9],
    'sepi_min': [16]
}
df_escenario_B_geo = pd.DataFrame(escenario_B_geo)
prediccion_B_geo = model.predict(df_escenario_B_geo)


print("\n--- Comparación de Riesgo Geográfico en Semana 16 ---")
print(f"Predicción para una provincia del NORTE (Salta): {int(prediccion_A_geo[0])} casos.")
print(f"Predicción para una provincia del SUR (Chubut): {int(prediccion_B_geo[0])} casos.")
print("\nUtilidad: El modelo confirma que el riesgo geográfico es un factor determinante, validando la estrategia de focalizar recursos en el norte del país.")
# =======================================================================
# =======================================================================
