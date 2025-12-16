# Proyecto de Inteligencia Artificial

Repositorio académico para la organización de proyectos de Inteligencia Artificial,
donde cada proyecto se estructura de forma clara y reproducible.

Cada proyecto contiene:
- Descripción del problema
- Stack tecnológico
- Arquitectura del sistema (fases)
- Capturas del proceso
- Qué se logró
- Código fuente
- Resultados cuantificables
- Datasheet o fuente del dataset utilizado

Este repositorio está orientado a fines académicos y de aprendizaje práctico.


# Fine-Tuning de un MLP con MLflow

En este proyecto se realizó la optimización (fine-tuning) de un modelo de red neuronal
multicapa (MLP) aplicado a un problema de clasificación supervisada, utilizando MLflow
para el seguimiento de experimentos y métricas.

El objetivo principal fue mejorar el desempeño del modelo base mediante el ajuste
de hiperparámetros y evaluar cuantitativamente los resultados obtenidos.



## Descripción del problema

El problema abordado en este proyecto consiste en mejorar el desempeño de un modelo
de clasificación supervisada aplicado a un conjunto de datos estructurados. El modelo
base presenta limitaciones en métricas de evaluación como el accuracy y el F1-score,
debido al uso de hiperparámetros fijos y no optimizados.

Estas limitaciones pueden provocar subajuste o sobreajuste, reduciendo la capacidad
del modelo para generalizar correctamente sobre nuevos datos. Por esta razón, se
plantea la necesidad de aplicar técnicas de fine-tuning que permitan optimizar la
arquitectura de la red neuronal y los parámetros de entrenamiento.

El objetivo es obtener un modelo más robusto, estable y reproducible, mejorando su
rendimiento mediante un proceso sistemático de experimentación.

## Stack tecnológico

El desarrollo de este proyecto se realizó utilizando el siguiente stack tecnológico:

- Lenguaje de programación: Python 3
- Librerías principales:
  - scikit-learn
  - mlflow
  - pandas
  - numpy
- Entorno de desarrollo:
  - Jupyter Notebook / Google Colab
- Herramientas de seguimiento experimental:
  - MLflow Tracking
- Modelo utilizado:
  - MLPClassifier (Multi Layer Perceptron)

Este stack permitió implementar, entrenar, evaluar y documentar el proceso de
optimización del modelo de forma reproducible.

## Arquitectura del sistema – Fases

La arquitectura del proyecto se organizó en las siguientes fases:

### Fase 1 – Preparación de datos
- Carga del conjunto de datos
- Separación de variables predictoras y variable objetivo
- Preprocesamiento mediante pipelines

### Fase 2 – Modelo base
- Implementación de un MLPClassifier inicial
- Entrenamiento con hiperparámetros por defecto
- Evaluación preliminar del desempeño

### Fase 3 – Fine-tuning del modelo
- Ajuste del número de capas ocultas
- Ajuste del número de neuronas por capa
- Selección de la función de activación
- Configuración del batch size y número de iteraciones
- Registro de experimentos con MLflow

### Fase 4 – Evaluación y comparación
- Comparación de métricas entre configuraciones
- Selección del modelo con mejor desempeño global

## Capturas


## Qué se logró

Con el desarrollo de este proyecto se logró:

- Mejorar el desempeño del modelo base mediante fine-tuning
- Identificar una configuración óptima de hiperparámetros
- Reducir el riesgo de sobreajuste del modelo
- Implementar un flujo de trabajo reproducible
- Integrar MLflow como herramienta de seguimiento experimental
- Analizar de forma objetiva el impacto de cada ajuste realizado


## Código
[FineTunning.pdf](https://github.com/user-attachments/files/24174526/FineTunning.pdf)
pip install mlflow
import zipfile
import os
import pandas as pd

# Ruta REAL del ZIP en Descargas
zip_path = r"C:\Users\ASUS\Downloads\neurofibromatosis+type+1+clinical+symptoms+of+familial+and+sporadic+cases.zip"

# Carpeta donde se extraerán los archivos
extract_folder = "dataset_nf1"
os.makedirs(extract_folder, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_folder)
    print("Archivos extraídos correctamente en:", extract_folder)
    print("Contenido del ZIP:")
    print(zip_ref.namelist())

import pandas as pd
import copy
import os

# Ruta al archivo dentro de la carpeta extraída
data_path = os.path.join("dataset_nf1", "dataset-uci.xlsx")

# 1) Cargar el Excel
df = pd.read_excel(data_path)

# 2) (Por si acaso) eliminar columna de índice automática si existiera
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# 3) Renombrar las 19 columnas con nombres "tipo German Credit"
nombresVariables = [
    'CASE_TYPE',             # Case Type: 0 = esporádico, 1 = familiar
    'TUMOUR_CASE',           # Tumour Case: 0 = sin tumores, 1 = con tumores
    'AGE_MOTHER',            # Age of Mother
    'AGE_FATHER',            # Age of Father
    'AGE_FIRST_DIAGNOSIS',   # Age at First Diagnosis
    'CAFE_AU_LAIT',          # Café au lait (CLS)
    'AXILLARY_FRECKLES',     # Axillary Freckles
    'INGUINAL_FRECKLES',     # Inguinal Freckles
    'LISCH_NODULES',         # Lisch Nodules
    'DERMAL_NEUROFIBROMINS', # Dermal Neurofibromins
    'PLEXIFORM_NEUROFIBROMINS', # Plexiform Neurofibromins
    'OPTIC_GLIOMA',          # Optic Glioma
    'SKELETAL_DYSPLASIA',    # Skeletal Dysplasia
    'LEARNING_DISABILITY',   # Learning Disability
    'HYPERTENSION',          # Hypertension
    'ASTROCYTOMA',           # Astrocytoma
    'HAMARTOMA',             # Hamartoma
    'SCOLIOSIS',             # Scoliosis
    'OTHER_SYMPTOMS'         # Other Symptoms
]

# Asegurarnos de que el número de nombres coincide
print("Columnas originales en el Excel:", len(df.columns))
print(df.columns)

df.columns = nombresVariables  # Asignamos los nombres "limpios"

# 4) Copia de trabajo (como en tus otras prácticas)
dataframe = copy.deepcopy(df)

print("Dataset NF1 cargado correctamente")
print("Cantidad de observaciones (pacientes):", dataframe.shape[0])
print("Cantidad de variables (características):", dataframe.shape[1])
print("Dimensiones totales (filas, columnas):", dataframe.shape)

dataframe.head()


# Formato a la variable de salida (de estudio): FAMILIAL
# 0 = caso esporádico, 1 = caso familiar

# Verificar qué valores tiene actualmente
print("Valores únicos en FAMILIAL antes de formatear:")
print(dataframe['CASE_TYPE'].value_counts())

# Si ya está en 0 y 1, solo nos aseguramos de que sea entero
dataframe['CASE_TYPE'] = dataframe['CASE_TYPE'].astype(int)

# Vista rápida
dataframe[['CASE_TYPE']].head()



# Separación de variable de salida y eliminación de variables poco relevantes

# Variable de salida (de estudio): tipo de caso (0 = esporádico, 1 = familiar)
Y = dataframe['CASE_TYPE']

# Eliminamos variables consideradas menos relevantes o redundantes
dataframe = dataframe.drop(['AGE_FATHER'], axis=1)
dataframe = dataframe.drop(['HYPERTENSION'], axis=1)

# Eliminamos del X la columna de salida
dataframe = dataframe.drop(['CASE_TYPE'], axis=1)

print("Shape de X después de eliminar variables:", dataframe.shape)
dataframe.head()

import seaborn as sns
import matplotlib.pyplot as plt

# Asegúrate de que Y ya está definido como:
# Y = dataframe_original['CASE_TYPE']   (antes de hacer los drops en X)

plt.figure(figsize=(5,4))
sns.countplot(x=Y)

plt.xlabel('Tipo de caso (CASE_TYPE)')
plt.ylabel('Cantidad de pacientes')
plt.title('Frecuencia de casos: esporádicos vs familiares')
plt.xticks([0, 1], ['Esporádico', 'Familiar'])

plt.show()

# ==== Determinación de tipos de variables para NF1 ====

# Verificamos las columnas actuales de X (solo por seguridad)
print("Columnas actuales de X:")
print(list(dataframe.columns))

# 1) Variables categóricas ordinales
# En este dataset no hay una escala clara tipo "bajo/medio/alto",
# así que dejamos esta lista vacía.
categorical_ordinal_features = []

# 2) Variables categóricas nominales (binarias 0/1)
categorical_nominal_features = [
    'TUMOUR_CASE',
    'CAFE_AU_LAIT',
    'AXILLARY_FRECKLES',
    'INGUINAL_FRECKLES',
    'LISCH_NODULES',
    'DERMAL_NEUROFIBROMINS',
    'PLEXIFORM_NEUROFIBROMINS',
    'OPTIC_GLIOMA',
    'SKELETAL_DYSPLASIA',
    'LEARNING_DISABILITY',
    'ASTROCYTOMA',
    'HAMARTOMA',
    'SCOLIOSIS',
    'OTHER_SYMPTOMS'
]

# 3) Variables numéricas
numeric_features = [
    'AGE_MOTHER',
    'AGE_FIRST_DIAGNOSIS'
]

def analisisVariables(dataframe, categorical_ordinal_features, categorical_nominal_features):
    cantidadTotalVariables = len(dataframe.columns)
    print('Cantidad de variables antes de transformación de variables: ', cantidadTotalVariables)

    cantidadVariablesNominales = len(categorical_nominal_features)
    cantidadVariablesBinarias = 0

    for variable in categorical_nominal_features:
        cantidadCategorias = dataframe[variable].nunique()
        cantidadVariablesBinarias = cantidadVariablesBinarias + cantidadCategorias
        print('Cantidad de categorías en la variable categórica nominal', variable, ':', cantidadCategorias)

    print('Cantidad de variables binarias que reemplazarán a las variables categóricas nominales: ', cantidadVariablesBinarias)

    cantidadTotalVariablesConTransformacion = cantidadTotalVariables - cantidadVariablesNominales + cantidadVariablesBinarias
    return cantidadTotalVariablesConTransformacion

cantidadTotalVariablesConTransformacion = analisisVariables(
    dataframe,
    categorical_ordinal_features,
    categorical_nominal_features
)

print('Cantidad de variables que habrá después de la transformación de variables: ',
      cantidadTotalVariablesConTransformacion)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer

# OJO: aquí se asume que YA definiste antes:
# categorical_ordinal_features
# categorical_nominal_features
# cantidadTotalVariablesConTransformacion

# 1. Preprocesador de variables categóricas a numéricas (ordinales y nominales)
categorical_ordinal_transformer = Pipeline(steps=[
    ('ordinal', OrdinalEncoder())
])

categorical_nominal_transformer_ConNombres = Pipeline(steps=[
    ('onehot', OneHotEncoder(sparse_output=False, handle_unknown="ignore"))  # corregido
])

preprocesorCategoricoNumericoConNombres = ColumnTransformer(
    transformers=[
        ('catord', categorical_ordinal_transformer, categorical_ordinal_features),
        ('catnom', categorical_nominal_transformer_ConNombres, categorical_nominal_features)
    ],
    remainder='passthrough',
    n_jobs=-1
)

# 2. Normalización y Scaling: MinMaxScaler
minmax_transformer = Pipeline(steps=[
    ('minmax', MinMaxScaler(feature_range=(0, 1)))
])

preprocesorMinMax = ColumnTransformer(
    transformers=[
        ('tranminmax', minmax_transformer, list(range(cantidadTotalVariablesConTransformacion)))
    ],
    remainder='passthrough'
)

# 3. Estandarización: StandardScaler
standardscaler_transformer = Pipeline(steps=[
    ('scaler', StandardScaler(with_mean=True, with_std=True))
])

preprocesorStandardScaler = ColumnTransformer(
    transformers=[
        ('transcaler', standardscaler_transformer, list(range(cantidadTotalVariablesConTransformacion)))
    ],
    remainder='passthrough'
)

# 4. Imputación de valores faltantes: SimpleImputer
simpleimputer_transformer = Pipeline(steps=[
    ('simpleimputer', SimpleImputer(strategy='median'))  # también puedes usar 'most_frequent'
])

preprocesorSimpleImputer = ColumnTransformer(
    transformers=[
        ('transimpleimputer', simpleimputer_transformer, list(range(cantidadTotalVariablesConTransformacion)))
    ],
    remainder='passthrough'
)

from sklearn.pipeline import Pipeline
from sklearn import set_config

# Opcional: para ver el diagrama bonito del pipeline
set_config(display='diagram')

# Construcción de Pipeline con transformadores
pipe = Pipeline(steps=[
    ('prepcn', preprocesorCategoricoNumericoConNombres),  # 1) categóricas → numéricas (One-Hot)
    ('prepstandard', preprocesorStandardScaler)           # 2) escalado StandardScaler
    # Si en lugar de StandardScaler quisieras MinMax, usas esta línea:
    # ('prepminmax', preprocesorMinMax)
])

pipe


import copy
import pandas as pd

# ==============================
# 1. Ejecutar el pipeline
# ==============================

# Trabajamos solo con las variables de entrada (X)
dataframeTransformado = copy.deepcopy(dataframe)

X_Transformado = pipe.fit_transform(dataframeTransformado)

print('********** Pipeline aplicado')
print('********** Transformador categórico nominal:')
print(pipe.named_steps['prepcn'].transformers_[1])   # bloque 'catnom'

# ==============================
# 2. Obtener los nombres de columnas nuevas
# ==============================

cnamesDataset1 = []

# (1) Si hubiera ordinales (en tu caso está vacío, pero lo dejamos general)
if len(categorical_ordinal_features) != 0:
    cnamesDataset1.extend(categorical_ordinal_features)

# (2) Nombres creados por OneHotEncoder para las variables nominales
if len(categorical_nominal_features) != 0:
    cnamesDataset2 = (
        pipe.named_steps['prepcn']        # ColumnTransformer
            .transformers_[1][1]          # [1] = ('catnom', pipeline, columnas) → [1] es el Pipeline interno
            .named_steps['onehot']        # dentro del Pipeline
            .get_feature_names_out(categorical_nominal_features)
    )
    cnamesDataset1.extend(list(cnamesDataset2))

# (3) Agregar las variables numéricas al final
cnamesDataset3 = numeric_features
cnamesDataset1.extend(cnamesDataset3)

print('********** Cantidad de variables:', len(cnamesDataset1))
print('********** Lista de variables:')
print(cnamesDataset1)

# ==============================
# 3. Formar el DataFrame transformado
# ==============================

dataframeTransformado = pd.DataFrame(
    data=X_Transformado,
    columns=cnamesDataset1
)

# Guardar dataset SIN etiquetas (solo X)
dataframeTransformado.to_csv(
    "NF1_DatasetTransformadoSinEtiquetas.csv",
    sep=";",
    index=False
)

# Agregar la variable de salida (Y = CASE_TYPE) al final
dataframeTransformado = pd.concat(
    [dataframeTransformado, Y.reset_index(drop=True)],
    axis=1
)

# Guardar dataset CON etiquetas (X + Y)
dataframeTransformado.to_csv(
    "NF1_DatasetTransformadoConEtiquetas.csv",
    sep=";",
    index=False
)

# Ver primeras filas
dataframeTransformado.head()


import pickle

# Función para guardar un Pipeline (o cualquier modelo entrenado)
def guardarPipeline(pipeline, nombreArchivo):
    with open(nombreArchivo + '.pickle', 'wb') as handle:
        pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f" Pipeline guardado como {nombreArchivo}.pickle")

# Función para cargar un Pipeline (o modelo) previamente guardado
def cargarPipeline(nombreArchivo):
    with open(nombreArchivo + '.pickle', 'rb') as handle:
        pipeline = pickle.load(handle)
    print(f" Pipeline cargado desde {nombreArchivo}.pickle")
    return pipeline


# El pipe ya está "fit" porque usaste fit_transform
guardarPipeline(pipe, "NF1_Pipeline_Preprocesamiento")


import pickle

# Función para guardar un pipeline
def guardarPipeline(pipeline, nombreArchivo):
    with open(nombreArchivo + '.pickle', 'wb') as handle:
        pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Nombre que usaremos SIEMPRE para este dataset
nombreArchivoPreprocesador = 'NF1_pipePreprocesadores'

# Guardar tu pipeline 'pipe' en un archivo .pickle
guardarPipeline(pipe, nombreArchivoPreprocesador)

print(" Pipeline guardado como:", nombreArchivoPreprocesador + ".pickle")

# FASE 2 - Celda 2.1
# Imports extra y funciones para guardar / cargar

import copy
import pickle
import joblib
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import set_config

# Función para cargar el pipeline de preprocesamiento desde disco (.pickle)
def cargarPipeline(nombreArchivo):
    with open(nombreArchivo + '.pickle', 'rb') as handle:
        pipeline = pickle.load(handle)
    return pipeline

# Guardar la red neuronal (pipeline + MLPClassifier) en un archivo .joblib
def guardarNN(model, nombreArchivo):
    print("Guardando Red Neuronal (pipeline con MLPClassifier) en archivo...")
    joblib.dump(model, nombreArchivo + '.joblib')
    print("Red Neuronal guardada en", nombreArchivo + '.joblib')

# Cargar la red neuronal (pipeline + MLPClassifier) desde un .joblib
def cargarNN(nombreArchivo):
    print("Cargando Red Neuronal (pipeline con MLPClassifier) desde archivo...")
    model = joblib.load(nombreArchivo + '.joblib')
    print("Red Neuronal cargada correctamente")
    return model
from sklearn.model_selection import train_test_split 
import copy

# Y ya lo definiste en la Fase 1 como:
# Y = df['CASE_TYPE']   (o dataframe_original['CASE_TYPE'])
Yval = Y.values

# X es tu dataframe de entrada YA SIN:
# 'CASE_TYPE', 'AGE_FATHER', 'HYPERTENSION'
X = copy.deepcopy(dataframe)

print("Shape de X:", X.shape)
print("Shape de Y:", Yval.shape)


# ==============================
# 2. Revisar y corregir NaN en X
# ==============================

print("\nValores faltantes por columna ANTES de imputar:")
print(X.isna().sum())

# Columnas numéricas con NaN en este dataset
columnas_con_nan = ['AGE_MOTHER', 'AGE_FIRST_DIAGNOSIS']

for col in columnas_con_nan:
    mediana = X[col].median()
    X[col] = X[col].fillna(mediana)

print("\nValores faltantes por columna DESPUÉS de imputar:")
print(X.isna().sum())

# ==============================
# 3. División 80% / 20%
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    Yval,
    test_size=0.2,
    random_state=42,
    stratify=Yval
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


# FASE 2 - Celda 2.2
# Definir X (entradas) e y (salida) y corregir NaN

# y: variable de salida (CASE_TYPE), viene de la Fase 1
Yval = Y.values

# X: todas las variables de entrada (dataframe ya SIN CASE_TYPE, AGE_FATHER, HYPERTENSION)
X = copy.deepcopy(dataframe)

print("Shape de X:", X.shape)
print("Shape de Y:", Yval.shape)

print("\nValores faltantes por columna ANTES de imputar:")
print(X.isna().sum())

# Imputación por la mediana en las columnas numéricas con NaN
for col in ['AGE_MOTHER', 'AGE_FIRST_DIAGNOSIS']:
    mediana = X[col].median()
    X[col] = X[col].fillna(mediana)

print("\nValores faltantes por columna DESPUÉS de imputar:")
print(X.isna().sum())


# FASE 2 - Celda 2.3
# Carga del pipeline NF1_pipePreprocesadores que hiciste en la Fase 1

nombreArchivoPreprocesador = 'NF1_pipePreprocesadores'

pipe_prep = cargarPipeline(nombreArchivoPreprocesador)

cantidadPasos = len(pipe_prep.steps)
print("Cantidad de pasos en pipe_prep:", cantidadPasos)

# Mostrar el diagrama del pipeline
set_config(display='diagram')
pipe_prep


# Imprime los pasos del pipeline para verificar el nombre del paso con el modelo
print(pipeNN.named_steps)


from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import seaborn as sns
import matplotlib.pyplot as plt
import copy

# ==============================
# 1. Definir X e y (igual que en el ejemplo)
# ==============================

# y: variable de salida (tipo de caso)
Yval = Y.values   # Y viene de la fase 1: Y = dataframe_original['CASE_TYPE']

# X: todas las variables de entrada (dataframe ya SIN CASE_TYPE, AGE_FATHER, HYPERTENSION)
X = copy.deepcopy(dataframe)

print("Shape de X:", X.shape)
print("Shape de Y:", Yval.shape)

# ==============================
# 2. Imputar valores faltantes en columnas numéricas
# ==============================

print("\nValores faltantes por columna ANTES de imputar:")
print(X.isna().sum())

for col in ['AGE_MOTHER', 'AGE_FIRST_DIAGNOSIS']:
    mediana = X[col].median()
    X[col] = X[col].fillna(mediana)

print("\nValores faltantes por columna DESPUÉS de imputar:")
print(X.isna().sum())

# ==============================
# 3. Partición entrenamiento / prueba (80% / 20%)
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    Yval,
    test_size=0.2,
    random_state=42,
    stratify=Yval
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

# ==============================
# 4. Crear una copia del pipeline de preprocesamiento
#    (el que cargaste desde NF1_pipePreprocesadores.pickle)
# ==============================

pipeNN = copy.deepcopy(pipe)   # pipe tiene: prepcn + prepstandard

# ==============================
# 5. Definir la "Red Neuronal" (MLPClassifier)
#    Aquí se ven los análogos de optimizer, epochs y batch_size
# ==============================

modelNN = MLPClassifier(
    hidden_layer_sizes=(4,),  # capas ocultas
    activation='logistic',
    solver='adam',               # ≈ optimizer
    batch_size=16,               # ≈ batch_size
    max_iter=10,                # ≈ epochs
    random_state=42
)

# ==============================
# 6. Añadir el modelo como último paso del pipeline
#    (equivalente a pipe.steps.append(['modelNN', model]) del ejemplo)
# ==============================

pipeNN.steps.append(('modelNN', modelNN))

print("Pasos de pipeNN:", [name for name, _ in pipeNN.steps])

# ==============================
# 7. Entrenar el pipeline completo
# ==============================

pipeNN.fit(X_train, y_train)
print(' Modelo base entrenado')

# ==============================
# 8. Guardar el modelo entrenado
# ==============================

guardarNN(pipeNN, 'NF1_modeloRedNeuronalBase')
print(' Modelo Base Guardado')

# ==============================
# 9. Gráfico de frecuencia de clases en entrenamiento
# ==============================

sns.countplot(x=y_train)
plt.xlabel('CASE_TYPE (0 = esporádico, 1 = familiar)')
plt.ylabel('Cantidad de pacientes')
plt.title('Frecuencia de clases en el conjunto de entrenamiento')
plt.show()


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y_pred = pipeNN.predict(X_test)

print("Accuracy en test:", accuracy_score(y_test, y_pred))
print("\nMatriz de confusión:\n", confusion_matrix(y_test, y_pred))
print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))


import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ==============================
# 1. Predicción con el modelo base
# ==============================

# pipeNN es tu pipeline entrenado (prepcn + prepstandard + modelNN)
y_pred = pipeNN.predict(X_test)          # ya devuelve 0 o 1 directamente

# Si también quieres las probabilidades (por ejemplo, para ver el umbral 0.5)
y_prob = pipeNN.predict_proba(X_test)[:, 1]   # prob de clase 1

# DataFrame con valores reales vs predichos
dataframeFinal = pd.DataFrame({
    'real': y_test,
    'prediccion': y_pred,
    'prob_clase_1': y_prob
})

np.set_printoptions(formatter={'float': lambda X: "{0:0.0f}".format(X)})
dataframeFinal.head(20)


import numpy as np
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# y_pred viene de tu modelo:
# y_pred = pipeNN.predict(X_test)

# ---- Errores tipo regresión (con etiquetas 0/1) ----
MAE  = metrics.mean_absolute_error(y_test, y_pred)
MSE  = metrics.mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)

MAE  = str(round(MAE, 4))
MSE  = str(round(MSE, 4))
RMSE = str(round(RMSE, 4))

print('Mean Absolute Error (MAE):', MAE)  
print('Mean Squared Error (MSE):', MSE)  
print('Root Mean Squared Error (RMSE):', RMSE) 
print('Accuracy:', round(accuracy,4)) 

# ---- Matriz de confusión ----
print('Confusion_matrix:')
y_test_transformado = y_test
y_pred_transformado = y_pred

cm = confusion_matrix(y_test_transformado, y_pred_transformado)  
print(cm)  
tn, fp, fn, tp = cm.ravel()
print(tn, fp, fn, tp)

# ---- Precision, Recall, F1 ----
precision = precision_score(y_test_transformado, y_pred_transformado)
recall    = recall_score(y_test_transformado, y_pred_transformado)
f1        = f1_score(y_test_transformado, y_pred_transformado)

print('Precision:', round(precision,4))
print('Recall   :', round(recall,4))
print('F1       :', round(f1,4))



import mlflow
import mlflow.sklearn
import copy
import numpy as np
import pickle
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

# --------------------------
# 1. Cargar pipeline de PREPROCESAMIENTO
# --------------------------
def cargarPipeline(nombreArchivo):
    with open(nombreArchivo + '.pickle', 'rb') as handle:
        pipeline = pickle.load(handle)
    return pipeline

nombreArchivoPreprocesador = 'NF1_pipePreprocesadores'
pipe_prep = cargarPipeline(nombreArchivoPreprocesador)

print("Pasos del pipeline de preprocesamiento:")
print([name for name, _ in pipe_prep.steps])

# --------------------------
# 2. Definir X e Y para todo el dataset
# --------------------------
Yval = Y.values  # Salida
X = copy.deepcopy(dataframe)  # Datos de entrada

# Asegurarnos de que no haya valores NaN en las columnas críticas
for col in ['AGE_MOTHER', 'AGE_FIRST_DIAGNOSIS']:
    mediana = X[col].median()
    X[col] = X[col].fillna(mediana)

print("Shape de X completo:", X.shape)
print("Shape de Y completo:", Yval.shape)

# --------------------------
# 3. Función de construcción del modelo con Fine Tuning (MLflow)
# --------------------------

def build_model():
    """
    Construye y devuelve un MLPClassifier que actúa como 'Red Neuronal'.
    """
    modelNN = MLPClassifier(
        hidden_layer_sizes=(4,4,),  # capas ocultas
        activation='logistic',
        solver='adam',               # ≈ optimizer
        batch_size=32,               # ≈ batch_size
        max_iter=100,                # ≈ epochs
        random_state=42
    )
    return modelNN

# Crear el modelo base
estimator = build_model()

# --------------------------
# 4. Construir pipeline COMPLETO = preprocesamiento + modeloNN
# --------------------------
pipeCV = Pipeline(steps=[
    ('prepcn', pipe_prep.named_steps['prepcn']),
    ('prepstandard', pipe_prep.named_steps['prepstandard']),
    ('modelNN', estimator)
])

print("Pasos en pipeCV:", [name for name, _ in pipeCV.steps])

# --------------------------
# 5. Configuración de validación cruzada (K-Fold con ShuffleSplit)
# --------------------------
numFolds = 5
kfold = ShuffleSplit(
    n_splits=numFolds,
    test_size=0.2,       
    random_state=42
)

# --------------------------
# 6. Ejecución de Cross-Validation con el pipeline completo y MLflow
# --------------------------

# Abrir el experimento MLflow para seguimiento
with mlflow.start_run():
    
    # Registrar el modelo y las configuraciones en MLflow
    mlflow.log_param("numFolds", numFolds)
    mlflow.log_param("batch_size", 32)
    mlflow.log_param("hidden_layer_sizes", (4, 4))
    
    # Ejecutar cross_val_score con el pipeline completo
    cvs = cross_val_score(
        pipeCV,
        X,
        Yval,
        cv=kfold,
        n_jobs=-1,  # usar todos los núcleos disponibles
        error_score="raise"
    )
    
    # Registrar el resultado en MLflow
    mlflow.log_metric("accuracy_mean", cvs.mean())
    
    print("Scores (accuracy por fold):")
    for i, score in enumerate(cvs):
        print(f"Fold {i+1}: {score:.4f}")

    print("Accuracy promedio del Modelo Base: ", cvs.mean())
    
    # Registrar el modelo final
    mlflow.sklearn.log_model(pipeCV, "model")
    
# Aquí termina la optimización y almacenamiento del modelo con MLflow.

import mlflow

# Obtener la última corrida activa
latest_run = mlflow.active_run()

if latest_run is not None:
    print("Último run_id:", latest_run.info.run_id)

    # Acceder a los parámetros y métricas de la última corrida
    print("Parámetros:", latest_run.data.params)
    print("Métricas:", latest_run.data.metrics)

else:
    print("No hay una corrida activa en este momento.")


import mlflow
import mlflow.sklearn
import copy
import pickle
import numpy as np
from datetime import datetime, timedelta
from time import time
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

# ==============================
# 1. Función para mostrar el tiempo en formato bonito
# ==============================

def GetTime(gs_time):
    sec = timedelta(seconds=gs_time)
    d = datetime(1, 1, 1) + sec
    tiempoTotal = "%d días: %d horas: %d min: %d seg" % (d.day-1, d.hour, d.minute, d.second)
    return tiempoTotal

# ==============================
# 2. Cargar pipeline de PREPROCESAMIENTO
# ==============================

def cargarPipeline(nombreArchivo):
    with open(nombreArchivo + '.pickle', 'rb') as handle:
        pipeline = pickle.load(handle)
    return pipeline

nombreArchivoPreprocesador = 'NF1_pipePreprocesadores'
pipe_prep = cargarPipeline(nombreArchivoPreprocesador)

print("Pasos del pipeline de preprocesamiento:")
print([name for name, _ in pipe_prep.steps])

# ==============================
# 3. Definir X e Y para TODO el dataset
# ==============================

Yval = Y.values  # Salida
X = copy.deepcopy(dataframe)  # Datos de entrada

# Imputar NaN en las numéricas
for col in ['AGE_MOTHER', 'AGE_FIRST_DIAGNOSIS']:
    mediana = X[col].median()
    X[col] = X[col].fillna(mediana)

print("Shape de X:", X.shape)
print("Shape de Y:", Yval.shape)

# ==============================
# 4. Construir el modelo base (MLPClassifier)
# ==============================

def build_model(batch_size=16, max_iter=200, solver='adam'):
    """
    Crea y devuelve un MLPClassifier con los hiperparámetros que se optimizarán.
    """
    modelNN = MLPClassifier(
        hidden_layer_sizes=(6, 6),
        activation='logistic',
        solver=solver,     # Optimizer
        batch_size=batch_size,     # Tamaño del batch
        max_iter=max_iter,      # Número de épocas
        random_state=42
    )
    return modelNN

# ==============================
# 5. Construir pipeline completo para MLflow
# ==============================

pipe = Pipeline(steps=[
    ('prepcn', pipe_prep.named_steps['prepcn']),
    ('prepstandard', pipe_prep.named_steps['prepstandard']),
    ('modelNN', build_model())
])

print("Pasos en pipeline de optimización:", [name for name, _ in pipe.steps])

# ==============================
# 6. Definir la rejilla de hiperparámetros
# ==============================

batch_size = [16, 32, 64]
epochs = [5, 10, 20]  # Puedes ajustar si quieres que demore menos
optimizer = ['adam', 'sgd']

# Crear la rejilla de parámetros
parameters = {
    'modelNN__batch_size': batch_size,
    'modelNN__max_iter': epochs,
    'modelNN__solver': optimizer
}

print("\nParámetros que se van a explorar en el GridSearch:")
print(parameters)

# ==============================
# 7. Ejecutar optimización con MLflow
# ==============================

# Iniciar un nuevo experimento en MLflow
with mlflow.start_run():
    # Búsqueda manual de hiperparámetros
    best_score = 0
    best_params = {}

    for bs in batch_size:
        for ep in epochs:
            for opt in optimizer:
                # Iniciar una sub-corrida para cada combinación de hiperparámetros
                with mlflow.start_run(nested=True):
                    # Registra los hiperparámetros para esta sub-corrida
                    mlflow.log_param("batch_size", bs)
                    mlflow.log_param("max_iter", ep)
                    mlflow.log_param("solver", opt)

                    # Ajustar el modelo con los parámetros
                    model = build_model(batch_size=bs, max_iter=ep, solver=opt)
                    pipe.set_params(modelNN=model)

                    # Configuración de validación cruzada
                    kfold = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
                    tic = time()
                    scores = []

                    for train_index, test_index in kfold.split(X):
                        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                        Y_train, Y_test = Yval[train_index], Yval[test_index]

                        # Ajuste y evaluación del modelo
                        pipe.fit(X_train, Y_train)
                        score = pipe.score(X_test, Y_test)
                        scores.append(score)

                    # Registrar métricas en MLflow
                    mean_score = np.mean(scores)
                    mlflow.log_metric("accuracy", mean_score)

                    # Verificar si el modelo actual es el mejor
                    if mean_score > best_score:
                        best_score = mean_score
                        best_params = {'batch_size': bs, 'max_iter': ep, 'solver': opt}

                    # Calcular el tiempo de ejecución
                    gs_time = time() - tic
                    tiempoTotal = GetTime(gs_time)
                    mlflow.log_metric("tiempo", gs_time)
                    print(f"Completado {bs=}, {ep=}, {opt=}. Tiempo: {tiempoTotal} | Accuracy: {mean_score:.4f}")

    # Imprimir y registrar los mejores parámetros encontrados
    print("\nMejores parámetros encontrados:")
    print(best_params)
    print(f"Mejor accuracy promedio: {best_score:.4f}")
    
    # Guardar el modelo en MLflow
    mlflow.sklearn.log_model(pipe, "best_model")

pipeNN.get_params().keys()       # si estás usando el pipeline con el MLP y el GridSearch


import numpy as np

# Supongamos que tienes una función o proceso para calcular las métricas de precisión
# durante la validación cruzada.

# Aquí tienes un ejemplo para calcular los resultados de cada combinación de parámetros:

all_means = []   # Lista de medias de accuracy
all_stds = []    # Lista de desviaciones estándar del accuracy
all_params = []  # Lista de combinaciones de hiperparámetros

# Iterar sobre las combinaciones de hiperparámetros
for bs in batch_size:
    for ep in epochs:
        for opt in optimizer:
            # Crear y ajustar el modelo para esta combinación de parámetros
            model = build_model(batch_size=bs, max_iter=ep, solver=opt)
            pipe.set_params(modelNN=model)

            # Validación cruzada
            kfold = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
            scores = []

            for train_index, test_index in kfold.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                Y_train, Y_test = Yval[train_index], Yval[test_index]

                # Ajustar el modelo con los datos de entrenamiento
                pipe.fit(X_train, Y_train)

                # Evaluar el modelo con los datos de prueba
                score = pipe.score(X_test, Y_test)
                scores.append(score)

            # Calcular la media y la desviación estándar de la precisión
            mean_score = np.mean(scores)
            stdev_score = np.std(scores)

            # Almacenar los resultados de esta combinación
            all_means.append(mean_score)
            all_stds.append(stdev_score)
            all_params.append({'batch_size': bs, 'max_iter': ep, 'solver': opt})

# Mostrar los resultados detallados de cada combinación
print("\nResultados detallados de cada combinación de hiperparámetros:\n")
for mean, stdev, param in zip(all_means, all_stds, all_params):
    print("%.4f (± %.4f) con: %r" % (mean, stdev, param))

# Si tienes los mejores parámetros y score
best_score = max(all_means)
best_params = all_params[all_means.index(best_score)]

# Mostrar el mejor resultado encontrado
print("\nBest: %.4f using %s" % (best_score, best_params))

print("Best: %.4f using %s" % (best_score, best_params))

# =====================================================
# 3. Definir X e Y (SIN NaN, ya imputado aquí)
# =====================================================
Yval = Y.values

X = copy.deepcopy(dataframe)

print("Shape de X:", X.shape)
print("Shape de Y:", Yval.shape)

# 🔍 Revisar NaN antes de imputar
print("\nNaNs por columna ANTES de imputar:")
print(X.isna().sum())

# Imputar NaN columna por columna
for col in X.columns:
    if X[col].isna().any():
        # Si la columna es binaria (0/1), usamos la moda
        valores_unicos = set(X[col].dropna().unique())
        if valores_unicos.issubset({0, 1}):
            moda = X[col].mode()[0]
            X[col] = X[col].fillna(moda)
            print(f"Imputando columna binaria {col} con la moda:", moda)
        else:
            mediana = X[col].median()
            X[col] = X[col].fillna(mediana)
            print(f"Imputando columna numérica {col} con la mediana:", mediana)

print("\nNaNs por columna DESPUÉS de imputar:")
print(X.isna().sum())


import mlflow

# Obtener la última corrida activa
latest_run = mlflow.active_run()

if latest_run is not None:
    # Si hay una corrida activa, puedes obtener el run_id
    print("Último run_id:", latest_run.info.run_id)
else:
    print("No hay una corrida activa en este momento.")

import mlflow
import mlflow.sklearn
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from time import time

# =====================================================
# PASO 2 – Optimización de la arquitectura (capas) y regularización
# =====================================================

# 1. Definir modelo base MLP
model_base = MLPClassifier(
    hidden_layer_sizes=(6, 6),
    activation='logistic',
    solver='adam',     # Este será sobreescrito en la búsqueda
    batch_size=16,     # También será sobreescrito en la búsqueda
    max_iter=200,      # Este será sobreescrito en la búsqueda
    random_state=42
)

# 2. Construir el pipeline completo: preprocesamiento + MLP
pipeNN_opt2 = Pipeline(steps=[
    ('prep', pipe_prep),   # pipeline de preprocesamiento que cargaste con cargarPipeline(...)
    ('modelNN', model_base)
])

print("Pasos en pipeNN_opt2:")
print([name for name, _ in pipeNN_opt2.steps])

# 3. Definir el grid de búsqueda:
param_grid_2 = {
    'modelNN__hidden_layer_sizes': [
        (4,), (8,), (16,), (32,), (64,), (128,),
        (8, 4), (16, 8), (32, 16), (64, 32)
    ],
    'modelNN__alpha': [0.0001, 0.001, 0.01, 0.1]
}

# 4. Definir esquema de validación cruzada tipo ShuffleSplit
numFolds = 5
kfold = ShuffleSplit(
    n_splits=numFolds,
    test_size=0.2,
    random_state=42
)

# 5. Ejecutar optimización de hiperparámetros usando MLflow
tic = time()

# Iniciar un experimento en MLflow
with mlflow.start_run():
    best_score = 0
    best_params = {}

    # Búsqueda manual de hiperparámetros
    for hidden_layer in param_grid_2['modelNN__hidden_layer_sizes']:
        for alpha in param_grid_2['modelNN__alpha']:
            # Registrar los parámetros en MLflow
            mlflow.log_param("hidden_layer_sizes", hidden_layer)
            mlflow.log_param("alpha", alpha)
            
            # Crear y ajustar el modelo con los parámetros actuales
            model = MLPClassifier(
                hidden_layer_sizes=hidden_layer,
                alpha=alpha,
                activation='logistic',
                solver='adam',
                batch_size=16,
                max_iter=200,
                random_state=42
            )
            
            pipeNN_opt2.set_params(modelNN=model)

            # Configuración de validación cruzada
            scores = []
            for train_index, test_index in kfold.split(X):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
                Y_train, Y_test = Yval[train_index], Yval[test_index]

                # Ajustar el modelo
                pipeNN_opt2.fit(X_train, Y_train)

                # Evaluar el modelo
                score = pipeNN_opt2.score(X_test, Y_test)
                scores.append(score)

            # Calcular la media y la desviación estándar de la precisión
            mean_score = np.mean(scores)
            stdev_score = np.std(scores)

            # Registrar las métricas en MLflow
            mlflow.log_metric("mean_accuracy", mean_score)
            mlflow.log_metric("stdev_accuracy", stdev_score)

            # Actualizar el mejor puntaje y parámetros
            if mean_score > best_score:
                best_score = mean_score
                best_params = {'hidden_layer_sizes': hidden_layer, 'alpha': alpha}

    # Mostrar los mejores parámetros y el mejor puntaje
    print("\nMejor combinación encontrada (Paso 2):")
    print("Best: %.4f usando %s" % (best_score, best_params))

    # Tiempo de ejecución
    gs_time2 = time() - tic
    print('\nTiempo en segundos: ', gs_time2)
    print('Tiempo formateado: ', GetTime(gs_time2))

    # Guardar el modelo con los mejores parámetros
    mlflow.sklearn.log_model(pipeNN_opt2, "best_model")

    # Resultados detallados
    print("\nResultados detallados:\n")
    for hidden_layer in param_grid_2['modelNN__hidden_layer_sizes']:
        for alpha in param_grid_2['modelNN__alpha']:
            print("Combinación: hidden_layer_sizes=%r, alpha=%r" % (hidden_layer, alpha))
            print("mean_accuracy: %.4f, stdev_accuracy: %.4f" % (mean_score, stdev_score))


means = gs2.cv_results_['mean_test_score']
stds = gs2.cv_results_['std_test_score']
params = gs2.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


cv_results= pd.DataFrame(gs2.cv_results_)
cv_results.head(3)

cv_results_3 = cv_results.head(3).rename(columns={
    "mean_test_score": "accuracy_promedio",
    "std_test_score": "std_accuracy",
    "param_modelNN__hidden_layer_sizes": "capas_neuronas",
    "param_modelNN__alpha": "regularizacion_L2"
})

cv_results_3


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==============================
# 1. Pasar resultados de gs2 a DataFrame
# ==============================
cv_results = pd.DataFrame(gs2.cv_results_)

# Creamos una columna string para que las tuplas se vean bonito en el eje X
cv_results["hidden_str"] = cv_results["param_modelNN__hidden_layer_sizes"].astype(str)

# Pivot: filas = alpha, columnas = hidden_layer_sizes, valores = mean_test_score
scores_matrix = pd.pivot_table(
    data=cv_results.sort_values('mean_test_score', ascending=False),
    index='param_modelNN__alpha',
    columns='hidden_str',
    values='mean_test_score'
)

print("Matriz para el heatmap (alpha vs hidden_layer_sizes):")
print(scores_matrix)

# ==============================
# 2. Función para dibujar el heatmap
# ==============================
def make_heatmap(ax, scores_matrix, make_cbar=False, cmap='RdBu'):
    # Matriz numérica
    data = scores_matrix.values

    im = ax.imshow(data, cmap=cmap, aspect='auto')

    # Ejes X (arquitecturas)
    x_labels = list(scores_matrix.columns)
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_xlabel('hidden_layer_sizes', fontsize=12)

    # Ejes Y (alpha)
    y_labels = list(scores_matrix.index)
    ax.set_yticks(np.arange(len(y_labels)))
    ax.set_yticklabels(y_labels)
    ax.set_ylabel('alpha (regularización L2)', fontsize=12)

    # Colorbar opcional
    if make_cbar:
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Accuracy promedio (cross-validation)', rotation=-90, va="bottom", fontsize=12)

# ==============================
# 3. Dibujar el heatmap
# ==============================
fig, ax = plt.subplots(figsize=(10, 4))

make_heatmap(ax, scores_matrix, make_cbar=True, cmap='RdBu')
ax.set_title('Heatmap: Accuracy según arquitectura y alpha', fontsize=14)

plt.tight_layout()
plt.show()


from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import copy
import numpy as np

# =====================================================
# 1. Asegurarnos de tener X (entradas) y Y (salida)
# =====================================================
# Y ya la definiste en la fase 1 como:
# Y = df['CASE_TYPE']  o dataframe_original['CASE_TYPE']

Yval = Y.values          # igual que en el ejemplo del German Credit

# X es TU dataframe de entrada YA SIN:
# 'CASE_TYPE', 'AGE_FATHER', 'HYPERTENSION'
X = copy.deepcopy(dataframe)

# Por si acaso, revisamos NaN
print("NaN por columna ANTES de imputar:")
print(X.isna().sum())

# Imputación simple (mediana) en las columnas numéricas con NaN
columnas_con_nan = ['AGE_MOTHER', 'AGE_FIRST_DIAGNOSIS']
for col in columnas_con_nan:
    mediana = X[col].median()
    X[col] = X[col].fillna(mediana)

print("\nNaN por columna DESPUÉS de imputar:")
print(X.isna().sum())

# =====================================================
# 2. Volvemos a cargar SOLO el pipeline de preprocesamiento
#    (el .pickle que creaste en la fase 1)
# =====================================================
pipe_prep = cargarPipeline(nombreArchivoPreprocesador)
print("\nPasos en pipe_prep (preprocesamiento):")
print([name for name, _ in pipe_prep.steps])

# =====================================================
# 3. Tomar los MEJORES hiperparámetros del Paso 2 (gs2)
#    (ajusta estos valores si tu GridSearch dio otros)
# =====================================================
print("\nMejores hiperparámetros encontrados en Paso 2:")
print(gs2.best_params_)

best_hls   = gs2.best_params_['modelNN__hidden_layer_sizes']  # arquitectura óptima
best_alpha = gs2.best_params_['modelNN__alpha']               # regularización óptima

print("\nArquitectura óptima:", best_hls)
print("Alpha óptimo:", best_alpha)

# =====================================================
# 4. Definir el modelo MLP FINAL (optimizado)
#    Aquí usamos los mejores parámetros encontrados
# =====================================================
modelNN_opt = MLPClassifier(
    hidden_layer_sizes = best_hls,
    activation         = 'logistic',
    solver             = 'adam',
    alpha              = best_alpha,
    batch_size         = 16,     # puedes dejar el que usaste en el Paso 1
    max_iter           = 200,    # ≈ epochs
    random_state       = 42
)

# =====================================================
# 5. Construir el pipeline COMPLETO (prep + red)
# =====================================================
pipe_opt = Pipeline(steps=[
    ('prepcn',      pipe_prep.named_steps['prepcn']),
    ('prepstandard',pipe_prep.named_steps['prepstandard']),
    ('modelNN',     modelNN_opt)
])

print("\nPasos en pipe_opt (modelo optimizado):")
print([name for name, _ in pipe_opt.steps])

# =====================================================
# 6. Evaluación del modelo optimizado con Cross-Validation
#    (equivalente a tu código con cvs = cross_val_score(...))
# =====================================================
numFolds = 5
kfold = ShuffleSplit(
    n_splits=numFolds,
    test_size=0.2,
    random_state=42
)

cvs = cross_val_score(
    pipe_opt,
    X,
    Yval,
    cv=kfold,
    n_jobs=-1,
    scoring='accuracy'
)

print("\nScores (accuracy) en cada fold:")
for i, sc in enumerate(cvs):
    print(f"Fold {i+1}: {sc:.4f}")

print("\nMedia de scores (Accuracy Modelo Optimizado):", cvs.mean())
accuracyModeloOptimizado = cvs.mean()


from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import copy
import numpy as np

# =====================================================
# 1. Cargar pipeline de preprocesamiento guardado (fase 1)
# =====================================================
pipe_prep = cargarPipeline(nombreArchivoPreprocesador)   # 'NF1_pipePreprocesadores'
print("Pasos del pipeline de preprocesamiento cargado:")
print([name for name, _ in pipe_prep.steps])

# =====================================================
# 2. Definir X e Y y asegurar que NO haya NaN
# =====================================================

# Y = variable de salida (Case Type)
Yval = Y.values

# X = tus variables de entrada (ya SIN CASE_TYPE, AGE_FATHER, HYPERTENSION)
X = copy.deepcopy(dataframe)

print("\nNaN por columna ANTES de imputar:")
print(X.isna().sum())

# Imputar medianas en las columnas numéricas que tienen NaN
columnas_con_nan = ['AGE_MOTHER', 'AGE_FIRST_DIAGNOSIS']
for col in columnas_con_nan:
    mediana = X[col].median()
    X[col] = X[col].fillna(mediana)

print("\nNaN por columna DESPUÉS de imputar:")
print(X.isna().sum())

# =====================================================
# 3. Definir el modelo optimizado (versión scikit-learn)
#    Aquí usamos la arquitectura (64, 4) que en el ejemplo
#    venía de l1=64 y l2=4 (paso de optimización 2).
#    Dropout no existe en MLPClassifier, pero 'alpha'
#    actúa como regularización (similar a controlar sobreajuste).
# =====================================================

modeloOptimizado = MLPClassifier(
    hidden_layer_sizes=(4, 4),  # equivalente a l1=64, l2=4
    activation='logistic',
    solver='adam',               # como tu optimizer='Adam'
    alpha=0.01,                  # puedes usar el alpha óptimo de gs2 si ya lo tienes
    batch_size=10,               # equivalente a batch_size=10
    max_iter=75,                 # equivalente a epochs=75
    random_state=42
)

# =====================================================
# 4. Construir pipeline COMPLETO: preprocesamiento + modelo optimizado
# =====================================================

pipeNN_opt = Pipeline(steps=[
    ('prepcn',      pipe_prep.named_steps['prepcn']),
    ('prepstandard',pipe_prep.named_steps['prepstandard']),
    ('modelNN',     modeloOptimizado)
])

print("\nPasos en pipeNN_opt:")
print([name for name, _ in pipeNN_opt.steps])

# =====================================================
# 5. Separar train / test (80% / 20%) y entrenar
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    Yval,
    test_size=0.2,
    random_state=42,
    stratify=Yval
)

pipeNN_opt.fit(X_train, y_train)

print("\n Modelo optimizado entrenado")

# =====================================================
# 6. Guardar modelo optimizado COMPLETO (pipeline + red)
# =====================================================

guardarNN(pipeNN_opt, 'NF1_modeloRedNeuronalOptimizada')

print(" Modelo optimizado guardado en 'NF1_modeloRedNeuronalOptimizada.joblib'")

# (Opcional) evaluar rápido en el conjunto de prueba
from sklearn.metrics import accuracy_score

y_pred_test = pipeNN_opt.predict(X_test)
acc_test = accuracy_score(y_test, y_pred_test)
print(f"\nAccuracy en test del modelo optimizado: {acc_test:.4f}")


import numpy as np
import pandas as pd

# Probabilidad de clase 1
y_proba = pipeNN_opt.predict_proba(X_test)[:, 1]

# Aplicar umbral 0.5
y_pred = (y_proba > 0.5).astype("int32")

dataframeFinal = pd.DataFrame({
    'real': y_test,
    'predicción': y_pred
})

np.set_printoptions(formatter={'float': lambda X: "{0:0.0f}".format(X)})

dataframeFinal.tail(10)

import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# y_test  -> reales
# y_pred  -> predicciones del modelo (0/1), ya las calculaste antes

# --- Errores ---
MAE  = metrics.mean_absolute_error(y_test, y_pred)
MSE  = metrics.mean_squared_error(y_test, y_pred)
RMSE = np.sqrt(MSE)

accuracy = accuracy_score(y_test, y_pred)   # ← float

# Si quieres guardar el accuracy del modelo optimizado:
accuracyModeloOptimizado = accuracy

# Redondeo para impresión
MAE_str  = str(round(MAE, 4))
MSE_str  = str(round(MSE, 4))
RMSE_str = str(round(RMSE, 4))

print('Mean Absolute Error (MAE):', MAE_str)  
print('Mean Squared Error (MSE):', MSE_str)  
print('Root Mean Squared Error (RMSE):', RMSE_str) 
print('Accuracy:', round(accuracy, 4))

# --- Matriz de confusión ---
print('\nConfusion_matrix:')
y_test_transformado = y_test
y_pred_transformado = y_pred

cm = confusion_matrix(y_test_transformado, y_pred_transformado)  
print(cm)  

tn, fp, fn, tp = cm.ravel()  
print('TN, FP, FN, TP:', tn, fp, fn, tp)

# --- Métricas de clasificación ---
precision = precision_score(y_test_transformado, y_pred_transformado)
recall    = recall_score(y_test_transformado, y_pred_transformado)
f1        = f1_score(y_test_transformado, y_pred_transformado)

print('Precision:', round(precision, 4))
print('Recall   :', round(recall, 4))
print('F1       :', round(f1, 4))


# Asegúrate de que sean floats
accuracyModeloBase = float(accuracyModeloBase)
accuracyModeloOptimizado = float(accuracyModeloOptimizado)

print('Accuracy Modelo Base       :', round(accuracyModeloBase, 4))
print('Accuracy Modelo Optimizado :', round(accuracyModeloOptimizado, 4))

mejora = round((accuracyModeloOptimizado - accuracyModeloBase) * 100, 4)
print('Mejora en %:', mejora)


import numpy as np
import pandas as pd

# ------------------------------
# Fase 4: Función de interpretación
# ------------------------------
def obtenerResultadosyCertezas(lista_prob):
    """
    lista_prob: lista o array de probabilidades de ser clase 1 (Caso familiar).
    Devuelve:
        - predicciones (probabilidades originales)
        - marcas (etiquetas textuales)
        - certezas (porcentaje de confianza como string 'xx%')
    """
    predicciones = list(lista_prob)
    marcas = []
    certezas = []
    
    for prob in predicciones:
        # Evitamos problemas numéricos extremos
        prob = float(prob)

        if prob < 0.5:
            # Clase 0: Caso esporádico
            marca = 'Caso esporádico'
            # certeza crece al acercarse a 0 (prob ∈ [0,0.5])
            certeza = 1 - (prob / 0.5)
        else:
            # Clase 1: Caso familiar
            marca = 'Caso familiar'
            # certeza crece al acercarse a 1 (prob ∈ [0.5,1])
            certeza = (prob - 0.5) / 0.5

        certezas.append(f"{int(certeza * 100)}%")
        marcas.append(marca)

    return predicciones, marcas, certezas


import joblib

# Función genérica para cargar tu modelo (pipeline completo)
def cargarNN(nombreArchivo):
    print("Cargando modelo (Pipeline con MLPClassifier) desde archivo...")
    model = joblib.load(nombreArchivo + '.joblib')
    print("Modelo cargado correctamente desde", nombreArchivo + '.joblib')
    return model

# Nombre del archivo que guardaste en la fase 3
nombreModeloOptimizado = 'NF1_modeloRedNeuronalOptimizada'  # ajusta si usaste otro

# Este objeto YA es un Pipeline completo: preprocesamiento + red neuronal
pipe_final = cargarNN(nombreModeloOptimizado)


def predecirNuevoPaciente(
    TUMOUR_CASE=1,
    AGE_MOTHER=30.0,
    AGE_FIRST_DIAGNOSIS=5.0,
    CAFE_AU_LAIT=1,
    AXILLARY_FRECKLES=1,
    INGUINAL_FRECKLES=0,
    LISCH_NODULES=0,
    DERMAL_NEUROFIBROMINS=0,
    PLEXIFORM_NEUROFIBROMINS=0,
    OPTIC_GLIOMA=0,
    SKELETAL_DYSPLASIA=0,
    LEARNING_DISABILITY=0,
    ASTROCYTOMA=0,
    HAMARTOMA=0,
    SCOLIOSIS=0,
    OTHER_SYMPTOMS=1
):
    """
    Devuelve un DataFrame con:
      - Probabilidad_clase_1 (Caso familiar)
      - Resultado (Caso esporádico / Caso familiar)
      - Certeza (%)
    """

    # Nombres de columnas EXACTOS como en tu X de entrenamiento
    cnames = [
        'TUMOUR_CASE',
        'AGE_MOTHER',
        'AGE_FIRST_DIAGNOSIS',
        'CAFE_AU_LAIT',
        'AXILLARY_FRECKLES',
        'INGUINAL_FRECKLES',
        'LISCH_NODULES',
        'DERMAL_NEUROFIBROMINS',
        'PLEXIFORM_NEUROFIBROMINS',
        'OPTIC_GLIOMA',
        'SKELETAL_DYSPLASIA',
        'LEARNING_DISABILITY',
        'ASTROCYTOMA',
        'HAMARTOMA',
        'SCOLIOSIS',
        'OTHER_SYMPTOMS'
    ]

    # Registro nuevo (1 solo paciente)
    Xnew = [[
        TUMOUR_CASE,
        AGE_MOTHER,
        AGE_FIRST_DIAGNOSIS,
        CAFE_AU_LAIT,
        AXILLARY_FRECKLES,
        INGUINAL_FRECKLES,
        LISCH_NODULES,
        DERMAL_NEUROFIBROMINS,
        PLEXIFORM_NEUROFIBROMINS,
        OPTIC_GLIOMA,
        SKELETAL_DYSPLASIA,
        LEARNING_DISABILITY,
        ASTROCYTOMA,
        HAMARTOMA,
        SCOLIOSIS,
        OTHER_SYMPTOMS
    ]]

    Xnew_df = pd.DataFrame(Xnew, columns=cnames)

    # 1) Probabilidad de CASE_TYPE = 1 (familiar)
    proba = pipe_final.predict_proba(Xnew_df)[0, 1]  # columna de la clase 1

    # 2) Interpretar probabilidad
    predicciones, marcas, certezas = obtenerResultadosyCertezas([proba])

    # 3) DataFrame final con resultados
    df_resultado = pd.DataFrame({
        'Probabilidad_clase_1(Familiar)': predicciones,
        'Resultado': marcas,
        'Certeza': certezas
    })

    # Opcional: formato de impresión
    np.set_printoptions(formatter={'float': lambda X: "{0:0.3f}".format(X)})

    return df_resultado
resultado = predecirNuevoPaciente(
    TUMOUR_CASE=1,
    AGE_MOTHER=28,
    AGE_FIRST_DIAGNOSIS=6,
    CAFE_AU_LAIT=1,
    AXILLARY_FRECKLES=1,
    INGUINAL_FRECKLES=1,
    LISCH_NODULES=0,
    DERMAL_NEUROFIBROMINS=0,
    PLEXIFORM_NEUROFIBROMINS=0,
    OPTIC_GLIOMA=0,
    SKELETAL_DYSPLASIA=0,
    LEARNING_DISABILITY=0,
    ASTROCYTOMA=0,
    HAMARTOMA=0,
    SCOLIOSIS=0,
    OTHER_SYMPTOMS=1
)

resultado











## Resultados cuantificables

Los resultados obtenidos evidencian una mejora respecto al modelo base:

- Incremento del accuracy
- Incremento del F1-score
- Mayor estabilidad entre ejecuciones
- Reducción del error de clasificación

Gracias al uso de MLflow, se registraron métricas, parámetros y tiempos de ejecución,
permitiendo una comparación objetiva entre las distintas configuraciones evaluadas.
