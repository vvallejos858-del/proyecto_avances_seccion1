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
