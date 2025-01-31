# DataScienceIntegrationTest

Este proyecto implementa un pipeline de machine learning diseñado para predecir si un usuario utiliza un "Monthly Pass" en el sistema de bicicletas compartidas de Los Ángeles. La solución integra procesamiento de datos, ingeniería de características y entrenamiento de modelos mediante **Azure Functions** y contenedores Docker.

---

## Índice

1. [Contexto del Problema](#contexto-del-problema)
2. [Estructura del Proyecto](#estructura-del-proyecto)
3. [Requisitos](#requisitos)
4. [Ejecución del Proyecto](#ejecución-del-proyecto)
   - [1. Construcción y Ejecución con Docker](#1-construcción-y-ejecución-con-docker)
   - [2. Interacción con Azure Functions](#2-interacción-con-azure-functions)
5. [Resultados Esperados](#resultados-esperados)
6. [Dashboard en Power BI](#dashboard-en-power-bi)
7. [Contribución](#contribución)

---

## Contexto del Problema

En **Los Ángeles**, el sistema compartido de bicicletas brinda datos anónimos sobre el uso del servicio. Este dataset incluye información histórica de viajes, y contiene diversas columnas relevantes para analizar el comportamiento de los usuarios.

### Objetivo del Proyecto

El objetivo principal de este proyecto es abordar la siguiente pregunta analítica:

> **¿Es posible predecir si el tipo de pase utilizado es “Monthly Pass” u otro basado en las demás variables de viaje?**

### Descripción de las Variables

El dataset contiene las siguientes columnas:

- **`trip_id`**: Identificador único para cada viaje.
- **`duration`**: Duración del viaje en minutos.
- **`start_time`**: Fecha y hora en que inicia el viaje (formato ISO 8601, tiempo local).
- **`end_time`**: Fecha y hora en que termina el viaje (formato ISO 8601, tiempo local).
- **`start_station`**: Estación donde inició el viaje.
- **`start_lat`**: Latitud de la estación de origen.
- **`start_lon`**: Longitud de la estación de origen.
- **`end_station`**: Estación donde terminó el viaje.
- **`end_lat`**: Latitud de la estación de destino.
- **`end_lon`**: Longitud de la estación de destino.
- **`bike_id`**: Identificador único para cada bicicleta.
- **`plan_duration`**: Días de validez del pase. (0 indica un pase único, "Walk-up plan").
- **`trip_route_category`**: Categoría del viaje:
  - `"Round trip"`: Viajes que comienzan y terminan en la misma estación.
  - Otros valores representan viajes de un solo sentido.
- **`passholder_type`**: Tipo de pase utilizado:
  - `"Monthly Pass"`
  - `"Walk-up"`
  - `"Flex Pass"`
  - `"Staff Annual"`

### Cómo se Aborda el Problema

1. **Preprocesamiento de Datos**:
   - Limpieza de valores nulos (e.g., coordenadas).
   - Manejo de valores atípicos en variables numéricas como `duration` y `distance`.
   - Codificación de variables categóricas como `trip_route_category`.

2. **Ingeniería de Características**:
   - Generación de nuevas variables:
     - Duración del viaje (`trip_duration_calculated`).
     - Distancia entre estaciones calculada con la fórmula de Haversine.
     - Identificación de viajes en fin de semana (`is_weekend`).
     - Clasificación por franjas horarias (`time_slot`).

3. **Modelado**:
   - Modelos supervisados de clasificación.
   - Técnicas de balanceo de datos (SMOTE y submuestreo).
   - Selección de características basada en importancia.

4. **Evaluación**:
   - Métricas: precisión, recall, F1-score.
   - Matriz de confusión para evaluar desempeño.

---

## Estructura del Proyecto

```plaintext
DataScienceIntegrationTest
├── assets
│   ├── screenshot_powerbi_dashboard.png
│   ├── powerbi_dashboard.pbix
├── azure_func_data_processing   # Azure Functions para procesamiento y modelado.
├── Cleaning                     # Scripts de limpieza de datos.
├── engineered_data              # Datos con ingeniería de características aplicada.
├── extracted_data               # Datos extraídos y limpios.
├── htmlcov                      # Reporte de cobertura de tests.
├── ml_results                   # Resultados de modelos y predicciones.
├── Model                        # Código para modelado y preprocesamiento.
├── tests                        # Tests unitarios y de integración.
├── .coverage                    # Archivo de cobertura de código.
├── .flake8                      # Configuración de linting.
├── .gitignore                   # Archivos ignorados por git.
├── azure-pipelines.yml          # Configuración de Azure Pipelines.
├── Dockerfile                   # Dockerfile para el contenedor.
├── README.md                    # Este archivo.
├── requirements.txt             # Dependencias del proyecto.
└── run_pipeline.sh              # Script para ejecutar el pipeline en Docker.
```

# Script para ejecutar el pipeline en Docker

## Requisitos

Antes de comenzar, asegúrate de tener instalado:

- **Docker** (última versión).
- **Python 3.9 o superior**.
- **Azure Functions Core Tools** (opcional, para pruebas locales).
- Dependencias especificadas en `requirements.txt`.
- Para la ejecución de las **Azure Functions Core Tools** se requiere de crear un ambiente virtual de python:

```bash
python -m venv env
source env/bin/activate
```

Instala las dependencias con:

```bash
pip install -r requirements.txt
```
---

## Ejecución del proyecto

1. **Construcción y Ejecución con Docker**:
El script **`run_pipeline.sh`** automatiza la construcción y ejecución del contenedor Docker. Sigue estos pasos:

Dar permisos con (linux/MacOs):

```bash
chmod +x run_pipeline.sh
```

Ejecuta el script:

```bash
./run_pipeline.sh
```
# Script para ejecutar el pipeline en Docker

## Este script realiza las siguientes acciones:

1. **Construye la imagen Docker** (`azure_functions-pipeline`).
2. **Monta carpetas locales en el contenedor**.
3. **Instala dependencias**.
4. **Ejecuta tests unitarios y de linting**.
5. **Inicia las Azure Functions en el puerto `7071`**.

  <div style="margin-top: 20px; text-align: center;">
        <img src="assets/Screenshot 2025-01-31 at 1.17.04 a.m..png" alt="Azure Function run" />
    </div>

### Interactúa con las Azure Functions:

Usa `curl` o Postman para probar las APIs:

**Ejemplo de solicitud POST:**

1. **DataProcessignFunctionr**

```bash
curl -X POST -H "Content-Type: application/json" \
-d '{"kaggle_competition": "ds-programming-test"}' \
http://localhost:7071/api/DataProcessingFunction
```

<div style="margin-top: 20px; text-align: center;">
        <img src="assets/Screenshot 2025-01-31 at 1.18.14 a.m..png" alt="Azure post run" />
    </div>

2. **MLPipelineFunction**

```bash
curl -X POST -H "Content-Type: application/json" -d '{}' http://localhost:7071/api/MLPipelineFunction
```
<div style="margin-top: 20px; text-align: center;">
        <img src="assets/Captura desde 2025-01-30 14-05-03.png" alt="Azure ML run" />
    </div>


### Detén el contenedor:

Presiona `Ctrl+C` para finalizar la ejecución.

---

## 2. Interacción con Azure Functions

### `DataProcessingFunction`

- **Ruta**: `/api/DataProcessingFunction`
- **Método**: POST
- **Descripción**: Descarga y procesa datos desde un concurso de Kaggle.
- **Entradas**:
  - `kaggle_competition`: Nombre del concurso.
- **Salidas**:
  - Lista de archivos procesados.

### `MLPipelineFunction`

- **Ruta**: `/api/MLPipelineFunction`
- **Método**: POST
- **Descripción**: Realiza ingeniería de características, entrena modelos y genera predicciones.
- **Entradas**:
  - Datos limpios en `extracted_data`.
- **Salidas**:
  - Predicciones, métricas y modelos entrenados.

  <div style="margin-top: 20px; text-align: center;">
        <img src="assets/Screenshot 2025-01-29 at 1.27.24 p.m..png" alt="Azure ML run" />
    </div>

---

## Resultados Esperados

- **`extracted_data/`**: Datos extraídos y procesados.
- **`engineered_data/`**: Datos con ingeniería de características.
- **`ml_results/`**: Predicciones y métricas de los modelos.

---

## Contribución

Si deseas contribuir:

1. Ejecuta los tests unitarios:

   ```bash
   pytest --cov=azure_func_data_processing --cov-report=html
   ```

2. Asegúrate de cumplir con las reglas de estilo definidas en `.flake8`:

   ```bash
   flake8 azure_func_data_processing Cleaning Model tests
   ```