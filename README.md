# Social Data Science

Repositorio del curso de **Social Data Science** que combina métodos de ciencia de datos con teoría y metodología de ciencias sociales para estudiar fenómenos sociales utilizando trazas digitales. Impartido por Alex Ojeda Copa.

## Descripción del Curso

Este curso explora la intersección entre ciencia de datos y ciencias sociales, aprovechando la digitalización masiva de interacciones humanas para generar nuevos conocimientos sobre comportamientos sociales, políticos y económicos.

### Objetivos

- Desarrollar habilidades para analizar datos sociales digitales
- Aplicar métodos computacionales a problemas de ciencias sociales
- Comprender los retos metodológicos y éticos del análisis de datos sociales

## Estructura del Curso

### Módulo 0: Introducción y Fundamentos
**Archivo**: `0.ipynb`

- Introducción a Social Data Science y sus diferencias con Data Science tradicional
- Habilidades núcleo: sustantivas, estadística y programación
- Retos típicos: sesgo, representatividad, privacidad, validez
- Flujo de trabajo completo desde problema hasta comunicación

### Módulo 1: Análisis de Datos y APIs
**Archivos**: `1.1. pandas.ipynb` y `1.2 internet.ipynb`

- **1.1 Pandas**: Estructuras de datos y operaciones básicas con datos tabulares
  - Series y DataFrames con datos de estudiantes
  - Operaciones de filtrado, agregación y manipulación
  - Estadísticas descriptivas y análisis exploratorio
- **1.2 Internet**: APIs y datos web
  - Consumo de APIs y datos abiertos
  - Ejercicio práctico con datos demográficos y adopción digital

### Módulo 2: Procesamiento de Lenguaje Natural
**Archivos**: `2.1. spacy.ipynb` y `2.2. noticias.ipynb`

- **2.1 spaCy**: Fundamentos de procesamiento de texto
  - Estructuras de datos (Doc, Token, Span)
  - Análisis morfológico y sintáctico
  - Entidades nombradas y análisis básico
  - Ejercicios con textos de ejemplo
- **2.2 Noticias**: Análisis de narrativas mediáticas
  - Análisis de sentimiento avanzado
  - Topic modeling y extracción de temas
  - Ejercicio práctico con medios bolivianos

### Módulo 3: Análisis de Redes Sociales
**Archivos**: `3.1. networkx.ipynb` y `3.2. reddit.ipynb`

- **3.1 NetworkX**: Fundamentos de análisis de redes
  - Tipos de grafos (dirigidos, no dirigidos, ponderados)
  - Métricas de centralidad e importancia
  - Visualización de redes y análisis básico
  - Ejercicios con redes sociales de ejemplo
- **3.2 Reddit**: Análisis de comunidades digitales
  - Red de interacciones en subreddits latinoamericanos
  - Detección de comunidades y polarización
  - Análisis de difusión de información

### Módulo 4: Análisis Espacial y Geográfico
**Archivos**: `4.1. geopandas.ipynb` y `4.2. salud.ipynb`

- **4.1 GeoPandas**: Fundamentos de datos geoespaciales
  - Geometrías básicas (puntos, líneas, polígonos)
  - GeoDataFrames y operaciones espaciales
  - Análisis de distancias, buffers y relaciones espaciales
  - Visualización de mapas temáticos y análisis de cobertura
- **4.2 Salud**: Análisis territorial de servicios de salud
  - Disponibilidad territorial de centros de salud en Cochabamba
  - Análisis de accesibilidad y cobertura geográfica
  - Mapas coropléticos y análisis de inequidades espaciales


## Tecnologías y Herramientas

### Lenguajes y Frameworks
- **Python**: Lenguaje principal del curso
- **Jupyter Notebooks**: Entorno de desarrollo interactivo

### Librerías Principales
- **Análisis de datos**: pandas, numpy
- **Visualización**: matplotlib, seaborn, plotly
- **Procesamiento de texto**: spaCy, scikit-learn, pysentimiento
- **Redes sociales**: networkx, community
- **Análisis espacial**: geopandas, folium, osmnx, shapely
- **APIs y datos**: requests, wbgapi
- **Análisis estadístico**: scipy, statsmodels

### Librerías de Apoyo
- **Geometrías**: shapely (integrada con geopandas)
- **Mapas interactivos**: folium, contextily
- **Machine Learning**: scikit-learn, nltk
- **Manejo de datos**: openpyxl, json

### Fuentes de Datos
- World Bank Open Data
- GDELT Project (eventos globales)
- Reddit API
- OpenStreetMap/Overpass API
- GeoBolivia (IDE-EPB)
- INE Bolivia

## Estructura del Repositorio

```
social-data-science/
├── README.md                 # Este archivo
├── fuentes-datos.csv        # Catálogo de fuentes de datos
├── 0.ipynb                  # Módulo 0: Introducción y fundamentos
├── 1.1. pandas.ipynb       # Módulo 1.1: Estructuras de datos con pandas
├── 1.2 internet.ipynb      # Módulo 1.2: APIs y datos web
├── 2.1. spacy.ipynb        # Módulo 2.1: Procesamiento de texto con spaCy
├── 2.2. noticias.ipynb     # Módulo 2.2: Análisis de narrativas mediáticas
├── 3.1. networkx.ipynb     # Módulo 3.1: Análisis de redes con NetworkX
├── 3.2. reddit.ipynb       # Módulo 3.2: Comunidades digitales en Reddit
├── 4.1. geopandas.ipynb    # Módulo 4.1: Análisis geoespacial con GeoPandas
├── 4.2. salud.ipynb        # Módulo 4.2: Análisis territorial de salud
├── cache/                   # Cache de consultas API
└── data/                    # Datos de ejemplo
```

## Requisitos Previos

### Conocimientos
- Fundamentos básicos bde programación en Python
- Estadística descriptiva básica
- Conceptos básicos de ciencias sociales

### Instalación
```bash
# Clonar el repositorio
git clone https://github.com/alex-roc/social-data-science.git
cd social-data-science
```
