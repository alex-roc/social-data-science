# Social Data Science

Repositorio del curso de **Social Data Science** que combina métodos de ciencia de datos con teoría y metodología de ciencias sociales para estudiar fenómenos sociales utilizando trazas digitales. Impartido por Alex Ojeda Copa.

## Descripción del Curso

Este curso explora la intersección entre ciencia de datos y ciencias sociales, aprovechando la digitalización masiva de interacciones humanas para generar nuevos conocimientos sobre comportamientos sociales, políticos y económicos.

### Objetivos

- Desarrollar habilidades para analizar datos sociales digitales
- Aplicar métodos computacionales a problemas de ciencias sociales
- Comprender los retos metodológicos y éticos del análisis de datos sociales

## Estructura del Curso

### Módulo 1: Fundamentos y Flujo de Trabajo
**Archivo**: `notebook-0.ipynb` y `notebook-1.ipynb`

- Introducción a Social Data Science y sus diferencias con Data Science tradicional
- Habilidades núcleo: sustantivas, estadística y programación
- Retos típicos: sesgo, representatividad, privacidad, validez
- Flujo de trabajo completo desde problema hasta comunicación
- **Ejercicio práctico**: Análisis demográfico de municipios bolivianos (2012-2024)
- **Ejercicio práctico**: Adopción digital y desigualdad en Sudamérica (2000-2024)

### Módulo 2: Análisis de Texto
**Archivo**: `notebook-2.ipynb`

- Fundamentos de NLP en Social Data Science
- Preprocesamiento con spaCy para español
- Representación de texto: de bolsa de palabras a embeddings
- Análisis de sentimiento con múltiples enfoques
- Topic modeling con LDA y BERTopic
- **Ejercicio práctico**: Análisis de narrativas mediáticas en Bolivia, Argentina y Perú

### Módulo 3: Análisis de Redes Sociales
**Archivo**: `notebook-3.ipynb`

- Modelado de redes sociales como grafos
- Métricas de centralidad e influencia
- Detección de comunidades
- Análisis de difusión y polarización
- **Ejercicio práctico**: Red de interacciones en subreddits latinoamericanos

### Módulo 4: Análisis Espacial
**Archivo**: `notebook-4.ipynb`

- Fundamentos de datos geoespaciales
- Servicios y fuentes de datos abiertos (OSM, WFS/WMS)
- Operaciones espaciales con GeoPandas
- Análisis territorial y mapas coropléticos
- **Ejercicio práctico**: Disponibilidad territorial de centros de salud en Cochabamba


## Tecnologías y Herramientas

### Lenguajes y Frameworks
- **Python**: Lenguaje principal del curso
- **Jupyter Notebooks**: Entorno de desarrollo interactivo

### Librerías Principales
- **Análisis de datos**: pandas, numpy
- **Visualización**: matplotlib, seaborn, plotly
- **Procesamiento de texto**: spaCy, scikit-learn, pysentimiento
- **Redes sociales**: networkx, community
- **Análisis espacial**: geopandas, folium, osmnx
- **APIs y datos**: requests, wbgapi

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
├── notebook-0.ipynb        # Módulo 1: Fundamentos
├── notebook-1.ipynb        # APIs y datos abiertos
├── notebook-2.ipynb        # Módulo 2: Análisis de texto
├── notebook-3.ipynb        # Módulo 3: Redes sociales
├── notebook-4.ipynb        # Módulo 4: Análisis espacial
├── cache/                   # Cache de consultas API
└── data/                    # Datos de ejemplo
```

## Requisitos Previos

### Conocimientos
- Fundamentos de programación en Python
- Estadística descriptiva básica
- Conceptos básicos de ciencias sociales

### Instalación
```bash
# Clonar el repositorio
git clone https://github.com/alex-roc/social-data-science.git
cd social-data-science

# Instalar dependencias (recomendado usar conda o virtualenv)
pip install pandas numpy matplotlib seaborn
pip install spacy scikit-learn networkx geopandas
pip install requests wbgapi pysentimiento
python -m spacy download es_core_news_md
```
