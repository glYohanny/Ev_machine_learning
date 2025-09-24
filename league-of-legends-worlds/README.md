# 🏆 League of Legends Worlds - Análisis de Datos con Kedro

## 📋 Descripción del Proyecto

Este proyecto implementa un análisis completo de datos de League of Legends Worlds Championship siguiendo la metodología **CRISP-DM** y utilizando **Kedro** como framework de pipeline de datos.

## 🎯 Objetivos

- **Análisis de rendimiento**: Identificar patrones de éxito en equipos y jugadores
- **Predicción de resultados**: Desarrollar modelos de ML para predecir victorias
- **Análisis del meta**: Entender la evolución del meta competitivo
- **Insights accionables**: Generar recomendaciones para equipos profesionales

## 🏗️ Arquitectura del Proyecto

```
league-of-legends-worlds/
├── conf/                    # Configuración de Kedro
│   ├── base/
│   │   ├── catalog.yml     # Catálogo de datasets
│   │   └── parameters.yml  # Parámetros del proyecto
│   └── local/              # Configuración local
├── data/                   # Datos organizados por etapas
│   ├── 01_raw/            # Datos originales
│   ├── 02_intermediate/   # Datos procesados
│   ├── 03_primary/        # Datos consolidados
│   ├── 04_feature/        # Características derivadas
│   ├── 05_model_input/    # Datos para ML
│   ├── 06_models/         # Modelos entrenados
│   ├── 07_model_output/   # Resultados de modelos
│   └── 08_reporting/      # Reportes generados
├── notebooks/             # Jupyter notebooks
├── src/                   # Código fuente
│   └── league_of_legends_worlds/
│       └── pipelines/     # Pipelines de Kedro
└── tests/                 # Tests unitarios
```

## 🚀 Instalación y Configuración

### 1. Clonar el repositorio
```bash
git clone <repository-url>
cd league-of-legends-worlds
```

### 2. Crear entorno virtual
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Configurar Kedro
```bash
kedro install
```

## 📊 Datasets

El proyecto utiliza tres datasets principales:

1. **champions_stats.csv**: Estadísticas de campeones por temporada
2. **matchs_stats.csv**: Datos de partidos individuales
3. **players_stats.csv**: Estadísticas de jugadores por temporada

### Formatos Soportados
- CSV (formato principal)
- JSON (para intercambio de datos)
- TXT (formato tabular)

## 🔄 Pipelines Disponibles

### Pipeline de Exploración de Datos
```bash
kedro run --pipeline data_exploration
```

### Pipeline de Preparación de Datos
```bash
kedro run --pipeline data_preparation
```

### Pipeline Completo
```bash
kedro run --pipeline full_pipeline
```

## 📈 Análisis Implementados

### 1. Análisis Univariado
- Estadísticas descriptivas (media, mediana, desviación estándar)
- Análisis de distribuciones (asimetría, curtosis)
- Detección de valores atípicos

### 2. Análisis Bivariado
- Matrices de correlación
- Análisis de relaciones entre variables
- Identificación de correlaciones fuertes

### 3. Análisis Multivariado
- Análisis de Componentes Principales (PCA)
- Clustering con K-Means
- Reducción de dimensionalidad

### 4. Feature Engineering
- Cálculo de KDA mejorado
- Métricas de eficiencia
- Categorización por rendimiento
- Indicadores de impacto

## 🎯 Targets para Machine Learning

### Regresión
- **Champions**: `win_rate` (predecir tasa de victorias)
- **Matches**: `duration` (predecir duración del partido)
- **Players**: `player_efficiency` (predecir eficiencia)

### Clasificación
- **Champions**: `popularity_tier` (tier de popularidad)
- **Matches**: `game_length_category` (categoría de duración)
- **Players**: `performance_tier` (tier de rendimiento)

### Clasificación Binaria
- **Champions**: `win` (victoria/derrota)
- **Players**: `win` (victoria/derrota)

## 🛠️ Tecnologías Utilizadas

- **Kedro**: Framework de pipeline de datos
- **Pandas**: Manipulación de datos
- **NumPy**: Operaciones numéricas
- **Scikit-learn**: Machine Learning
- **Matplotlib/Seaborn**: Visualizaciones
- **Jupyter**: Notebooks interactivos

## 📝 Metodología CRISP-DM

El proyecto sigue las 6 etapas de CRISP-DM:

1. ✅ **Comprensión del Negocio**: Objetivos y criterios de éxito definidos
2. ✅ **Comprensión de los Datos**: Análisis exploratorio completo
3. ✅ **Preparación de los Datos**: Limpieza y feature engineering
4. 🔄 **Modelado**: Implementación de modelos de ML
5. 🔄 **Evaluación**: Validación y métricas de rendimiento
6. 🔄 **Despliegue**: Reportes y visualizaciones finales

## 🧪 Testing

Ejecutar tests:
```bash
kedro test
```

## 📊 Reportes Generados

- **Reporte de Calidad de Datos**: Análisis de valores faltantes y duplicados
- **Reporte de EDA**: Análisis exploratorio completo
- **Reporte de Modelos**: Resultados de ML y métricas

## 🤝 Contribución

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.

## 👥 Autores

- **Tu Nombre** - *Trabajo inicial* - [TuGitHub](https://github.com/tuusername)

## 🙏 Agradecimientos

- Riot Games por los datos de League of Legends
- La comunidad de Kedro por el framework
- La comunidad de Python por las librerías utilizadas

---

**¡Que ganes el juego! 🎮**