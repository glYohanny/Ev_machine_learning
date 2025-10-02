# 🚀 Guía de Instalación - League of Legends Worlds Analysis

## 📋 Tabla de Contenidos

- [Requisitos del Sistema](#requisitos-del-sistema)
- [Instalación Paso a Paso](#instalación-paso-a-paso)
- [Configuración del Entorno](#configuración-del-entorno)
- [Verificación de la Instalación](#verificación-de-la-instalación)
- [Solución de Problemas](#solución-de-problemas)
- [Configuración Avanzada](#configuración-avanzada)

---

## 💻 Requisitos del Sistema

### Requisitos Mínimos
- **Sistema Operativo:** Windows 10/11, macOS 10.15+, Ubuntu 18.04+
- **Python:** 3.9 o superior
- **RAM:** 4GB mínimo, 8GB recomendado
- **Espacio en Disco:** 2GB libres
- **Procesador:** Intel i3/AMD Ryzen 3 o superior

### Requisitos Recomendados
- **RAM:** 16GB o más
- **Espacio en Disco:** 5GB libres
- **Procesador:** Intel i5/AMD Ryzen 5 o superior
- **GPU:** Para análisis más intensivos (opcional)

---

## 🔧 Instalación Paso a Paso

### 1. Verificar Python

Antes de comenzar, verifica que tienes Python instalado:

```bash
python --version
# Debe mostrar Python 3.9 o superior

python -m pip --version
# Debe mostrar pip instalado
```

Si no tienes Python instalado:
- **Windows:** Descarga desde [python.org](https://python.org)
- **macOS:** Usa Homebrew: `brew install python`
- **Linux:** `sudo apt update && sudo apt install python3 python3-pip`

### 2. Clonar el Repositorio

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/league-of-legends-worlds.git
cd league-of-legends-worlds
```

### 3. Crear Entorno Virtual

**Windows:**
```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
venv\Scripts\activate

# Verificar activación (debe mostrar la ruta del venv)
where python
```

**macOS/Linux:**
```bash
# Crear entorno virtual
python3 -m venv venv

# Activar entorno virtual
source venv/bin/activate

# Verificar activación (debe mostrar la ruta del venv)
which python
```

### 4. Actualizar pip

```bash
python -m pip install --upgrade pip
```

### 5. Instalar Dependencias

```bash
# Instalar dependencias principales
pip install -r requirements.txt

# Verificar instalación
pip list
```

### 6. Configurar Kedro

```bash
# Instalar Kedro
kedro install

# Verificar configuración
kedro info
```

---

## ⚙️ Configuración del Entorno

### Variables de Entorno

Crea un archivo `.env` en la raíz del proyecto:

```bash
# .env
KEDRO_ENV=local
LOG_LEVEL=INFO
DATA_PATH=./data
```

### Configuración de Kedro

El proyecto usa la estructura estándar de Kedro:

```
conf/
├── base/           # Configuración base
│   ├── catalog.yml    # Catálogo de datasets
│   └── parameters.yml # Parámetros del proyecto
├── local/          # Configuración local (no versionar)
│   ├── credentials.yml
│   └── parameters.yml
└── logging.yml     # Configuración de logs
```

### Configuración de Jupyter

```bash
# Instalar Jupyter Lab
pip install jupyterlab

# Configurar kernel de Python
python -m ipykernel install --user --name=league-worlds --display-name="League Worlds"
```

---

## ✅ Verificación de la Instalación

### 1. Verificar Kedro

```bash
# Verificar que Kedro está instalado correctamente
kedro info

# Debe mostrar información del proyecto
```

### 2. Verificar Pipelines

```bash
# Listar pipelines disponibles
kedro pipeline list

# Debe mostrar:
# - data_exploration
# - data_preparation
# - full_pipeline
```

### 3. Ejecutar Test Básico

```bash
# Ejecutar tests
kedro test

# Debe ejecutar todos los tests sin errores
```

### 4. Verificar Datos

```bash
# Verificar estructura de datos
ls -la data/

# Debe mostrar las carpetas:
# - 01_raw/
# - 02_intermediate/
# - 03_primary/
# - 04_feature/
# - 05_model_input/
# - 06_models/
# - 07_model_output/
# - 08_reporting/
```

### 5. Ejecutar Pipeline de Prueba

```bash
# Ejecutar pipeline de exploración
kedro run --pipeline data_exploration

# Debe ejecutar sin errores y generar reportes
```

---

## 🔧 Solución de Problemas

### Error: "Python no encontrado"

**Solución:**
```bash
# Verificar instalación de Python
python --version
python3 --version

# Si no está instalado, instalarlo desde python.org
```

### Error: "pip no encontrado"

**Solución:**
```bash
# Instalar pip
python -m ensurepip --upgrade

# O descargar get-pip.py
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python get-pip.py
```

### Error: "kedro no encontrado"

**Solución:**
```bash
# Verificar que el entorno virtual está activado
# Debe mostrar (venv) al inicio del prompt

# Reinstalar Kedro
pip install kedro==0.19.0
```

### Error: "Módulo no encontrado"

**Solución:**
```bash
# Verificar que todas las dependencias están instaladas
pip install -r requirements.txt

# Si persiste, reinstalar
pip uninstall -r requirements.txt -y
pip install -r requirements.txt
```

### Error: "Permisos denegados"

**Solución:**
```bash
# Windows: Ejecutar como administrador
# macOS/Linux: Usar sudo si es necesario
sudo pip install -r requirements.txt
```

### Error: "Encoding de archivos"

**Solución:**
```bash
# Verificar encoding del sistema
python -c "import locale; print(locale.getpreferredencoding())"

# Configurar encoding en el archivo .env
PYTHONIOENCODING=utf-8
```

---

## 🔧 Configuración Avanzada

### Configuración de Desarrollo

Para desarrollo activo, instala dependencias adicionales:

```bash
# Instalar dependencias de desarrollo
pip install -r requirements-dev.txt

# Configurar pre-commit hooks
pre-commit install
```

### Configuración de Producción

Para entornos de producción:

```bash
# Instalar solo dependencias de producción
pip install -r requirements-prod.txt

# Configurar variables de entorno de producción
export KEDRO_ENV=production
export LOG_LEVEL=WARNING
```

### Configuración de GPU

Para análisis con GPU (opcional):

```bash
# Instalar PyTorch con soporte GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verificar disponibilidad de GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Configuración de Docker

Si prefieres usar Docker:

```bash
# Construir imagen
docker build -t league-worlds .

# Ejecutar contenedor
docker run -p 8888:8888 league-worlds
```

---

## 📊 Verificación Final

Ejecuta este script para verificar que todo está funcionando:

```python
# test_installation.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import kedro

print("✅ Pandas:", pd.__version__)
print("✅ NumPy:", np.__version__)
print("✅ Matplotlib:", plt.matplotlib.__version__)
print("✅ Seaborn:", sns.__version__)
print("✅ Scikit-learn:", sklearn.__version__)
print("✅ Kedro:", kedro.__version__)

# Test básico de funcionalidad
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
print("✅ Test de DataFrame:", len(df) == 3)

print("\n🎉 ¡Instalación completada exitosamente!")
```

Ejecutar:
```bash
python test_installation.py
```

---

## 🆘 Obtener Ayuda

### Recursos de Ayuda

1. **Documentación Oficial:** [README.md](../README.md)
2. **API Reference:** [API_REFERENCE.md](API_REFERENCE.md)
3. **Ejemplos:** [notebooks/](../notebooks/)
4. **Issues:** [GitHub Issues](https://github.com/tu-usuario/league-of-legends-worlds/issues)

### Comandos de Diagnóstico

```bash
# Información del sistema
kedro info

# Estado de los datos
kedro catalog list

# Verificar pipelines
kedro pipeline list

# Logs del sistema
tail -f info.log
```

### Contacto

- **Email:** tu-email@ejemplo.com
- **GitHub:** [tu-usuario](https://github.com/tu-usuario)
- **Discord:** [Servidor del proyecto](https://discord.gg/tu-servidor)

---

**🎮 ¡Que ganes el juego!** Una vez completada la instalación, puedes comenzar con el análisis de datos de League of Legends Worlds.
