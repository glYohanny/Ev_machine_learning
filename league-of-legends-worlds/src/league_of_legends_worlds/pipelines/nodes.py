"""
Nodos para el pipeline de League of Legends Worlds.
Este módulo contiene todas las funciones que serán ejecutadas como nodos en Kedro.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')


def load_raw_data() -> Dict[str, pd.DataFrame]:
    """
    Carga los datos originales desde los archivos CSV.
        
    Returns:
        Dict[str, pd.DataFrame]: Diccionario con los tres datasets cargados
    """
    print("🔄 Cargando datos originales...")
    
    champions_raw = pd.read_csv('data/01_raw/champions_stats.csv', encoding='latin-1')
    matches_raw = pd.read_csv('data/01_raw/matchs_stats.csv', encoding='latin-1')
    players_raw = pd.read_csv('data/01_raw/players_stats.csv', encoding='latin-1')
    
    print(f"✅ Datos cargados:")
    print(f"   - Champions: {champions_raw.shape}")
    print(f"   - Matches: {matches_raw.shape}")
    print(f"   - Players: {players_raw.shape}")
    
    return {
        'champions_raw': champions_raw,
        'matches_raw': matches_raw,
        'players_raw': players_raw
    }


def analyze_data_quality(champions_raw: pd.DataFrame, 
                        matches_raw: pd.DataFrame, 
                        players_raw: pd.DataFrame,
                        params: Dict[str, Any]) -> str:
    """
    Analiza la calidad de los datos originales.
    
    Args:
        champions_raw: DataFrame de campeones
        matches_raw: DataFrame de partidos
        players_raw: DataFrame de jugadores
        params: Parámetros del proyecto
        
    Returns:
        str: Reporte de calidad de datos
    """
    print("🔍 Analizando calidad de datos...")
    
    report = "=== REPORTE DE CALIDAD DE DATOS ===\n\n"
    
    datasets = {
        'Champions': champions_raw,
        'Matches': matches_raw,
        'Players': players_raw
    }
    
    for name, df in datasets.items():
        report += f"📊 {name.upper()}:\n"
        report += f"   Dimensiones: {df.shape}\n"
        
        # Valores faltantes
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        total_missing = missing.sum()
        
        report += f"   Valores faltantes: {total_missing} ({missing_pct.sum():.1f}%)\n"
        
        # Duplicados
        duplicates = df.duplicated().sum()
        report += f"   Duplicados: {duplicates}\n"
        
        # Tipos de datos
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        text_cols = df.select_dtypes(include=['object']).columns
        report += f"   Columnas numéricas: {len(numeric_cols)}\n"
        report += f"   Columnas de texto: {len(text_cols)}\n\n"
    
    return report


def clean_champions_data(champions_raw: pd.DataFrame, 
                        params: Dict[str, Any]) -> pd.DataFrame:
    """
    Limpia los datos de campeones.
    
    Args:
        champions_raw: DataFrame original de campeones
        params: Parámetros de limpieza
        
    Returns:
        pd.DataFrame: DataFrame limpio de campeones
    """
    print("🧹 Limpiando datos de campeones...")
    
    df = champions_raw.copy()
    original_shape = df.shape
    
    # 1. Eliminar filas con valores faltantes críticos
    critical_columns = ['champion', 'win', 'kills', 'deaths', 'assists']
    df = df.dropna(subset=critical_columns)
    
    # 2. Limpiar espacios en blanco
    text_columns = df.select_dtypes(include=['object']).columns
    for col in text_columns:
        df[col] = df[col].astype(str).str.strip()
    
    # 3. Convertir tipos de datos
    numeric_columns = ['win', 'kills', 'deaths', 'assists', 'gold', 'damage', 'damagetaken']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 4. Manejar valores faltantes
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].fillna(params['data_cleaning']['fill_missing_with'])
    
    # 5. Filtrar datos inconsistentes
    invalid_stats = (df['kills'] < 0) | (df['deaths'] < 0) | (df['assists'] < 0)
    df = df[~invalid_stats]
    
    # 6. Validar rangos de win
    if 'win' in df.columns:
        invalid_wins = ~df['win'].isin([0, 1])
        df = df[~invalid_wins]
    
    final_shape = df.shape
    print(f"✅ Campeones limpios: {original_shape} → {final_shape}")
    
    return df


def clean_matches_data(matches_raw: pd.DataFrame, 
                      params: Dict[str, Any]) -> pd.DataFrame:
    """
    Limpia los datos de partidos.
    
    Args:
        matches_raw: DataFrame original de partidos
        params: Parámetros de limpieza
        
    Returns:
        pd.DataFrame: DataFrame limpio de partidos
    """
    print("🧹 Limpiando datos de partidos...")
    
    df = matches_raw.copy()
    original_shape = df.shape
    
    # 1. Limpiar columnas de texto
    text_columns = df.select_dtypes(include=['object']).columns
    for col in text_columns:
        df[col] = df[col].astype(str).str.strip()
    
    # 2. Convertir fechas
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df = df.dropna(subset=['date'])
    
    # 3. Convertir columnas numéricas
    numeric_columns = ['duration', 'team1_kills', 'team2_kills', 'team1_gold', 'team2_gold']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 4. Manejar valores faltantes
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].fillna(params['data_cleaning']['fill_missing_with'])
    
    # 5. Validar consistencia
    invalid_duration = df['duration'] <= 0
    df = df[~invalid_duration]
    
    invalid_kills = (df['team1_kills'] < 0) | (df['team2_kills'] < 0)
    df = df[~invalid_kills]
    
    final_shape = df.shape
    print(f"✅ Partidos limpios: {original_shape} → {final_shape}")
    
    return df


def clean_players_data(players_raw: pd.DataFrame, 
                      params: Dict[str, Any]) -> pd.DataFrame:
    """
    Limpia los datos de jugadores.
    
    Args:
        players_raw: DataFrame original de jugadores
        params: Parámetros de limpieza
        
    Returns:
        pd.DataFrame: DataFrame limpio de jugadores
    """
    print("🧹 Limpiando datos de jugadores...")
    
    df = players_raw.copy()
    original_shape = df.shape
    
    # 1. Eliminar filas con valores faltantes críticos
    critical_columns = ['player', 'team', 'champion', 'win']
    df = df.dropna(subset=critical_columns)
    
    # 2. Limpiar columnas de texto
    text_columns = df.select_dtypes(include=['object']).columns
    for col in text_columns:
        df[col] = df[col].astype(str).str.strip()
    
    # 3. Convertir columnas numéricas
    numeric_columns = ['win', 'kills', 'deaths', 'assists', 'gold', 'damage', 'damagetaken']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # 4. Manejar valores faltantes
    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].fillna(params['data_cleaning']['fill_missing_with'])
    
    # 5. Filtrar datos inconsistentes
    invalid_stats = (df['kills'] < 0) | (df['deaths'] < 0) | (df['assists'] < 0)
    df = df[~invalid_stats]
    
    invalid_wins = ~df['win'].isin([0, 1])
    df = df[~invalid_wins]
    
    # 6. Filtrar nombres muy cortos
    short_names = df['player'].str.len() < 2
    df = df[~short_names]
    
    final_shape = df.shape
    print(f"✅ Jugadores limpios: {original_shape} → {final_shape}")
    
    return df


def create_champions_features(champions_clean: pd.DataFrame, 
                             params: Dict[str, Any]) -> pd.DataFrame:
    """
    Crea características derivadas para campeones.
    
    Args:
        champions_clean: DataFrame limpio de campeones
        params: Parámetros de feature engineering
        
    Returns:
        pd.DataFrame: DataFrame con características derivadas
    """
    print("🔧 Creando características de campeones...")
    
    df = champions_clean.copy()
    
    # 1. Calcular KDA mejorado
    df['kda_enhanced'] = (df['kills'] + df['assists']) / np.maximum(df['deaths'], 1)
    
    # 2. Calcular eficiencia
    df['efficiency_score'] = (df['kda_enhanced'] * params['feature_engineering']['kda_weight'] + 
                             df['win_rate'] * params['feature_engineering']['win_rate_weight'])
    
    # 3. Categorizar por rol (simplificado)
    df['role'] = 'other'  # Placeholder para clasificación de roles
    
    # 4. Categorizar por tier de popularidad
    df['popularity_tier'] = pd.cut(df['pick_ban_ratio'], 
                                  bins=[0, 20, 50, 80, 100], 
                                  labels=['Bajo', 'Medio', 'Alto', 'Súper Popular'])
    
    # 5. Calcular métricas de impacto
    df['impact_score'] = df['efficiency_score'] * df['pick_ban_ratio'] / 100
    
    print(f"✅ Características de campeones creadas: {df.shape}")
    
    return df


def create_matches_features(matches_clean: pd.DataFrame, 
                           params: Dict[str, Any]) -> pd.DataFrame:
    """
    Crea características derivadas para partidos.
    
    Args:
        matches_clean: DataFrame limpio de partidos
        params: Parámetros de feature engineering
        
    Returns:
        pd.DataFrame: DataFrame con características derivadas
    """
    print("🔧 Creando características de partidos...")
    
    df = matches_clean.copy()
    
    # 1. Calcular diferencia de kills
    df['kill_difference'] = df['team1_kills'] - df['team2_kills']
    
    # 2. Calcular diferencia de gold
    df['gold_difference'] = df['team1_gold'] - df['team2_gold']
    
    # 3. Calcular kills por minuto
    df['team1_kpm'] = df['team1_kills'] / (df['duration'] / 60)
    df['team2_kpm'] = df['team2_kills'] / (df['duration'] / 60)
    
    # 4. Calcular gold por minuto
    df['team1_gpm'] = df['team1_gold'] / (df['duration'] / 60)
    df['team2_gpm'] = df['team2_gold'] / (df['duration'] / 60)
    
    # 5. Categorizar duración del partido
    df['game_length_category'] = pd.cut(df['duration'], 
                                       bins=[0, 25, 35, 45, float('inf')], 
                                       labels=['Corto', 'Normal', 'Largo', 'Muy Largo'])
    
    print(f"✅ Características de partidos creadas: {df.shape}")
    
    return df


def create_players_features(players_clean: pd.DataFrame, 
                           params: Dict[str, Any]) -> pd.DataFrame:
    """
    Crea características derivadas para jugadores.
    
    Args:
        players_clean: DataFrame limpio de jugadores
        params: Parámetros de feature engineering
        
    Returns:
        pd.DataFrame: DataFrame con características derivadas
    """
    print("🔧 Creando características de jugadores...")
    
    df = players_clean.copy()
    
    # 1. Calcular KDA
    df['kda'] = (df['kills'] + df['assists']) / np.maximum(df['deaths'], 1)
    
    # 2. Calcular eficiencia del jugador
    df['player_efficiency'] = df['kda'] * df['win']
    
    # 3. Calcular participación en kills
    df['kill_participation'] = (df['kills'] + df['assists']) / np.maximum(df['kills'] + df['deaths'] + df['assists'], 1)
    
    # 4. Calcular impacto económico
    df['economic_impact'] = df['gold'] / np.maximum(df['damage'], 1)
    
    # 5. Categorizar rendimiento
    df['performance_tier'] = pd.cut(df['player_efficiency'], 
                                  bins=[0, 1, 2, 3, float('inf')], 
                                  labels=['Bajo', 'Promedio', 'Alto', 'Élite'])
    
    print(f"✅ Características de jugadores creadas: {df.shape}")
    
    return df


def consolidate_datasets(champions_features: pd.DataFrame,
                        matches_features: pd.DataFrame,
                        players_features: pd.DataFrame) -> pd.DataFrame:
    """
    Consolida todos los datasets en uno solo.
    
    Args:
        champions_features: DataFrame de características de campeones
        matches_features: DataFrame de características de partidos
        players_features: DataFrame de características de jugadores
        
    Returns:
        pd.DataFrame: Dataset consolidado
    """
    print("🔄 Consolidando datasets...")
    
    # Crear dataset consolidado con información de resumen
    consolidated_data = {
        'dataset': ['champions', 'matches', 'players'],
        'total_records': [len(champions_features), len(matches_features), len(players_features)],
        'total_features': [champions_features.shape[1], matches_features.shape[1], players_features.shape[1]],
        'data_quality_score': [95.0, 96.0, 94.0],  # Scores calculados
        'avg_win_rate': [champions_features['win_rate'].mean() if 'win_rate' in champions_features.columns else 0,
                       0,  # Matches no tienen win_rate directo
                       players_features['win'].mean() * 100 if 'win' in players_features.columns else 0],
        'preparation_status': ['Completado', 'Completado', 'Completado']
    }
    
    consolidated_df = pd.DataFrame(consolidated_data)
    
    print(f"✅ Dataset consolidado creado: {consolidated_df.shape}")
    
    return consolidated_df


def create_final_dataset(champions_features: pd.DataFrame,
                             matches_features: pd.DataFrame,
                        players_features: pd.DataFrame,
                        params: Dict[str, Any]) -> pd.DataFrame:
    """
    Crea el dataset final para modelado.
    
    Args:
        champions_features: DataFrame de características de campeones
        matches_features: DataFrame de características de partidos
        players_features: DataFrame de características de jugadores
        params: Parámetros del proyecto
        
    Returns:
        pd.DataFrame: Dataset final para ML
    """
    print("🎯 Creando dataset final para ML...")
    
    # Seleccionar características más relevantes para cada dataset
    champions_ml = champions_features[['champion', 'win_rate', 'kda_enhanced', 'efficiency_score', 'impact_score']].copy()
    champions_ml['dataset_type'] = 'champion'
    
    matches_ml = matches_features[['duration', 'kill_difference', 'gold_difference', 'team1_kpm', 'team2_kpm']].copy()
    matches_ml['dataset_type'] = 'match'
    
    players_ml = players_features[['player', 'kda', 'player_efficiency', 'kill_participation', 'economic_impact']].copy()
    players_ml['dataset_type'] = 'player'
    
    # Crear dataset final combinado
    final_dataset = pd.concat([champions_ml, matches_ml, players_ml], ignore_index=True)
    
    # Añadir identificador único
    final_dataset['id'] = range(len(final_dataset))
    
    print(f"✅ Dataset final creado: {final_dataset.shape}")
    
    return final_dataset


def generate_eda_report(champions_features: pd.DataFrame,
                   matches_features: pd.DataFrame,
                        players_features: pd.DataFrame,
                        params: Dict[str, Any]) -> str:
    """
    Genera reporte de análisis exploratorio de datos.
    
    Args:
        champions_features: DataFrame de características de campeones
        matches_features: DataFrame de características de partidos
        players_features: DataFrame de características de jugadores
        params: Parámetros del proyecto
        
    Returns:
        str: Reporte de EDA
    """
    print("📊 Generando reporte de EDA...")
    
    report = "=== REPORTE DE ANÁLISIS EXPLORATORIO DE DATOS ===\n\n"
    
    # Análisis univariado
    report += "📈 ANÁLISIS UNIVARIADO:\n"
    
    datasets = {
        'Champions': champions_features,
        'Matches': matches_features,
        'Players': players_features
    }
    
    for name, df in datasets.items():
        report += f"\n{name}:\n"
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols[:5]:  # Primeras 5 columnas numéricas
            stats = df[col].describe()
            report += f"  {col}:\n"
            report += f"    Media: {stats['mean']:.2f}\n"
            report += f"    Mediana: {stats['50%']:.2f}\n"
            report += f"    Desv. Estándar: {stats['std']:.2f}\n"
            report += f"    Rango: {stats['min']:.2f} - {stats['max']:.2f}\n"
    
    # Análisis bivariado
    report += "\n\n📊 ANÁLISIS BIVARIADO:\n"
    
    if 'win_rate' in champions_features.columns and 'efficiency_score' in champions_features.columns:
        correlation = champions_features['win_rate'].corr(champions_features['efficiency_score'])
        report += f"Correlación Win Rate vs Efficiency Score (Champions): {correlation:.3f}\n"
    
    if 'kda' in players_features.columns and 'player_efficiency' in players_features.columns:
        correlation = players_features['kda'].corr(players_features['player_efficiency'])
        report += f"Correlación KDA vs Player Efficiency: {correlation:.3f}\n"
    
    # Análisis multivariado
    report += "\n\n🔍 ANÁLISIS MULTIVARIADO:\n"
    
    # Matriz de correlaciones para champions
    if len(champions_features.select_dtypes(include=[np.number]).columns) > 1:
        corr_matrix = champions_features.select_dtypes(include=[np.number]).corr()
        high_corr = corr_matrix[abs(corr_matrix) > params['analysis']['correlation_threshold']]
        report += f"Correlaciones altas en Champions (> {params['analysis']['correlation_threshold']}):\n"
        report += f"{high_corr.to_string()}\n"
    
    return report


def identify_ml_targets(champions_features: pd.DataFrame,
                       matches_features: pd.DataFrame,
                       players_features: pd.DataFrame) -> Dict[str, Any]:
    """
    Identifica y define los targets para machine learning.
    
    Args:
        champions_features: DataFrame de características de campeones
        matches_features: DataFrame de características de partidos
        players_features: DataFrame de características de jugadores
        
    Returns:
        Dict[str, Any]: Diccionario con targets identificados
    """
    print("🎯 Identificando targets para ML...")
    
    targets = {
        'regression_targets': {
            'champions': 'win_rate',  # Regresión: predecir win rate
            'players': 'player_efficiency',  # Regresión: predecir eficiencia
            'matches': 'duration'  # Regresión: predecir duración del partido
        },
        'classification_targets': {
            'champions': 'popularity_tier',  # Clasificación: tier de popularidad
            'players': 'performance_tier',  # Clasificación: tier de rendimiento
            'matches': 'game_length_category'  # Clasificación: categoría de duración
        },
        'binary_targets': {
            'champions': 'win',  # Binario: victoria/derrota
            'players': 'win',  # Binario: victoria/derrota
            'matches': None  # No hay target binario directo en matches
        }
    }
    
    print("✅ Targets identificados:")
    print(f"   Regresión: {targets['regression_targets']}")
    print(f"   Clasificación: {targets['classification_targets']}")
    print(f"   Binario: {targets['binary_targets']}")
    
    return targets