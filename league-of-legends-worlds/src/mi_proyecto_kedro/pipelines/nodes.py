"""Nodos para el pipeline de exploración de datos de League of Legends Worlds."""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


def explore_champions_data(champions_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Explorar y analizar datos de campeones.
    
    Args:
        champions_stats: DataFrame con estadísticas de campeones
        
    Returns:
        DataFrame con análisis de campeones
    """
    logger.info(f"Explorando datos de campeones: {champions_stats.shape}")
    
    # Crear análisis básico
    analysis = champions_stats.copy()
    
    # Calcular métricas adicionales
    analysis['pick_rate'] = (analysis['played_games'] / 
                           analysis['games_contests'] * 100).round(2)
    analysis['ban_rate'] = (analysis['banned_games'] / 
                          analysis['games_contests'] * 100).round(2)
    
    # Clasificar campeones por popularidad
    analysis['popularity_tier'] = pd.cut(
        analysis['pick_ban_ratio'], 
        bins=[0, 25, 50, 75, 100], 
        labels=['Poco Popular', 'Moderado', 'Popular', 'Súper Popular']
    )
    
    # Calcular eficiencia (KDA * win_rate)
    analysis['efficiency_score'] = (
        analysis['kill_death_assist_ratio'] * analysis['win_rate'] / 100
    ).round(2)
    
    logger.info(f"Análisis de campeones completado: {analysis.shape}")
    return analysis


def explore_matches_data(matchs_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Explorar y analizar datos de partidos.
    
    Args:
        matchs_stats: DataFrame con estadísticas de partidos
        
    Returns:
        DataFrame con análisis de partidos
    """
    logger.info(f"Explorando datos de partidos: {matchs_stats.shape}")
    
    analysis = matchs_stats.copy()
    
    # Convertir fecha
    analysis['date'] = pd.to_datetime(analysis['date'])
    analysis['year'] = analysis['date'].dt.year
    analysis['month'] = analysis['date'].dt.month
    
    # Extraer campeones únicos por partido
    champion_columns = [col for col in analysis.columns if 'pick_' in col or 'ban_' in col]
    
    # Crear lista de todos los campeones en cada partido
    analysis['all_champions'] = analysis[champion_columns].apply(
        lambda row: [champ for champ in row.dropna()], axis=1
    )
    
    # Contar campeones únicos por partido
    analysis['unique_champions_count'] = analysis['all_champions'].apply(len)
    
    logger.info(f"Análisis de partidos completado: {analysis.shape}")
    return analysis


def explore_players_data(players_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Explorar y analizar datos de jugadores.
    
    Args:
        players_stats: DataFrame con estadísticas de jugadores
        
    Returns:
        DataFrame con análisis de jugadores
    """
    logger.info(f"Explorando datos de jugadores: {players_stats.shape}")
    
    analysis = players_stats.copy()
    
    # Calcular métricas adicionales
    analysis['total_games'] = analysis['wins'] + analysis['loses']
    analysis['avg_kda'] = analysis['kill_death_assist_ratio'].round(2)
    
    # Clasificar jugadores por rendimiento
    analysis['performance_tier'] = pd.cut(
        analysis['win_rate'], 
        bins=[0, 40, 55, 70, 100], 
        labels=['Bajo', 'Promedio', 'Alto', 'Élite']
    )
    
    # Calcular impacto del jugador
    analysis['player_impact'] = (
        analysis['kill_participation'] * analysis['win_rate'] / 100
    ).round(2)
    
    logger.info(f"Análisis de jugadores completado: {analysis.shape}")
    return analysis


def generate_data_summary(champions_stats: pd.DataFrame, 
                         matchs_stats: pd.DataFrame, 
                         players_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Generar resumen consolidado de todos los datos.
    
    Args:
        champions_stats: DataFrame de campeones
        matchs_stats: DataFrame de partidos
        players_stats: DataFrame de jugadores
        
    Returns:
        DataFrame consolidado
    """
    logger.info("Generando resumen consolidado de datos")
    
    # Crear resumen de datos
    summary_data = {
        'dataset': ['champions_stats', 'matchs_stats', 'players_stats'],
        'total_records': [
            len(champions_stats), 
            len(matchs_stats), 
            len(players_stats)
        ],
        'total_columns': [
            len(champions_stats.columns), 
            len(matchs_stats.columns), 
            len(players_stats.columns)
        ],
        'missing_values': [
            champions_stats.isnull().sum().sum(),
            matchs_stats.isnull().sum().sum(),
            players_stats.isnull().sum().sum()
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    
    # Agregar estadísticas generales
    summary_df['data_quality_score'] = (
        (1 - summary_df['missing_values'] / summary_df['total_records']) * 100
    ).round(2)
    
    logger.info(f"Resumen consolidado generado: {summary_df.shape}")
    return summary_df


def create_data_quality_report(champions_stats: pd.DataFrame,
                              matchs_stats: pd.DataFrame,
                              players_stats: pd.DataFrame) -> None:
    """
    Crear reporte de calidad de datos.
    
    Args:
        champions_stats: DataFrame de campeones
        matchs_stats: DataFrame de partidos  
        players_stats: DataFrame de jugadores
    """
    logger.info("Creando reporte de calidad de datos")
    
    datasets = {
        'champions_stats': champions_stats,
        'matchs_stats': matchs_stats,
        'players_stats': players_stats
    }
    
    for name, df in datasets.items():
        logger.info(f"\n=== REPORTE DE CALIDAD: {name.upper()} ===")
        logger.info(f"Forma del dataset: {df.shape}")
        logger.info(f"Columnas: {list(df.columns)}")
        logger.info(f"Valores faltantes: {df.isnull().sum().sum()}")
        logger.info(f"Duplicados: {df.duplicated().sum()}")
        logger.info(f"Tipos de datos:\n{df.dtypes}")
        
        # Estadísticas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            logger.info(f"Estadísticas numéricas:\n{df[numeric_cols].describe()}")
    
    logger.info("Reporte de calidad completado")
