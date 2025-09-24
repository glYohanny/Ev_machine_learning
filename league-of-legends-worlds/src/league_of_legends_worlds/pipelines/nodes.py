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


# ==================== FUNCIONES DE PREPARACIÓN DE DATOS ====================

def clean_champions_data(champions_analysis: pd.DataFrame) -> pd.DataFrame:
    """
    Limpiar y preparar datos de campeones.
    
    Args:
        champions_analysis: DataFrame con análisis de campeones
        
    Returns:
        DataFrame limpio de campeones
    """
    logger.info("Limpiando datos de campeones")
    
    df = champions_analysis.copy()
    
    # Rellenar valores faltantes
    df['kill_participation'] = df['kill_participation'].fillna(0)
    df['kill_share'] = df['kill_share'].fillna(0)
    df['gold_share'] = df['gold_share'].fillna(0)
    
    # Limpiar nombres de campeones
    df['champion'] = df['champion'].str.strip()
    
    # Convertir columnas numéricas
    numeric_columns = ['pick_ban_ratio', 'win_rate', 'kill_death_assist_ratio', 
                      'cs/min', 'gold/min', 'damage/min']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Eliminar filas con datos críticos faltantes
    df = df.dropna(subset=['champion', 'win_rate', 'pick_ban_ratio'])
    
    logger.info(f"Datos de campeones limpios: {df.shape}")
    return df


def clean_matches_data(matchs_analysis: pd.DataFrame) -> pd.DataFrame:
    """
    Limpiar y preparar datos de partidos.
    
    Args:
        matchs_analysis: DataFrame con análisis de partidos
        
    Returns:
        DataFrame limpio de partidos
    """
    logger.info("Limpiando datos de partidos")
    
    df = matchs_analysis.copy()
    
    # Limpiar nombres de equipos
    df['blue_team'] = df['blue_team'].str.strip()
    df['red_team'] = df['red_team'].str.strip()
    df['winner'] = df['winner'].str.strip()
    
    # Estandarizar nombres de equipos (ejemplo básico)
    team_mapping = {
        'TSM': 'Team SoloMid',
        'Fnatic': 'FNATIC',
        'SKT': 'SK Telecom T1',
        'KT': 'KT Rolster'
    }
    
    for old_name, new_name in team_mapping.items():
        df['blue_team'] = df['blue_team'].replace(old_name, new_name)
        df['red_team'] = df['red_team'].replace(old_name, new_name)
        df['winner'] = df['winner'].replace(old_name, new_name)
    
    # Limpiar nombres de campeones en picks y bans
    champion_columns = [col for col in df.columns if 'pick_' in col or 'ban_' in col]
    for col in champion_columns:
        df[col] = df[col].str.strip()
    
    # Convertir fecha
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    logger.info(f"Datos de partidos limpios: {df.shape}")
    return df


def clean_players_data(players_analysis: pd.DataFrame) -> pd.DataFrame:
    """
    Limpiar y preparar datos de jugadores.
    
    Args:
        players_analysis: DataFrame con análisis de jugadores
        
    Returns:
        DataFrame limpio de jugadores
    """
    logger.info("Limpiando datos de jugadores")
    
    df = players_analysis.copy()
    
    # Limpiar nombres de jugadores y equipos
    df['player'] = df['player'].str.strip()
    df['team'] = df['team'].str.strip()
    
    # Rellenar valores faltantes
    df['kill_participation'] = df['kill_participation'].fillna(0)
    df['kill_share'] = df['kill_share'].fillna(0)
    df['gold_share'] = df['gold_share'].fillna(0)
    
    # Convertir columnas numéricas
    numeric_columns = ['win_rate', 'kill_death_assist_ratio', 'cs/min', 
                      'gold/min', 'damage/min']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Filtrar jugadores con al menos 5 partidos para análisis más confiable
    df = df[df['games_played'] >= 5]
    
    logger.info(f"Datos de jugadores limpios: {df.shape}")
    return df


def create_derived_features(champions_clean: pd.DataFrame,
                           matches_clean: pd.DataFrame,
                           players_clean: pd.DataFrame) -> tuple:
    """
    Crear características derivadas para análisis.
    
    Args:
        champions_clean: DataFrame limpio de campeones
        matches_clean: DataFrame limpio de partidos
        players_clean: DataFrame limpio de jugadores
        
    Returns:
        Tupla con DataFrames con características derivadas
    """
    logger.info("Creando características derivadas")
    
    # Características derivadas para campeones
    champs_features = champions_clean.copy()
    
    # Clasificación de campeones por rol (simplificada)
    champion_roles = {
        'top': ['Garen', 'Darius', 'Jax', 'Fiora', 'Camille', 'Irelia', 'Riven'],
        'jungle': ['Lee Sin', 'Jarvan IV', 'Elise', 'Rek\'Sai', 'Graves', 'Nidalee'],
        'mid': ['Zed', 'Yasuo', 'Ahri', 'Orianna', 'Syndra', 'Azir', 'LeBlanc'],
        'adc': ['Caitlyn', 'Jinx', 'Vayne', 'Tristana', 'Sivir', 'Ashe'],
        'support': ['Thresh', 'Blitzcrank', 'Leona', 'Braum', 'Janna', 'Soraka']
    }
    
    def assign_role(champion):
        for role, champions in champion_roles.items():
            if champion in champions:
                return role
        return 'other'
    
    champs_features['role'] = champs_features['champion'].apply(assign_role)
    
    # Calcular métricas de eficiencia
    champs_features['efficiency_score'] = (
        champs_features['kill_death_assist_ratio'] * 
        champs_features['win_rate'] / 100
    ).round(2)
    
    # Características derivadas para partidos
    matches_features = matches_clean.copy()
    
    # Crear indicador de duración del partido (simulado)
    matches_features['match_duration'] = np.random.randint(20, 60, len(matches_features))
    
    # Crear indicador de equipo favorito (basado en win rate histórico)
    team_stats = players_clean.groupby('team')['win_rate'].mean().to_dict()
    matches_features['blue_team_win_rate'] = matches_features['blue_team'].map(team_stats)
    matches_features['red_team_win_rate'] = matches_features['red_team'].map(team_stats)
    matches_features['blue_team_favorite'] = (
        matches_features['blue_team_win_rate'] > matches_features['red_team_win_rate']
    )
    
    # Características derivadas para jugadores
    players_features = players_clean.copy()
    
    # Clasificar jugadores por rendimiento
    players_features['performance_category'] = pd.cut(
        players_features['win_rate'],
        bins=[0, 40, 55, 70, 100],
        labels=['Bajo', 'Promedio', 'Alto', 'Élite']
    )
    
    # Calcular impacto del jugador
    players_features['player_impact_score'] = (
        players_features['kill_participation'] * 
        players_features['win_rate'] / 100
    ).round(2)
    
    logger.info("Características derivadas creadas")
    return champs_features, matches_features, players_features


def validate_data_consistency(champions_features: pd.DataFrame,
                             matches_features: pd.DataFrame,
                             players_features: pd.DataFrame) -> None:
    """
    Validar consistencia de los datos preparados.
    
    Args:
        champions_features: DataFrame de campeones con características
        matches_features: DataFrame de partidos con características
        players_features: DataFrame de jugadores con características
    """
    logger.info("Validando consistencia de datos")
    
    # Validaciones básicas
    validations = []
    
    # 1. Verificar rangos de win_rate
    champs_wr_valid = ((champions_features['win_rate'] >= 0) & 
                      (champions_features['win_rate'] <= 100)).all()
    validations.append(f"Champions win_rate válido: {champs_wr_valid}")
    
    players_wr_valid = ((players_features['win_rate'] >= 0) & 
                       (players_features['win_rate'] <= 100)).all()
    validations.append(f"Players win_rate válido: {players_wr_valid}")
    
    # 2. Verificar que no hay duplicados críticos
    champs_duplicates = champions_features.duplicated(subset=['champion', 'season']).sum()
    validations.append(f"Champions duplicados: {champs_duplicates}")
    
    # 3. Verificar consistencia de fechas
    matches_date_valid = matches_features['date'].notna().all()
    validations.append(f"Fechas de partidos válidas: {matches_date_valid}")
    
    # 4. Verificar que los equipos en partidos existen en datos de jugadores
    match_teams = set(matches_features['blue_team'].tolist() + 
                     matches_features['red_team'].tolist())
    player_teams = set(players_features['team'].tolist())
    teams_consistent = len(match_teams - player_teams) < 5  # Tolerar algunas diferencias
    validations.append(f"Equipos consistentes: {teams_consistent}")
    
    logger.info("Validaciones completadas:")
    for validation in validations:
        logger.info(f"  - {validation}")
    
    # Reportar problemas críticos
    if not champs_wr_valid or not players_wr_valid:
        logger.warning("⚠️  PROBLEMA CRÍTICO: Valores de win_rate fuera de rango")
    
    if champs_duplicates > 0:
        logger.warning(f"⚠️  PROBLEMA: {champs_duplicates} campeones duplicados")


def merge_datasets(champions_features: pd.DataFrame,
                   matches_features: pd.DataFrame,
                   players_features: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidar todos los datasets en uno final para análisis.
    
    Args:
        champions_features: DataFrame de campeones con características
        matches_features: DataFrame de partidos con características
        players_features: DataFrame de jugadores con características
        
    Returns:
        DataFrame consolidado
    """
    logger.info("Consolidando datasets")
    
    # Crear resumen consolidado
    summary_data = {
        'dataset': ['champions', 'matches', 'players'],
        'total_records': [
            len(champions_features),
            len(matches_features),
            len(players_features)
        ],
        'total_features': [
            len(champions_features.columns),
            len(matches_features.columns),
            len(players_features.columns)
        ],
        'data_quality_score': [
            (1 - champions_features.isnull().sum().sum() / 
             (len(champions_features) * len(champions_features.columns))) * 100,
            (1 - matches_features.isnull().sum().sum() / 
             (len(matches_features) * len(matches_features.columns))) * 100,
            (1 - players_features.isnull().sum().sum() / 
             (len(players_features) * len(players_features.columns))) * 100
        ]
    }
    
    consolidated_df = pd.DataFrame(summary_data)
    
    # Agregar estadísticas generales
    consolidated_df['avg_win_rate'] = [
        champions_features['win_rate'].mean(),
        None,  # Los partidos no tienen win_rate directo
        players_features['win_rate'].mean()
    ]
    
    consolidated_df['preparation_status'] = 'Completado'
    
    logger.info(f"Dataset consolidado creado: {consolidated_df.shape}")
    logger.info("ETAPA 3 - PREPARACIÓN DE DATOS COMPLETADA ✅")
    
    return consolidated_df
