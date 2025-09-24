"""
Registry de pipelines para League of Legends Worlds.
"""

from kedro.pipeline import Pipeline
from .data_exploration import create_data_exploration_pipeline
from .data_preparation import create_data_preparation_pipeline


def register_pipelines() -> dict:
    """
    Registra todos los pipelines disponibles.
    
    Returns:
        dict: Diccionario con pipelines registrados
    """
    return {
        "data_exploration": create_data_exploration_pipeline(),
        "data_preparation": create_data_preparation_pipeline(),
        "full_pipeline": create_data_exploration_pipeline() + create_data_preparation_pipeline(),
        "__default__": create_data_exploration_pipeline() + create_data_preparation_pipeline()
    }