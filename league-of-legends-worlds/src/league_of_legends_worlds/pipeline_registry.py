"""Project pipelines."""
from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    
    # Agregar pipelines específicos de CRISP-DM
    from .pipelines.data_exploration import create_pipeline as create_data_exploration_pipeline
    from .pipelines.data_preparation import create_pipeline as create_data_preparation_pipeline
    
    pipelines["data_exploration"] = create_data_exploration_pipeline()
    pipelines["data_preparation"] = create_data_preparation_pipeline()
    
    # Pipeline completo que incluye exploración y preparación
    pipelines["crisp_dm_etapas_1_3"] = (
        create_data_exploration_pipeline() + 
        create_data_preparation_pipeline()
    )
    
    pipelines["__default__"] = pipelines["crisp_dm_etapas_1_3"]
    return pipelines
