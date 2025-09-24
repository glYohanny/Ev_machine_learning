"""
Pipeline de exploración de datos para League of Legends Worlds.
"""

from kedro.pipeline import Pipeline, node
from .nodes import (
    load_raw_data,
    analyze_data_quality,
    generate_eda_report
)


def create_data_exploration_pipeline() -> Pipeline:
    """
    Crea el pipeline de exploración de datos.
    
    Returns:
        Pipeline: Pipeline de Kedro para exploración
    """
    return Pipeline(
        [
            node(
                func=load_raw_data,
                inputs=None,
                outputs=["champions_raw", "matches_raw", "players_raw"],
                name="load_raw_data_node",
                tags=["data_loading"]
            ),
            node(
                func=analyze_data_quality,
                inputs=["champions_raw", "matches_raw", "players_raw", "params:data_cleaning"],
                outputs="data_quality_report",
                name="analyze_data_quality_node",
                tags=["data_quality"]
            ),
            node(
                func=generate_eda_report,
                inputs=["champions_raw", "matches_raw", "players_raw", "params:analysis"],
                outputs="eda_report",
                name="generate_eda_report_node",
                tags=["eda"]
            )
        ],
        tags=["data_exploration"]
    )