"""Pipeline para exploración y análisis inicial de datos de League of Legends Worlds."""

from kedro.pipeline import Pipeline, node
from .nodes import (
    explore_champions_data,
    explore_matches_data,
    explore_players_data,
    generate_data_summary,
    create_data_quality_report
)


def create_pipeline(**kwargs) -> Pipeline:
    """Crear pipeline de exploración de datos."""
    return Pipeline(
        [
            node(
                func=explore_champions_data,
                inputs="champions_stats",
                outputs="champions_analysis",
                name="explore_champions_data_node",
                tags=["exploration", "champions"]
            ),
            node(
                func=explore_matches_data,
                inputs="matchs_stats",
                outputs="matchs_analysis",
                name="explore_matches_data_node",
                tags=["exploration", "matches"]
            ),
            node(
                func=explore_players_data,
                inputs="players_stats",
                outputs="players_analysis",
                tags=["exploration", "players"]
            ),
            node(
                func=generate_data_summary,
                inputs=["champions_stats", "matchs_stats", "players_stats"],
                outputs="worlds_consolidated",
                name="generate_data_summary_node",
                tags=["exploration", "summary"]
            ),
            node(
                func=create_data_quality_report,
                inputs=["champions_stats", "matchs_stats", "players_stats"],
                outputs=None,
                name="create_data_quality_report_node",
                tags=["exploration", "quality"]
            ),
        ]
    )
