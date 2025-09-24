"""Pipeline para preparación y limpieza de datos de League of Legends Worlds."""

from kedro.pipeline import Pipeline, node
from .nodes import (
    clean_champions_data,
    clean_matches_data,
    clean_players_data,
    create_derived_features,
    validate_data_consistency,
    merge_datasets
)


def create_pipeline(**kwargs) -> Pipeline:
    """Crear pipeline de preparación de datos."""
    return Pipeline(
        [
            node(
                func=clean_champions_data,
                inputs="champions_analysis",
                outputs="champions_clean",
                name="clean_champions_data_node",
                tags=["preparation", "champions"]
            ),
            node(
                func=clean_matches_data,
                inputs="matchs_analysis",
                outputs="matches_clean",
                name="clean_matches_data_node",
                tags=["preparation", "matches"]
            ),
            node(
                func=clean_players_data,
                inputs="players_analysis",
                outputs="players_clean",
                name="clean_players_data_node",
                tags=["preparation", "players"]
            ),
            node(
                func=create_derived_features,
                inputs=["champions_clean", "matches_clean", "players_clean"],
                outputs=["champions_features", "matches_features", "players_features"],
                name="create_derived_features_node",
                tags=["preparation", "features"]
            ),
            node(
                func=validate_data_consistency,
                inputs=["champions_features", "matches_features", "players_features"],
                outputs=None,
                name="validate_data_consistency_node",
                tags=["preparation", "validation"]
            ),
            node(
                func=merge_datasets,
                inputs=["champions_features", "matches_features", "players_features"],
                outputs="worlds_final",
                name="merge_datasets_node",
                tags=["preparation", "merge"]
            ),
        ]
    )
