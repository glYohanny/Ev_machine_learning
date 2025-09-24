"""
Pipeline de preparación de datos para League of Legends Worlds.
"""

from kedro.pipeline import Pipeline, node
from .nodes import (
    clean_champions_data,
    clean_matches_data,
    clean_players_data,
    create_champions_features,
    create_matches_features,
    create_players_features,
    consolidate_datasets,
    create_final_dataset,
    identify_ml_targets
)


def create_data_preparation_pipeline() -> Pipeline:
    """
    Crea el pipeline de preparación de datos.
    
    Returns:
        Pipeline: Pipeline de Kedro para preparación
    """
    return Pipeline(
        [
            # Limpieza de datos
            node(
                func=clean_champions_data,
                inputs=["champions_raw", "params:data_cleaning"],
                outputs="champions_clean",
                name="clean_champions_data_node",
                tags=["data_cleaning"]
            ),
            node(
                func=clean_matches_data,
                inputs=["matches_raw", "params:data_cleaning"],
                outputs="matches_clean",
                name="clean_matches_data_node",
                tags=["data_cleaning"]
            ),
            node(
                func=clean_players_data,
                inputs=["players_raw", "params:data_cleaning"],
                outputs="players_clean",
                name="clean_players_data_node",
                tags=["data_cleaning"]
            ),
            
            # Feature Engineering
            node(
                func=create_champions_features,
                inputs=["champions_clean", "params:feature_engineering"],
                outputs="champions_features",
                name="create_champions_features_node",
                tags=["feature_engineering"]
            ),
            node(
                func=create_matches_features,
                inputs=["matches_clean", "params:feature_engineering"],
                outputs="matches_features",
                name="create_matches_features_node",
                tags=["feature_engineering"]
            ),
            node(
                func=create_players_features,
                inputs=["players_clean", "params:feature_engineering"],
                name="create_players_features_node",
                outputs="players_features",
                tags=["feature_engineering"]
            ),
            
            # Consolidación
            node(
                func=consolidate_datasets,
                inputs=["champions_features", "matches_features", "players_features"],
                outputs="worlds_consolidated",
                name="consolidate_datasets_node",
                tags=["consolidation"]
            ),
            node(
                func=create_final_dataset,
                inputs=["champions_features", "matches_features", "players_features", "params:modeling"],
                outputs="worlds_final",
                name="create_final_dataset_node",
                tags=["final_dataset"]
            ),
            node(
                func=identify_ml_targets,
                inputs=["champions_features", "matches_features", "players_features"],
                outputs="ml_targets",
                name="identify_ml_targets_node",
                tags=["ml_targets"]
            )
        ],
        tags=["data_preparation"]
    )