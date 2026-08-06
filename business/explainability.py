"""Helpers for model explainability."""

SUPPORTED_TREE_MODEL_TYPES = {"lightgbm"}


def get_tree_explainable_model(model_artifact: object, model_type: str):
    """Return the raw tree model from a trained model bundle."""
    normalized_model_type = str(model_type).strip().lower()

    if normalized_model_type not in SUPPORTED_TREE_MODEL_TYPES:
        raise ValueError(
            f"SHAP TreeExplainer does not support model type '{normalized_model_type}'."
        )

    if not isinstance(model_artifact, dict):
        raise TypeError(
            "Expected a model artifact bundle represented by dict, "
            f"received {type(model_artifact).__name__}."
        )

    model = model_artifact.get("model")
    if model is None:
        raise ValueError("Model artifact bundle does not contain the 'model' key.")

    if not hasattr(model, "predict"):
        raise TypeError(
            "The model inside the artifact bundle does not provide predict(). "
            f"Received {type(model).__name__}."
        )

    return model
