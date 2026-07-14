import pandas as pd
import joblib
from lightgbm import LGBMRegressor


def train_lgbm_model(
    X_train, y_train, X_valid, y_valid, best_params, random_state, callbacks=None
):
    """
    Huấn luyện mô hình LGBMRegressor với bộ tham số tốt nhất.
    """
    model = LGBMRegressor(
        n_estimators=best_params["n_estimators"],
        learning_rate=best_params["learning_rate"],
        num_leaves=best_params["num_leaves"],
        random_state=random_state,
        verbose=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=callbacks,
    )

    return model


def calculate_feature_importance(model, feature_columns):
    """
    Tính toán Feature Importance theo cả hai loại: Split và Gain.
    """
    split_importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "split_importance": model.booster_.feature_importance(
                importance_type="split"
            ),
        }
    ).sort_values("split_importance", ascending=False)

    gain_importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "gain_importance": model.booster_.feature_importance(
                importance_type="gain"
            ),
        }
    ).sort_values("gain_importance", ascending=False)

    return split_importance, gain_importance


def save_model(model, filepath="model.pkl"):
    """
    Lưu mô hình đã huấn luyện ra file local.
    """
    joblib.dump(model, filepath)
    return filepath
