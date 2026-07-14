import optuna
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_percentage_error


def run_hpo_optimization(X_train, y_train, X_valid, y_valid, n_trials, random_state):
    """
    Thực hiện tối ưu hóa Hyperparameters bằng Optuna.
    Hàm này hoàn toàn độc lập với ClearML Task.
    """

    def objective(trial: optuna.Trial) -> float:
        model_params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 100),
            "n_estimators": trial.suggest_int("num_estimators", 100, 1000),
            "random_state": random_state,
        }

        model = LGBMRegressor(**model_params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_valid)
        mape = mean_absolute_percentage_error(y_valid, y_pred)

        return mape

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )

    study.optimize(objective, n_trials=n_trials)

    return study
