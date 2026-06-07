from __future__ import annotations

import re
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import optuna
import pandas as pd

from ..evaluation.cv import TimeSeriesCV
from ..evaluation.metrics import mase

optuna.logging.set_verbosity(optuna.logging.WARNING)

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "regression_l1",
    "verbosity": -1,
    "n_jobs": -1,
    "num_leaves": 127,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
}

_LAG_PATTERN = re.compile(r"^lag_(\d+)$")


class LGBMGlobalForecaster:
    """LightGBM trained globally on all series simultaneously.

    Accepts pre-engineered feature DataFrames (unique_id, ds, y + feature cols).
    Uses predict_recursive() for forecasting because future feature rows must be
    supplied by the caller — the model cannot reconstruct calendar or hierarchical
    features internally, so it does not implement the BaseForecaster interface.

    Recursive vs. direct:
      With SAFE_LAGS >= horizon=28, every future lag value references a
      historical date, so predict_recursive() degenerates to a single forward
      pass (no step-by-step updating). The implementation handles the general
      case correctly for shorter lags too.
    """

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
        random_state: int = 42,
    ) -> None:
        self.params = {**DEFAULT_PARAMS, **(params or {}), "seed": random_state}
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state

        self._model: lgb.Booster | None = None
        self._feature_cols: list[str] = []
        self._target_col: str = "y"
        self._lag_offsets: dict[str, int] = {}  # col_name -> lag days

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str = "y",
        val_df: pd.DataFrame | None = None,
    ) -> LGBMGlobalForecaster:
        """Train on df[feature_cols] → df[target_col].

        Args:
            df: Training rows. Must have unique_id, ds, target_col, + feature_cols.
            feature_cols: Ordered list of feature column names.
            target_col: Name of the target column.
            val_df: Optional validation set for early stopping (same schema as df).
        """
        self._feature_cols = feature_cols
        self._target_col = target_col
        self._lag_offsets = {
            col: int(m.group(1))
            for col in feature_cols
            if (m := _LAG_PATTERN.match(col))
        }

        clean = df.dropna(subset=feature_cols + [target_col])
        train_data = lgb.Dataset(
            clean[feature_cols].values,
            label=clean[target_col].values,
            feature_name=feature_cols,
            free_raw_data=False,
        )

        callbacks: list[Any] = [lgb.log_evaluation(period=-1)]
        valid_sets = [train_data]
        valid_names = ["train"]

        if val_df is not None:
            clean_val = val_df.dropna(subset=feature_cols + [target_col])
            val_data = lgb.Dataset(
                clean_val[feature_cols].values,
                label=clean_val[target_col].values,
                feature_name=feature_cols,
                free_raw_data=False,
                reference=train_data,
            )
            valid_sets.append(val_data)
            valid_names.append("val")
            callbacks.append(
                lgb.early_stopping(self.early_stopping_rounds, verbose=False)
            )

        self._model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_recursive(
        self,
        df: pd.DataFrame,
        horizon: int = 28,
    ) -> pd.DataFrame:
        """Forecast horizon steps ahead from a feature-engineered history.

        Args:
            df: Full feature-engineered DataFrame (history + future rows).
                Future rows have target_col == NaN. All feature columns for
                future rows must be pre-computed by the caller. For lags >=
                horizon (the default pipeline), every future feature value
                already references a historical date — no recursion is needed
                and all horizon steps are predicted in one forward pass.
                For pipelines with lags < horizon, this method iterates
                step-by-step and back-fills predicted values into subsequent
                feature rows.
            horizon: Number of steps to forecast (used only to validate the
                     number of NaN target rows per series).

        Returns:
            DataFrame with columns: unique_id, ds, y_pred.
        """
        if self._model is None:
            raise RuntimeError("Call fit() first.")

        target = self._target_col
        future_mask = df[target].isna()
        if not future_mask.any():
            raise ValueError(
                f"No future rows found (all '{target}' values are non-NaN). "
                "Future rows must have target_col == NaN."
            )

        future_df = df[future_mask].copy().sort_values(["unique_id", "ds"])

        min_lag = min(self._lag_offsets.values()) if self._lag_offsets else horizon + 1

        if min_lag >= horizon:
            # Direct: all lag values are in training history — predict in one pass
            x_future = future_df[self._feature_cols].values
            future_df["y_pred"] = np.maximum(0.0, self._model.predict(x_future))
        else:
            # True recursive: predict step by step, filling shorter lags
            future_df = self._predict_step_by_step(df, future_df)

        return future_df[["unique_id", "ds", "y_pred"]].reset_index(drop=True)

    def _predict_step_by_step(
        self,
        df_full: pd.DataFrame,
        future_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Iterative prediction for pipelines with lags < horizon."""
        assert self._model is not None
        target = self._target_col
        preds_by_uid: dict[str, dict[pd.Timestamp, float]] = {}

        for uid, group in future_df.groupby("unique_id"):
            history_y = (
                df_full[df_full["unique_id"] == uid]
                .dropna(subset=[target])
                .sort_values("ds")
                .set_index("ds")[target]
            )
            uid_preds: dict[pd.Timestamp, float] = {}

            for _, row in group.sort_values("ds").iterrows():
                feat_row = row[self._feature_cols].copy()

                # Back-fill lags that reference already-predicted future dates
                for col, k in self._lag_offsets.items():
                    lookup = pd.Timestamp(row["ds"]) - pd.Timedelta(days=k)
                    if lookup in uid_preds:
                        feat_row[col] = uid_preds[lookup]
                    elif lookup in history_y.index:
                        feat_row[col] = history_y[lookup]

                pred = float(
                    np.maximum(0.0, self._model.predict(feat_row.values.reshape(1, -1)))
                )
                uid_preds[pd.Timestamp(row["ds"])] = pred

            preds_by_uid[str(uid)] = uid_preds

        future_df = future_df.copy()
        future_df["y_pred"] = future_df.apply(
            lambda r: preds_by_uid[str(r["unique_id"])][pd.Timestamp(r["ds"])],
            axis=1,
        )
        return future_df

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """Return feature importances sorted descending."""
        if self._model is None:
            raise RuntimeError("Call fit() first.")
        imp = self._model.feature_importance(importance_type=importance_type)
        return pd.Series(
            imp, index=self._feature_cols, name=importance_type
        ).sort_values(ascending=False)


# ------------------------------------------------------------------
# Optuna hyperparameter optimisation
# ------------------------------------------------------------------


def optimize_lgbm(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    cv: TimeSeriesCV,
    n_trials: int = 100,
    num_boost_round: int = 500,
    mlflow_experiment: str = "lgbm_optuna",
) -> dict[str, Any]:
    """Bayesian hyperparameter search for LGBMGlobalForecaster.

    Minimises mean MASE across CV folds. Each trial is logged to MLflow.

    Search space:
      num_leaves          [31, 512]
      learning_rate       [0.01, 0.3]  log-uniform
      feature_fraction    [0.5, 1.0]
      bagging_fraction    [0.5, 1.0]
      min_child_samples   [10, 200]
      reg_alpha           [1e-8, 10]   log-uniform
      reg_lambda          [1e-8, 10]   log-uniform

    Returns:
        Best params dict, ready to pass to LGBMGlobalForecaster(params=...).
    """
    mlflow.set_experiment(mlflow_experiment)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 31, 512),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 1,
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        fold_scores: list[float] = []
        for fold_idx, (train_dates, val_dates) in enumerate(cv.split(train_df)):
            fold_train = cv.filter_fold(train_df, train_dates)
            fold_val = cv.filter_fold(train_df, val_dates)

            model = LGBMGlobalForecaster(params=params, num_boost_round=num_boost_round)
            model.fit(fold_train, feature_cols, target_col, val_df=fold_val)

            # Predict on validation (requires pre-computed features)
            val_with_nan = fold_val.copy()
            val_with_nan[target_col] = np.nan
            combined = pd.concat([fold_train, val_with_nan], ignore_index=True)

            preds = model.predict_recursive(combined, horizon=cv.horizon)
            series_ids = sorted(fold_val["unique_id"].unique())

            y_true = np.array(
                [
                    fold_val[fold_val["unique_id"] == uid]
                    .sort_values("ds")[target_col]
                    .values
                    for uid in series_ids
                ]
            )
            y_pred = np.array(
                [
                    preds[preds["unique_id"] == uid].sort_values("ds")["y_pred"].values
                    for uid in series_ids
                ]
            )
            y_train_mat = np.array(
                [
                    fold_train[fold_train["unique_id"] == uid]
                    .sort_values("ds")[target_col]
                    .values
                    for uid in series_ids
                ]
            )

            h = min(y_true.shape[1], y_pred.shape[1])
            score = mase(y_true[:, :h], y_pred[:, :h], y_train_mat)
            fold_scores.append(score)

            trial.report(score, step=fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_mase = float(np.mean(fold_scores))

        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("cv_mase", mean_mase)
            mlflow.log_metric("trial_number", trial.number)

        return mean_mase

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=10)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name=mlflow_experiment,
    )

    with mlflow.start_run(run_name=f"{mlflow_experiment}_study"):
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        best = study.best_params
        mlflow.log_params({f"best_{k}": v for k, v in best.items()})
        mlflow.log_metric("best_cv_mase", study.best_value)

    return best
