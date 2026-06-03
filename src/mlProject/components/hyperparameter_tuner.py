import os
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path

from mlProject.entity.config_entity import ModelSearchConfig
from mlProject.utils.common import save_json


class HyperparameterTuner:
    def __init__(self, config: ModelSearchConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def run(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        train_x = train_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]

        os.makedirs(self.config.root_dir, exist_ok=True)

        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
        mlflow.set_tracking_uri(mlflow_uri)

        results = []
        best_result = None

        for alpha in self.config.alpha_grid:
            for l1_ratio in self.config.l1_ratio_grid:
                model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
                model.fit(train_x, train_y)

                predicted = model.predict(test_x)
                rmse, mae, r2 = self.eval_metrics(test_y, predicted)

                model_name = f"{self.config.model_name_prefix}_alpha_{alpha}_l1_{l1_ratio}.joblib"
                model_path = Path(self.config.root_dir) / model_name
                joblib.dump(model, model_path)

                run_name = f"ElasticNet_alpha_{alpha}_l1_{l1_ratio}"
                with mlflow.start_run(run_name=run_name):
                    mlflow.log_param("alpha", alpha)
                    mlflow.log_param("l1_ratio", l1_ratio)
                    mlflow.log_metric("rmse", float(rmse))
                    mlflow.log_metric("mae", float(mae))
                    mlflow.log_metric("r2", float(r2))
                    try:
                        mlflow.log_artifact(str(model_path), artifact_path="models")
                    except Exception:
                        pass

                result = {
                    "alpha": float(alpha),
                    "l1_ratio": float(l1_ratio),
                    "rmse": float(rmse),
                    "mae": float(mae),
                    "r2": float(r2),
                    "model_path": str(model_path)
                }
                results.append(result)

                if best_result is None or rmse < best_result["rmse"]:
                    best_result = result
                    best_model_path = model_path

        if best_result is not None:
            best_path = Path(self.config.root_dir) / "best_model.joblib"
            joblib.dump(joblib.load(best_model_path), best_path)
            best_result["best_model_path"] = str(best_path)

        save_json(path=Path(self.config.result_file), data={"results": results, "best": best_result})
        return best_result
