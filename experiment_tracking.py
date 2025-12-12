from typing import Any
import mlflow


class ExperimentTracker:
    def __init__(self, experiment_name):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        mlflow.autolog(log_input_examples=True)
        self.experiment_name = experiment_name
        mlflow.set_experiment(self.experiment_name)

    def log_params(self, params: dict):
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: dict, step: int | None = None):
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def start_run(
        self,
        run_id: str | None = None,
        experiment_id: str | None = None,
        run_name: str | None = None,
        nested: bool = False,
        parent_run_id: str | None = None,
        tags: dict[str, Any] | None = None,
        description: str | None = None,
        log_system_metrics: bool | None = None,
    ):
        return mlflow.start_run(
            run_id=run_id,
            experiment_id=experiment_id,
            run_name=run_name,
            nested=nested,
            parent_run_id=parent_run_id,
            tags=tags,
            description=description,
            log_system_metrics=log_system_metrics,
        )

    def end_run(self):
        mlflow.end_run()
