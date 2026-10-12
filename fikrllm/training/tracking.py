"""
tracking.py -- recording what happened during a run
"""

from collections.abc import Sequence
from dataclasses import asdict

import trackio

from fikrllm.config import ModelConfig, TrainingConfig


class Tracker:
    """Logs metrics to Trackio, or to nothing at all.
    """

    # Initialize this object with the state it needs later.
    def __init__(
        self,
        training_config: TrainingConfig,
        model_config: ModelConfig | None = None,
    ) -> None:
        self.run_name = training_config.run_name
        self.project = training_config.project
        self.run_id = None

        run_config = {
            "training": asdict(training_config)
        }

        if model_config is not None:
            run_config["model"] = asdict(model_config)

        run = trackio.init(
            project=self.project,
            name=self.run_name,
            config=run_config
        )

        self.run_id = getattr(run, "id", None)

        print(
            f"trackio: run '{self.run_name}' (id {self.run_id}) "
            f"in project '{self.project}'"
        )

        print(f"  open the dashboard:  trackio show --project {self.project}")

    # Purpose: log.
    def log(self, metrics: dict[str, float], step: int) -> None:
        trackio.log(metrics, step=step)

    # Purpose: log table.
    def log_table(
        self,
        name: str,
        columns: Sequence[str],
        rows: Sequence[Sequence[object]],
        step: int,
    ) -> None:

        table_data = [ list(row) for row in rows ]
        table = trackio.Table(
            columns=list(columns),
            data=table_data
        )

        trackio.log({
            name: table
        }, step=step)

    # Purpose: finish.
    def finish(self) -> None:
        trackio.finish()

    # Enter the context manager and return the active object.
    def __enter__(self) -> "Tracker":
        return self

    # Leave the context manager and release any open resources.
    def __exit__(self, *exception: object) -> bool:
        self.finish()
        return False
    
