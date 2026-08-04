# src/core/tracking.py

from __future__ import annotations
from typing import Dict, Any

import wandb

class Tracker:
    def log(self, metrics: Dict[str, float], **context: Any):
        raise NotImplementedError

    def finish(self):
        pass

class WandBTracker(Tracker):
    def __init__(
        self,
        project: str,
        run_name: str | None = None,
        config: dict | None = None,
        enabled: bool = True,
    ):
        self.enabled = enabled

        if not enabled:
            self.run = None
            return

        self.run = wandb.init(
            project=project,
            name=run_name,
            config=config or {},
        )

    def log(self, metrics: dict, **context):
        if not self.enabled:
            return

        payload = {**metrics, **context}
        wandb.log(payload)

    def finish(self):
        if self.enabled and self.run is not None:
            self.run.finish()

class PrintTracker(Tracker):
    def log(self, metrics: dict, **context):
        print({**context, **metrics})

