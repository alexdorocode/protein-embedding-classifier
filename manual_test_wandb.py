# manual_wandb_check.py
from src.core.tracking import WandBTracker

tracker = WandBTracker(
    project="pec_develop",
    run_name="manual-wandb-check",
    config={"foo": 1},
    enabled=True,
)

tracker.log({"accuracy": 0.9}, step=1)
tracker.finish()