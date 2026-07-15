"""Model utilities for local ADMET workflows.

Training helpers are imported lazily so lightweight artifact utilities can be
used inside Processing containers without installing training dependencies.
"""

__all__ = ["train_local_baseline"]


def __getattr__(name: str):
    if name == "train_local_baseline":
        from admet_platform.models.baseline import train_local_baseline

        return train_local_baseline
    raise AttributeError(name)
