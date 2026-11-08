import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WANDB_DOWNLOAD_DIR = REPO_ROOT / "outputs" / "wandb_checkpoints"


def resolve_local_path(path: str | None) -> str | None:
    """Resolve a user-given path against the directory the command was launched from.

    Hydra changes the working directory to the run's output folder, so relative paths are
    resolved with ``hydra.utils.to_absolute_path`` (falls back to the repo root).
    """
    if path is None or str(path).startswith("run:") or os.path.isabs(path):
        return path
    try:
        from hydra.utils import to_absolute_path
        candidate = to_absolute_path(path)
    except Exception:
        candidate = os.path.abspath(path)
    if not os.path.exists(candidate) and (REPO_ROOT / path).exists():
        candidate = str(REPO_ROOT / path)
    return candidate


def parse_checkpoint_path(path: str | None = None, project: str | None = None) -> str | None:
    """Return a local checkpoint path.

    Accepted forms:
        - a local file path (absolute, or relative to where the command was launched)
        - ``run:<run_id>[:<iteration>]``                   (uses your default W&B entity and ``project``)
        - ``run:<entity>/<project>/<run_id>[:<iteration>]``
    W&B checkpoints are downloaded to ``outputs/wandb_checkpoints/<run_id>/``.
    Without ``<iteration>`` the latest checkpoint of the run is used.
    """
    if path is None:
        return None
    if not str(path).startswith("run:"):
        local = resolve_local_path(path)
        if not os.path.isfile(local):
            raise FileNotFoundError(f"Checkpoint not found: {local}")
        return local

    import wandb

    run_path, _, target = path[len("run:"):].partition(":")
    api = wandb.Api()
    if "/" not in run_path:
        entity = api.default_entity
        run_path = f"{entity}/{project}/{run_path}" if project else f"{entity}/{run_path}"
    run = api.run(run_path)
    root = WANDB_DOWNLOAD_DIR / run.id
    root.mkdir(parents=True, exist_ok=True)

    def iteration(file) -> int:
        number = Path(file.name).stem.split("_")[-1]
        return 10**9 if number == "final" else int(number)

    checkpoints = sorted(
        (f for f in run.files(per_page=10000) if Path(f.name).name.startswith("checkpoint_")),
        key=iteration,
    )
    if not checkpoints:
        raise ValueError(f"No checkpoints found in W&B run {run_path}.")
    if target:
        matches = [f for f in checkpoints if Path(f.name).stem.split("_")[-1] == target]
        if not matches:
            available = [Path(f.name).stem.split("_")[-1] for f in checkpoints]
            raise ValueError(f"Checkpoint {target} not found in {run_path}. Available: {available}")
        checkpoint = matches[0]
    else:
        checkpoint = checkpoints[-1]

    local = root / checkpoint.name
    if not local.exists():
        print(f"Downloading {run_path}/{checkpoint.name} to {local}")
        checkpoint.download(str(root), replace=True)
    return str(local)
