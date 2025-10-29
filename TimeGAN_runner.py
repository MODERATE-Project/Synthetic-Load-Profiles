import os
import pathlib

import numpy as np
import sys
import tensorflow as tf

from DoppelGANger_runner import _load_inputs

TIMEGAN_ROOT = pathlib.Path("third_party") / "timegan"
resolved_timegan_path = str(TIMEGAN_ROOT.resolve())
if resolved_timegan_path not in sys.path:
    sys.path.insert(0, resolved_timegan_path)

from third_party.timegan.timegan import timegan  # noqa: E402




def _prepare_timegan_training_data():
    """Return data shaped for TimeGAN (N, T, D) using the shared Fluvius corpus."""
    _, aligned_series = _load_inputs()
    data = aligned_series.T.to_numpy(dtype=np.float32)
    return np.expand_dims(data, axis=-1)


def train(
    module: str = "gru",
    hidden_dim: int = 24,
    num_layers: int = 3,
    iterations: int = 5000,
    batch_size: int = 128,
):
    """Train TimeGAN on the same Fluvius dataset used by DoppelGANger."""
    ori_data = _prepare_timegan_training_data()
    parameters = {
        "module": module,
        "hidden_dim": hidden_dim,
        "num_layer": num_layers,
        "iterations": iterations,
        "batch_size": batch_size,
    }

    generated_data = timegan(ori_data, parameters)

    run_dir = pathlib.Path("runs") / "timegan"
    run_dir.mkdir(parents=True, exist_ok=True)
    np.save(run_dir / "generated.npy", generated_data)

    return generated_data


if __name__ == "__main__":
    train()
