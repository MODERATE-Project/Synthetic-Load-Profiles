import os
import pathlib
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf

from third_party.doppelganger.gan.doppelganger import DoppelGANger
from third_party.doppelganger.gan.network import (
    AttrDiscriminator,
    Discriminator,
    DoppelGANgerGenerator,
)
from third_party.doppelganger.gan.output import Normalization, Output, OutputType
from third_party.doppelganger.gan.util import add_gen_flag, normalize_per_sample
from model.utils import calc_features, compute_trends

tf.compat.v1.disable_v2_behavior()
# Alias common TF1 symbols expected by DoppelGANger code.
tf.placeholder = tf.compat.v1.placeholder
tf.Session = tf.compat.v1.Session
tf.ConfigProto = tf.compat.v1.ConfigProto
tf.global_variables_initializer = tf.compat.v1.global_variables_initializer
tf.variable_scope = tf.compat.v1.variable_scope
tf.AUTO_REUSE = tf.compat.v1.AUTO_REUSE
tf.layers = tf.compat.v1.layers
tf.nn.rnn_cell = tf.compat.v1.nn.rnn_cell

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
try:
    tf.config.experimental.set_visible_devices([], "GPU")
except (RuntimeError, ValueError):
    pass


DATA_ROOT = pathlib.Path("data") / "sample_data" / "opendata_fluvius"
STATIC_COLS = ["has_PV", "has_EV", "has_HP"]
SAMPLE_LEN = 24  # number of timesteps per RNN block (24 hours)


def _load_inputs():
    """Load and align raw labels and time series."""
    labels_df = pd.read_csv(DATA_ROOT / "fluvius_indicators.csv")
    timeseries_df = pd.read_csv(DATA_ROOT / "fluvius_wide_format.csv", index_col=0)

    category_map = {
        "standard": "None",
        "PV": "PV",
        "heat pump+PV": "HP+PV",
        "EV": "EV",
        "EV+PV": "EV+PV",
    }
    labels_df.rename(columns={"EAN_ID": "ID", "label": "Category"}, inplace=True)
    labels_df["Category"] = labels_df["Category"].map(category_map).fillna("None")

    labels_df["ID"] = labels_df["ID"].astype(str)
    timeseries_df.columns = timeseries_df.columns.astype(str)

    # binary attribute indicators
    labels_df["has_PV"] = labels_df["Category"].str.contains("PV")
    labels_df["has_EV"] = labels_df["Category"].str.contains("EV")
    labels_df["has_HP"] = labels_df["Category"].str.contains("HP")
    labels_df[STATIC_COLS] = labels_df[STATIC_COLS].astype(np.int32)

    available_mask = labels_df["ID"].isin(timeseries_df.columns)
    labels_df = labels_df[available_mask].reset_index(drop=True)
    aligned_series = timeseries_df[labels_df["ID"].values].copy()

    return labels_df, aligned_series


def _prepare_training_data():
    labels_df, aligned_series = _load_inputs()

    load_array = aligned_series.T.to_numpy(dtype=np.float32)  # (N, T)
    data_feature = load_array[:, :, None]  # (N, T, 1)

    data_gen_flag = np.ones(
        (data_feature.shape[0], data_feature.shape[1]), dtype=np.float32
    )

    data_feature_outputs = [
        Output(
            type_=OutputType.CONTINUOUS,
            dim=1,
            normalization=Normalization.ZERO_ONE,
            is_gen_flag=False,
        )
    ]

    attribute_blocks = []
    data_attribute_outputs = []
    for col in STATIC_COLS:
        values = labels_df[col].to_numpy(dtype=np.float32)
        one_hot = np.stack([1.0 - values, values], axis=1)
        attribute_blocks.append(one_hot)
        data_attribute_outputs.append(
            Output(type_=OutputType.DISCRETE, dim=2, normalization=None)
        )

    data_attribute = np.concatenate(attribute_blocks, axis=1).astype(np.float32)

    (
        data_feature,
        data_attribute,
        data_attribute_outputs,
        real_attribute_mask,
    ) = normalize_per_sample(
        data_feature,
        data_attribute,
        data_feature_outputs,
        data_attribute_outputs,
    )

    data_feature, data_feature_outputs = add_gen_flag(
        data_feature, data_gen_flag, data_feature_outputs, SAMPLE_LEN
    )

    return (
        data_feature,
        data_attribute,
        real_attribute_mask,
        data_gen_flag,
        data_feature_outputs,
        data_attribute_outputs,
        aligned_series,
    )


def _init_networks(
    data_feature_outputs,
    data_attribute_outputs,
    real_attribute_mask,
):
    generator = DoppelGANgerGenerator(
        feed_back=False,
        noise=True,
        feature_outputs=data_feature_outputs,
        attribute_outputs=data_attribute_outputs,
        real_attribute_mask=real_attribute_mask,
        sample_len=SAMPLE_LEN,
    )
    discriminator = Discriminator()
    attr_discriminator = AttrDiscriminator()
    return generator, discriminator, attr_discriminator


def train():
    (
        data_feature,
        data_attribute,
        real_attribute_mask,
        data_gen_flag,
        data_feature_outputs,
        data_attribute_outputs,
        aligned_series,
    ) = _prepare_training_data()

    run_dir = pathlib.Path("runs") / "doppelganger"
    checkpoint_dir = run_dir / "checkpoints"
    sample_dir = run_dir / "samples"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    time_path = run_dir / "time.txt"

    tf.compat.v1.reset_default_graph()

    arr_real = aligned_series.to_numpy(dtype=np.float32)
    arr_dt = pd.to_datetime(aligned_series.index)
    arr_features_real = calc_features(arr_real, axis=0)
    arr_time_features_real = calc_features(arr_real, axis=1)
    trend_real_dict = compute_trends(arr_real, arr_dt)
    timestamp_strings = arr_dt.astype(str).to_numpy()

    generator, discriminator, attr_discriminator = _init_networks(
        data_feature_outputs,
        data_attribute_outputs,
        real_attribute_mask,
    )

    run_config = tf.compat.v1.ConfigProto()
    run_config.allow_soft_placement = True
    with tf.compat.v1.Session(config=run_config) as sess:
        gan = DoppelGANger(
            sess=sess,
            checkpoint_dir=str(checkpoint_dir),
            sample_dir=str(sample_dir),
            time_path=str(time_path),
            epoch=200,
            batch_size=64,
            data_feature=data_feature,
            data_attribute=data_attribute,
            real_attribute_mask=real_attribute_mask,
            data_gen_flag=data_gen_flag,
            sample_len=SAMPLE_LEN,
            data_feature_outputs=data_feature_outputs,
            data_attribute_outputs=data_attribute_outputs,
            vis_freq=100,
            vis_num_sample=5,
            generator=generator,
            discriminator=discriminator,
            attr_discriminator=attr_discriminator,
            d_gp_coe=10.0,
            attr_d_gp_coe=10.0,
            g_attr_d_coe=1.0,
            d_rounds=1,
            g_rounds=1,
            num_packing=1,
            extra_checkpoint_freq=10,
        )
        gan.build()
        gan.arr_real = arr_real
        gan.arr_features_real = arr_features_real
        gan.arr_time_features_real = arr_time_features_real
        gan.arr_dt = arr_dt
        gan.trend_real_dict = trend_real_dict
        gan.real_timestamps = timestamp_strings
        gan.train()


def save_model(destination_dir=None):
    """Copy the latest checkpoint produced by training into destination_dir."""
    run_dir = pathlib.Path("runs") / "doppelganger"
    checkpoint_dir = run_dir / "checkpoints"
    if not checkpoint_dir.exists():
        raise FileNotFoundError("No training checkpoints found. Run train() first.")

    latest_ckpt = tf.compat.v1.train.latest_checkpoint(str(checkpoint_dir))
    if latest_ckpt is None:
        raise FileNotFoundError("No checkpoint files present in checkpoint directory.")

    export_dir = pathlib.Path(destination_dir or (run_dir / "exported_model"))
    export_dir.mkdir(parents=True, exist_ok=True)

    ckpt_prefix = pathlib.Path(latest_ckpt)
    for ckpt_file in ckpt_prefix.parent.glob(f"{ckpt_prefix.name}*"):
        shutil.copy2(ckpt_file, export_dir / ckpt_file.name)

    checkpoint_file = ckpt_prefix.parent / "checkpoint"
    if checkpoint_file.exists():
        shutil.copy2(checkpoint_file, export_dir / checkpoint_file.name)

    return export_dir


def load_model(checkpoint_dir=None, batch_size=64):
    """Restore a trained DoppelGANger model and return (gan, session)."""
    run_dir = pathlib.Path("runs") / "doppelganger"
    checkpoint_dir = pathlib.Path(checkpoint_dir or (run_dir / "checkpoints"))
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

    latest_ckpt = tf.compat.v1.train.latest_checkpoint(str(checkpoint_dir))
    if latest_ckpt is None:
        raise FileNotFoundError("No checkpoint files found to load.")

    sample_dir = run_dir / "samples"
    sample_dir.mkdir(parents=True, exist_ok=True)
    time_path = run_dir / "time.txt"

    (
        data_feature,
        data_attribute,
        real_attribute_mask,
        data_gen_flag,
        data_feature_outputs,
        data_attribute_outputs,
        aligned_series,
    ) = _prepare_training_data()

    tf.compat.v1.reset_default_graph()
    generator, discriminator, attr_discriminator = _init_networks(
        data_feature_outputs,
        data_attribute_outputs,
        real_attribute_mask,
    )

    run_config = tf.compat.v1.ConfigProto()
    run_config.allow_soft_placement = True
    sess = tf.compat.v1.Session(config=run_config)
    gan = DoppelGANger(
        sess=sess,
        checkpoint_dir=str(checkpoint_dir),
        sample_dir=str(sample_dir),
        time_path=str(time_path),
        epoch=1,
        batch_size=batch_size,
        data_feature=data_feature,
        data_attribute=data_attribute,
        real_attribute_mask=real_attribute_mask,
        data_gen_flag=data_gen_flag,
        sample_len=SAMPLE_LEN,
        data_feature_outputs=data_feature_outputs,
        data_attribute_outputs=data_attribute_outputs,
        vis_freq=100,
        vis_num_sample=5,
        generator=generator,
        discriminator=discriminator,
        attr_discriminator=attr_discriminator,
        d_gp_coe=10.0,
        attr_d_gp_coe=10.0,
        g_attr_d_coe=1.0,
        d_rounds=1,
        g_rounds=1,
        num_packing=1,
        extra_checkpoint_freq=10,
    )
    gan.build()
    gan.load()

    arr_real = aligned_series.to_numpy(dtype=np.float32)
    arr_dt = pd.to_datetime(aligned_series.index)
    gan.arr_real = arr_real
    gan.arr_features_real = calc_features(arr_real, axis=0)
    gan.arr_time_features_real = calc_features(arr_real, axis=1)
    gan.arr_dt = arr_dt
    gan.trend_real_dict = compute_trends(arr_real, arr_dt)
    gan.real_timestamps = arr_dt.astype(str).to_numpy()

    return gan, sess


def _generate_samples(gan, num_profiles):
    noise = gan.gen_attribute_input_noise(num_profiles)
    add_noise = gan.gen_attribute_input_noise(num_profiles)
    feat_noise = gan.gen_feature_input_noise(num_profiles, gan.sample_time)
    feat_data = gan.gen_feature_input_data_free(num_profiles)
    features, attributes, gen_flags, lengths = gan.sample_from(
        noise,
        add_noise,
        feat_noise,
        feat_data,
        return_gen_flag_feature=True,
    )
    # Collapse to raw load values (first feature dim) per sample
    load_dim = gan.data_feature_outputs[0].dim
    loads = features[:, :, :load_dim]
    # Zero-out padded tail using generation flags
    mask = np.expand_dims(gen_flags, axis=2)
    loads = loads * mask
    return loads, attributes, lengths


def train_save_generate(num_profiles, export_dir=None):
    """Train DoppelGANger, save the checkpoint, and export synthetic samples."""
    train()
    saved_dir = save_model(export_dir)
    gan, sess = load_model(saved_dir)
    try:
        loads, attributes, lengths = _generate_samples(gan, num_profiles)
    finally:
        sess.close()

    df = pd.DataFrame(loads[:, :, 0])
    df.index.name = "profile_id"
    df.columns = [f"t_{i}" for i in range(df.shape[1])]
    out_dir = pathlib.Path("runs") / "doppelganger"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{num_profiles}_DoppelGANger.parquet"
    df.to_parquet(out_path, index=True)
    return out_path, saved_dir


if __name__ == "__main__":
    train_save_generate(num_profiles=1300, export_dir="runs/doppelganger/exported_model")
