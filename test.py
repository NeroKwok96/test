from pathlib import Path
from matplotlib import pyplot as plt
from utils.dataloader import load_sensor_vel_freq
from models.low_vel_gan import LowVelGAN
from datetime import datetime
from utils.logger import logger

import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import os


def get_threshold(model: LowVelGAN, filter: float):
    normal_dataset, _, _, _ = load_sensor_vel_freq(
        "data/vel", selected_row_num=400, unsatisfactory=filter
    )

    normal_temp_dataset = tf.expand_dims(normal_dataset, axis=-1)
    _, normal_recon = model.generator(normal_temp_dataset)
    reconstruction_scores = tf.keras.losses.mae(
        tf.reshape(normal_recon, [len(normal_recon), -1]),
        tf.reshape(normal_temp_dataset, [len(normal_temp_dataset), -1]),
    )

    threshold = {}
    mean = np.mean(reconstruction_scores)
    std = np.std(reconstruction_scores)
    threshold["good"] = mean + 2 * std
    threshold["usable"] = mean + 5 * std
    threshold["unsatisfactory"] = mean + 9 * std

    logger.info(f"{np.mean(reconstruction_scores)=}")
    logger.info(f"{np.std(reconstruction_scores)=}")
    logger.info(f"{threshold=}")

    return threshold


def plot_data(model, dataset, filepath):
    sample_size = 4
    fig, axs = plt.subplots(sample_size, 2, figsize=(50, 30))
    rand_sample_idx = np.random.choice(len(dataset), size=sample_size)
    dataset = dataset[rand_sample_idx]
    for i, test_input in enumerate(dataset):
        test_input = test_input.reshape(1, *test_input.shape, 1)
        _, pred = model.generator(test_input, training=False)
        pred = pred.numpy()

        for j in range(2):
            axs[i, j].plot(pred.reshape(-1, 4)[:, j * 2], label="pred")
            axs[i, j].plot(test_input.reshape(-1, 4)[:, j * 2], label="ground")
            axs[i, j].legend(loc="upper right")

    fig.savefig(filepath)


def evaluate(args: argparse.Namespace):
    normal_dataset, abnormal_dataset, meta_normal, meta_abnormal = load_sensor_vel_freq(
        "data/vel",
        selected_row_num=400,
        unsatisfactory=args.fail,
        select_year=args.select_year,
    )

    logger.info("Imported dataset")
    logger.info(f"{normal_dataset.shape=}")
    logger.info(f"{meta_normal.shape=}")
    logger.info(f"{abnormal_dataset.shape=}")
    logger.info(f"{meta_abnormal.shape=}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_directory = os.path.join("out", "evaluate", timestamp)
    Path(out_directory).mkdir(parents=True, exist_ok=True)

    meta_normal.to_csv(os.path.join(out_directory, "meta_normal.csv"))
    meta_abnormal.to_csv(os.path.join(out_directory, "meta_abnormal.csv"))

    model = LowVelGAN()
    model.load_weights(
        input_shape=(None, 400, 4, 1), save_dir=f"out/models/{args.model}"
    )

    plot_data(model, normal_dataset, os.path.join(out_directory, "normal.png"))
    plot_data(model, abnormal_dataset, os.path.join(out_directory, "abnormal.png"))

    # Define the threshold
    thresholds = get_threshold(model, args.threshold)

    # Create CSV file for reporting
    df = pd.DataFrame()

    # Evaluate with threshold
    normal_temp_dataset = tf.expand_dims(normal_dataset, axis=-1)
    _, normal_recon = model.generator(normal_temp_dataset)
    reconstruction_scores = tf.keras.losses.mae(
        tf.reshape(normal_recon, [len(normal_recon), -1]),
        tf.reshape(normal_temp_dataset, [len(normal_temp_dataset), -1]),
    )
    good = tf.cast(tf.math.less(reconstruction_scores, thresholds["good"]), tf.uint8)
    usable = tf.cast(
        tf.math.greater_equal(reconstruction_scores, thresholds["good"]),
        tf.uint8,
    )
    unsatisfactory = tf.cast(
        tf.math.greater_equal(reconstruction_scores, thresholds["usable"]),
        tf.uint8,
    )
    un_usable = tf.cast(
        tf.math.greater_equal(reconstruction_scores, thresholds["unsatisfactory"]),
        tf.uint8,
    )
    meta_normal["Abnormal (Formula)"] = 0
    meta_normal["Good (Model)"] = good
    meta_normal["Usable (Model)"] = usable
    meta_normal["Unsatisfactory (Model)"] = unsatisfactory
    meta_normal["Un-usable (Model)"] = un_usable
    meta_normal["Match"] = 0 == usable
    df = pd.concat([df, meta_normal])

    if len(meta_abnormal) > 0:
        abnormal_temp_dataset = abnormal_dataset
        abnormal_temp_dataset = tf.expand_dims(abnormal_temp_dataset, axis=-1)
        _, abnormal_temp_recon = model.generator(abnormal_temp_dataset)
        abnormal_temp_loss = tf.keras.losses.mae(
            tf.reshape(abnormal_temp_recon, [len(abnormal_temp_recon), -1]),
            tf.reshape(abnormal_temp_dataset, [len(abnormal_temp_dataset), -1]),
        )
        good = tf.cast(tf.math.less(abnormal_temp_loss, thresholds["good"]), tf.uint8)
        usable = tf.cast(
            tf.math.greater_equal(abnormal_temp_loss, thresholds["good"]),
            tf.uint8,
        )
        unsatisfactory = tf.cast(
            tf.math.greater_equal(abnormal_temp_loss, thresholds["usable"]),
            tf.uint8,
        )
        un_usable = tf.cast(
            tf.math.greater_equal(abnormal_temp_loss, thresholds["unsatisfactory"]),
            tf.uint8,
        )
        meta_abnormal["Abnormal (Formula)"] = 1
        meta_abnormal["Good (Model)"] = good
        meta_abnormal["Usable (Model)"] = usable
        meta_abnormal["Unsatisfactory (Model)"] = unsatisfactory
        meta_abnormal["Un-usable (Model)"] = un_usable
        meta_abnormal["Match"] = np.any([usable, unsatisfactory, un_usable], axis=0)
        df = pd.concat([df, meta_abnormal])
    else:
        abnormal_temp_dataset = abnormal_dataset
        abnormal_temp_dataset = tf.expand_dims(abnormal_temp_dataset, axis=-1)
        _, abnormal_temp_recon = model.generator(abnormal_temp_dataset)
        abnormal_temp_loss = tf.keras.losses.mae(
            tf.reshape(abnormal_temp_recon, [len(abnormal_temp_recon), -1]),
            tf.reshape(abnormal_temp_dataset, [len(abnormal_temp_dataset), -1]),
        )
        un_usable = tf.cast(
            tf.math.greater_equal(abnormal_temp_loss, thresholds["unsatisfactory"]),
            tf.uint8,
        )
        logger.info(
            f"Testing unusable data - unusable number: {np.count_nonzero(un_usable)}, total: {len(abnormal_dataset)}"
        )

    df.to_csv(os.path.join(out_directory, "report.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation for LowVelGAN")
    parser.add_argument("--model", "-m", default="model", help="Model name")
    parser.add_argument("--fail", "-f", type=float, default=16.2, help="Fail Threshold")
    parser.add_argument(
        "--threshold", "-t", type=float, default=8.1, help="Loss Threshold"
    )
    parser.add_argument(
        "--select_year", "-sy", type=int, default=2023, help="Select year for dataset"
    )

    args = parser.parse_args()
    evaluate(args)