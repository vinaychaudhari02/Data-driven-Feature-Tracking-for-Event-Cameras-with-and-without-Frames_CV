""" Predict tracks for a sequence with a network """
import logging
import os
import sys
from pathlib import Path

import hydra
import imageio
import IPython
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf, open_dict
from prettytable import PrettyTable
from tqdm import tqdm

# --- ADD PROJECT ROOT TO PATH AND IMPORT YOUR MODEL ---
sys.path.append('/data/projects/vchaudhari/CV_project/deep_ev_tracker')
from models.MOE_model import TrackerNetHeat

from utils.dataset import CornerConfig, ECSubseq, EDSSubseq, EvalDatasetType
from utils.timers import CudaTimer, cuda_timers
from utils.track_utils import (
    TrackObserver,
    get_gt_corners,
)
from utils.visualization import generate_track_colors, render_pred_tracks, render_tracks

# Configure GPU order
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Logging
logger = logging.getLogger(__name__)
results_table = PrettyTable()
results_table.field_names = ["Inference Time"]

# Configure datasets
corner_config = CornerConfig(30, 0.3, 15, 0.15, False, 11)

EvalDatasetConfigDict = {
    EvalDatasetType.EC: {"dt": 0.010, "root_dir": "<path>"},
    EvalDatasetType.EDS: {"dt": 0.01, "root_dir": "/data/projects/vchaudhari/CV_project/eds_subseq/"},
}

EVAL_DATASETS = [
    ("peanuts_running_2360_2460", EvalDatasetType.EDS),
    ("peanuts_light_160_386",      EvalDatasetType.EDS),
]


def evaluate(model, sequence_dataset, dt_track_vis, sequence_name, visualize):
    # === Save results to this directory ===
    output_dir = "/data/projects/vchaudhari/CV_project/Results/model2/fog/"
    os.makedirs(output_dir, exist_ok=True)

    tracks_pred = TrackObserver(
        t_init=sequence_dataset.t_init, u_centers_init=sequence_dataset.u_centers
    )

    model.reset(sequence_dataset.n_tracks)
    event_generator = sequence_dataset.events()
    cuda_timer = CudaTimer(model.device, sequence_dataset.sequence_name)

    with torch.no_grad():
        # Predict network tracks
        for t, x in tqdm(
            event_generator,
            total=sequence_dataset.n_events - 1,
            desc="Predicting tracks with network...",
        ):
            with cuda_timer:
                x = x.to(model.device)
                y_hat = model.forward(x)
                sequence_dataset.accumulate_y_hat(y_hat)
            tracks_pred.add_observation(t, sequence_dataset.u_centers.cpu().numpy())

        if visualize:
            # Visualize network tracks
            gif_img_arr = []
            tracks_pred_interp = tracks_pred.get_interpolators()
            track_colors = generate_track_colors(sequence_dataset.n_tracks)
            for i, (t, img_now) in enumerate(
                tqdm(
                    sequence_dataset.frames(),
                    total=sequence_dataset.n_frames - 1,
                    desc="Rendering predicted tracks... ",
                )
            ):
                fig_arr = render_pred_tracks(
                    tracks_pred_interp, t, img_now, track_colors, dt_track=dt_track_vis
                )
                gif_img_arr.append(fig_arr)
            imageio.mimsave(
                os.path.join(output_dir, f"{sequence_name}_tracks_pred.gif"), gif_img_arr
            )

    # Save predicted tracks
    np.savetxt(
        os.path.join(output_dir, f"{sequence_name}.txt"),
        tracks_pred.track_data,
        fmt=["%i", "%.9f", "%i", "%i"],
        delimiter=" ",
    )

    return {"latency": sum(cuda_timers[sequence_dataset.sequence_name])}


@hydra.main(config_path="configs", config_name="eval_real_defaults")
def track(cfg):
    pl.seed_everything(1234)
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.model.representation = cfg.representation
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    # ── BUILD THE MODEL AND LAZY-INIT VIA DUMMY FORWARD ──
    ckpt = torch.load(cfg.weights_path, map_location="cpu")
    hyper = ckpt.get("hyper_parameters", {})
    model = TrackerNetHeat(**hyper)

    # strip any DataParallel prefixes from checkpoint
    sd = ckpt.get("state_dict", ckpt)
    state_dict = {k.replace("module.", ""): v for k, v in sd.items()}

    # run a dummy forward on CPU to lazy-init all submodules with correct shapes
    with torch.no_grad():
        batch_size = 4
        # derive reference channel count from checkpoint
        ref_ch = state_dict["reference_encoder.conv_bottom_0.model.0.weight"].shape[1]
        C_total = hyper["target_channels"] + ref_ch
        # derive spatial resolution so freq_embed sizes match checkpoint
        patch = hyper["patch_size"]
        r0 = state_dict["heat_backbone_t.freq_embed.0"].shape[0]
        H = W = patch * r0
        dummy = torch.zeros(batch_size, C_total, H, W)
        _ = model(dummy)

    # now load the real weights
    model.load_state_dict(state_dict, strict=True)

    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()

    # ── RUN EVALUATION ──
    for seq_name, seq_type in EVAL_DATASETS:
        if seq_type == EvalDatasetType.EC:
            dataset_class = ECSubseq
        elif seq_type == EvalDatasetType.EDS:
            dataset_class = EDSSubseq
        else:
            raise ValueError

        dataset = dataset_class(
            EvalDatasetConfigDict[seq_type]["root_dir"],
            seq_name,
            -1,
            cfg.patch_size,
            cfg.representation,
            EvalDatasetConfigDict[seq_type]["dt"],
            corner_config,
        )

        # override with your GT keypoints
        gt_features_path = str(Path(cfg.gt_path) / f"{seq_name}.gt.txt")
        gt_start_corners = get_gt_corners(gt_features_path)
        dataset.override_keypoints(gt_start_corners)

        metrics = evaluate(model, dataset, cfg.dt_track_vis, seq_name, cfg.visualize)
        logger.info(f"=== DATASET: {seq_name} ===")
        logger.info(f"Latency: {metrics['latency']} s")
        results_table.add_row([metrics["latency"]])

    logger.info(f"\n{results_table.get_string()}")


if __name__ == "__main__":
    track()