import logging
import os
import sys

import hydra
import h5py
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import tqdm
from matplotlib import cm
from omegaconf import OmegaConf, open_dict
from hydra.utils import instantiate

# adjust to your repo root
sys.path.append(
    r'D:\Academics\PhD\SLU\Computer Vision\Data-driven-Feature-Tracking-for-Event-Cameras-with-and-without-Frames_CV\deep_ev_tracker'
)

from utils.utils import *
from disp_dataloader.m3ed_loader import M3EDTestDataModule

logger = logging.getLogger(__name__)
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
torch.set_num_threads(1)
torch.backends.cudnn.benchmark = True


def propagate_keys_disp(cfg):
    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.data.patch_size        = cfg.patch_size
        cfg.data.min_track_length  = cfg.min_track_length
        cfg.data.tracks_per_sample = cfg.tracks_per_sample
        cfg.data.disp_patch_range  = cfg.disp_patch_range

        cfg.model.patch_size        = cfg.patch_size
        cfg.model.min_track_length  = cfg.min_track_length
        cfg.model.tracks_per_sample = cfg.tracks_per_sample
        cfg.model.disp_patch_range  = cfg.disp_patch_range


def create_attn_mask(seq_frame_idx, device):
    m = torch.from_numpy(seq_frame_idx[:, None] == seq_frame_idx[None, :]).to(device)
    return torch.logical_not(m).bool()


def test_run(model, dataloader, cfg, visualize=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.eval().to(device)

    # accumulators
    list_disp_pred   = []
    list_disp_gt     = []
    list_seq_names   = []
    list_img_points  = []

    # if visualizing, open HDF5 once and prepare GIF list
    if visualize:
        h5_file   = h5py.File(cfg.data.h5_path, 'r')
        ovcleft   = h5_file['ovcleft']                # shape: (N_frames, H, W)
        gif_frames = []
        cmap       = cm.get_cmap('jet')

    print(f"Number of batches in dataloader: {len(dataloader)}")
    attn_mask = None

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Batches")):
        (
            seq_frame_patches,     # np [B, T, H, W, C]
            seq_event_patches,     # np [B, T, H, W, C]
            seq_y_gt_disp,         # np [B, T]
            seq_frame_idx,         # np [B, T] (or [T] if B=1)
            seq_names,             # list[str] of length B
            img_points             # list/array: per-sample shape [T,2] or [N, T,2]
        ) = batch

        B, T, H, W, C = seq_frame_patches.shape

        # to torch: [B, T, C, H, W]
        frames = (torch.from_numpy(seq_frame_patches)
                  .permute(0,1,4,2,3)
                  .to(device))
        events = (torch.from_numpy(seq_event_patches)
                  .permute(0,1,4,2,3)
                  .to(device))

        # update mask if batch size changed
        if attn_mask is None or attn_mask.shape[0] != B:
            attn_mask = create_attn_mask(seq_frame_idx, device)

        # predict disparities
        y_pred_disp = np.zeros((B, T), dtype=np.float32)
        model.reset(None)
        for t in range(T):
            f   = frames[:, t]
            e   = events[:, t]
            out = model.forward(f, e, attn_mask)         # [B,2]
            y_pred_disp[:, t] = out[:, 1].detach().cpu().numpy()

        # store numeric results
        list_disp_pred.append(y_pred_disp)
        list_disp_gt.append(seq_y_gt_disp)
        list_seq_names.append(seq_names)
        list_img_points.append(img_points)

        print(f"Batch {batch_idx} done. preds shape: {y_pred_disp.shape}")

        # visualization: overlay tracks on full frames from HDF5
        if visualize:
            max_disp = float(y_pred_disp.max() or 1.0)

            for i in range(B):
                # handle 2D or 1D seq_frame_idx
                if seq_frame_idx.ndim == 2:
                    frame_idxs = seq_frame_idx[i]
                else:
                    frame_idxs = seq_frame_idx
                start_idx = int(frame_idxs[0])

                # read full left frame from HDF5
                bg = ovcleft[start_idx]                   # [H, W]

                fig, ax = plt.subplots(figsize=(8,6))
                ax.imshow(bg, cmap='gray')
                ax.axis('off')

                # prepare tracks
                pts = np.array(img_points[i])
                if pts.ndim == 2 and pts.shape[1] == 2:
                    tracks = [pts]                         # single track
                elif pts.ndim == 3 and pts.shape[2] == 2:
                    tracks = list(pts)                     # multiple tracks
                else:
                    raise ValueError(f"Unexpected img_points[{i}].shape = {pts.shape}")

                # plot each segment coloured by disparity
                for track in tracks:
                    for t in range(T-1):
                        x0, y0 = track[t]
                        x1, y1 = track[t+1]
                        c = cmap(y_pred_disp[i, t] / max_disp)
                        ax.plot([x0, x1], [y0, y1], color=c, linewidth=2)

                fig.canvas.draw()
                arr = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
                arr = arr.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                plt.close(fig)
                gif_frames.append(arr)

    if visualize:
        h5_file.close()

    # sanity check
    if not list_disp_pred:
        raise RuntimeError("No predictions generated; check your dataloader.")

    # save numeric outputs
    np.savez_compressed(
        'results.npz',
        disparity_pred=np.concatenate(list_disp_pred, axis=0),
        seq_names=np.concatenate(list_seq_names, axis=0).flatten(),
    )
    np.savez_compressed(
        'ground_truth.npz',
        disparity_gt=np.concatenate(list_disp_gt, axis=0),
        image_points=np.concatenate(list_img_points, axis=0),
        seq_names=np.concatenate(list_seq_names, axis=0).flatten(),
    )

    # write GIF
    if visualize and gif_frames:
        imageio.mimsave('disp_evaluation.gif', gif_frames, fps=5)
        logger.info(f"Saved GIF with {len(gif_frames)} frames to disp_evaluation.gif")


@hydra.main(config_path="disp_configs", config_name="m3ed_test")
def test(cfg):
    pl.seed_everything(1234)
    propagate_keys_disp(cfg)
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    # write out exact config
    with open('test_config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    # instantiate and load model
    model = instantiate(cfg.model, _recursive_=False)
    ckpt  = torch.load(cfg.checkpoint_path)
    model.load_state_dict(ckpt['state_dict'], strict=True)

    # prepare data
    dm = M3EDTestDataModule(**cfg.data)
    dm.setup()
    dl = dm.test_dataloader()

    with torch.no_grad():
        test_run(
            model,
            dl,
            cfg,
            visualize=getattr(cfg, 'visualize', False),
        )


if __name__ == '__main__':
    test()
