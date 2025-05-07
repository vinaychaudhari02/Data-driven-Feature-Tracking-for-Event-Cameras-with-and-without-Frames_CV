import logging
import os
import hydra
import numpy as np
import pytorch_lightning as pl
import torch
import sys
import tqdm
import matplotlib.pyplot as plt
import imageio

from omegaconf import OmegaConf, open_dict

sys.path.append('D:\Academics\PhD\SLU\Computer Vision\Data-driven-Feature-Tracking-for-Event-Cameras-with-and-without-Frames_CV\deep_ev_tracker')
from utils.utils import *
from disp_dataloader.m3ed_loader import M3EDTestDataModule

logger = logging.getLogger(__name__)
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
torch.set_num_threads(1)
torch.backends.cudnn.benchmark = True


def propagate_keys_disp(cfg):
    OmegaConf.set_struct(cfg, True)

    with open_dict(cfg):
        cfg.data.patch_size = cfg.patch_size
        cfg.data.min_track_length = cfg.min_track_length
        cfg.data.tracks_per_sample = cfg.tracks_per_sample
        cfg.data.disp_patch_range = cfg.disp_patch_range

        cfg.model.patch_size = cfg.patch_size
        cfg.model.min_track_length = cfg.min_track_length
        cfg.model.tracks_per_sample = cfg.tracks_per_sample
        cfg.model.disp_patch_range = cfg.disp_patch_range


def create_attn_mask(seq_frame_idx, device):
    attn_mask = torch.from_numpy(seq_frame_idx[:, None] == seq_frame_idx[None, :]).to(device)
    attn_mask = torch.logical_not(attn_mask).bool()
    return attn_mask


def save_disparity_gif(seq_frames, pred_disp, gt_disp, sample_idx, save_path, min_track_length):
    os.makedirs(save_path, exist_ok=True)
    images = []

    for t in range(min_track_length):
        frame = seq_frames[sample_idx, t].permute(1, 2, 0).cpu().numpy()

        # Convert to RGB if grayscale
        if frame.shape[2] == 1:
            frame = np.repeat(frame, 3, axis=2)

        # Plot with disparity info
        fig, ax = plt.subplots()
        ax.imshow(frame.astype(np.uint8))
        ax.set_title(f"t={t}, Pred: {pred_disp[sample_idx, t]:.2f}, GT: {gt_disp[sample_idx, t]:.2f}")
        ax.axis('off')

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)
        plt.close(fig)

    gif_name = os.path.join(save_path, f"sample_{sample_idx}_eval.gif")
    imageio.mimsave(gif_name, images, fps=2)


def test_run(model, dataloader, cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.eval()
    model = model.to(device)

    attn_mask = None

    list_disp_pred, list_disp_gt, list_seq_names, list_img_points = [], [], [], []
    for batch_idx, batch_sample in enumerate(tqdm.tqdm(dataloader)):
        seq_frame_patches, seq_event_patches, seq_y_gt_disp_samples, seq_frame_idx, seq_names, img_points = batch_sample
        seq_frame_patches = torch.from_numpy(seq_frame_patches).permute([0, 1, 4, 2, 3]).to(device)
        seq_event_patches = torch.from_numpy(seq_event_patches).permute([0, 1, 4, 2, 3]).to(device)
        n_samples = seq_frame_patches.shape[0]

        assert cfg.min_track_length == seq_frame_patches.shape[1]
        y_pred_disp_samples = np.zeros([n_samples, cfg.min_track_length])

        if attn_mask is None or seq_frame_patches.shape[0] != attn_mask.shape[0]:
            attn_mask = create_attn_mask(seq_frame_idx, device)

        model.reset(None)
        for i_unroll in range(cfg.min_track_length):
            frame_patches = seq_frame_patches[:, i_unroll, :, :, :]
            event_patches = seq_event_patches[:, i_unroll, :, :, :]
            y_disp_pred = model.forward(frame_patches, event_patches, attn_mask)
            y_pred_disp_samples[:, i_unroll] = y_disp_pred[:, 1].detach().cpu().numpy()

        list_disp_pred.append(y_pred_disp_samples)
        list_disp_gt.append(seq_y_gt_disp_samples)
        list_seq_names.append(seq_names)
        list_img_points.append(img_points)

        # Save GIFs for the first batch only
        if batch_idx == 0:
            for sample_idx in range(min(3, n_samples)):
                save_disparity_gif(
                    seq_frames=seq_frame_patches,
                    pred_disp=y_pred_disp_samples,
                    gt_disp=seq_y_gt_disp_samples,
                    sample_idx=sample_idx,
                    save_path='eval_gifs',
                    min_track_length=cfg.min_track_length
                )

    # Save results
    np.savez_compressed('results.npz',
                        disparity_pred=np.concatenate(list_disp_pred, axis=0),
                        seq_names=np.concatenate(list_seq_names, axis=0).flatten())

    np.savez_compressed('ground_truth.npz',
                        disparity_gt=np.concatenate(list_disp_gt, axis=0),
                        image_points=np.concatenate(list_img_points, axis=0),
                        seq_names=np.concatenate(list_seq_names, axis=0).flatten())


@hydra.main(config_path="disp_configs", config_name="m3ed_test")
def test(cfg):
    pl.seed_everything(1234)

    propagate_keys_disp(cfg)
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    with open('test_config.yaml', 'w') as outfile:
        OmegaConf.save(cfg, outfile)

    model = hydra.utils.instantiate(
        cfg.model,
        recursive=False,
    )

    if cfg.checkpoint_path.lower() == 'none':
        print("Provide Checkpoints")

    checkpoint = torch.load(cfg.checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'], strict=True)

    data_module = M3EDTestDataModule(**cfg.data)
    data_module.setup()
    dataloader = data_module.test_dataloader()

    with torch.no_grad():
        test_run(model, dataloader, cfg)


if __name__ == '_main_':
    test()