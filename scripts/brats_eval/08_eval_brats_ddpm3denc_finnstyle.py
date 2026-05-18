from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import argparse
import random

import numpy as np
import pandas as pd
import torch
import torchio as tio
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.metrics import average_precision_score
from scipy.ndimage import median_filter, binary_erosion, label as cc_label

from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader

from src.models.DDPM_2D_3DEnc import DDPM_2D_3DEnc
from src.datamodules.create_dataset import Eval
from src.utils.generate_noise import gen_noise


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def filter_small_components(binary_vol: np.ndarray, min_size: int = 7) -> np.ndarray:
    binary_vol = binary_vol.astype(bool)
    lab, n = cc_label(binary_vol)

    if n == 0:
        return binary_vol

    out = np.zeros_like(binary_vol, dtype=bool)
    for i in range(1, n + 1):
        comp = lab == i
        if int(comp.sum()) >= min_size:
            out |= comp
    return out


def best_dice_from_scores_volume(
    y_true_vol: np.ndarray,
    score_vol: np.ndarray,
    eval_mask: np.ndarray,
    n_thresh: int = 300,
    component_min_size: int = 0,
):
    y_true_vol = y_true_vol.astype(bool)
    eval_mask = eval_mask.astype(bool)
    score_vol = score_vol.astype(np.float64)

    y_true_flat = y_true_vol[eval_mask].astype(bool)
    score_flat = score_vol[eval_mask]

    if y_true_flat.sum() == 0:
        return np.nan, np.nan

    thresholds = np.unique(np.quantile(score_flat, np.linspace(0.0, 1.0, n_thresh)))

    best_dice = -1.0
    best_thr = None

    for thr in thresholds:
        pred = (score_vol > thr) & eval_mask

        if component_min_size and component_min_size > 0:
            pred = filter_small_components(pred, min_size=component_min_size)

        pred_flat = pred[eval_mask].astype(bool)

        tp = np.logical_and(pred_flat, y_true_flat).sum()
        fp = np.logical_and(pred_flat, ~y_true_flat).sum()
        fn = np.logical_and(~pred_flat, y_true_flat).sum()

        denom = 2 * tp + fp + fn
        dice = (2 * tp / denom) if denom > 0 else 0.0

        if dice > best_dice:
            best_dice = dice
            best_thr = thr

    return float(best_dice), float(best_thr)


def make_eval_mask(brain_mask: np.ndarray, erode_iter: int):
    brain = brain_mask > 0.5

    if erode_iter and erode_iter > 0:
        eroded = binary_erosion(brain, iterations=erode_iter)
        if eroded.sum() > 0:
            return eroded.astype(bool)

    return brain.astype(bool)


def save_case_figure(out_path, name, input_vol, reco_vol, score_vol, tumour, eval_mask, threshold):
    D = input_vol.shape[-1]
    z = int(np.argmax(tumour.sum(axis=(0, 1)))) if tumour.sum() > 0 else D // 2

    pred = (score_vol > threshold) & eval_mask

    fig, axes = plt.subplots(1, 6, figsize=(24, 4))

    axes[0].imshow(input_vol[:, :, z].T, origin="lower", cmap="gray")
    axes[0].set_title(f"{name} input z={z}")
    axes[0].axis("off")

    axes[1].imshow(reco_vol[:, :, z].T, origin="lower", cmap="gray")
    axes[1].set_title("Reconstruction")
    axes[1].axis("off")

    axes[2].imshow(score_vol[:, :, z].T, origin="lower")
    axes[2].set_title("Score map")
    axes[2].axis("off")

    axes[3].imshow(input_vol[:, :, z].T, origin="lower", cmap="gray")
    axes[3].imshow(tumour[:, :, z].T, origin="lower", alpha=0.4)
    axes[3].set_title("Tumour")
    axes[3].axis("off")

    axes[4].imshow(input_vol[:, :, z].T, origin="lower", cmap="gray")
    axes[4].imshow(eval_mask[:, :, z].T, origin="lower", alpha=0.25)
    axes[4].set_title("Eval mask")
    axes[4].axis("off")

    axes[5].imshow(input_vol[:, :, z].T, origin="lower", cmap="gray")
    axes[5].imshow(pred[:, :, z].T, origin="lower", alpha=0.4)
    axes[5].set_title("Best-threshold pred")
    axes[5].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def reconstruct_volume(model, batch, device, timesteps, slice_batch_size):
    with torch.no_grad():
        vol = batch["vol"][tio.DATA]  # [B,C,H,W,D]
        cond = model._encode_3d(batch)

        B, C, H, W, D = vol.shape
        assert B == 1, "This evaluator expects batch size 1."

        reco_sum = None

        for t in timesteps:
            reco_slices = []

            for start in range(0, D, slice_batch_size):
                end = min(start + slice_batch_size, D)

                xs = []
                for z in range(start, end):
                    xs.append(vol[:, :, :, :, z])
                x_batch = torch.cat(xs, dim=0)  # [slice_batch,1,H,W]

                cond_batch = cond.repeat(x_batch.shape[0], 1)

                noise = (
                    gen_noise(model.cfg, x_batch.shape).to(device)
                    if model.cfg.get("noisetype") is not None
                    else None
                )

                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device == "cuda")):
                    _, reco = model.diffusion(
                        x_batch,
                        cond=cond_batch,
                        t=int(t) - 1,
                        noise=noise,
                    )

                reco_slices.append(reco.detach().float().cpu())

            reco_cat = torch.cat(reco_slices, dim=0)  # [D,1,H,W]
            reco_vol = reco_cat.squeeze(1).permute(1, 2, 0).numpy()  # [H,W,D]

            if reco_sum is None:
                reco_sum = reco_vol
            else:
                reco_sum += reco_vol

        reco_mean = reco_sum / float(len(timesteps))
        return reco_mean


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", default="Data/splits/Brats21_goldstyle_t2_all_absolute.csv")
    ap.add_argument("--ddpm-ckpt", default="checkpoints/stage2/fold0_split_ddpm_best.ckpt")
    ap.add_argument("--encoder-ckpt", default="checkpoints/stage1/fold0_split_encoder.ckpt")
    ap.add_argument("--outdir", default="docs/brats_eval/finnstyle_ddpm3denc")

    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--slice-batch-size", type=int, default=10)
    ap.add_argument("--save-figures", type=int, default=10)
    ap.add_argument("--resume", action="store_true")

    ap.add_argument("--timesteps", type=int, nargs="+", default=[500],
                    help="Use one value like 500, or ensemble values like 250 500 750.")

    ap.add_argument("--postprocess-finn", action="store_true",
                    help="Apply Finn-style median filtering, mask erosion, and connected-component filtering.")

    ap.add_argument("--median-size", type=int, default=5)
    ap.add_argument("--erode-iter", type=int, default=3)
    ap.add_argument("--component-min-size", type=int, default=7)
    ap.add_argument("--n-thresh", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    figdir = outdir / "figures"
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    results_csv = outdir / "brats_ddpm3denc_finnstyle_metrics_per_case.csv"
    summary_csv = outdir / "brats_ddpm3denc_finnstyle_summary.csv"

    ddpm_ckpt = Path(args.ddpm_ckpt).resolve()
    encoder_ckpt = Path(args.encoder_ckpt).resolve()
    csv_path = Path(args.csv)

    if not ddpm_ckpt.exists():
        raise FileNotFoundError(f"Missing DDPM checkpoint: {ddpm_ckpt}")
    if not encoder_ckpt.exists():
        raise FileNotFoundError(f"Missing encoder checkpoint: {encoder_ckpt}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("DDPM checkpoint:", ddpm_ckpt)
    print("Encoder checkpoint:", encoder_ckpt)
    print("CSV:", csv_path)
    print("Device:", device)
    print("timesteps:", args.timesteps)
    print("postprocess_finn:", args.postprocess_finn)

    model_yaml = OmegaConf.load("configs/model/DDPM_2D_3DEnc.yaml")
    model_cfg = model_yaml.cfg if "cfg" in model_yaml else model_yaml

    with open_dict(model_cfg):
        model_cfg.encoder_path = str(encoder_ckpt)
        model_cfg.condition = True

    model = DDPM_2D_3DEnc(model_cfg)
    ckpt = torch.load(str(ddpm_ckpt), map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device)
    model.eval()

    dm_yaml = OmegaConf.load("configs/datamodule/Gold_700_split.yaml")
    data_cfg = dm_yaml.cfg if "cfg" in dm_yaml else dm_yaml

    df = pd.read_csv(csv_path)

    if args.limit and args.limit > 0:
        df = df.head(args.limit).copy()

    done = set()
    rows = []

    if args.resume and results_csv.exists():
        old = pd.read_csv(results_csv)
        rows = old.to_dict("records")
        done = set(old["img_name"].astype(str))
        print("Resuming. Already done:", len(done))

    for local_i, (idx, row) in enumerate(tqdm(list(df.iterrows()), total=len(df))):
        name = str(row["img_name"])

        if name in done:
            continue

        set_seed(args.seed + int(idx))

        try:
            one_df = pd.DataFrame([row])
            ds = Eval(one_df, data_cfg)
            dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
            batch = next(iter(dl))

            for k, v in list(batch.items()):
                if isinstance(v, dict) and tio.DATA in v and torch.is_tensor(v[tio.DATA]):
                    batch[k][tio.DATA] = v[tio.DATA].to(device)
                elif torch.is_tensor(v):
                    batch[k] = v.to(device)

            with torch.no_grad():
                vol = batch["vol"][tio.DATA]
                mask = batch["mask"][tio.DATA]
                seg = batch["seg"][tio.DATA] if "seg" in batch else torch.zeros_like(mask)

                input_vol = vol.detach().float().cpu().squeeze(0).squeeze(0).numpy()
                mask_vol = mask.detach().float().cpu().squeeze(0).squeeze(0).numpy()
                seg_vol = seg.detach().float().cpu().squeeze(0).squeeze(0).numpy()

                reco_vol = reconstruct_volume(
                    model=model,
                    batch=batch,
                    device=device,
                    timesteps=args.timesteps,
                    slice_batch_size=args.slice_batch_size,
                )

            residual_raw = np.abs(input_vol - reco_vol)

            brain = mask_vol > 0.5
            tumour = seg_vol > 0.5

            if args.postprocess_finn:
                score_vol = median_filter(residual_raw, size=args.median_size)
                eval_mask = make_eval_mask(brain, erode_iter=args.erode_iter)
                cc_min = args.component_min_size
                eval_name = "finn_postprocessed"
            else:
                score_vol = residual_raw
                eval_mask = brain
                cc_min = 0
                eval_name = "raw_residual"

            y_true = tumour[eval_mask].astype(int).ravel()
            scores = score_vol[eval_mask].ravel()

            if y_true.sum() == 0:
                auprc = np.nan
            else:
                auprc = float(average_precision_score(y_true, scores))

            best_dice, best_thr = best_dice_from_scores_volume(
                y_true_vol=tumour,
                score_vol=score_vol,
                eval_mask=eval_mask,
                n_thresh=args.n_thresh,
                component_min_size=cc_min,
            )

            result = {
                "img_name": name,
                "eval_name": eval_name,
                "timesteps": "_".join(map(str, args.timesteps)),
                "postprocess_finn": bool(args.postprocess_finn),
                "auprc_evalmask": auprc,
                "best_dice_evalmask_oracle": best_dice,
                "best_threshold_evalmask": best_thr,
                "tumour_voxels_total": int(tumour.sum()),
                "brainmask_voxels_total": int(brain.sum()),
                "evalmask_voxels": int(eval_mask.sum()),
                "tumour_voxels_evalmask": int((tumour & eval_mask).sum()),
                "tumour_prevalence_evalmask": float(y_true.mean()) if len(y_true) else np.nan,
                "residual_raw_min": float(residual_raw.min()),
                "residual_raw_mean": float(residual_raw.mean()),
                "residual_raw_max": float(residual_raw.max()),
                "score_min": float(score_vol.min()),
                "score_mean": float(score_vol.mean()),
                "score_max": float(score_vol.max()),
                "status": "ok",
                "error": "",
            }

            if len([r for r in rows if r.get("status") == "ok"]) < args.save_figures:
                save_case_figure(
                    figdir / f"{name}_{eval_name}_reconstruction_residual.png",
                    name,
                    input_vol,
                    reco_vol,
                    score_vol,
                    tumour,
                    eval_mask,
                    best_thr,
                )

        except Exception as e:
            result = {
                "img_name": name,
                "eval_name": "finn_postprocessed" if args.postprocess_finn else "raw_residual",
                "timesteps": "_".join(map(str, args.timesteps)),
                "postprocess_finn": bool(args.postprocess_finn),
                "auprc_evalmask": np.nan,
                "best_dice_evalmask_oracle": np.nan,
                "best_threshold_evalmask": np.nan,
                "tumour_voxels_total": np.nan,
                "brainmask_voxels_total": np.nan,
                "evalmask_voxels": np.nan,
                "tumour_voxels_evalmask": np.nan,
                "tumour_prevalence_evalmask": np.nan,
                "residual_raw_min": np.nan,
                "residual_raw_mean": np.nan,
                "residual_raw_max": np.nan,
                "score_min": np.nan,
                "score_mean": np.nan,
                "score_max": np.nan,
                "status": "error",
                "error": repr(e),
            }

        rows.append(result)
        pd.DataFrame(rows).to_csv(results_csv, index=False)

        if device == "cuda":
            torch.cuda.empty_cache()

    res = pd.DataFrame(rows)
    ok = res[res["status"] == "ok"].copy()

    summary = {
        "n_total": len(res),
        "n_ok": len(ok),
        "n_error": int((res["status"] == "error").sum()),
        "timesteps": "_".join(map(str, args.timesteps)),
        "postprocess_finn": bool(args.postprocess_finn),
        "mean_auprc": float(ok["auprc_evalmask"].mean()) if len(ok) else np.nan,
        "median_auprc": float(ok["auprc_evalmask"].median()) if len(ok) else np.nan,
        "std_auprc": float(ok["auprc_evalmask"].std()) if len(ok) else np.nan,
        "mean_best_dice_oracle": float(ok["best_dice_evalmask_oracle"].mean()) if len(ok) else np.nan,
        "median_best_dice_oracle": float(ok["best_dice_evalmask_oracle"].median()) if len(ok) else np.nan,
        "std_best_dice_oracle": float(ok["best_dice_evalmask_oracle"].std()) if len(ok) else np.nan,
        "mean_tumour_prevalence_evalmask": float(ok["tumour_prevalence_evalmask"].mean()) if len(ok) else np.nan,
    }

    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    print("\nSummary:")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print("\nSaved:")
    print(results_csv)
    print(summary_csv)
    print(figdir)


if __name__ == "__main__":
    main()
