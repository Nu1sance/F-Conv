#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize Fconv_PCA:
- basis functions (Basis)
- linear combination coefficients (weights)
- reconstructed spatial convolution kernels

This script assumes the Fconv_PCA and GetBasis_PCA definitions are available to import
from a module (e.g., the file you pasted). You can also paste this script into the same
file and run it, or adjust the import path accordingly.

Usage examples:
  # Visualize with freshly initialized weights
  python visualize_fconv_pca.py \
      --sizeP 7 --inNum 3 --outNum 4 --tranNum 8 --inP 7 \
      --m-index 0 --n-index 0 \
      --num-basis-show 16 --basis-orient 0 \
      --outdir viz_out

  # Visualize with a trained checkpoint (state_dict)
  python visualize_fconv_pca.py \
      --sizeP 7 --inNum 64 --outNum 64 --tranNum 8 --inP 7 \
      --m-index 0 --n-index 0 \
      --ckpt path/to/checkpoint.pth \
      --outdir viz_out_trained

Notes:
- "basis" has shape [H, W, tranNum, Rank].
- "weights" has shape [outNum, inNum, expand, Rank] (expand = tranNum if ifIni=0 else 1).
- Reconstructed per-(m,n) kernels have shape [tranNum, expand, H, W] before the final reshape.
- The final conv filter used by conv2d has shape [outNum*tranNum, inNum*expand, H, W].
"""

import argparse
import os
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
# Import your Fconv_PCA and GetBasis_PCA from your module.
# If your definitions are in the same directory in a file named fconv_pca_def.py, do:
#   from fconv_pca_def import Fconv_PCA, GetBasis_PCA
# Otherwise, if you paste this script into the same file containing those classes,
# this import is not required.
# --------------------------------------------------------------------------------------
from F_Conv import Fconv_PCA, GetBasis_PCA

# If you run this file standalone, paste the definitions of Fconv_PCA and GetBasis_PCA
# above this line, or adjust the import according to your project structure.


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _save_grid(images: np.ndarray,
               grid_shape: Tuple[int, int],
               out_path: str,
               title: Optional[str] = None,
               cmap: str = "RdBu_r",
               share_scale: bool = True):
    """
    Save a grid of 2D images using matplotlib.

    Args:
        images: np.ndarray of shape [N, H, W]
        grid_shape: (rows, cols) such that rows*cols == N
        out_path: file path to save
        title: optional grid title
        cmap: colormap
        share_scale: if True, use the same vmin/vmax across all tiles
    """
    N, H, W = images.shape
    rows, cols = grid_shape
    assert rows * cols == N, f"Grid {rows}x{cols} != N={N}"

    if share_scale:
        amax = np.max(np.abs(images))
        vmin, vmax = -amax, amax
    else:
        vmin = vmax = None

    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.2 * rows))
    if title:
        fig.suptitle(title, fontsize=12)

    axes = np.array(axes).reshape(rows, cols)
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            ax = axes[i, j]
            im = ax.imshow(images[idx], cmap=cmap, vmin=vmin, vmax=vmax)
            ax.axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_heatmap(mat: np.ndarray, out_path: str, title: Optional[str] = None,
                  xlabel: str = "", ylabel: str = "", cmap: str = "viridis"):
    """
    Plot a heatmap for 2D matrix.
    """
    fig, ax = plt.subplots(figsize=(8, 3.5))
    if title:
        ax.set_title(title)
    im = ax.imshow(mat, aspect="auto", cmap=cmap)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.set_ylabel("value", rotation=270, labelpad=10)
    plt.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def reconstruct_tempW_from_module(conv: nn.Module) -> torch.Tensor:
    """
    Reconstruct tempW exactly as in Fconv_PCA.forward (before reshape),
    but returning the full 6D tensor:

        tempW: [outNum, tranNum, inNum, expand, H, W]

    The subsequent cyclic shift across the 'expand' axis per-orientation block is applied.
    """
    with torch.no_grad():
        Basis = conv.Basis  # [H, W, tranNum, Rank]
        weights = conv.weights  # [outNum, inNum, expand, Rank]
        outNum, inNum, expand, Rank = weights.shape
        H, W, tranNum, RankB = Basis.shape
        assert RankB == Rank, f"Rank mismatch: Basis({RankB}) vs weights({Rank})"
        assert tranNum % expand == 0, f"tranNum({tranNum}) must be divisible by expand({expand})"

        # einsum dims: Basis 'ijok', weights 'mnak' -> 'monaij'
        tempW = torch.einsum('ijok,mnak->monaij', Basis, weights)
        # tempW: [m=outNum, o=tranNum, n=inNum, a=expand, i=H, j=W]

        # Apply the cyclic shift per expand block to match the forward() logic
        Num = tranNum // expand
        tempWList = []
        for i in range(expand):
            # concat last-i and first-(a-i) slices along 'a' dimension to rotate it
            # Note: -i means last i entries; when i=0, -0 is 0 so we use a guard
            if i == 0:
                rotated = tempW[:, i*Num:(i+1)*Num, :, :, :, :]
            else:
                rotated = torch.cat([
                    tempW[:, i*Num:(i+1)*Num, :, -i:, :, :],
                    tempW[:, i*Num:(i+1)*Num, :, :-i, :, :]
                ], dim=3)
            tempWList.append(rotated)
        tempW = torch.cat(tempWList, dim=1)  # [outNum, tranNum, inNum, expand, H, W]
        return tempW


def main():
    parser = argparse.ArgumentParser(description="Visualize Fconv_PCA basis, coefficients, and reconstructed kernels.")
    # Module params
    parser.add_argument("--sizeP", type=int, default=7, help="Kernel size")
    parser.add_argument("--inNum", type=int, default=3, help="Input channels")
    parser.add_argument("--outNum", type=int, default=4, help="Output channels")
    parser.add_argument("--tranNum", type=int, default=4, help="Number of rotation orientations")
    parser.add_argument("--inP", type=int, default=None, help="Number of Fourier frequencies (defaults to sizeP)")
    parser.add_argument("--ifIni", type=int, default=0, choices=[0, 1], help="If 1, expand=1; otherwise expand=tranNum")
    parser.add_argument("--no-bias", action="store_true", help="Disable bias parameter")
    parser.add_argument("--smooth", action="store_true", help="Apply Smooth scaling (same as GetBasis_PCA Smooth=True)")
    parser.add_argument("--iniScale", type=float, default=1.0, help="Init scale for weights (not critical)")

    # Selection
    parser.add_argument("--m-index", type=int, default=0, help="Output channel index to visualize")
    parser.add_argument("--n-index", type=int, default=0, help="Input channel index to visualize")
    parser.add_argument("--expand-index", type=int, default=-1, help="Expand index to select; -1 to show all")
    parser.add_argument("--collapse-expand", action="store_true", help="Sum over expand axis when visualizing kernels")

    # Basis visualization
    parser.add_argument("--num-basis-show", type=int, default=16, help="Number of basis to show for a fixed orientation")
    parser.add_argument("--basis-orient", type=int, default=0, help="Orientation index used when showing multiple basis")
    parser.add_argument("--basis-index", type=int, default=0, help="A single basis index to show across all orientations")

    # Checkpoint and output
    parser.add_argument("--ckpt", type=str, default="", help="Optional path to a state_dict checkpoint to load")
    parser.add_argument("--outdir", type=str, default="viz_out", help="Output directory")
    args = parser.parse_args()

    _ensure_dir(args.outdir)

    # Instantiate module
    bias_flag = not args.no_bias
    expand = 1 if args.ifIni == 1 else args.tranNum

    # You must import Fconv_PCA from your definitions. If this import fails,
    # please paste this script into the same file where Fconv_PCA is defined.
    try:
        from F_Conv import Fconv_PCA  # adjust to your module name if needed
    except Exception:
        # If the classes are in the same file, this import may not be required.
        # Attempt to reference Fconv_PCA directly (works if pasted together).
        pass

    conv = Fconv_PCA(
        sizeP=args.sizeP,
        inNum=args.inNum,
        outNum=args.outNum,
        tranNum=args.tranNum,
        inP=args.inP,
        padding=0,
        ifIni=args.ifIni,
        bias=bias_flag,
        Smooth=args.smooth,
        iniScale=args.iniScale,
    )
    x = torch.randn(1, 12, 512, 512)
    y = conv(x)

    # Optionally load trained weights
    if args.ckpt and os.path.isfile(args.ckpt):
        print(f"[INFO] Loading checkpoint: {args.ckpt}")
        state = torch.load(args.ckpt, map_location="cpu")
        # Accept either raw state_dict or {'state_dict': ...}
        sd = state if isinstance(state, dict) and all(k.startswith(("weights", "c", "Basis")) for k in state.keys()) else state.get("state_dict", state)
        # Try strict=False to be more robust to key prefixing
        conv.load_state_dict(sd, strict=False)
    else:
        if args.ckpt:
            print(f"[WARN] Checkpoint not found: {args.ckpt}. Proceeding with random initialization.")

    conv.eval()  # we only need buffers/params; no forward required
    with torch.no_grad():
        Basis = conv.Basis.cpu().numpy()  # [H, W, tranNum, Rank]
        weights = conv.weights.detach().cpu().numpy()  # [outNum, inNum, expand, Rank]

    H, W, tranNum_B, Rank = Basis.shape
    outNum, inNum, expand_param, Rank_w = weights.shape
    assert tranNum_B == args.tranNum, "Basis tranNum inconsistent"
    assert Rank == Rank_w, "Basis rank != weights rank"
    assert expand_param == expand, "expand value mismatch"

    m = np.clip(args.m_index, 0, outNum - 1)
    n = np.clip(args.n_index, 0, inNum - 1)
    e_sel = args.expand_index
    if e_sel >= 0:
        e_sel = np.clip(e_sel, 0, expand - 1)

    print(f"[INFO] Shapes: Basis={Basis.shape}, weights={weights.shape}")
    print(f"[INFO] Visualizing out m={m}, in n={n}, expand={'all' if e_sel<0 else e_sel}")

    # ---------------------------------------------
    # 1) Visualize basis
    # ---------------------------------------------
    # a) Fix one orientation, show many basis indices (first K)
    K = min(args.num_basis_show, Rank)
    orient_idx = np.clip(args.basis_orient, 0, args.tranNum - 1)
    basis_fixed_orient = Basis[:, :, orient_idx, :K].transpose(2, 0, 1)  # [K, H, W]
    _save_grid(
        images=basis_fixed_orient,
        grid_shape=(int(math.ceil(K / 8)), min(8, K)),
        out_path=os.path.join(args.outdir, f"basis_many_k_at_orient_{orient_idx}.png"),
        title=f"Basis (K={K}) at orientation {orient_idx}",
        cmap="RdBu_r",
        share_scale=True,
    )

    # b) Fix one basis index, show all orientations
    k_one = np.clip(args.basis_index, 0, Rank - 1)
    basis_all_orients = Basis[:, :, :, k_one].transpose(2, 0, 1)  # [tranNum, H, W]
    _save_grid(
        images=basis_all_orients,
        grid_shape=(int(math.ceil(args.tranNum / 8)), min(8, args.tranNum)),
        out_path=os.path.join(args.outdir, f"basis_orientations_for_k_{k_one}.png"),
        title=f"One basis (k={k_one}) across {args.tranNum} orientations",
        cmap="RdBu_r",
        share_scale=True,
    )

    # ---------------------------------------------
    # 2) Visualize coefficients (weights[m, n, a, k])
    # ---------------------------------------------
    coeff_mn = weights[m, n]  # [expand, Rank]
    _plot_heatmap(
        coeff_mn,
        out_path=os.path.join(args.outdir, f"coeff_heatmap_m{m}_n{n}.png"),
        title=f"Coefficients heatmap (weights[m={m}, n={n}, a, k])",
        xlabel="basis index k",
        ylabel="expand index a",
        cmap="magma",
    )

    # Also show per-expand lines of top abs coefficients (optional quick view)
    # Identify top L by absolute average across expand
    L = min(24, Rank)
    mean_abs = np.mean(np.abs(coeff_mn), axis=0)
    top_idx = np.argsort(-mean_abs)[:L]
    coeff_top = coeff_mn[:, top_idx]  # [expand, L]
    _plot_heatmap(
        coeff_top,
        out_path=os.path.join(args.outdir, f"coeff_heatmap_TOP{L}_m{m}_n{n}.png"),
        title=f"Top-{L} coefficients by |mean| across expand (m={m}, n={n})",
        xlabel="top-k basis (sorted by |mean|)",
        ylabel="expand index a",
        cmap="magma",
    )

    # ---------------------------------------------
    # 3) Reconstruct spatial kernels for (m, n)
    # ---------------------------------------------
    tempW = reconstruct_tempW_from_module(conv)  # [outNum, tranNum, inNum, expand, H, W]
    kern_mn = tempW[m, :, n, :, :, :].cpu().numpy()  # [tranNum, expand, H, W]

    if args.collapse_expand:
        # Sum over expand dimension to show a single kernel per orientation
        kernels = np.sum(kern_mn, axis=1)  # [tranNum, H, W]
        _save_grid(
            images=kernels,
            grid_shape=(int(math.ceil(args.tranNum / 8)), min(8, args.tranNum)),
            out_path=os.path.join(args.outdir, f"kernels_SUM_expand_m{m}_n{n}.png"),
            title=f"Reconstructed kernels (sum over expand) for m={m}, n={n}",
            cmap="RdBu_r",
            share_scale=True,
        )
    else:
        if e_sel >= 0:
            # Show only one expand index across orientations
            kernels = kern_mn[:, e_sel, :, :]  # [tranNum, H, W]
            _save_grid(
                images=kernels,
                grid_shape=(int(math.ceil(args.tranNum / 8)), min(8, args.tranNum)),
                out_path=os.path.join(args.outdir, f"kernels_expand{e_sel}_m{m}_n{n}.png"),
                title=f"Reconstructed kernels for m={m}, n={n}, expand={e_sel}",
                cmap="RdBu_r",
                share_scale=True,
            )
        else:
            # Show all expand indices for all orientations (grid: rows=tranNum, cols=expand)
            # Flatten to [tranNum*expand, H, W], then grid
            t, a, h, w = kern_mn.shape
            kernels = kern_mn.reshape(t * a, h, w)
            _save_grid(
                images=kernels,
                grid_shape=(t, a),
                out_path=os.path.join(args.outdir, f"kernels_all_expand_m{m}_n{n}.png"),
                title=f"Reconstructed kernels for m={m}, n={n} (grid: rows=orient, cols=expand)",
                cmap="RdBu_r",
                share_scale=True,
            )

    # ---------------------------------------------
    # 4) (Optional) Visualize the actual conv filter block used by conv2d
    #    for this output channel m: shape [tranNum, inNum*expand, H, W].
    #    This is what gets reshaped into [outNum*tranNum, inNum*expand, H, W].
    # ---------------------------------------------
    with torch.no_grad():
        # Reuse tempW to assemble the full _filter like forward()
        # tempW: [m, o, n, a, i, j] -> reshape to [outNum*tranNum, inNum*expand, H, W]
        outNum, o, inNum, expand, H, W = tempW.shape
        filt = tempW.reshape(outNum * o, inNum * expand, H, W)  # [out*tranNum, in*expand, H, W]
        block_m = filt[m*o:(m+1)*o]  # [tranNum, inNum*expand, H, W]
        # For display, show L2 over input planes to compress to one 2D per orientation
        block_m_np = block_m.cpu().numpy()
        block_energy = np.sqrt(np.sum(block_m_np**2, axis=1))  # [tranNum, H, W]
        _save_grid(
            images=block_energy,
            grid_shape=(int(math.ceil(args.tranNum / 8)), min(8, args.tranNum)),
            out_path=os.path.join(args.outdir, f"conv_filter_block_energy_m{m}.png"),
            title=f"Conv filter block energy per orientation (m={m})",
            cmap="inferno",
            share_scale=False,
        )

    print(f"[DONE] Visualizations saved to: {args.outdir}")


if __name__ == "__main__":
    main()