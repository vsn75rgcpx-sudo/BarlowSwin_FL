"""
vis3d.py
--------
Visualization utilities for 3D MRI segmentation results.

Functions:
 - show_slice_overlay(): visualize 2D slice with segmentation overlay
 - show_volume_slices(): scrollable slice viewer
 - render_3d_surface(): 3D surface visualization (Plotly)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import plotly.graph_objects as go


# ------------------------------------------------------------
# 2D slice overlay
# ------------------------------------------------------------
def show_slice_overlay(volume, mask, slice_idx=None, axis=0, alpha=0.4):
    """
    Display a slice of MRI with segmentation overlay.

    Args:
        volume: numpy array (D,H,W)
        mask: numpy array same shape as volume
        slice_idx: which slice to visualize
        axis: slicing direction (0=D, 1=H, 2=W)
        alpha: transparency of mask
    """

    if slice_idx is None:
        slice_idx = volume.shape[axis] // 2

    if axis == 0:
        img = volume[slice_idx]
        m   = mask[slice_idx]
    elif axis == 1:
        img = volume[:, slice_idx, :]
        m   = mask[:, slice_idx, :]
    else:
        img = volume[:, :, slice_idx]
        m   = mask[:, :, slice_idx]

    plt.figure(figsize=(6,6))
    plt.imshow(img, cmap="gray")
    plt.imshow(np.ma.masked_where(m == 0, m), cmap="jet", alpha=alpha)
    plt.title(f"Slice {slice_idx} (axis={axis})")
    plt.axis("off")
    plt.show()


# ------------------------------------------------------------
# Scrollable 2D slice viewer
# ------------------------------------------------------------
def show_volume_slices(volume, mask=None, axis=0, alpha=0.4):
    """
    Interactive scroll-through-slices view.

    Args:
        volume: (D,H,W)
        mask: optional segmentation mask (D,H,W)
    """

    vol = volume
    max_idx = vol.shape[axis] - 1

    # initial slice
    idx0 = max_idx // 2

    def get_slice(v, idx):
        if axis == 0: return v[idx]
        elif axis == 1: return v[:, idx, :]
        else: return v[:, :, idx]

    # figure
    fig, ax = plt.subplots(figsize=(6,6))
    plt.subplots_adjust(bottom=0.15)

    img_show = get_slice(vol, idx0)
    im = ax.imshow(img_show, cmap="gray")

    if mask is not None:
        mask_show = get_slice(mask, idx0)
        seg = ax.imshow(np.ma.masked_where(mask_show == 0, mask_show),
                        cmap="jet", alpha=alpha)

    ax.set_title(f"Slice {idx0}/{max_idx}")
    ax.axis("off")

    # slider
    ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
    slider = Slider(ax_slider, "slice", 0, max_idx, valinit=idx0, valfmt="%0.0f")

    def update(val):
        idx = int(slider.val)
        im.set_data(get_slice(vol, idx))
        ax.set_title(f"Slice {idx}/{max_idx}")

        if mask is not None:
            seg.set_data(np.ma.masked_where(get_slice(mask, idx) == 0,
                                             get_slice(mask, idx)))
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()


# ------------------------------------------------------------
# 3D surface rendering with Plotly
# ------------------------------------------------------------
def render_3d_surface(mask, level=0.5, opacity=0.6):
    """
    3D segmentation mask rendering via Plotly volume visualization.

    Args:
        mask: segmentation volume (D,H,W)
        level: threshold for surface
    """
    vol = mask.astype(float)

    fig = go.Figure(data=go.Volume(
        x=np.arange(vol.shape[2]).repeat(vol.shape[0]*vol.shape[1]),
        y=np.tile(np.repeat(np.arange(vol.shape[1]), vol.shape[2]), vol.shape[0]),
        z=np.repeat(np.arange(vol.shape[0]), vol.shape[1]*vol.shape[2]),
        value=vol.flatten(),
        opacity=opacity,
        surface_count=2,
        isomin=level,
        isomax=vol.max(),
        colorscale="Turbo",
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=True),
            yaxis=dict(visible=True),
            zaxis=dict(visible=True)
        ),
        width=800,
        height=800,
        title="3D Segmentation Surface Rendering"
    )

    fig.show()
