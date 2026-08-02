"""
plot_comparisons.py
-------------------
Reproduces the three comparison figures from the saved .npy result files:

  1. 'mse_channels' -> Reconstruction MSE vs quantization (Eb/N0 = 5 dB)
                       reads data/pltdata/mse_{raytracing,cdl,awgn}.npy
  2. 'ber_cdl'      -> BER on 3GPP CDL-A channel
                       reads data/pltdata/bler_cdl.npy
  3. 'per_scene'    -> Per-scene BER (Neural Receiver, 2P)
                       reads data/pltdata/ber_munich.npy, ber_etoile.npy

These functions are pure numpy + matplotlib (no Sionna/TensorFlow), so they can be
called from main.py's plot_figure(), or run standalone:

    python plot_comparisons.py mse_channels
    python plot_comparisons.py ber_cdl
    python plot_comparisons.py per_scene
    python plot_comparisons.py all
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from plot_style import (RECEIVER_STYLES, CHANNEL_STYLES, RX_2P, RX_1P,
                        plot_series, style_axis, panel_tag, save, finalize_dual,
                        plot_receiver_split, FIGSIZE_DUAL, LABEL_SIZE)

PLT_DIR = 'data/pltdata'
FIG_DIR = 'data/figures'
os.makedirs(FIG_DIR, exist_ok=True)


def _load(name):
    """Load a dict-style .npy result file (or None if missing)."""
    path = os.path.join(PLT_DIR, name)
    if not os.path.exists(path):
        print('  [skip] {} not found'.format(path))
        return None
    obj = np.load(path, allow_pickle=True)
    try:
        return obj.item()
    except (ValueError, AttributeError):
        return obj


def plot_mse_channels(channels=('raytracing', 'cdl', 'awgn')):
    """Reconstruction MSE vs quantization (Eb/N0 = 5 dB), two panels (Fig. 9).

    (a) neural receiver (2P), (b) LS-LMMSE baseline (1P); each panel compares the
    ray-tracing / CDL-A / AWGN channels using the shared channel style.
    """
    q = np.arange(4, 11)
    data = {ch: _load('mse_{}.npy'.format(ch)) for ch in channels}
    data = {ch: d for ch, d in data.items() if d is not None}
    if not data:
        print('No MSE files found; nothing to plot.')
        return
    # Perfect-CSI reference (channel-independent); take from any available channel
    ref = next(iter(data.values())).get('baseline-perfect-csi')

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DUAL, sharey=True)
    panels = [('neural-receiver-2p', '(a) Neural Receiver (2P)'),
              ('baseline-ls-estimation-1p', '(b) LS-LMMSE (1P)')]
    for axi, (key, tag) in zip(axes, panels):
        for ch, d in data.items():
            if key in d:
                plot_series(axi, q, d[key], CHANNEL_STYLES[ch])
        if ref is not None:
            axi.semilogy(q, np.asarray(ref), 'k:', alpha=0.5, label='Perfect-CSI (ref)')
        style_axis(axi, 'Quantization level (bits)', None, legend=False)
        panel_tag(axi, tag)
    finalize_dual(fig, axes, 'MSE', os.path.join(FIG_DIR, 'mse_channels.pdf'), ncol=4)


def plot_ber_cdl(channel='cdl'):
    """BER vs Eb/N0 on the CDL-A channel (Fig. 7), split into (a) 2P | (b) 1P."""
    d = _load('bler_{}.npy'.format(channel))
    if d is None:
        return
    ebno = np.arange(-5.0, 16.0, 1.0)
    plot_receiver_split(ebno, d,
                        xlabel=r'$E_b/N_0$ (dB)', ylabel='BER',
                        out_path=os.path.join(FIG_DIR, 'ber_cdl.pdf'))


def plot_per_scene(scenes=('munich', 'etoile')):
    """Per-scene BER (Neural Receiver, 2P): Munich vs Etoile (shared style, no title)."""
    styles = {'munich': dict(color='C0', fmt='s-',  label='Munich (test)'),
              'etoile': dict(color='C3', fmt='o--', label='Etoile (test)')}
    ebno = np.arange(-5.0, 16.0, 1.0)
    from plot_style import FIGSIZE_SINGLE
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    plotted = False
    for sc in scenes:
        arr = _load('ber_{}.npy'.format(sc))
        if arr is None:
            continue
        plot_series(ax, ebno, arr, styles.get(sc, dict(color=None, fmt='-', label=sc)))
        plotted = True
    if not plotted:
        print('No per-scene BER files found; nothing to plot.')
        plt.close(fig)
        return
    style_axis(ax, r'$E_b/N_0$ (dB)', 'BER', legend_ncol=1)
    save(fig, os.path.join(FIG_DIR, 'ber_per_scene.pdf'))


if __name__ == '__main__':
    import sys
    which = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if which in ('mse_channels', 'all'):
        plot_mse_channels()
    if which in ('ber_cdl', 'all'):
        plot_ber_cdl()
    if which in ('per_scene', 'all'):
        plot_per_scene()