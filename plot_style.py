"""
plot_style.py
-------------
Single source of truth for the styling of every comparison figure in this repo.

Design goals (shared across all figures):
  * identical font sizes  -> LABEL_SIZE / TICK_SIZE / LEGEND_SIZE
  * no titles on any figure (panel identity comes from '(a)'/'(b)' tags + legends;
    descriptive text belongs in the LaTeX caption)
  * one canonical line style per method, reused in EVERY figure it appears in:
      - COLOR encodes the method family (Neural Receiver, Neural Receiver FC,
        LS-LMMSE, Perfect-CSI), so a family keeps the same color across panels
      - LINESTYLE encodes the pilot scenario: 2P = solid, 1P = dashed
      - MARKER encodes the family (square / triangle / star / circle)

Import this module from every plotting routine instead of hand-writing styles.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Global size constants -- change here to restyle every figure at once.
# --------------------------------------------------------------------------- #
# Font sizes are deliberately large: these figures are included at
# width=\linewidth, so a wide figure is scaled DOWN by LaTeX (~0.5x for the
# dual-panel ones in a single column).  Large source fonts + a compact figure
# width mean the printed text lands close to the caption size.
LABEL_SIZE      = 18
TICK_SIZE       = 15
LEGEND_SIZE     = 14
PANEL_TAG_SIZE  = 17
MARKERSIZE      = 6.5
LINEWIDTH       = 2.2

FIGSIZE_SINGLE  = (6.4, 4.6)     # one-panel figure
FIGSIZE_DUAL    = (7.4, 4.3)     # 1x2 panel figure (room for a shared top legend)

GRID_KW   = dict(which='both', alpha=0.35)
# near-opaque legend so it stays readable where it sits over the curves
LEGEND_KW = dict(fontsize=LEGEND_SIZE, framealpha=0.9)

# --------------------------------------------------------------------------- #
# Canonical receiver styles.  fmt = marker + linestyle (e.g. 's-').
# Keys match the dict keys used in the saved .npy result files.
# --------------------------------------------------------------------------- #
RECEIVER_STYLES = {
    'neural-receiver-2p':        dict(color='C0', fmt='s-',  label='Neural Receiver - 2P'),
    'neural-receiver-1p':        dict(color='C0', fmt='s--', label='Neural Receiver - 1P'),
    'neural-receiver-fc-2p':     dict(color='C5', fmt='^-',  label='Neural Receiver FC - 2P'),
    'neural-receiver-fc-1p':     dict(color='C5', fmt='^--', label='Neural Receiver FC - 1P'),
    'baseline-ls-estimation-2p': dict(color='C2', fmt='*-',  label='LS-LMMSE - 2P'),
    'baseline-ls-estimation-1p': dict(color='C2', fmt='*--', label='LS-LMMSE - 1P'),
    'baseline-perfect-csi':      dict(color='C4', fmt='o--', label='Perfect-CSI'),
}

# Ordered receiver groups for the 2P | 1P split figures (Perfect-CSI in both).
RX_2P = ['neural-receiver-2p', 'neural-receiver-fc-2p',
         'baseline-ls-estimation-2p', 'baseline-perfect-csi']
RX_1P = ['neural-receiver-1p', 'neural-receiver-fc-1p',
         'baseline-ls-estimation-1p', 'baseline-perfect-csi']

# Receivers shown in the vs-quantization figures (no FC receiver there).
RX_QUANT = ['neural-receiver-2p', 'neural-receiver-1p',
            'baseline-ls-estimation-2p', 'baseline-ls-estimation-1p',
            'baseline-perfect-csi']

# --------------------------------------------------------------------------- #
# Channel styles (Fig. 9) and pose-network styles (Fig. 13).
# --------------------------------------------------------------------------- #
CHANNEL_STYLES = {
    'raytracing': dict(color='C0', fmt='s-',  label='Ray tracing'),
    'cdl':        dict(color='C3', fmt='^--', label='CDL-A'),
    'awgn':       dict(color='C2', fmt='o:',  label='AWGN (flat)'),
}

NETWORK_STYLES = {
    'mlp':    dict(color='C0', fmt='s-',  label='Proposed MLP'),
    'lstm':   dict(color='C1', fmt='^--', label='LSTM'),
    'bi_rnn': dict(color='C3', fmt='o--', label='Bi-RNN (DIP-IMU [14])'),
}


# --------------------------------------------------------------------------- #
# Low-level helpers
# --------------------------------------------------------------------------- #
def _clean(y, log):
    y = np.asarray(y, dtype=float)
    if log:
        y = np.where(y <= 0, np.nan, y)   # a log axis cannot show values <= 0
    return y


def plot_series(ax, x, y, style, log=True):
    """Plot one styled series on `ax` from a style dict {color, fmt, label}."""
    y = _clean(y, log)
    x = np.asarray(x)[:len(y)]
    ax.semilogy(x, y, style['fmt'], color=style['color'], label=style['label'],
                markersize=MARKERSIZE, linewidth=LINEWIDTH)


def style_axis(ax, xlabel, ylabel=None, ylim=None, legend_ncol=1, legend=True):
    """Apply the shared axis styling (labels, ticks, grid, legend)."""
    ax.set_xlabel(xlabel, fontsize=LABEL_SIZE)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.grid(**GRID_KW)
    if ylim is not None:
        ax.set_ylim(ylim)
    if legend:
        ax.legend(ncol=legend_ncol, **LEGEND_KW)


def panel_tag(ax, tag):
    """Register a '(a)'/'(b)' identifier for a panel; drawn by save() below the
    x-label at the panel's final position (so it never collides with the label)."""
    ax._panel_tag = tag


def save(fig, out_path, tagged=False):
    """Lay out (leaving room for panel tags below the x-labels), save, and close."""
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    # Reserve bottom room for both the x-label and the '(a)/(b)' tag beneath it.
    rect = (0, 0.15, 1, 1) if tagged else (0, 0, 1, 1)
    fig.tight_layout(rect=rect)
    if tagged:
        for ax in fig.axes:
            tag = getattr(ax, '_panel_tag', None)
            if tag:
                bbox = ax.get_position()   # final position after tight_layout
                fig.text(bbox.x0 + bbox.width / 2.0, 0.02, tag,
                         ha='center', va='bottom', fontsize=PANEL_TAG_SIZE)
    fig.savefig(out_path)
    plt.close(fig)
    print('Saved {}'.format(out_path))


# --------------------------------------------------------------------------- #
# Unified layout: shared legend ON TOP of the panel(s), tags close below.
# Used by every comparison figure so they all look the same.
# --------------------------------------------------------------------------- #
def _row_major(handles, labels, ncol):
    """Reorder so matplotlib's column-major legend fill renders row by row."""
    import math
    n = len(labels)
    nrow = max(1, math.ceil(n / ncol))
    order = []
    for c in range(ncol):
        for r in range(nrow):
            i = r * ncol + c
            if i < n:
                order.append(i)
    return [handles[i] for i in order], [labels[i] for i in order]


def _finalize(fig, axes, ylabel, out_path, handles, labels, ncol,
              tagged=False, left=0.12, right=0.985, wspace=0.06):
    """Place a shared legend directly above the panel(s) with a small gap, add
    '(a)/(b)' tags just below the x-labels (if any), then save.  Gaps are kept
    tight by driving the margins from the actual legend row count and the
    measured x-label position rather than fixed reserved bands."""
    import math
    if ylabel is not None:
        axes[0].set_ylabel(ylabel, fontsize=LABEL_SIZE)

    H = fig.get_figheight()
    nrow = max(1, math.ceil(len(labels) / ncol))
    row_frac = (LEGEND_SIZE * 1.7) / (H * 72.0)      # height of one legend row
    top_axes = 0.988 - (row_frac * nrow + 0.015)
    bottom_axes = 0.205 if tagged else 0.15
    fig.subplots_adjust(left=left, right=right, top=top_axes,
                        bottom=bottom_axes, wspace=wspace)

    # Grow the left margin if the y-axis label / tick labels would be clipped
    # (their width depends on the actual tick text, e.g. '10^-6' vs '3x10^0').
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = fig.transFigure.inverted()
    x0 = inv.transform((axes[0].get_tightbbox(renderer).x0, 0))[0]
    pad = 0.015
    if x0 < pad:
        left = min(0.30, left + (pad - x0))
        fig.subplots_adjust(left=left)
        fig.canvas.draw()

    # Matplotlib fills legend columns top-to-bottom; reorder so the entries read
    # left-to-right, row by row (families on the first row, 2P/1P key below).
    handles, labels = _row_major(handles, labels, ncol)

    # Legend sits just above the panels (its bottom edge ~ the axes top).
    fig.legend(handles, labels, loc='lower center',
               bbox_to_anchor=(0.5, top_axes + 0.008), ncol=ncol,
               fontsize=LEGEND_SIZE, framealpha=0.9,
               columnspacing=1.1, handletextpad=0.4, borderaxespad=0.0)

    # '(a)/(b)' tags placed just under each measured x-label.
    if tagged:
        fig.canvas.draw()
        inv = fig.transFigure.inverted()
        for ax in axes:
            tag = getattr(ax, '_panel_tag', None)
            if not tag:
                continue
            ext = ax.xaxis.get_label().get_window_extent(fig.canvas.get_renderer())
            _, y_bottom = inv.transform((ext.x0, ext.y0))
            bbox = ax.get_position()
            fig.text(bbox.x0 + bbox.width / 2.0, y_bottom - 0.015, tag,
                     ha='center', va='top', fontsize=PANEL_TAG_SIZE)

    fig.savefig(out_path)
    plt.close(fig)
    print('Saved {}'.format(out_path))


def finalize_dual(fig, axes, ylabel, out_path, ncol=4, handles=None, labels=None):
    """Two-panel wrapper. If handles/labels are not supplied they are collected
    from both panels and de-duplicated by label (used by Fig. 9 and Fig. 13)."""
    if handles is None:
        handles, labels, seen = [], [], set()
        for ax in axes:
            for h, l in zip(*ax.get_legend_handles_labels()):
                if l not in seen:
                    seen.add(l); handles.append(h); labels.append(l)
    _finalize(fig, axes, ylabel, out_path, handles, labels, ncol,
              tagged=True, left=0.11, wspace=0.06)


# --------------------------------------------------------------------------- #
# High-level figure builders
# --------------------------------------------------------------------------- #
def receiver_legend(keys):
    """Compact receiver legend: one colored entry per method family present in
    `keys`, plus a solid/dashed key for the 2P/1P scenario when both appear
    (color = family, linestyle = scenario, so labels need no '- 2P'/'- 1P')."""
    from matplotlib.lines import Line2D
    families = [('neural-receiver',        'Neural Receiver'),
                ('neural-receiver-fc',     'Neural Receiver FC'),
                ('baseline-ls-estimation', 'LS-LMMSE'),
                ('baseline-perfect-csi',   'Perfect-CSI')]
    present = set(keys)
    handles, labels = [], []
    has_2p = has_1p = False
    for base, name in families:
        variants = [base, base + '-2p', base + '-1p']
        if not any(v in present for v in variants):
            continue
        style_key = base + '-2p' if base + '-2p' in RECEIVER_STYLES else base
        st = RECEIVER_STYLES[style_key]
        handles.append(Line2D([], [], color=st['color'], marker=st['fmt'][0],
                              linestyle='-', markersize=MARKERSIZE, linewidth=LINEWIDTH))
        labels.append(name)
        has_2p = has_2p or (base + '-2p') in present
        has_1p = has_1p or (base + '-1p') in present
    if has_2p and has_1p:
        handles += [Line2D([], [], color='0.25', linestyle='-',  linewidth=LINEWIDTH),
                    Line2D([], [], color='0.25', linestyle='--', linewidth=LINEWIDTH)]
        labels += ['2P', '1P']
    return handles, labels


def plot_receiver_split(x, data, xlabel, ylabel, out_path, ylim=None):
    """Two-panel '(a) 2P | (b) 1P' receiver comparison sharing one style set.

    Used for BER vs Eb/N0 on ray tracing (Fig. 6) and CDL-A (Fig. 7)."""
    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_DUAL, sharey=True)
    for ax, group, tag in zip(axes, (RX_2P, RX_1P), ('(a) 2P', '(b) 1P')):
        for key in group:
            if key in data:
                plot_series(ax, x, data[key], RECEIVER_STYLES[key])
        style_axis(ax, xlabel, None, ylim=ylim, legend=False)
        panel_tag(ax, tag)
    handles, labels = receiver_legend(list(RX_2P) + list(RX_1P))
    # 3 columns keeps each legend row within the figure width (the family names
    # are long); families fill the first rows, the 2P/1P key follows.
    _finalize(fig, axes, ylabel, out_path, handles, labels, ncol=3,
              tagged=True, left=0.11)


def plot_receiver_combined(x, data, keys, xlabel, ylabel, out_path, ylim=None):
    """Single-panel receiver comparison with the SAME shared top legend
    (Fig. 8 MSE, Fig. 12 MPJAE)."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    for key in keys:
        if key in data:
            plot_series(ax, x, data[key], RECEIVER_STYLES[key])
    style_axis(ax, xlabel, ylabel, ylim=ylim, legend=False)
    handles, labels = receiver_legend([k for k in keys if k in data])
    # Put the method families on the first legend row, the 2P/1P key below.
    ncol = len(labels) - 2 if ('2P' in labels and '1P' in labels) else len(labels)
    _finalize(fig, [ax], None, out_path, handles, labels, ncol=max(1, ncol),
              tagged=False, left=0.145, right=0.97)
