"""
gen_channel_datasets.py
-----------------------
Generate CDL and AWGN(flat) channel-impulse-response datasets in the SAME format as
the ray-traced datasets (data/cirdata/a_dataset_*.npy, tau_dataset_*.npy), so they
drop straight into the existing CIRGenerator -> CIRDataset -> OFDMChannel pipeline.

Why this approach (R5.3 AWGN, R5.4 CDL):
    Your pipeline already consumes (a, tau) arrays of shape
        a   : [num_cirs, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
        tau : [num_cirs, num_rx, num_tx, num_paths]
    so the cleanest, lowest-risk way to add a CDL or AWGN comparison is to produce
    (a, tau) in this format and reuse the existing BER (eval_mode 2) and MSE/MPJAE
    (eval_mode 3) evaluation unchanged. No receiver retraining is required for the
    headline channel-realism claim (see CHANNEL_BASELINES.md).

VERSION NOTE (important):
    The doc snippets you pasted use the Sionna v1.x API (`sionna.phy.channel...`, and
    an AWGN example that imports torch). Your codebase pins sionna==0.19.0 (TensorFlow,
    `sionna.channel...`, `sionna.rt...`). This file targets 0.19 to stay consistent with
    your trained weights and ray-tracing pipeline. The v1.x equivalents are noted in
    comments in case you ever upgrade.

Outputs:
    data/cirdata/a_dataset_cdl.npy,  data/cirdata/tau_dataset_cdl.npy
    data/cirdata/a_dataset_awgn.npy, data/cirdata/tau_dataset_awgn.npy
"""

import os
import argparse
import numpy as np

# OFDM / array parameters -- keep in sync with ofdm_params in main.py / Table III
DEFAULTS = dict(
    carrier_frequency=3.5e9,
    subcarrier_spacing=30e3,   # used as the Doppler sampling frequency (match rg.subcarrier_spacing)
    num_time_steps=14,         # number of OFDM symbols
    num_rx=1,
    num_rx_ant=16,             # base station (uplink receiver)
    num_tx=1,
    num_tx_ant=1,              # user (uplink transmitter)
    min_speed=13.6,            # m/s, matches the ray-traced rx velocity range
    max_speed=18.8,
    num_paths=75,              # MUST match ofdm_params['num_rt_paths']; CIRDataset is built
                               # with this fixed path count, so all CIRs are padded to it.
)

CIRDIR = 'data/cirdata'
os.makedirs(CIRDIR, exist_ok=True)


# =====================================================================================
# CDL (R5.4) -- Sionna 0.19 API
# =====================================================================================
def gen_cdl(num_cirs=12000, batch_size_cir=200, model='A', delay_spread=300e-9, p=DEFAULTS):
    """
    Generate a CDL CIR dataset (3GPP TR 38.901) matching the ray-traced format.

    model         : 'A','B','C','D','E' (CDL-A/C are the common dispersive NLOS models)
    delay_spread  : RMS delay spread in seconds (300 ns is a typical urban value)
    """
    import tensorflow as tf
    # --- Sionna 0.19 imports (NOT sionna.phy, which is v1.x) ---
    from sionna.channel.tr38901 import AntennaArray, CDL

    fc = p['carrier_frequency']

    # Uplink: user terminal (UT) transmits, base station (BS) receives.
    # UT: single omni antenna (num_tx_ant = 1).
    ut_array = AntennaArray(num_rows=1, num_cols=1,
                            polarization='single', polarization_type='V',
                            antenna_pattern='omni', carrier_frequency=fc)
    # BS: dual-polarised 38.901 panel; num_cols * 2 = num_rx_ant (16 -> 8 cols x 2 pol).
    assert p['num_rx_ant'] % 2 == 0, 'num_rx_ant must be even for dual polarisation'
    bs_array = AntennaArray(num_rows=1, num_cols=p['num_rx_ant'] // 2,
                            polarization='dual', polarization_type='VH',
                            antenna_pattern='38.901', carrier_frequency=fc)

    cdl = CDL(model=model, delay_spread=delay_spread, carrier_frequency=fc,
              ut_array=ut_array, bs_array=bs_array, direction='uplink',
              min_speed=p['min_speed'], max_speed=p['max_speed'])
    # v1.x equivalent: from sionna.phy.channel.tr38901 import Antenna, AntennaArray, CDL
    #   ut_array = AntennaArray(antenna=Antenna(pattern="omni",   polarization="single"), num_rows=1, num_cols=1)
    #   bs_array = AntennaArray(antenna=Antenna(pattern="38.901", polarization="dual"),   num_rows=1, num_cols=8)

    a_list, tau_list = [], []
    collected = 0
    while collected < num_cirs:
        bs = min(batch_size_cir, num_cirs - collected)
        # a: [bs, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
        # tau: [bs, num_rx, num_tx, num_paths]
        a, tau = cdl(batch_size=bs, num_time_steps=p['num_time_steps'],
                     sampling_frequency=p['subcarrier_spacing'])
        a_list.append(a.numpy().astype(np.complex64))
        tau_list.append(tau.numpy().astype(np.float32))
        collected += bs
        print('  CDL-{}: {}/{} CIRs'.format(model, collected, num_cirs))

    a_arr = np.concatenate(a_list, axis=0)
    tau_arr = np.concatenate(tau_list, axis=0)
    a_arr, tau_arr = _fit_paths(a_arr, tau_arr, p['num_paths'])
    _save('cdl', a_arr, tau_arr)
    return a_arr, tau_arr


# =====================================================================================
# AWGN / flat channel (R5.3) -- reproduces the idealised AWGN assumption of [16]
# within the OFDM system: a single unit-gain, zero-delay tap => flat H over subcarriers.
# =====================================================================================
def gen_awgn(num_cirs=2000, p=DEFAULTS):
    """
    Generate a flat (AWGN) channel dataset: one unit-gain path at zero delay for every
    receive antenna and OFDM symbol. After cir_to_ofdm_channel(normalize=True) this gives
    H[n] = 1 across all subcarriers, so the OFDMChannel (add_awgn=True) reduces to AWGN.
    The 16-antenna array is retained (as in the ray-traced case) so the comparison is fair:
    the ONLY thing removed is multipath + Doppler, which is exactly the "realism" being
    quantified against the AWGN-based prior work [16].
    """
    nrx, nra, ntx, nta, nts = (p['num_rx'], p['num_rx_ant'], p['num_tx'],
                               p['num_tx_ant'], p['num_time_steps'])
    target = p['num_paths']
    # First path: unit gain at zero delay (flat channel). Remaining paths: zero gain.
    a = np.zeros([num_cirs, nrx, nra, ntx, nta, target, nts], dtype=np.complex64)
    a[..., 0, :] = 1.0
    tau = np.zeros([num_cirs, nrx, ntx, target], dtype=np.float32)
    _save('awgn', a, tau)
    return a, tau
    # v0.19 alternative (direct, without the OFDM pipeline):
    #   from sionna.channel import AWGN ; awgn = AWGN() ; y = awgn([x, no])
    # but the flat-CIR route above reuses your full E2E pipeline + BER/MSE/MPJAE metrics,
    # which is what makes it directly comparable to the ray-traced results.


def _fit_paths(a, tau, target):
    """
    Pad (or truncate) the path dimension so every CIR has exactly `target` paths, matching
    the fixed num_paths the CIRDataset is constructed with (ofdm_params['num_rt_paths']=75).
    Padded paths have zero gain (and zero delay), so they add nothing to the channel
    H[n] = sum_i a_i exp(-j2*pi*f*tau_i): the result is identical, just shape-compatible.
      a   : [..., num_paths, num_time_steps]  (path axis = -2)
      tau : [..., num_paths]                  (path axis = -1)
    """
    P = a.shape[-2]
    if P == target:
        return a, tau
    if P < target:
        pad_a = np.zeros(a.shape[:-2] + (target - P,) + a.shape[-1:], dtype=a.dtype)
        a = np.concatenate([a, pad_a], axis=-2)
        pad_tau = np.zeros(tau.shape[:-1] + (target - P,), dtype=tau.dtype)
        tau = np.concatenate([tau, pad_tau], axis=-1)
        print('  padded paths {} -> {}'.format(P, target))
    else:  # keep the first `target` paths (CDL models all have < 75, so this rarely triggers)
        a = a[..., :target, :]
        tau = tau[..., :target]
        print('  truncated paths {} -> {}'.format(P, target))
    return a, tau


def _save(name, a, tau):
    ap = os.path.join(CIRDIR, 'a_dataset_{}.npy'.format(name))
    tp = os.path.join(CIRDIR, 'tau_dataset_{}.npy'.format(name))
    np.save(ap, a)
    np.save(tp, tau)
    print('Saved {}  a={}  tau={}'.format(name, a.shape, tau.shape))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Generate CDL / AWGN CIR datasets (Sionna 0.19)')
    ap.add_argument('--channel', type=str, required=True, choices=['cdl', 'awgn'])
    ap.add_argument('--num_cirs', type=int, default=12000)
    ap.add_argument('--batch_cir', type=int, default=200)
    ap.add_argument('--cdl_model', type=str, default='A')
    ap.add_argument('--delay_spread', type=float, default=300e-9)
    ap.add_argument('--num_paths', type=int, default=DEFAULTS['num_paths'],
                    help="Target path count; MUST equal ofdm_params['num_rt_paths'] (75).")
    ap.add_argument('--repad', action='store_true',
                    help='Do not regenerate; just re-pad an existing dataset to --num_paths.')
    args = ap.parse_args()
    DEFAULTS['num_paths'] = args.num_paths

    if args.repad:
        # Fix an already-generated file in place (no Sionna / no regeneration needed).
        ap_ = os.path.join(CIRDIR, 'a_dataset_{}.npy'.format(args.channel))
        tp_ = os.path.join(CIRDIR, 'tau_dataset_{}.npy'.format(args.channel))
        a = np.load(ap_); tau = np.load(tp_)
        print('Loaded {}  a={}  tau={}'.format(args.channel, a.shape, tau.shape))
        a, tau = _fit_paths(a, tau, args.num_paths)
        _save(args.channel, a, tau)
    elif args.channel == 'cdl':
        gen_cdl(num_cirs=args.num_cirs, batch_size_cir=args.batch_cir,
                model=args.cdl_model, delay_spread=args.delay_spread)
    else:
        gen_awgn(num_cirs=min(args.num_cirs, 2000))