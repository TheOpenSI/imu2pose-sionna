import os
import platform
import pickle as pkl
import numpy as np

# ----------------------------------------------------------------------------------
# Paths (mirror pose_generator.py / imu_functions.py)
# ----------------------------------------------------------------------------------
_os_name = platform.system()
if _os_name == 'Linux':
    imu_dataset_path = os.path.expanduser('~/Data/datasets/DIP_IMU_and_Others/')
else:
    imu_dataset_path = os.path.expanduser('~/datasets/DIP_IMU_and_Others/')

WEIGHTS_DIR = 'data/weights'
os.makedirs(WEIGHTS_DIR, exist_ok=True)


# ==================================================================================
# 1. Model builders (Keras)
# ==================================================================================
def build_bi_rnn_model(input_dim=204, output_dim=72, hidden_units=256, num_layers=2,
                       learning_rate=1e-3):
    """
    DIP-IMU-style bidirectional RNN (LSTM cells). Maps a sequence of IMU frames
    [T, input_dim] to a sequence of SMPL pose parameters [T, output_dim].

    Reference architecture: Huang et al., "Deep Inertial Poser" [14] -- two stacked
    bidirectional LSTM layers. (We use an MSE objective for parity with the proposed
    MLP and the paper's loss; DIP-IMU's original log-likelihood objective can be
    substituted but is not necessary for this comparison.)
    """
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Bidirectional, LSTM, TimeDistributed, Dense
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers.legacy import Adam

    inputs = Input(shape=(None, input_dim))           # variable-length sequence
    x = inputs
    for _ in range(num_layers):
        x = Bidirectional(LSTM(hidden_units, return_sequences=True))(x)
    outputs = TimeDistributed(Dense(output_dim, activation='linear'))(x)
    model = Model(inputs=inputs, outputs=outputs, name='dip_birnn')
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model


def build_lstm_model(input_dim=204, output_dim=72, hidden_units=256, num_layers=2,
                     learning_rate=1e-3):
    """
    Unidirectional LSTM baseline. Same depth/width as the Bi-RNN but causal
    (no access to future frames), representing a simpler temporal model.
    """
    import tensorflow as tf
    from tensorflow.keras.layers import Input, LSTM, TimeDistributed, Dense
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers.legacy import Adam

    inputs = Input(shape=(None, input_dim))
    x = inputs
    for _ in range(num_layers):
        x = LSTM(hidden_units, return_sequences=True)(x)
    outputs = TimeDistributed(Dense(output_dim, activation='linear'))(x)
    model = Model(inputs=inputs, outputs=outputs, name='lstm')
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse', metrics=['mae'])
    return model


# ==================================================================================
# 2. Sequence loading (preserve per-file motion boundaries) + windowing
#    Mirrors pose_generator.get_data_chunks / process_datasets normalisation, but
#    keeps each recording as a separate sequence so temporal models train fairly.
# ==================================================================================
def _read_file_sequence(f):
    """Return (imu[L,204], gt[L,72]) for a single DIP-IMU .pkl file, NaN rows dropped."""
    d = pkl.load(open(f, 'rb'), encoding='latin1')
    imu_ori = d['imu_ori']                       # [L, 17, 3, 3]
    imu_acc = d['imu_acc']                        # [L, 17, 3]
    gt = np.asarray(d['gt'])                      # [L, 72]
    L = imu_ori.shape[0]
    imu_ori = np.reshape(imu_ori, [L, 17 * 9])    # 153
    imu_acc = np.reshape(imu_acc, [L, 17 * 3])    # 51
    imu = np.concatenate((imu_ori, imu_acc), axis=1)   # [L, 204]
    merged = np.concatenate((imu, gt), axis=1)         # [L, 276]
    keep = ~np.isnan(merged).any(axis=1)
    merged = merged[keep]
    imu = merged[:, :204]
    gt = merged[:, 204:]
    return imu, gt


def get_sequence_chunks(split='train'):
    """
    Load DIP-IMU as a LIST of per-recording sequences (boundaries preserved).

    Normalisation matches pose_generator.process_datasets:
      * orientation columns [:153] are left unchanged,
      * acceleration columns [153:204] are scaled by a MaxAbsScaler fit over ALL
        frames of the split.

    Returns
    -------
    seqs_imu : list of [L_f, 204] float arrays
    seqs_gt  : list of [L_f, 72]  float arrays
    """
    from sklearn.preprocessing import MaxAbsScaler

    path = os.path.join(imu_dataset_path, 'DIP_IMU')
    if split == 'train':
        subjects = ['s_01', 's_02', 's_03', 's_04', 's_05', 's_06', 's_07', 's_08']
    else:
        subjects = ['s_09', 's_10']

    files = []
    for s in subjects:
        sp = os.path.join(path, s)
        for fn in os.listdir(sp):
            if fn.endswith('.pkl'):
                files.append(os.path.join(sp, fn))

    seqs_imu, seqs_gt = [], []
    for f in files:
        imu, gt = _read_file_sequence(f)
        if imu.shape[0] > 0:
            seqs_imu.append(imu)
            seqs_gt.append(gt)

    # Fit MaxAbsScaler on acceleration over all frames of the split (as in process_datasets)
    all_acc = np.concatenate([s[:, 153:] for s in seqs_imu], axis=0)
    scaler = MaxAbsScaler().fit(all_acc)
    seqs_imu = [np.concatenate((s[:, :153], scaler.transform(s[:, 153:])), axis=1)
                for s in seqs_imu]
    return seqs_imu, seqs_gt


def make_windows(seq, T, stride=None):
    """Chop a [L, D] sequence into [num_win, T, D] non-overlapping (default) windows."""
    if stride is None:
        stride = T
    L = seq.shape[0]
    if L < T:
        return np.empty((0, T, seq.shape[1]), dtype=seq.dtype)
    starts = range(0, L - T + 1, stride)
    return np.stack([seq[s:s + T] for s in starts], axis=0)


def build_windowed_dataset(seqs_imu, seqs_gt, T, stride=None, batch_size=64, shuffle=True):
    """Build a tf.data.Dataset of (imu_window[T,204], gt_window[T,72]) pairs."""
    import tensorflow as tf
    Xs, Ys = [], []
    for imu, gt in zip(seqs_imu, seqs_gt):
        wx = make_windows(imu, T, stride)
        wy = make_windows(gt, T, stride)
        if wx.shape[0] > 0:
            Xs.append(wx)
            Ys.append(wy)
    X = np.concatenate(Xs, axis=0).astype(np.float32)
    Y = np.concatenate(Ys, axis=0).astype(np.float32)
    print('Windowed training set: X={}, Y={}'.format(X.shape, Y.shape))
    ds = tf.data.Dataset.from_tensor_slices((X, Y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(X), 50000))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def train_temporal(model_type='bi_rnn', T=20, epochs=50, batch_size=64,
                   hidden_units=256, num_layers=2):
    """
    Train a temporal pose network on CLEAN, properly-sequenced DIP-IMU data.
    Saves weights to data/weights/{model_type}_smpl.h5.
    """
    seqs_imu, seqs_gt = get_sequence_chunks('train')
    ds = build_windowed_dataset(seqs_imu, seqs_gt, T=T, batch_size=batch_size, shuffle=True)

    if model_type == 'bi_rnn':
        model = build_bi_rnn_model(hidden_units=hidden_units, num_layers=num_layers)
    elif model_type == 'lstm':
        model = build_lstm_model(hidden_units=hidden_units, num_layers=num_layers)
    else:
        raise ValueError("model_type must be 'bi_rnn' or 'lstm'")

    print('Training {} for {} epochs (T={})...'.format(model_type, epochs, T))
    model.fit(ds, epochs=epochs)
    out = os.path.join(WEIGHTS_DIR, '{}_smpl.h5'.format(model_type))
    model.save(out)
    print('Saved {} to {}'.format(model_type, out))
    return model


# ==================================================================================
# 3. Geodesic MPJAE (copied verbatim from pose_generator.py for a standalone module)
# ==================================================================================
def batch_rodrigues_numpy(rot_vecs):
    """Convert (N,3) axis-angle vectors to (N,3,3) rotation matrices (Rodrigues)."""
    theta = np.linalg.norm(rot_vecs, axis=1, keepdims=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        k = rot_vecs / theta
    k = np.nan_to_num(k)
    kx, ky, kz = k[:, 0], k[:, 1], k[:, 2]
    ct = np.cos(theta).squeeze()
    st = np.sin(theta).squeeze()
    vt = 1 - ct
    N = rot_vecs.shape[0]
    R = np.zeros((N, 3, 3))
    R[:, 0, 0] = ct + kx ** 2 * vt
    R[:, 1, 1] = ct + ky ** 2 * vt
    R[:, 2, 2] = ct + kz ** 2 * vt
    R[:, 0, 1] = kx * ky * vt - kz * st
    R[:, 0, 2] = kx * kz * vt + ky * st
    R[:, 1, 0] = kx * ky * vt + kz * st
    R[:, 1, 2] = ky * kz * vt - kx * st
    R[:, 2, 0] = kx * kz * vt - ky * st
    R[:, 2, 1] = ky * kz * vt + kx * st
    return R


def compute_geodesic_error(v1, v2):
    """Geodesic distance (degrees) between two batches of axis-angle vectors [N,3]."""
    R1 = batch_rodrigues_numpy(v1)
    R2 = batch_rodrigues_numpy(v2)
    R_diff = np.einsum('nij,nkj->nik', R1, R2)
    trace = np.trace(R_diff, axis1=1, axis2=2)
    val = (trace - 1.0) / 2.0
    val = np.clip(val, -1.0, 1.0)
    return np.degrees(np.arccos(val))


# ==================================================================================
# 4. MPJAE comparison across pose networks on the SAME distorted signals
# ==================================================================================
def _predict_poses(model, imu_flat, kind, T):
    """
    Run a pose network on a flat [N,204] IMU array, return [N,72] poses.
      * kind='mlp'    : per-frame prediction.
      * kind in {'lstm','bi_rnn'} : non-overlapping windows of length T.
    """
    if kind == 'mlp':
        return model.predict(imu_flat, verbose=0)          # [N, 72], per-frame
    # Temporal: window, predict, flatten back
    N = (imu_flat.shape[0] // T) * T
    x = imu_flat[:N].reshape(-1, T, imu_flat.shape[1]).astype(np.float32)
    p = model.predict(x, verbose=0)          # [num_win, T, 72]
    return p.reshape(-1, 72)


def mpjae_compare(quantization_range=range(4, 11), batch_size=100, T=20,
                  systems=('neural-receiver',), scenarios=('1p', '2p'),
                  ebno=5.0, reference='gt'):
    """
    Compute MPJAE for {MLP, LSTM, Bi-RNN} on identical distorted IMU signals.

    Parameters
    ----------
    reference : 'gt'    -> absolute MPJAE vs dataset ground-truth SMPL poses (default,
                           directly answers "how does pose accuracy compare").
                'clean' -> MPJAE of distorted-input prediction vs same network's
                           clean-input prediction (matches paper Fig. 10 methodology).

    Requires (already produced by your existing pipeline):
      * data/weights/mlp_smpl.h5        (pose_generator.py --train 1)
      * data/weights/lstm_smpl.h5       (temporal_receivers.train_temporal('lstm'))
      * data/weights/bi_rnn_smpl.h5     (temporal_receivers.train_temporal('bi_rnn'))
      * data/imu/ori_imu_<system>_<scn>_<q>_<ebno>.npy  (main.py --eval_mode 3)
      * data/imu/rec_imu_<system>_<scn>_<q>_<ebno>.npy
      * <imu_dataset_path>/processed_test.npz  (for reference='gt')

    Saves data/pltdata/mpjae_compare.npy and data/figures/mpjae_compare.pdf.
    """
    import tensorflow as tf
    import matplotlib.pyplot as plt

    nets = {
        'mlp':    ('data/weights/mlp_smpl.h5',    'mlp'),
        'lstm':   ('data/weights/lstm_smpl.h5',   'lstm'),
        'bi_rnn': ('data/weights/bi_rnn_smpl.h5', 'bi_rnn'),
    }
    loaded = {}
    for name, (path, kind) in nets.items():
        if os.path.exists(path):
            loaded[name] = (tf.keras.models.load_model(path), kind)
        else:
            print('WARNING: {} not found ({}), skipping.'.format(name, path))

    gt_all = None
    if reference == 'gt':
        test = np.load(os.path.join(imu_dataset_path, 'processed_test.npz'), allow_pickle=True)
        gt_all = np.asarray(test['gt'])          # [N_test, 72], frame-aligned with ori_imu

    results = {}   # results[net][system-scenario] = list over q
    for net in loaded:
        results[net] = {}

    for system in systems:
        for scn in scenarios:
            tag = '{}-{}'.format(system, scn)
            for net in loaded:
                results[net][tag] = []
            for q in quantization_range:
                ori = np.load('data/imu/ori_imu_{}_{}_{}_{}.npy'.format(system, scn, q, ebno))
                rec = np.load('data/imu/rec_imu_{}_{}_{}_{}.npy'.format(system, scn, q, ebno))
                n = min(len(ori), len(rec))
                ori, rec = ori[:n], rec[:n]
                gt = gt_all[:n] if reference == 'gt' else None

                for net, (model, kind) in loaded.items():
                    rec_pose = _predict_poses(model, rec, kind, T)
                    if reference == 'gt':
                        m = min(len(rec_pose), len(gt))
                        target = gt[:m]
                    else:  # 'clean'
                        clean_pose = _predict_poses(model, ori, kind, T)
                        m = min(len(rec_pose), len(clean_pose))
                        target = clean_pose[:m]
                    err = compute_geodesic_error(rec_pose[:m].reshape(-1, 3),
                                                 target.reshape(-1, 3))
                    results[net][tag].append(float(np.mean(err)))
                print('q={} {}: '.format(q, tag) +
                      ', '.join('{}={:.3f}'.format(k, results[k][tag][-1]) for k in loaded))

    os.makedirs('data/pltdata', exist_ok=True)
    os.makedirs('data/figures', exist_ok=True)
    np.save('data/pltdata/mpjae_compare.npy', results)

    # Fig. 13: one (a) 2P | (b) 1P figure per system comparing the pose networks.
    from plot_style import (NETWORK_STYLES, plot_series, style_axis, panel_tag,
                            finalize_dual, FIGSIZE_DUAL)
    qs = list(quantization_range)
    # Fixed panel order (a) 2P | (b) 1P regardless of the `scenarios` argument.
    panel_scns = [s for s in ('2p', '1p') if s in scenarios]
    tag_map = {'2p': '(a) 2P', '1p': '(b) 1P'}
    for system in systems:
        fig, axes = plt.subplots(1, len(panel_scns), figsize=FIGSIZE_DUAL,
                                 sharey=True, squeeze=False)
        axes = axes[0]
        for axi, scn in zip(axes, panel_scns):
            key = '{}-{}'.format(system, scn)
            for net in loaded:
                plot_series(axi, qs, results[net][key], NETWORK_STYLES[net])
            style_axis(axi, 'Quantization level (bits)', None, legend=False)
            panel_tag(axi, tag_map.get(scn, scn))
        fn = 'data/figures/mpjae_compare_{}.pdf'.format(system)
        finalize_dual(fig, axes, r'MPJAE ($^\circ$)', fn, ncol=3)
    return results


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Temporal IMU-receiver baselines (Bi-RNN / LSTM)')
    ap.add_argument('--train', type=str, default=None, choices=['bi_rnn', 'lstm'],
                    help='Train a temporal model on clean DIP-IMU sequences')
    ap.add_argument('--num_ep', type=int, default=50)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--win', type=int, default=20, help='Sequence window length T')
    ap.add_argument('--compare', type=int, default=0,
                    help='Run MPJAE comparison across {MLP, LSTM, Bi-RNN}')
    ap.add_argument('--reference', type=str, default='gt', choices=['gt', 'clean'])
    ap.add_argument('--ebno', type=float, default=5.0)
    args = ap.parse_args()

    if args.train is not None:
        train_temporal(model_type=args.train, T=args.win,
                       epochs=args.num_ep, batch_size=args.batch)
    if args.compare:
        mpjae_compare(quantization_range=range(4, 11), batch_size=args.batch,
                      T=args.win, ebno=args.ebno, reference=args.reference)
