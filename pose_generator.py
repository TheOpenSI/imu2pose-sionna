import os
import platform
import argparse
import pickle as pkl
import torch
import smplx
import numpy as np
import pyvista as pv
import tensorflow as tf
from sklearn.preprocessing import MaxAbsScaler
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers.legacy import Adam
from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer

# File paths to load SMPL model and IMU dataset
os_name = platform.system()
if os_name == 'Linux':
    body_model_path = os.path.expanduser('~/Data/datasets/smpl/smpl/SMPL_MALE.pkl') 
    imu_dataset_path = os.path.expanduser('~/Data/datasets/DIP_IMU_and_Others/') 
else:
    body_model_path = os.path.expanduser('~/datasets/SMPLs/models/smpl/SMPL_MALE.pkl') 
    imu_dataset_path = os.path.expanduser('~/datasets/DIP_IMU_and_Others/') 
    
if not os.path.isdir('data/weights'):
    os.mkdir('data/weights')

def build_mlp_model(input_dim, output_dim):
    """
    Builds a simple MLP model for predicting SMPL pose parameters from IMU data.

    Args:
        input_dim (int): Number of input features (e.g., 204 for IMU data).
        output_dim (int): Number of output features (e.g., 72 for SMPL pose parameters).

    Returns:
        keras.Model: Compiled MLP model.
    """
    inputs = Input(shape=(input_dim,))
    x = Dense(512, activation='relu')(inputs)  # First hidden layer
    x = Dense(256, activation='relu')(x)      # Second hidden layer
    x = Dense(128, activation='relu')(x)      # Third hidden layer
    outputs = Dense(output_dim, activation='linear')(x)  # Output layer
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def get_data_chunks(pkl_files):
    """
    Return the IMU dataset. The output format is a tuple of imu with shape (seq_len, 1, 204)
    and gt with shape (seq_len, 1, 72)
    """
    imu_out = []
    gt_out = []
    for f in pkl_files:
        imu_ori_data = pkl.load(open(f, 'rb'), encoding='latin1')['imu_ori']  # [seq_len, 17, 3, 3]
        imu_acc_data = pkl.load(open(f, 'rb'), encoding='latin1')['imu_acc']   # [seq_len, 17, 3]
        gt_data = pkl.load(open(f, 'rb'), encoding='latin1')['gt']  # [seq_len, 72]
        seq_len = imu_ori_data.shape[0]

        imu_ori_data = np.reshape(imu_ori_data, [seq_len, 17 * 9])
        imu_acc_data = np.reshape(imu_acc_data, [seq_len, 17 * 3])
        # print('One ori sample: {}'.format(imu_acc_data[0, :]))
        imu_data = np.concatenate((imu_ori_data, imu_acc_data), axis=1)
        merged_data = np.concatenate((imu_data, gt_data), axis=1)  # [seq_len, 276]
        # count number of Nan entries
        nan_mask = np.isnan(merged_data)
        row_nan_mask = np.any(nan_mask, axis=1)
        num_nan = np.count_nonzero(row_nan_mask)
        # discard entries with Nan values
        print('-- discard {} Nan entries out of {} entries----'.format(num_nan, seq_len))
        clean_merged_data = merged_data[~np.isnan(merged_data).any(axis=1)]
        new_seq_len = seq_len - num_nan
        # split again
        clean_imu_data = clean_merged_data[:, :17 * 12]  # [seq_len, 204]
        clean_gt_data = clean_merged_data[:, 17 * 12:]  # [seq_len, 72]
        # output data [1, nb_imus, nb_imu_features]
        print('--- file: {}, seq_len: {}, imu_data: {}, gt_data: {}'.format(
            f, new_seq_len, clean_imu_data.shape, clean_gt_data.shape))

        for i in range(new_seq_len):
            imu_out.append(clean_imu_data[i])  # [1, 204]
            gt_out.append(clean_gt_data[i])  # [1, 72]

    return imu_out, gt_out

def process_datasets():
    "Process training and testing IMU datasets"
    path = imu_dataset_path + 'DIP_IMU/'
    train_subjects = ['s_01', 's_02', 's_03', 's_04', 's_05', 's_06', 's_07', 's_08']
    test_subjects = ['s_09', 's_10']
    train_files, test_files = [], []
    
    print('Process IMU training data \n')
    for s in train_subjects:
            subject_path = os.path.join(path, '{}/'.format(s))
            for f in os.listdir(subject_path):
                if f.endswith('.pkl'):
                    train_files.append(os.path.join(subject_path, f))
                    
    scaler = MaxAbsScaler()
    imu_train, gt_train = get_data_chunks(train_files)
    imu_train = np.squeeze(imu_train)
    seq_len = len(imu_train)
    ori_train = imu_train[:, :153]
    # normalize acceleration
    acc_train = imu_train[:, 153:]
    acc_train_abs = scaler.fit_transform(acc_train)
    imu_train = np.concatenate((ori_train, acc_train_abs), axis=1)
    imu_train = np.reshape(imu_train, [seq_len, 1, 204])
    print('Save train dataset to {}processed_train.npz'.format(imu_dataset_path))
    np.savez(os.path.join(imu_dataset_path, 'processed_train.npz'), imu=imu_train, gt=gt_train)
    
    print('Process IMU testing data \n')
    for s in test_subjects:
            subject_path = os.path.join(path, '{}/'.format(s))
            for f in os.listdir(subject_path):
                if f.endswith('.pkl'):
                    test_files.append(os.path.join(subject_path, f))
    scaler = MaxAbsScaler()
    imu_test, gt_test = get_data_chunks(test_files)
    imu_test = np.squeeze(imu_test)
    seq_len = len(imu_test)
    ori_test = imu_test[:, :153]
    # normalize acceleration
    acc_test = imu_test[:, 153:]
    acc_test_abs = scaler.fit_transform(acc_test)
    imu_test = np.concatenate((ori_test, acc_test_abs), axis=1)
    imu_test = np.reshape(imu_test, [seq_len, 1, 204])
    print('Save test dataset to {}processed_test.npz'.format(imu_dataset_path))
    np.savez(os.path.join(imu_dataset_path, 'processed_test.npz'), imu=imu_test, gt=gt_test)

def load_datasets(batch_size=32, shuffle=True):
    """
    Load train and test datasets and return tf.data.Dataset loaders.

    Args:
        batch_size (int): Batch size for training and testing datasets.
        shuffle (bool): Whether to shuffle the training dataset.

    Returns:
        train_loader (tf.data.Dataset): Train dataset loader.
        test_loader (tf.data.Dataset): Test dataset loader.
    """
    train_path = imu_dataset_path + 'processed_train.npz'
    test_path = imu_dataset_path + 'processed_test.npz'
    train_data = np.load(train_path, allow_pickle=True)
    test_data = np.load(test_path, allow_pickle=True)
    
    # Print datasets
    keys = list(train_data.keys())
    print('train_data keys: {}'.format(keys))
    for k in keys:
        if k != 'statistics':
            print('key: {}, shape: {}, dtype: {}, sample: {}'.format(k, train_data[k].shape, train_data[k].dtype, train_data[k][0].shape))

    keys = list(test_data.keys())
    print('test_data keys: {}'.format(keys))
    for k in keys:
        if k != 'statistics':
            print('key: {}, shape: {}, dtype: {}, sample: {}'.format(k, test_data[k].shape, test_data[k].dtype, test_data[k][0].shape))
    
    # Extract IMU (input) and GT (target) data
    X_train = train_data['imu'].reshape(-1, 204)  # Reshape (seq_len, 1, 204) to (seq_len, 204)
    y_train = train_data['gt']                    # Shape: (seq_len, 72)
    X_test = test_data['imu'].reshape(-1, 204)   # Reshape (seq_len, 1, 204) to (seq_len, 204)
    y_test = test_data['gt']                     # Shape: (seq_len, 72)
    
    # Print dataset shapes and examples
    print(f"Train Data: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"Test Data: X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    # Create TensorFlow Dataset objects
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    
    # Shuffle, batch, and prefetch for training dataset
    if shuffle:
        train_dataset = train_dataset.shuffle(buffer_size=len(X_train))
    train_dataset = train_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    # Batch and prefetch for test dataset
    test_dataset = test_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_dataset, test_dataset
            
def train_mlp(train_loader, input_dim, output_dim, epochs=10, batch_size=64):
    # Build the model
    # Build and compile the model
    model = build_mlp_model(input_dim, output_dim)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    
    # Train the model
    print("Start training...")
    history = model.fit(
        train_loader,
        epochs=epochs
    )
    file_name = 'data/weights/mlp_smpl.h5'
    print("Save MLP_SMPL model to {}".format(file_name))
    model.save(file_name)
    
    return model, history

def render_mesh(vertices, faces, animation=False, color=None):
    if color == 'gt':
        mesh_color = [255.0 / 255, 51.0 / 255, 51.0 / 255]
    elif color == 'vae':
        mesh_color = [224.0 / 255, 224.0 / 255, 225.0 / 255]
    elif color == 'lasso':
        mesh_color = [51.0 / 255, 153.0 / 255, 255.0 / 255]
    elif color == 'lasso-opt':
        mesh_color = [102.0 / 255, 255.0 / 255, 102.0 / 255]
    elif color == 'dip':
        mesh_color = [255.0 / 255, 153.0 / 255, 255.0 / 255]
    else:
        mesh_color = [224.0 / 255, 224.0 / 255, 225.0 / 255]

    # Convert faces to PyVista-compatible format
    # Each face must be preceded by the number of vertices (e.g., [3, v1, v2, v3])
    faces_pv = np.hstack([[3] + list(face) for face in faces])
    
    seq_len = vertices.shape[0]  # Number of frames in the animation
    # Create the initial mesh
    mesh = pv.PolyData(vertices[0], faces_pv)

    # Set up PyVista plotter
    pl = pv.Plotter()
    actor = pl.add_mesh(mesh, color=mesh_color, show_edges=False, smooth_shading=True)

    # Animation callback
    def callback(step):
        current_frame = step % seq_len  # Loop through the frames
        mesh.points = vertices[current_frame]  # Update vertex positions
        pl.render()  # Trigger render after updating the mesh

    # Add timer event for animation
    max_steps = seq_len * 3 if animation else 1
    pl.add_timer_event(max_steps=max_steps, duration=1000, callback=callback)
    # Camera position
    cpos = [(0.0, 0.0, 10.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    pl.show(cpos=cpos)
    

def aitview(pose, faces, color):
    """Create animation with AitViewer

    Args:
        pose (np.float64): SMPL's pose parameter (seq_len, 72)
        faces (np.float64): SMPL's face parameter (seq_len, 6890, 3)
        color (list): RGBA color
    """
    C.update_conf({"run_animations": True,
                   'smplx_models': body_model_path,
                   'export_dir': 'data'
                   })
    
    # Downsample to 30 Hz.
    pose = pose[::2]
    betas = torch.zeros((pose.shape[0], 10)).float().to(C.device)
    smpl_layer = SMPLLayer(model_type="smpl", gender='male', device=C.device)
    _, joints = smpl_layer(
        poses_body=pose[:, 3:].to(C.device),
        poses_root=pose[:, :3].to(C.device),
        betas=betas,
    )
    
    smpl_seq = SMPLSequence(poses_body=pose[:, 3:], smpl_layer=smpl_layer, poses_root=pose[:, :3])
    smpl_seq.mesh_seq.color = smpl_seq.mesh_seq.color[:3] + (1.0,)
    
    # Change color for SMPL model
    if smpl_seq.mesh_seq.face_colors is None:
        num_frames = pose.shape[0]  # N frames
        num_faces = faces.shape[0]  # F faces
        smpl_seq.mesh_seq.face_colors = np.full((num_frames, num_faces, 4), color)  # Red color (R=0.6, G=0, B=0, A=1.0)
        

    # Add everything to the scene and display at 30 fps.
    v = Viewer()
    v.playback_fps = 30.0

    v.scene.add(smpl_seq)
    v.run()
    
def smpl_forward(model, pose, batch_size):
    """Perfom batch forward of SMPL pose parameter (72) to vertices of the 3D meshes

    Args:
        model (SMPL model): SMPL model
        pose (torch.float64): SMPL's pose parameter (72)
        batch_size (int): Batch size

    Returns:
        torch.float64: vertices and joints
    """
    global_orient = pose[:, :3].reshape(batch_size, 3)
    body_pose = pose[:, 3:].reshape(batch_size, 69)
    # print('body_pose.shape: {}'.format(body_pose.shape))
    res = model(global_orient=global_orient, body_pose=body_pose)
    vertices = res.vertices.detach().cpu().numpy().squeeze()
    joints = res.joints.detach().cpu().numpy().squeeze()

    return vertices, joints

def generate_pose_animation(data_loader, color):
    file_name = 'data/weights/mlp_smpl.h5'
    print('Load MLP_SMPL model from: {}'.format(file_name))
    model = tf.keras.models.load_model(file_name)
    imu, gt = next(iter(data_loader))
    print('imu shape: {}'.format(imu.shape))
    # load body model
    body_model = smplx.create(model_path=body_model_path, model_type='smpl', gender='male', dtype=torch.float64)
    
    pose = model(imu)
    pose = pose.numpy()
    pose = pose[::2]  # downsample
    print('pose shape: {}'.format(pose.shape))
   
    pose = torch.from_numpy(pose)  # [b, 1, 72]
    faces = body_model.faces
    batsz = pose.shape[0]
    
    vertices, _ = smpl_forward(body_model, pose, batsz)
    print('vertices: {}'.format(vertices.shape))
    # Visualize
     # Enable this line to have better animation
    aitview(pose=pose, faces=faces, color=color)
    # render_mesh(vertices, faces, animation=True, color='vae')
    

if __name__ == '__main__':
    # Arg parser
    parser = argparse.ArgumentParser(description='Pose generator script')
    parser.add_argument('--process', type=int, help='Pre-process datasets', default=1)
    parser.add_argument('--train', type=int, help='Train MLP model from scratch', default=0)
    parser.add_argument('--num_ep', type=int, help='Number of training epochs', default=50)
    parser.add_argument('--batch', type=int, help='Batch size', default=100)
    parser.add_argument('--ebno', type=float, 
                        help='Ebno (dB) value when running SMPL simulation at the receiver', 
                        default=5.0,
                        )
    parser.add_argument('--quantz', type=int, help='Quantization level', default=6)
    args = parser.parse_args()
    
    if args.process:
        process_datasets()
        
    # Build datasets
    train_loader, _ = load_datasets(batch_size=args.batch)
    
    if args.train:
        model, history = train_mlp(
            train_loader=train_loader,
            input_dim=204, output_dim=72,
            epochs=args.num_ep, batch_size=args.batch
            )
    
    batch_size = 5000
    color_list = [
        [140.0 / 255, 140.0 / 255, 140.0 / 255, 1.0],  # Ground truth - even darker gray
        [120.0 / 255, 140.0 / 255, 180.0 / 255, 1.0],  # Neural-receiver-2p - muted gray-blue
        [100.0 / 255, 120.0 / 255, 160.0 / 255, 1.0],  # Neural-receiver-1p - darker and slightly cooler
        [200.0 / 255, 140.0 / 255, 120.0 / 255, 1.0],  # LS-estimation-2p - darker soft orange
        [180.0 / 255, 120.0 / 255, 100.0 / 255, 1.0],   # LS-estimation-1p - less bright
        [120.0 / 255, 160.0 / 255, 120.0 / 255, 1.0]  # Perfect-CSI - muted dark green
    ]
    
    # visualize TX data
    tx_data = np.load('data/imu/ori_imu_{}_{}_{}.npy'.format('baseline-perfect-csi', args.quantz, args.ebno))
    # qtz_data = np.load('data/imu/qtz_imu_{}_{}_{}_{}.npy'.format(system, args.quantz, args.ebno))
    X_test = tx_data
    print(f"Test Data: X_test shape: {X_test.shape}")
    y_test = None
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_dataset = test_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    # screenshot frames: 173, 510
    generate_pose_animation(test_dataset, color_list[0])

    # visualize RX data
    for system in ['neural-receiver_2p', 'neural-receiver_1p', 'baseline-ls-estimation_2p', 'baseline-ls-estimation_1p' , 'baseline-perfect-csi']: 
        rx_data = np.load('data/imu/rec_imu_{}_{}_{}.npy'.format(system, args.quantz, args.ebno))
        print('rx data shape: {}'.format(rx_data.shape))
        if system == 'neural-receiver_2p':
            color = color_list[1]
        elif system == 'neural-receiver_1p':
            color = color_list[2]
        elif system == 'baseline-ls-estimation_2p':
            color = color_list[3]
        elif system == 'baseline-ls-estimation_1p':
            color = color_list[4]
        elif system == 'baseline-perfect-csi':
            color = color_list[5]
            
        X_test = rx_data
        print(f"Test Data: X_test shape: {X_test.shape}")
        y_test = None
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test_dataset = test_dataset.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
        # screenshot frames: 173, 510
        generate_pose_animation(test_dataset, color)
        del rx_data, test_dataset