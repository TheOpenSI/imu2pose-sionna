import os
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.viewer import Viewer
from aitviewer.configuration import CONFIG as C

if __name__ == '__main__':
    C.update_conf({
        'smplx_models': os.path.expanduser('/Users/hieu/datasets/SMPLs/models')
    })
    v = Viewer()
    v.scene.add(SMPLSequence.t_pose())
    v.run()