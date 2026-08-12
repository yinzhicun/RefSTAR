import os.path as osp
from basicsr.train import train_pipeline

import refstar.archs
import refstar.data
import refstar.models
import refstar.losses

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
