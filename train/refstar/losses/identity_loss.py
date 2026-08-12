import os
import torch
from torch import nn as nn

from basicsr.utils.registry import LOSS_REGISTRY
from .ir_backbone import Backbone


@LOSS_REGISTRY.register()
class IDLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super(IDLoss, self).__init__()
        self.loss_weight = loss_weight
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(os.path.join(os.path.dirname(__file__), 'weights', 'model_ir_se50.pth')))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

    def extract_feats(self, x):
        x = x[:, :, 70:446, 64:440]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y):
        n_samples = y_hat.shape[0]
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()

        loss = 0
        count = 0

        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target
            count += 1

        return self.loss_weight * loss / count
