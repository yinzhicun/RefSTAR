'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
import os
import cv2
import torch
import numpy as np
import random
from .parse_net import ParseNet
from copy import deepcopy
import torch.nn.functional as F

_PACKAGE_MODELS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class FaceParse(object):
    def __init__(self, base_dir=_PACKAGE_MODELS_DIR, model='ParseNet-latest', device='cuda'):
        self.mfile = os.path.join(base_dir, 'weights', model+'.pth')
        self.size = 512
        self.device = device

        '''
        0: 'background' 1: 'skin'   2: 'nose'
        3: 'eye_g'  4: 'l_eye'  5: 'r_eye'
        6: 'l_brow' 7: 'r_brow' 8: 'l_ear'
        9: 'r_ear'  10: 'mouth' 11: 'u_lip'
        12: 'l_lip' 13: 'hair'  14: 'hat'
        15: 'ear_r' 16: 'neck_l'    17: 'neck'
        18: 'cloth'
        '''
        self.MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0]
        self.load_model()

    def load_model(self):
        self.faceparse = ParseNet(self.size, self.size, 32, 64, 19, norm_type='bn', relu_type='LeakyReLU', ch_range=[32, 256])
        self.faceparse.load_state_dict(torch.load(self.mfile, map_location=torch.device('cpu')))
        self.faceparse.to(self.device)
        self.faceparse.eval()

    def process(self, im):
        h, w = im.shape[:2]
        if h==self.size and w==self.size:
            pass
        else:
            im = cv2.resize(im, (self.size, self.size))
        imt = self.img2tensor(im)
        pred_mask, sr_img_tensor = self.faceparse(imt)
        mask = self.tenor2mask(pred_mask)

        return mask

    def process_tensor(self, imt, MASK_COLORMAP_=None):
        imt = F.interpolate(imt, (self.size, self.size))
        pred_mask, sr_img_tensor = self.faceparse(imt)

        MASK_COLORMAP = [0, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 0, 0, 0, 0]
        mask = pred_mask.argmax(dim=1)
        arr = range(1, 13)
        nums = range(0, 5)
        random_num = random.choice(nums)

        if random_num == 0:
            pass
        else:
            for x in random.sample(arr, random_num):
                MASK_COLORMAP[x] = 0

        if MASK_COLORMAP_:
            MASK_COLORMAP = MASK_COLORMAP_
        else:
            pass
        for idx, color in enumerate(MASK_COLORMAP):

            mask = torch.where(mask==idx, color, mask)
        mask = mask.unsqueeze(1).to(torch.uint8) / 255

        return mask, MASK_COLORMAP

    def img2tensor(self, img):
        img = img[..., ::-1]
        img = img / 255. * 2 - 1
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(self.device)
        return img_tensor.float()

    def tenor2mask(self, tensor):
        if len(tensor.shape) < 4:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[1] > 1:
            tensor = tensor.argmax(dim=1) 

        tensor = tensor.squeeze(1).data.cpu().numpy()
        color_maps = []
        for t in tensor:
            tmp_img = np.zeros(tensor.shape[1:])
            for idx, color in enumerate(self.MASK_COLORMAP):
                tmp_img[t == idx] = color
            color_maps.append(tmp_img.astype(np.uint8))
        return color_maps