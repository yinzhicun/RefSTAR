import os
import cv2
import torch
import numpy as np
import random
from parse_model import ParseNet
from copy import deepcopy
import torch.nn.functional as F

class FaceParse(object):
    def __init__(self, base_dir='./', model='model_final', device='cuda'):
        #self.mfile = os.path.join(base_dir, 'res', model+'.pth')
        self.mfile = './pretrained_models/RefSel/net_g_35000.pth'
        self.size = 512
        self.device = device


        atts = ['background','mask']

        self.MASK_COLORMAP = [0, 255]
        self.load_model()

    def load_model(self):
        self.faceparse = ParseNet(self.size, self.size, 32, 64, 2, norm_type='bn', relu_type='LeakyReLU', ch_range=[32, 256])
        try:
            self.faceparse.load_state_dict(torch.load(self.mfile, map_location=self.device)['params'], strict=True)
        except:
            self.faceparse.load_state_dict(torch.load(self.mfile, map_location=self.device), strict=True)

        self.faceparse.to(self.device)
        self.faceparse.eval()

    def process(self, img_input, img_ref):
        # h, w = img_input.shape[:2]
        # if h==self.size and w==self.size:
        #     pass
        # else:
        #     img_input = cv2.resize(img_input, (self.size, self.size))
        #     img_ref = cv2.resize(img_ref, (self.size, self.size))
        # imt_input = self.img2tensor(img_input)
        # imt_ref = self.img2tensor(img_ref)
        c_img = torch.cat((img_input,img_ref),dim=1)
        print(c_img.shape)
        pred_mask, sr_img_tensor = self.faceparse(c_img)
        # mask = self.tenor2mask(pred_mask)
        mask = F.softmax(pred_mask, dim=1)
        return mask[:,0,:,:]

    def process_tensor(self, imt):
        imt = F.interpolate(imt.flip(1)*2-1, (self.size, self.size))
        pred_mask, sr_img_tensor = self.faceparse(imt)

        mask = pred_mask.argmax(dim=1)
        for idx, color in enumerate(self.MASK_COLORMAP):
            mask = torch.where(mask==idx, color, mask)
        #mask = mask.repeat(3, 1, 1).unsqueeze(0) #.cpu().float().numpy()
        mask = mask.unsqueeze(0)

        return mask

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
            #tmp_img = np.zeros(tensor.shape[1:] + (3,))
            tmp_img = np.zeros(tensor.shape[1:])
            for idx, color in enumerate(self.MASK_COLORMAP):
                tmp_img[t == idx] = color
            color_maps.append(tmp_img.astype(np.uint8))
        return color_maps

