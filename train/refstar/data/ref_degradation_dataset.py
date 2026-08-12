import cv2
import math
import numpy as np
import os
import os.path as osp
import random
import time
import torch
from PIL import Image
from refstar.data.degradations import circular_lowpass_kernel, random_mixed_kernels
from basicsr.data.transforms import augment
from basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from basicsr.data.data_util import paths_from_folder
from basicsr.utils.registry import DATASET_REGISTRY
from torch.utils import data as data
from torchvision.transforms.functional import (adjust_brightness, adjust_contrast, adjust_hue, adjust_saturation,
                                               normalize)
@DATASET_REGISTRY.register()
class RefDegradationDataset(data.Dataset):
    """FFHQ dataset for GFPGAN.

    It reads high resolution images, and then generate low-quality (LQ) images on-the-fly.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            io_backend (dict): IO backend type and other kwarg.
            mean (list | tuple): Image mean.
            std (list | tuple): Image std.
            use_hflip (bool): Whether to horizontally flip.
            Please see more options in the codes.
    """

    def __init__(self, opt):
        super(RefDegradationDataset, self).__init__()

        self.opt = opt
        self.file_client = None
        self.io_backend_opt = self.opt['io_backend']

        self.id_folder = self.opt['dataroot_id']
        self.lm_id_folder = self.opt['dataroot_lm_id']
        self.id_names = os.listdir(self.id_folder)
        self.id_folders = {id_name: os.listdir(osp.join(self.id_folder, id_name)) for id_name in self.id_names}

      
        if self.io_backend_opt['type'] == 'lmdb':
            raise NotImplementedError('lmdb is not supported in DMDNet.')
            self.io_backend_opt['db_paths'] = self.gt_folder
            if not self.gt_folder.endswith('.lmdb'):
                raise ValueError("'dataroot_gt' should end with '.lmdb', "f'but received {self.gt_folder}')
            with open(osp.join(self.gt_folder, 'meta_info.txt')) as fin:
                self.paths = [line.split('.')[0] for line in fin]
        else:
            self.id_paths = paths_from_folder(self.id_folder)

        # blur settings for the first degradation
        self.blur_kernel_size = opt['blur_kernel_size']
        self.kernel_list = opt['kernel_list']
        self.kernel_prob = opt['kernel_prob']  # a list for each kernel probability
        self.blur_sigma = opt['blur_sigma']
        self.betag_range = opt['betag_range']  # betag used in generalized Gaussian blur kernels
        self.betap_range = opt['betap_range']  # betap used in plateau blur kernels
        self.sinc_prob = opt['sinc_prob']  # the probability for sinc filters

        try:
            self.motion_stdev = opt['motion_stdev']
            self.motion_mode_list = opt['motion_mode_list']
            self.motion_mode_prob = opt['motion_mode_prob']
        except KeyError:
            print('KeyError [motion 1]')

        # blur settings for the second degradation
        self.blur_kernel_size2 = opt['blur_kernel_size2']
        self.kernel_list2 = opt['kernel_list2']
        self.kernel_prob2 = opt['kernel_prob2']
        self.blur_sigma2 = opt['blur_sigma2']
        self.betag_range2 = opt['betag_range2']
        self.betap_range2 = opt['betap_range2']
        self.sinc_prob2 = opt['sinc_prob2']

        try:
            self.motion_stdev2 = opt['motion_stdev2']
            self.motion_mode_list2 = opt['motion_mode_list2']
            self.motion_mode_prob2 = opt['motion_mode_prob2']
        except KeyError:
            print('KeyError [motion 2]')

        # a final sinc filter
        self.final_sinc_prob = opt['final_sinc_prob']

        self.kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
        self.pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor[10, 10] = 1


        










    def get_part_location(self, landmarkpath, gt_path, downscale=1, w_offset=0, h_offset=0, resize=1024, flip=0):
        Landmarks = []
        Map_LE_B = list(np.hstack((range(17,22), range(36,42))))
        Map_RE_B = list(np.hstack((range(22,27), range(42,48))))
        Map_LE = list(range(36,42))
        Map_RE = list(range(42,48))
        Map_NO = list(range(29,36))
        Map_MO = list(range(48,68))

        with open(os.path.join(landmarkpath),'r') as f:
            for line in f:
                tmp = [np.float64(i) for i in line.split(' ') if i != '\n']
                Landmarks.append(tmp)

        Landmarks = np.array(Landmarks) / downscale # 512 * 512
        if flip:
            Landmarks[:,0] = 512 * resize / 512 - Landmarks[:,0]
            Map_LE_B, Map_RE_B = Map_RE_B, Map_LE_B
            Map_LE, Map_RE = Map_RE, Map_LE

        Landmarks = Landmarks - [w_offset, h_offset]
        Landmarks[Landmarks > (504 * resize / 512)]= 504 * resize / 512
        Landmarks[Landmarks < (8 * resize / 512)]= 8 * resize / 512
        
        #left eye
        Mean_LE = np.mean(Landmarks[Map_LE],0)
        L_LE1 = Mean_LE[1] - np.min(Landmarks[Map_LE_B,1])
        L_LE1 = L_LE1 * 1.6
        L_LE2 = L_LE1 / 1.9
        L_LE_xy = L_LE1 + L_LE2
        L_LE_lt = [L_LE_xy/2, L_LE1]
        L_LE_rb = [L_LE_xy/2, L_LE2]
        Location_LE = np.hstack((Mean_LE - L_LE_lt + 1, Mean_LE + L_LE_rb)).astype(int)

        #right eye
        Mean_RE = np.mean(Landmarks[Map_RE],0)
        L_RE1 = Mean_RE[1] - np.min(Landmarks[Map_RE_B,1])
        L_RE1 = L_RE1 * 1.6
        L_RE2 = L_RE1 / 1.9
        L_RE_xy = L_RE1 + L_RE2
        L_RE_lt = [L_RE_xy/2, L_RE1]
        L_RE_rb = [L_RE_xy/2, L_RE2]
        Location_RE = np.hstack((Mean_RE - L_RE_lt + 1, Mean_RE + L_RE_rb)).astype(int)

        #nose
        Mean_NO = np.mean(Landmarks[Map_NO],0)
        if flip:
            L_NO1 =( np.max([Mean_NO[0] - Landmarks[35][0], Landmarks[31][0] - Mean_NO[0]])) * 1.25
            L_NO2 = (Landmarks[33][1] - Mean_NO[1]) * 1.1
        else:
            L_NO1 =( np.max([Mean_NO[0] - Landmarks[31][0], Landmarks[35][0] - Mean_NO[0]])) * 1.25
            L_NO2 = (Landmarks[33][1] - Mean_NO[1]) * 1.1
        L_NO_xy = L_NO1 * 2
        L_NO_lt = [L_NO_xy/2, L_NO_xy - L_NO2]
        L_NO_rb = [L_NO_xy/2, L_NO2]
        Location_NO = np.hstack((Mean_NO - L_NO_lt + 1, Mean_NO + L_NO_rb)).astype(int)
        
        #mouth
        Mean_MO = np.mean(Landmarks[Map_MO],0)
        L_MO = np.max((np.max(np.max(Landmarks[Map_MO],0) - np.min(Landmarks[Map_MO],0))/2,16)) * 1.2
        MO_O = Mean_MO - L_MO + 1
        MO_T = Mean_MO + L_MO
        MO_T[MO_T> (510 * resize / 512)] = (510 * resize / 512)
        Location_MO = np.hstack((MO_O, MO_T)).astype(int)
        return torch.cat([torch.FloatTensor(Location_LE).unsqueeze(0), torch.FloatTensor(Location_RE).unsqueeze(0), torch.FloatTensor(Location_NO).unsqueeze(0), torch.FloatTensor(Location_MO).unsqueeze(0)], dim=0), torch.from_numpy(Landmarks)

    def __getitem__(self, index):
        
        if self.file_client is None:
            self.file_client = FileClient(self.io_backend_opt.pop('type'), **self.io_backend_opt)

        id_name = random.choice(self.id_names)
        id_folder = self.id_folders[id_name]
        id_folder = random.sample(id_folder, k=2)
        gt_path = id_folder[0]
        name_gt = osp.basename(gt_path)[:-4]
        lm_gt_path = osp.join(self.lm_id_folder, id_name, name_gt+'.txt')
        img_bytes = self.file_client.get(osp.join(self.id_folder, id_name, gt_path))
        img_gt = imfrombytes(img_bytes, float32=False) # BGR [0, 1]

        ref_path= id_folder[1]
        name_ref = osp.basename(ref_path)[:-4]
        lm_ref_path = osp.join(self.lm_id_folder, id_name, name_ref+'.txt')
        img_bytes = self.file_client.get(osp.join(self.id_folder, id_name, ref_path))
        img_ref = imfrombytes(img_bytes, float32=False) # BGR [0, 1]

        flip = self.opt['use_hflip']
        flip = flip and random.random() < 0.5
        HQ_Loc, HQ_Land = self.get_part_location(lm_gt_path, gt_path, resize=self.opt['gt_size'], flip=flip)
        HQ_Ref_Loc, HQ_Ref_Land = self.get_part_location(lm_ref_path, ref_path, resize=self.opt['gt_size'], flip=flip)

        

        # crop or pad to 400
        # TODO: 400 is hard-coded. You may change it accordingly
        h, w = img_gt.shape[0:2]
        target_size = self.opt["gt_size"]

        if h == target_size and w == target_size:
            img_gt = img_gt / 255.
            img_ref = img_ref / 255.

        else:
            img_gt = Image.fromarray(img_gt)
            img_gt = img_gt.resize((target_size, target_size), Image.BICUBIC)
            img_gt = np.array(img_gt)
            img_gt = img_gt / 255.

            img_ref = Image.fromarray(img_ref)
            img_ref = img_ref.resize((target_size, target_size), Image.BICUBIC)
            img_ref = np.array(img_ref)
            img_ref = img_ref / 255.

        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None,
                motion_stdev=self.motion_stdev,
                motion_mode_list=self.motion_mode_list,
                motion_mode_prob=self.motion_mode_prob)
        # pad kernel
        h, w = kernel.shape
        assert h == w and h % 2 == 1
        if h > 41:
            kernel = kernel[(h-41)//2:h-(h-41)//2, (h-41)//2:h-(h-41)//2]
            kernel /= kernel.sum()
            kernel = np.ascontiguousarray(kernel)
        elif h < 41:
            pad_size = (41 - h) // 2
            kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None,
                motion_stdev=self.motion_stdev2,
                motion_mode_list=self.motion_mode_list2,
                motion_mode_prob=self.motion_mode_prob2)

        # pad kernel
        h2, w2 = kernel2.shape
        assert h2 == w2 and h2 % 2 == 1
        if h2 > 41:
            kernel2 = kernel2[(h2-41)//2:h2-(h2-41)//2, (h2-41)//2:h2-(h2-41)//2]
            kernel2 /= kernel2.sum()
            kernel2 = np.ascontiguousarray(kernel2)
        elif h2 < 41:
            pad_size = (41 - h2) // 2
            kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))
        
        if np.random.uniform() < self.opt['final_sinc_prob']:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        img_ref = img2tensor([img_ref], bgr2rgb=True, float32=True)[0]
        
        if flip:
            img_gt = torch.flip(img_gt,[2])
            img_ref = torch.flip(img_ref,[2])


        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)


        return_d = {'gt': img_gt, 'ref':img_ref, 'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel, 'gt_path': gt_path, 'ref_path':ref_path, 'HQLoc':HQ_Loc,'HQLand':HQ_Land, 'HQRefLoc':HQ_Ref_Loc,'HQRefLand':HQ_Ref_Land}
        return return_d

    def __len__(self):
        return len(self.id_folders)
