import os.path
import logging
import torch

from utils_ import utils_logger
from utils_ import utils_image as util
from PIL import Image, ImageOps
# from utils import utils_model
from collections import OrderedDict
from FaceRefEnhancement import FaceRefRestoration
import cv2
import numpy as np
import time
import copy
import argparse

def rgb_to_ycbcr(im_rgb):
    im_ycbcr = copy.deepcopy(im_rgb)

    R = im_rgb[:, :, 0]
    G = im_rgb[:, :, 1]
    B = im_rgb[:, :, 2]

    im_ycbcr[:, :, 0] = 0.299 * R + 0.587 * G + 0.114 * B
    im_ycbcr[:, :, 1] = -0.1687 * R - 0.3313 * G + 0.5 * B
    im_ycbcr[:, :, 2] = 0.5 * R - 0.4187 * G - 0.0813 * B

    return np.uint8(im_ycbcr)

def ycbcr_to_rgb(im_ycbcr):

    im_rgb = copy.deepcopy(im_ycbcr)

    im_rgb = im_rgb.astype(np.float32)

    Y = im_rgb[:, :, 0]
    Cb = im_rgb[:, :, 1]
    Cr = im_rgb[:, :, 2]

    im_result = copy.deepcopy(im_rgb)
    im_result[:, :, 0] = 1.164*Y + 1.596*Cr - 0.871
    im_result[:, :, 1] = Y - 0.34414*Cb - 0.71414*Cr
    im_result[:, :, 2] = Y + 1.772*Cb

    return np.uint8(im_result)

def str2bool(v):
    return v.lower() in ('t', 'true', 'y', 'yes', '1')

def main():
    degradation_type = 2
    parser = argparse.ArgumentParser()
   
    parser.add_argument('--detect_faces', type=str2bool, default=False)
    parser.add_argument('--testsets', type=str, default=None)
    parser.add_argument('--testsets_ref', type=str, default=None)

    parser.add_argument('--FaceModel', type=str, default='RefSTAR')
    parser.add_argument('--ModelPath', type=str, default=None)
    parser.add_argument('--output_size', type=int, default=512)
    parser.add_argument('--scale_factor', type=int, default=1)
    parser.add_argument('--save_results', type=str2bool, default=True)
    parser.add_argument('--output_path', type=str, default='output')
    parser.add_argument('--output_postfix', type=str, default=None)
    parser.add_argument('--bit', type=int, default=32)
    parser.add_argument('--norm', type=str2bool, default=True)
    parser.add_argument('--checklandmark', type=str2bool, default=False)

    args = parser.parse_args()

    utils_logger.logger_info('blind_sr_log', log_path='blind_sr_log.log')
    logger = logging.getLogger('blind_sr_log')
    
    testsets = args.testsets       # fixed, set path of testsets
    testsets_ref = args.testsets_ref
    facemodel_names = { 
        'RefSTAR':'./pretrained_models/net_g_latest.pth',

    }

    SelectFaceModel = args.FaceModel
    ModelPath = args.ModelPath or facemodel_names[args.FaceModel]
    output_size = args.output_size

    sf = args.scale_factor
    save_results = args.save_results

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    FaceModel = FaceRefRestoration(ModelName=SelectFaceModel, FaceModelPath=ModelPath,
                                output_size=output_size, device=device, bit=args.bit, norm=args.norm, checklandmark=args.checklandmark)###lxm

    torch.cuda.empty_cache()
    
    L_path = testsets
    Ref_path = testsets_ref
    t = args.output_postfix or time.strftime("%m-%d_%H-%M", time.localtime())
    E_path = args.output_path

    util.mkdir(E_path)

    logger.info('{:>16s} : {:s}'.format('Input Path', L_path))
    logger.info('{:>16s} : {:s}'.format('Output Path', E_path))
    idx = 0
    test_results = OrderedDict()
    test_results['runtime'] = []
    

    for img in util.get_image_paths(L_path):
        
        # --------------------------------
        # (1) img_L
        # --------------------------------
        idx += 1
        img_name, ext = os.path.splitext(os.path.basename(img))
        if 'ref' in img_name:
            continue
        logger.info('{:->4d} --> {:<s} --> x{:<d}--> {:<s}'.format(idx, 'test_face_only', sf, img_name+ext))

        img_L = Image.open(img)
        img_L = ImageOps.exif_transpose(img_L)
        img_L = img_L.convert('RGB')
        img_L = np.array(img_L)
        # print(img_L.shape)

        ref_img_path = None
        for ext_option in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
            potential_path = os.path.join(Ref_path, img_name + ext_option)
            if os.path.exists(potential_path):
                ref_img_path = potential_path
                break
        if ref_img_path is None:
            logger.warning(f'Reference image not found for {img_name}')
            continue
        img_ref_L = Image.open(ref_img_path)

        img_ref_L = ImageOps.exif_transpose(img_ref_L)
        img_ref_L = img_ref_L.convert('RGB')
        img_ref_L = np.array(img_ref_L)

        # --------------------------------
        # (2) inference
        # --------------------------------

        start_time = time.time()

        ##############################
        #####Begin Restoring Face
        ##############################
        img, ori_faces, ref_face, en_faces, ref_face_with_mask = FaceModel.handle_faces(img=img_L, img_ref=img_ref_L, sf=args.scale_factor, detect_faces=args.detect_faces)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_time = time.time()
        test_results['runtime'].append((end_time - start_time) * 1000)  # milliseconds

        # if save_results:
        if args.detect_faces:
            for m, (ef, of) in enumerate(zip(en_faces, ori_faces)):
                of = cv2.resize(of, ef.shape[:2])
                cv2.imwrite(os.path.join(E_path, img_name +f'_face{m:02d}_{output_size}'+'.png'), np.hstack((of, ref_face_with_mask, ef)))
            cv2.imwrite(os.path.join(E_path, img_name+f'_from_{output_size}.png'), img[:,:,::-1])
        else:
            cv2.imwrite(os.path.join(E_path, img_name+f'_combine.png'), np.hstack((img_L[:,:,::-1], ref_face[:,:,::-1], ref_face_with_mask, img)))
            cv2.imwrite(os.path.join(E_path, img_name+'.png'), img)

        ave_runtime = sum(test_results['runtime']) / len(test_results['runtime']) / 1000.0
        logger.info('------> Average runtime of ({}) is : {:.6f} seconds'.format(L_path, ave_runtime))

if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(end-start)
