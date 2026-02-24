# -*- coding: utf-8 -*-
from turtle import exitonclick
import numpy as np
import cv2
import torch
import face_alignment

from utils_ import utils_image as util
import torchvision.transforms as transforms
import random
from scipy import ndimage
import scipy
import scipy.stats as ss
from scipy.interpolate import interp2d
from scipy.linalg import orth
from math import sqrt
from torch.autograd import Function
from PIL import Image

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), './retinaface'))
from retinaface.retinaface_detection import RetinaFaceDetection ##should be replaced and fine-tuned on low-quality face images

import torch.nn.utils.spectral_norm as SpectralNorm
import torch.nn as nn
from utils_ import utils_faces
import torch.nn.functional as F
import time
from face_parse.face_parsing import FaceParse
from mask_parsing import FaceParse as MaskParse

from models.sdxl_arch import SD_Ref_Dual

# from affine_final import affine_conv
from affine5 import affine_conv
from insightface.app import FaceAnalysis
from scipy.spatial.distance import cosine
from thop import profile
from thop import clever_format


def img_resize(image, size_h=512, size_w=512, mode='bicubic'):
    image_PIL = Image.fromarray(image)
    image_PIL = image_PIL.resize((size_w, size_h), Image.LANCZOS)
    return np.asarray(image_PIL)

def check_landmarks(img, landmarks):
    img_with_landmark = np.ascontiguousarray(img, dtype=np.uint8)
    landmarks = np.array(landmarks.cpu())
    for point in landmarks[17:,0:2]:
        cv2.circle(img_with_landmark, (int(point[0]), int(point[1])), 1, (0,255,0), -1)
    return img_with_landmark

class FaceRefRestoration(object):
    def __init__(self, ModelName='UNet', FaceModelPath='./model_zoo/1029_Face.pth',
                 output_size=1024, device='cuda', bit=32, norm=True, checklandmark=False):
        self.output_size = output_size
        self.device = device

        self.facedetector = RetinaFaceDetection('./')
        self.faceparser = FaceParse('./', device=device)
        self.masknet = MaskParse(device=device)

        self.facelandmarkdector = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False,device="cuda")
        self.ModelName = ModelName
        self.bit = bit
        self.norm = norm
        self.checklandmark = checklandmark
        self.modelFace = SD_Ref_Dual()
       
        Num_Parameter = utils_faces.print_networks(self.modelFace)

        print('[{}] Total Number of Parameters : {:.2f} M'.format(self.ModelName, Num_Parameter))
        self.modelFace.load_state_dict(torch.load(FaceModelPath, map_location='cpu')['params'], strict=False)
        self.modelFace.eval().cuda()

        for k, v in self.modelFace.named_parameters():
            v.requires_grad = False
            # print(k, v)
        self.modelFace = self.modelFace.to(self.device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.threshold = 0.9
        # the mask for pasting restored faces back
        self.mask = np.zeros((1024, 1024), np.float32)
        cv2.rectangle(self.mask, (26*output_size//512, 26*output_size//512), (486*output_size//512, 486*output_size//512), (1, 1, 1), -1, cv2.LINE_AA)
        self.mask = cv2.GaussianBlur(self.mask, (100*output_size//512+1, 100*output_size//512+1), int(10*output_size//512+1))
        self.mask = cv2.GaussianBlur(self.mask, (100*output_size//512+1, 100*output_size//512+1), int(10*output_size//512+1))
        self.kernel = np.array((
                [0.0625, 0.125, 0.0625],
                [0.125, 0.25, 0.125],
                [0.0625, 0.125, 0.0625]), dtype="float32")
        default_square = True
        inner_padding_factor = 0.25
        outer_padding = (0, 0)
        self.reference_5pts = utils_faces.get_reference_facial_points(
                (output_size, output_size), inner_padding_factor, outer_padding, default_square)
        self.FaceDistanceR = 0.09 ##Threshold for desiding distance of two boxex, 0~0.2 is preferred
        if bit==16:
            self.modelFace = self.modelFace.to(torch.float16)
            print('16-bit mode')
        elif bit == 32:
            print('32-bit mode')
        else:
            raise NotImplementedError
    
    def mask_postprocess(self, mask, thres=26):
        mask[:thres, :] = 0; mask[-thres:, :] = 0
        mask[:, :thres] = 0; mask[:, -thres:] = 0
        mask = cv2.GaussianBlur(mask, (101, 101), 4)
        mask = cv2.GaussianBlur(mask, (101, 101), 4)
        return mask.astype(np.float32)

    def handle_faces(self, img, img_ref, bg=None, sf=2, resizelands=0.3, detect_faces=True):
        height, width = img.shape[:2]
        height_ref, width_ref = img_ref.shape[:2]

        full_mask = np.zeros((height, width), dtype=np.float32)
        full_img = np.zeros(img.shape, dtype=np.uint8)
        # print(height, width)
        # img = cv2.resize(img, (width*sf, height*sf))
        if detect_faces:
            if resizelands:
                facebs, landms = self.facedetector.detect(
                    cv2.resize(img, (int(width*resizelands), int(height*resizelands)), interpolation=cv2.INTER_CUBIC))
                facebs_ref, landms_ref = self.facedetector.detect(
                    cv2.resize(img_ref, (int(width_ref*resizelands), int(height_ref*resizelands)), interpolation=cv2.INTER_CUBIC))
            else:
                facebs, landms = self.facedetector.detect(img)
            print('Detected {} faces...'.format(len(facebs)))
            orig_faces, enhanced_faces, masks_faces, ref_recon_faces,ref_mask_faces = [], [], [], [], []

            ### obtain list which should use face parsing map
            MaskLabels = np.zeros(len(facebs))

            # only one face in reference image
            for i, (faceb_ref, facialref5points) in enumerate(zip(facebs_ref, landms_ref)):
                if faceb_ref[4]<self.threshold: continue
                if resizelands:
                    faceb_ref[:3] = faceb_ref[:3] / resizelands
                    facialref5points = facialref5points / resizelands
                # faceb[:3] = faceb[:3] * sf
                # facial5points = facial5points * sf
                fh_ref, fw_ref = (faceb_ref[3]-faceb_ref[1]), (faceb_ref[2]-faceb_ref[0])  # [x0, y0, x1, y1]

                facialref5points = np.reshape(facialref5points, (2, 5))
                face_ref, tfm_inv = utils_faces.warp_and_crop_face(img_ref, facialref5points, reference_pts=self.reference_5pts, crop_size=(self.output_size, self.output_size))

                face_ref_landmarks = self.facelandmarkdector.get_landmarks(img_resize(face_ref, 256, 256))
                face_ref_landmarks_ = face_ref_landmarks[0] * (self.output_size  / 256)
                face_ref_landmarks_t = torch.from_numpy(face_ref_landmarks_).unsqueeze(0).to(self.device)

              
                # print(face_ref_landmarks_.shape)
                Ref = transforms.ToTensor()(face_ref)
                if self.norm == True:
                    Ref = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(Ref)
                Ref = Ref.unsqueeze(0).to(self.device)
                Ref = F.interpolate(Ref, (self.output_size//sf, self.output_size//sf), mode='bicubic', align_corners=False)
                

            for i, (faceb, facial5points) in enumerate(zip(facebs, landms)):
                if faceb[4]<self.threshold: continue
                if resizelands:
                    faceb[:3] = faceb[:3] / resizelands
                    facial5points = facial5points / resizelands
                # faceb[:3] = faceb[:3] * sf
                # facial5points = facial5points * sf
                fh, fw = (faceb[3]-faceb[1]), (faceb[2]-faceb[0])  # [x0, y0, x1, y1]

                facial5points = np.reshape(facial5points, (2, 5))
                of, tfm_inv = utils_faces.warp_and_crop_face(img, facial5points, reference_pts=self.reference_5pts, crop_size=(self.output_size, self.output_size))
                face_landmarks = self.facelandmarkdector.get_landmarks(img_resize(of, 256, 256))
                face_landmarks_ = face_landmarks[0] * (self.output_size / 256)
                face_landmarks_t = torch.from_numpy(face_landmarks_).unsqueeze(0).to(self.device)
                # print(face_landmarks_)
                
                of = cv2.resize(of, (16, 16), interpolation=cv2.INTER_LANCZOS4)
                of = cv2.resize(of, (512, 512), interpolation=cv2.INTER_LANCZOS4)
                LQ = transforms.ToTensor()(of)
                if self.norm == True:
                    LQ = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(LQ)
                LQ = LQ.unsqueeze(0).to(self.device)
              
                LQ = F.interpolate(LQ, (self.output_size//sf, self.output_size//sf), mode='bicubic', align_corners=False)

                start_time = time.time()
               
                Ref_Mask = self.masknet.process(LQ, Ref)[0]
                Ref_Mask = Ref_Mask.unsqueeze(0).to(self.device)
                Ref_with_Mask = Ref * Ref_Mask

                ef, *_ = self.modelFace(LQ, Ref, Ref * Ref_Mask)
                self.modelFace.reference_control_writer.clear()
                self.modelFace.reference_control_reader.clear()

                end_time = time.time()
                print("whole time:", end_time-start_time)


                if self.norm == True:
                    ef = ef * 0.5 + 0.5
                    Ref_with_Mask = Ref_with_Mask * 0.5 + 0.5
                ef = ef.squeeze(0).permute(1, 2, 0).flip(2) # RGB->BGR
                ef = np.clip(ef.float().cpu().numpy(), 0, 1) * 255.0
                
                Ref_with_Mask = Ref_with_Mask.squeeze(0).permute(1, 2, 0).flip(2) # RGB->BGR
                Ref_with_Mask = np.clip(Ref_with_Mask.float().cpu().numpy(), 0, 1) * 255.0
               
                if self.checklandmark:
                    orig_faces.append(check_landmarks(of[:,:,::-1], face_landmarks_t[0]))
                else:
                    orig_faces.append(of[:,:,::-1])

                enhanced_faces.append(ef)
                ref_mask_faces.append(Ref_with_Mask)

                tmp_mask = self.mask_postprocess(self.faceparser.process(ef)[0]/255.)
                tmp_mask = cv2.resize(tmp_mask, ef.shape[:2])
                tmp_mask = cv2.warpAffine(tmp_mask, tfm_inv, (width, height), flags=3)

                tmp_img = affine_conv(ef[:,:,::-1], tfm_inv, (height, width))

                mask = tmp_mask - full_mask
                full_mask[np.where(mask>0)] = tmp_mask[np.where(mask>0)]
                full_img[np.where(mask>0)] = tmp_img[np.where(mask>0)]
                break

            full_mask = full_mask[:, :, np.newaxis]
            if bg is not None:
                img = cv2.convertScaleAbs(bg*(1-full_mask) + full_img*full_mask)
            else:
                img = cv2.convertScaleAbs(img*(1-full_mask) + full_img*full_mask)

            return img, orig_faces, face_ref[:,:,::-1], enhanced_faces, Ref_with_Mask
        
        else:
            Ref = transforms.ToTensor()(img_ref)
            if self.norm == True:
                Ref = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(Ref)
            Ref = Ref.unsqueeze(0).to(self.device)
            Ref = F.interpolate(Ref, (self.output_size//sf, self.output_size//sf), mode='bicubic', align_corners=False)
                
                
            LQ = transforms.ToTensor()(img)
            if self.norm == True:
                LQ = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(LQ)
            LQ = LQ.unsqueeze(0).to(self.device)
            LQ = F.interpolate(LQ, (self.output_size//sf, self.output_size//sf), mode='bicubic', align_corners=False)

            Ref_Mask = self.masknet.process(LQ, Ref)[0]
            # Ref_Mask_np = Ref_Mask_save
            # Ref_Mask = transforms.ToTensor()(Ref_Mask)
            Ref_Mask = Ref_Mask.unsqueeze(0).to(self.device)
            Ref_with_Mask = Ref * Ref_Mask

            ef, *_ = self.modelFace(LQ, Ref, Ref_with_Mask)
            self.modelFace.reference_control_writer.clear()
            self.modelFace.reference_control_reader.clear()

            ef = ef * 0.5 + 0.5
            ef = ef.squeeze(0).permute(1, 2, 0).flip(2) # RGB->BGR
            ef = np.clip(ef.float().cpu().numpy(), 0, 1) * 255.0
            
            Ref_with_Mask = Ref_with_Mask * 0.5 + 0.5
            Ref_with_Mask = Ref_with_Mask.squeeze(0).permute(1, 2, 0).flip(2) # RGB->BGR
            Ref_with_Mask = np.clip(Ref_with_Mask.float().cpu().numpy(), 0, 1) * 255.0

            # overlay = np.zeros_like(img_ref)
            # overlay[Ref_Mask_np == 255] = (0, 255, 0)  # BGR格式，绿色
            # # 将overlay以50%透明度叠加到ref_img上
            # alpha = 0.5  # 透明度
            # blended_img = img_ref.copy()
            # # 找到需要叠加的位置
            # indices = np.where(Ref_Mask_np == 255)
            # # 进行叠加
            # blended_img[indices] = ((1 - alpha) * img_ref[indices] + alpha * overlay[indices]).astype(np.uint8)


            # return ef, None, img_ref, None, blended_img[:,:,::-1]
            return ef, None, img_ref, None, Ref_with_Mask

