import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import pi
import math
import time
from matplotlib import pyplot as plt

def affine_2to3(M):
    assert type(M) == np.ndarray and M.shape == (2, 3)
    return np.concatenate([M, np.array([[0, 0, 1]], dtype=M.dtype)], axis=0)

def affine_3to2(M):
    assert type(M) == np.ndarray and M.shape == (3, 3)
    return M[:2]

def cubic(x):
    absx = np.abs(x)
    absx2 = absx ** 2
    absx3 = absx ** 3
    return ((1.5*absx3 - 2.5*absx2 + 1) * (absx <= 1) +
            (-0.5*absx3 + 2.5*absx2 - 4*absx + 2) * ((1 < absx) & (absx <= 2)))

def lanczos2(x):
    return (((np.sin(pi*x) * np.sin(pi*x/2) + np.finfo(np.float32).eps) /
             ((pi**2 * x**2 / 2) + np.finfo(np.float32).eps))
            * (abs(x) < 2))

def lanczos3(x):
    return (((np.sin(pi*x) * np.sin(pi*x/3) + np.finfo(np.float32).eps) /
            ((pi**2 * x**2 / 3) + np.finfo(np.float32).eps))
            * (abs(x) < 3))

def get_kernel(M, method=cubic, kernel_width=4):
    # M: np.array@[2 3]
    M = np.array(M)
    M[:, 2] = 0
    M_ = affine_3to2(np.linalg.inv(affine_2to3(M)))
    r = kernel_width // 2
    v = np.array([[0, 0, 1], [-r, -r, 1], [-r, r, 1],
                  [r, -r, 1], [r, r, 1]],
                 dtype=np.float64)  # [center, lu, ld, ru, rd], [5, 3]@xy1
    v_ = v @ M_.T  # [5 2]@xy
    v_xmin, v_ymin = v_.min(axis=0)
    v_xmax, v_ymax = v_.max(axis=0)
    v_xmin, v_xmax = int(np.floor(min(v_xmin, -r))), int(np.ceil(max(v_xmax, r)))
    v_ymin, v_ymax = int(np.floor(min(v_ymin, -r))), int(np.ceil(max(v_ymax, r)))
    shape_x, shape_y = v_xmax - v_xmin + 1, v_ymax - v_ymin + 1

    M_shift_ = M_ + np.array([[0, 0, -v_xmin],
                              [0, 0, -v_ymin]], dtype=np.float64)
    center_x, center_y = -v_xmin, -v_ymin

    mask = np.ones((kernel_width, kernel_width), dtype=np.int32)
    mask_ = cv2.warpAffine(mask, M_shift_, (shape_x, shape_y), flags=cv2.INTER_NEAREST)
    mask_[center_y-r:center_y+r+1, center_x-r:center_x+r+1] = 1
    kernel_index_ = np.concatenate(np.meshgrid(np.arange(v_xmin, v_xmax+1),
                                               np.arange(v_ymin, v_ymax+1),
                                               1), dtype=np.float64, axis=-1)
    kernel_index = kernel_index_ @ M.T # [h w 2]@xy
    kernel_distance = np.linalg.norm(kernel_index, axis=-1)
    kernel = method(kernel_distance)
    # kernel_distance_ = np.linalg.norm(kernel_index_[:, :, :2], axis=-1)
    # density = (kernel_distance + 1e-6) / (kernel_distance_ + 1e-6)
    # density = density / density.min()
    kernel = kernel # * density # * density
    kernel[mask_ == 0] = 0
    return kernel / kernel.sum(), (center_x, center_y)

def affine_conv(img, M, dsize=None, warp_mask_img=None, method=cubic, kernel_width=4):
    H, W, C = img.shape
    img = torch.from_numpy(img.copy()).permute(2, 0, 1)[None].to(torch.float32) / 255. # [1 C H W]

    if dsize is not None:
        H_warp, W_warp = dsize
    else:  # show the whole warped image
        dst_pos = np.array([[0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1]], dtype=np.float32) @ M.T
        min_x, min_y = dst_pos.min(axis=0)
        max_x, max_y = dst_pos.max(axis=0)
        M[0, 2] -= min_x
        M[1, 2] -= min_y
        H_warp, W_warp = int(np.ceil(max_y - min_y)), int(np.ceil(max_x - min_x))

    if (dsize is None) or (dsize is not None and warp_mask_img is None):
        img_mask_img = np.ones((H, W), dtype=np.float32)
        warp_mask_img = cv2.warpAffine(img_mask_img, M, (H_warp, W_warp), flags=cv2.INTER_CUBIC) > 1e-6

    kernel, (center_x, center_y) = get_kernel(M, method, kernel_width)
    h, w = kernel.shape
    img = F.pad(img, (w//2, w-w//2-1, h//2, h-h//2-1), mode='reflect')
    kernel = torch.from_numpy(kernel)[None, None].repeat(3, 1, 1, 1).to(torch.float32)
    img = np.array(F.conv2d(img, kernel, groups=3)[0].permute(1, 2, 0))
    M_ = affine_3to2(np.linalg.inv(affine_2to3(M)))
    M_1 = M_ + np.array([[0, 0, center_x-w//2],
                         [0, 0, center_y-h//2]], dtype=np.float64)
    M__ = affine_3to2(np.linalg.inv(affine_2to3(M_1)))
    # M__ = M + np.array([[0, 0, center_x-w//2],
    #                     [0, 0, center_y-h//2]], dtype=np.float64)
    warp = cv2.warpAffine(img, M__, (W_warp, H_warp), flags=cv2.INTER_LINEAR)
    return (warp.clip(0, 1)*255).round().astype(np.uint8)

def affine(img, M, dsize=None, warp_mask_img=None, method=cubic, kernel_width=4.0):
    """
    Params:
        img: np.ndarray, [H W C]@np.uint8
        M: np.ndarray, [2 3]@np.float32, img to warp; M_: warp to img
        warp_mask_img: np.ndarray, [H_warp W_warp]@np.bool8
        (method, kernel_width): [(cubic, 4.0)|(lanczos2, 4.0)|(lanczos3, 6.0)]
    """
    t = time.time()
    H, W, C = img.shape
    img = img.astype(np.float32)
    if dsize is not None:
        H_warp, W_warp = dsize
    else:  # show the whole warped image
        dst_pos = np.array([[0, 0, 1], [W, 0, 1], [W, H, 1], [0, H, 1]], dtype=np.float32) @ M.T
        min_x, min_y = dst_pos.min(axis=0)
        max_x, max_y = dst_pos.max(axis=0)
        M[0, 2] -= min_x
        M[1, 2] -= min_y
        H_warp, W_warp = int(np.ceil(max_y - min_y)), int(np.ceil(max_x - min_x))

    if (dsize is None) or (dsize is not None and warp_mask_img is None):
        img_mask_img = np.ones((H, W), dtype=np.float32)
        warp_mask_img = cv2.warpAffine(img_mask_img, M, (H_warp, W_warp), flags=cv2.INTER_CUBIC) > 1e-6

    M_ = affine_3to2(np.linalg.inv(affine_2to3(M)))
    warp = np.zeros((H_warp, W_warp, 3), dtype=img.dtype)

    # calculate the density on each direction
    # center, left-up, left-down, right-down, right-up
    warp_index_points = np.array([[0, 0, 1], [-1, -1, 1], [-1, 1, 1], [1, 1, 1], [1, -1, 1]], dtype=np.float32) # [5 3]@xy1
    img_index_points = (warp_index_points @ M_.T)  # [5 2]@xy
    lu, ld, rd, ru = np.linalg.norm(img_index_points[0] - img_index_points[1:], axis=1)
    l, r, u, d = (lu + ld) / 2, (ru + rd) / 2, (lu + ru) / 2, (ld + rd) / 2

    l_axis = np.linspace(-kernel_width/2.0 * np.ceil(1 / l), 0, int(kernel_width/2.0 * np.ceil(l)), endpoint=False)
    r_axis = np.linspace(0, kernel_width/2.0 * np.ceil(1 / r), int(kernel_width/2.0 * np.ceil(r))+1, endpoint=True)
    u_axis = np.linspace(-kernel_width/2.0 * np.ceil(1 / u), 0, int(kernel_width/2.0 * np.ceil(u)), endpoint=False)
    d_axis = np.linspace(0, kernel_width/2.0 * np.ceil(1 / d), int(kernel_width/2.0 * np.ceil(d))+1, endpoint=True)

    x_axis = np.concatenate((l_axis, r_axis), axis=0)
    y_axis = np.concatenate((u_axis, d_axis), axis=0)

    warp_index_rel = np.stack(np.meshgrid(y_axis, x_axis, indexing='ij'), axis=-1).reshape(-1, 2) # [H_rel*W_rel 2]@ij
    num_rel = warp_index_rel.shape[0]

    warp_index_warp = np.stack(np.meshgrid(range(H_warp), range(W_warp), indexing='ij'), axis=-1) # [H_warp W_warp 2]@ij
    warp_index_img = warp_index_warp[warp_mask_img]  # [#pix 2]@ij
    num_pix = warp_index_img.shape[0]

    warp_index_sampling = warp_index_img[:, None, :] + warp_index_rel[None, :, :] # [#pix #rel 2]@ij
    img_index_sampling = (np.concatenate((warp_index_sampling.reshape(-1, 2)[:, ::-1],
                                          np.ones((num_rel*num_pix, 1), dtype=warp_index_sampling.dtype)),
                                         axis=-1) @ M_.T).reshape(num_pix, num_rel, 2)[..., ::-1] # [#pix #rel 2]@ij
    img_index_sampling_round = np.round(img_index_sampling).astype(np.int32)
    img_index_sampling_round_valid = (img_index_sampling_round[:, :, 0] >= 0) & \
                                     (img_index_sampling_round[:, :, 0] < H) & \
                                     (img_index_sampling_round[:, :, 1] >= 0) & \
                                     (img_index_sampling_round[:, :, 1] < W)
    img_index_sampling_round[~img_index_sampling_round_valid] = 0
    warp_index_sampling_round = (np.concatenate((img_index_sampling_round.reshape(-1, 2)[:, ::-1],
                                                 np.ones((num_rel*num_pix, 1), dtype=img_index_sampling_round.dtype)),
                                                axis=-1) @ M.T).reshape(num_pix, num_rel, 2)[..., ::-1] # [#pix #rel 2]@ij
    warp_index_sampling_round_distance = np.linalg.norm(warp_index_sampling_round - warp_index_img[:, None, :], axis=-1) # [#pix #rel]
    warp_index_sampling_round_weight = method(warp_index_sampling_round_distance)
    warp_index_sampling_round_weight[~img_index_sampling_round_valid] = 0.
    warp_index_sampling_round_weight = warp_index_sampling_round_weight / warp_index_sampling_round_weight.sum(axis=1, keepdims=True)
    img_value_sampling_round = img[img_index_sampling_round[..., 0], img_index_sampling_round[..., 1]] # [#pix #rel 3]@RGB
    warp_value_pix = (img_value_sampling_round * warp_index_sampling_round_weight[:, :, None]).sum(axis=1) # [#pix 3]@RGB

    warp[warp_index_img[..., 0], warp_index_img[..., 1]] = warp_value_pix
    return warp.clip(0, 255).round().astype(np.uint8)



def int2tuple(ints):
    ints = list(map(int, ints.split(' ')))
    assert(len(ints) in (1, 2))
    if len(ints) == 1:
        return (ints[0], ints[0])
    return (ints[0], ints[1])



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--M', type=str, required=True)
    parser.add_argument('--dsize', type=int2tuple, default=(1024, 1024))
    parser.add_argument('--warp_mask_img', type=str, default=None, choices=[None],
                        help='currently only calculated by default.')
    parser.add_argument('--impl', type=str, default='affine_conv', choices=['affine', 'affine_conv'])
    parser.add_argument('--method', type=str, default='cubic', choices=['cubic', 'lanczos2'])
    parser.add_argument('--kernel_width', type=int, default=4)
    args = parser.parse_args()
    img = np.ascontiguousarray(cv2.imread(args.img, cv2.IMREAD_COLOR)[..., ::-1])
    M = np.load(args.M)
    method = {'cubic':cubic, 'lanczos2':lanczos2}.get(args.method, cubic)
    method_tranditional = {'cubic': cv2.INTER_CUBIC, 'lanczos2':cv2.INTER_LANCZOS4}.get(args.method, cv2.INTER_CUBIC)
    warp = eval(args.impl)(img, M, args.dsize, args.warp_mask_img, method, args.kernel_width)
    cv2.imwrite(args.output+'_'+args.impl+'.png', warp[..., ::-1])


















