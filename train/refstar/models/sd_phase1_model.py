import torch
import time
import random
import cv2
import os
import numpy as np
from tqdm import tqdm
from os import path as osp
from collections import OrderedDict
from insightface.app import FaceAnalysis

from copy import deepcopy
from refstar.data.degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt, random_add_impulse_noise_pt
from basicsr.data.transforms import paired_random_crop
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.dist_util import master_only

from basicsr.utils.img_process_util import filter2D
from torch.nn import functional as F
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.base_model import BaseModel

                                        
def tensor2img(tensor):
    x = tensor.clone()
    output = (x.flip(0).permute(1, 2, 0) + 1) / 2
    img_save =  np.clip(output.detach().float().cpu().numpy(), 0, 1) * 255
    return img_save

def check_landmarks(img, landmarks):
    img_with_landmark = np.ascontiguousarray(img, dtype=np.uint8)
    landmarks = np.array(landmarks.cpu())
    for point in landmarks[17:,0:2]:
        cv2.circle(img_with_landmark, (int(point[0]), int(point[1])), 1, (0,255,0), -1)
    return img_with_landmark

@MODEL_REGISTRY.register()
class SDModel(BaseModel):
    """U-Net structure SR GAN model for single image super-resolution."""

    def __init__(self, opt):
        super(SDModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)

        self.print_network(self.net_g)



        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)


        if self.is_train:
            self.init_training_settings()

        self.jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        self.usm_sharpener = USMSharp().cuda()  # do usm sharpening
        self.queue_size = opt.get('queue_size', 180)

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # load pretrained discriminator models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)


        self.net_g.train()
        self.net_d.train()
        
        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('regress_opt'):
            self.cri_reg = build_loss(train_opt['regress_opt']).to(self.device)
        else:
            self.cri_reg = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None
        
        if train_opt.get('id_opt'):
            self.cri_id = build_loss(train_opt['id_opt']).to(self.device)
        else:
            self.cri_id = None

        if train_opt.get('lpips_opt'):
            self.cri_lpips = build_loss(train_opt['lpips_opt']).to(self.device)
        else:
            self.cri_lpips = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

        

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)

    def optimize_parameters(self, current_iter):

        scale_list = [1, 0.75, 0.5, 0.25]
        # usm sharpening
        l1_gt = self.gt_usm
        percep_gt = self.gt_usm
        gan_gt = self.gt_usm
       
        if self.opt.get('l1_gt_usm', True) is False:
            l1_gt = self.gt
        if self.opt.get('percep_gt_usm', True) is False:
            percep_gt = self.gt

        if self.opt.get('gan_gt_usm', True) is False:
            gan_gt = self.gt

        for p in self.net_d.parameters():
            p.requires_grad = False
        
        self.optimizer_g.zero_grad()
        

        self.output = self.net_g(self.lq, self.ref)


        if self.opt['rank'] == 0 and(current_iter == 10 or current_iter % 1000 == 0):
        
            print(f'rank_{self.opt["rank"]}')

            face_gt_save = tensor2img(self.gt[0])
            face_lq_save = tensor2img(self.lq[0])


            output_save = tensor2img(self.output[0])

            ref_save = tensor2img(self.ref[0])

            face_gt_with_landmark_save = check_landmarks(tensor2img(self.gt[0]), self.HQLand[0])
            ref_with_landmark_save = check_landmarks(tensor2img(self.ref[0]), self.HQRefLand[0])

            tmp_save_dir = self.opt['tmp_vis_save_dir']
            os.makedirs(tmp_save_dir, exist_ok=True)

            cv2.imwrite(f'{tmp_save_dir}/{current_iter}_face_gt.png', face_gt_save)
            cv2.imwrite(f'{tmp_save_dir}/{current_iter}_face_gt_land.png', face_gt_with_landmark_save)
            cv2.imwrite(f'{tmp_save_dir}/{current_iter}_face_lq.png', face_lq_save)

            cv2.imwrite(f'{tmp_save_dir}/{current_iter}_output.png', output_save)
            cv2.imwrite(f'{tmp_save_dir}/{current_iter}_ref.png', ref_save)
            cv2.imwrite(f'{tmp_save_dir}/{current_iter}_ref_land.png', ref_with_landmark_save)

        logger = get_root_logger()
        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.cri_pix:
                output_gray = self.output[:,0,:,:] * 0.2989 + self.output[:,1,:,:] * 0.5870 + self.output[:,2,:,:] * 0.1140
                gt_gray = self.gt[:,0,:,:] * 0.2989 + self.gt[:,1,:,:] * 0.5870 + self.gt[:,2,:,:] * 0.1140
                

                l_g_pix = self.cri_pix(output_gray, gt_gray)

                l_g_total += l_g_pix

                loss_dict['l_g_pix'] = l_g_pix



            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, self.gt)

                if l_g_percep is not None:
                    l_g_total += l_g_percep

                    loss_dict['l_g_percep'] = l_g_percep

                if l_g_style is not None:
                    l_g_total += l_g_style

                    loss_dict['l_g_style'] = l_g_style

            if self.cri_id:
                l_g_id = self.cri_id(self.output, self.gt)

                l_g_total += l_g_id

                loss_dict['l_g_id'] = l_g_id

            # gan loss
            fake_g_pred = self.net_d(self.output)

            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)

            l_g_total += l_g_gan

            loss_dict['l_g_gan'] = l_g_gan

          
            l_g_total.backward()

            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
      
        # real
        real_d_pred = self.net_d(self.gt)

        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)

        l_d_real_total = l_d_real

        loss_dict['l_d_real'] = l_d_real
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())

        l_d_real_total.backward()
      
        # fake
        fake_d_pred = self.net_d(self.output.detach().clone())  # clone for pt1.9

        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        l_d_fake_total = l_d_fake

        loss_dict['l_d_fake'] = l_d_fake

        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

        l_d_fake_total.backward()

        self.optimizer_d.step()
    
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def test_selfensemble(self):
        # TODO: to be tested
        # 8 augmentations

        def _transform(v, op):
            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)

            return ret

        # prepare augmented data
        lq_list = [self.lq]
        for tf in 'v', 'h', 't':
            lq_list.extend([_transform(t, tf) for t in lq_list])

        # inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
        else:
            self.net_g.eval()
            with torch.no_grad():
                out_list = [self.net_g_ema(aug) for aug in lq_list]
            self.net_g.train()

        # merge results
        for i in range(len(out_list)):
            if i > 3:
                out_list[i] = _transform(out_list[i], 't')
            if i % 4 > 1:
                out_list[i] = _transform(out_list[i], 'h')
            if (i % 4) % 2 == 1:
                out_list[i] = _transform(out_list[i], 'v')
        output = torch.cat(out_list, dim=0)

        self.output = output.mean(dim=0, keepdim=True)

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        # do not use the synthetic process during validation
        self.is_train = False
        self.validation(dataloader, current_iter, tb_logger, save_img)
        self.is_train = True

    def validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            self._initialize_best_metric_results(dataset_name)
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network_lora([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network_lora(self.net_g, 'net_g', current_iter)

        self.save_network(self.net_d, 'net_d', current_iter)
        self.save_training_state(epoch, current_iter)

    @master_only
    def save_network_lora(self, net, net_label, current_iter, param_key='params'):
        """Save networks.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """
        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{net_label}_{current_iter}.pth'
        save_path = os.path.join(self.opt['path']['models'], save_filename)

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            del_keys = []
            for key, param in state_dict.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    key = key[7:]
                if 'lora' in key:
                    state_dict[key] = param.cpu()
                else: 
                    del_keys.append(key)
            for key in del_keys:
                del state_dict[key]
            save_dict[param_key_] = state_dict

        # avoid occasional writing errors
        retry = 3
        while retry > 0:
            try:
                torch.save(save_dict, save_path)
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f'Save model error: {e}, remaining retry times: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            logger.warning(f'Still cannot save {save_path}. Just ignore it.')

    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, 'queue_lr'):
            assert self.queue_size % b == 0, f'queue size {self.queue_size} should be divisible by batch size {b}'
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ref_lr = torch.zeros(self.queue_size, c, h, w).cuda()

            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).cuda()
            self.queue_ref = torch.zeros(self.queue_size, c, h, w).cuda()

            self.queue_locs = torch.zeros(self.queue_size, 4, 4).cuda()
            self.queue_ref_locs = torch.zeros(self.queue_size, 4, 4).cuda()
            
            self.queue_lands = torch.zeros(self.queue_size, 68, 2).cuda()
            self.queue_ref_lands = torch.zeros(self.queue_size, 68, 2).cuda()

            self.queue_ref_id = torch.zeros(self.queue_size, 1, 512).cuda()

           
            self.queue_ptr = 0

        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_ref_lr = self.queue_ref_lr[idx]

            self.queue_gt = self.queue_gt[idx]
            self.queue_ref = self.queue_ref[idx]

            self.queue_locs = self.queue_locs[idx]
            self.queue_ref_locs = self.queue_ref_locs[idx]


            self.queue_lands = self.queue_lands[idx]
            self.queue_ref_lands = self.queue_ref_lands[idx]

            self.queue_ref_id = self.queue_ref_id[idx]

            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            ref_lq_dequeue = self.queue_ref_lr[0:b, :, :, :].clone()

            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            ref_dequeue = self.queue_ref[0:b, :, :, :].clone()

            locs_dequeue = self.queue_locs[0:b, :, :].clone()
            locs_ref_dequeue = self.queue_ref_locs[0:b, :, :].clone()

            lands_dequeue = self.queue_lands[0:b, :, :].clone()
            lands_ref_dequeue = self.queue_ref_lands[0:b, :, :].clone()

            ref_id_dequeue = self.queue_ref_id[0:b, :, :].clone()



            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_ref_lr[0:b, :, :, :] = self.ref_lq.clone()

            self.queue_gt[0:b, :, :, :] = self.gt.clone()
            self.queue_ref[0:b, :, :, :] = self.ref.clone()

            self.queue_locs[0:b, :, :] = self.HQLocs.clone()
            self.queue_ref_locs[0:b, :, :] = self.HQRefLocs.clone()

            self.queue_lands[0:b, :, :] = self.HQLand.clone()
            self.queue_ref_lands[0:b, :, :] = self.HQRefLand.clone()

            self.queue_ref_id[0:b, :, :] = self.ref_ids.clone()
            self.lq = lq_dequeue
            self.ref_lq = ref_lq_dequeue

            self.gt = gt_dequeue
            self.ref = ref_dequeue

            self.HQLocs = locs_dequeue
            self.HQRefLocs = locs_ref_dequeue

            self.HQLand = lands_dequeue
            self.HQRefLand = lands_ref_dequeue
            self.ref_ids = ref_id_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.lq.clone()
            self.queue_ref_lr[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.ref_lq.clone()

            self.queue_gt[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.gt.clone()
            self.queue_ref[self.queue_ptr:self.queue_ptr + b, :, :, :] = self.ref.clone()

            self.queue_locs[self.queue_ptr:self.queue_ptr + b, :, :] = self.HQLocs.clone()
            self.queue_ref_locs[self.queue_ptr:self.queue_ptr + b, :, :] = self.HQRefLocs.clone()

            self.queue_lands[self.queue_ptr:self.queue_ptr + b, :, :] = self.HQLand.clone()
            self.queue_ref_lands[self.queue_ptr:self.queue_ptr + b, :, :] = self.HQRefLand.clone()
            self.queue_ref_id[self.queue_ptr:self.queue_ptr + b, :, :] = self.ref_ids.clone()
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def feed_data(self, data):
        """Accept data from dataloader, and then add two-order degradations to obtain LQ images.
        """

        if self.is_train and self.opt.get('high_order_degradation', True):
            # training data synthesis
            self.gt = data['gt'].to(self.device)
            self.ref = data['ref'].to(self.device)


            

            self.ref_ids = torch.ones([self.gt.size()[0], 1, 512]).to(self.device)
            
            self.HQLocs = data['HQLoc']
            self.HQRefLocs = data['HQRefLoc']
            self.HQLand = data['HQLand']
            self.HQRefLand = data['HQRefLand']
            # USM sharpen the GT images
            self.gt_usm = self.usm_sharpener(self.gt)


            #     # random crop


            self.kernel1 = data['kernel1'].to(self.device)
            self.kernel2 = data['kernel2'].to(self.device)
            self.sinc_kernel = data['sinc_kernel'].to(self.device)

            
            ori_h, ori_w = self.gt.size()[2:4]

            # blur
            out = filter2D(self.gt, self.kernel1)
            out_ref = filter2D(self.ref, self.kernel1)

            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range'][0], 1)
            else:
                scale = 1

            mode = random.choice(['area', 'bilinear', 'bicubic'])
            if mode == 'area':
                out = F.interpolate(out, scale_factor=scale, mode=mode)
                out_ref = F.interpolate(out_ref, scale_factor=scale, mode=mode)

            else:
                out = F.interpolate(out, scale_factor=scale, mode=mode, align_corners=True)
                out_ref = F.interpolate(out_ref, scale_factor=scale, mode=mode, align_corners=True)

            # add noise
            gray_noise_prob = self.opt['gray_noise_prob']
            if np.random.uniform() < self.opt['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                out_ref = random_add_gaussian_noise_pt(
                    out_ref, sigma_range=self.opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                if self.opt.get('use_poisson_noise', True):
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.opt['poisson_scale_range'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)
                    out_ref = random_add_poisson_noise_pt(
                        out_ref,
                        scale_range=self.opt['poisson_scale_range'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)
                else:
                    out = random_add_impulse_noise_pt(
                        out,
                        replace_frac_range=self.opt['impulse_replace_frac_range'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)
                    out_ref = random_add_impulse_noise_pt(
                        out_ref,
                        replace_frac_range=self.opt['impulse_replace_frac_range'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)

            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range'])
            jpeg_p_ref = out.new_zeros(out_ref.size(0)).uniform_(*self.opt['jpeg_range'])

            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = self.jpeger(out, quality=jpeg_p)

            out_ref = torch.clamp(out_ref, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out_ref = self.jpeger(out_ref, quality=jpeg_p_ref)
            # blur
            if np.random.uniform() < self.opt['second_blur_prob']:
                out = filter2D(out, self.kernel2)
                out_ref = filter2D(out_ref, self.kernel2)

            # random resize
            updown_type = random.choices(['up', 'down', 'keep'], self.opt['resize_prob2'])[0]
            if updown_type == 'up':
                scale = np.random.uniform(1, self.opt['resize_range2'][1])
            elif updown_type == 'down':
                scale = np.random.uniform(self.opt['resize_range2'][0], 1)
            else:
                scale = 1

            mode = random.choice(['area', 'bilinear', 'bicubic'])
            if mode == 'area':
                out = F.interpolate(out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)
                out_ref = F.interpolate(out_ref, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode)

            else:
                out = F.interpolate(out, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode, align_corners=True)
                out_ref = F.interpolate(out_ref, size=(int(ori_h / self.opt['scale'] * scale), int(ori_w / self.opt['scale'] * scale)), mode=mode, align_corners=True)

            # add noise
            gray_noise_prob = self.opt['gray_noise_prob2']
            if np.random.uniform() < self.opt['gaussian_noise_prob2']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
                out_ref = random_add_gaussian_noise_pt(
                    out_ref, sigma_range=self.opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                if self.opt.get('use_poisson_noise', True):
                    out = random_add_poisson_noise_pt(
                        out,
                        scale_range=self.opt['poisson_scale_range2'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)
                    out_ref = random_add_poisson_noise_pt(
                        out_ref,
                        scale_range=self.opt['poisson_scale_range2'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)
                else:
                    out = random_add_impulse_noise_pt(
                        out,
                        replace_frac_range=self.opt['impulse_replace_frac_range2'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)
                    out_ref = random_add_impulse_noise_pt(
                        out_ref,
                        replace_frac_range=self.opt['impulse_replace_frac_range2'],
                        gray_prob=gray_noise_prob,
                        clip=True,
                        rounds=False)
                
            # JPEG compression + the final sinc filter
            # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
            # as one operation.
            # We consider two orders:
            #   1. [resize back + sinc filter] + JPEG compression
            #   2. JPEG compression + [resize back + sinc filter]
            # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
            if np.random.uniform() < 0.5:
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                if mode == 'area':
                    out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                    out_ref = F.interpolate(out_ref, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                
                else:
                    out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode, align_corners=True)
                    out_ref = F.interpolate(out_ref, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode, align_corners=True)

                out = filter2D(out, self.sinc_kernel)
                out_ref = filter2D(out_ref, self.sinc_kernel)

                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                jpeg_p_ref = out_ref.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])

                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)

                out_ref = torch.clamp(out_ref, 0, 1)
                out_ref = self.jpeger(out_ref, quality=jpeg_p_ref)
            else:
                # JPEG compression
                jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])
                jpeg_p_ref = out_ref.new_zeros(out.size(0)).uniform_(*self.opt['jpeg_range2'])

                out = torch.clamp(out, 0, 1)
                out = self.jpeger(out, quality=jpeg_p)
                out_ref = torch.clamp(out_ref, 0, 1)
                out_ref = self.jpeger(out_ref, quality=jpeg_p_ref)
                # resize back + the final sinc filter
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                if mode == 'area':
                    out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)
                    out_ref = F.interpolate(out_ref, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode)

                else:
                    out = F.interpolate(out, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode, align_corners=True)
                    out_ref = F.interpolate(out_ref, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode=mode, align_corners=True)

                out = filter2D(out, self.sinc_kernel)
                out_ref = filter2D(out_ref, self.sinc_kernel)

            if self.opt['scale'] != 1:
                mode = random.choice(['area', 'bilinear', 'bicubic'])
                out = F.interpolate(out, size=(ori_h, ori_w), mode=mode)
                out_ref = F.interpolate(out_ref, size=(ori_h, ori_w), mode=mode)

            if self.opt['higher_ref_degradation']:
                out_ref = F.interpolate(out_ref, size=(ori_h // 32, ori_w // 32), mode='bicubic')
                out_ref = F.interpolate(out_ref, size=(ori_h // self.opt['scale'], ori_w // self.opt['scale']), mode='bicubic')


            # clamp and round
            
            self.lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.
            self.ref_lq = torch.clamp((out_ref * 255.0).round(), 0, 255) / 255.
            
            self.lq = self.lq * 2 - 1 
            self.ref_lq = self.ref_lq * 2 - 1
            self.gt = self.gt * 2 - 1 
            self.ref = self.ref * 2 - 1 

            # random crop

            # training pair pool
            self._dequeue_and_enqueue()
            self.gt_usm = self.usm_sharpener(self.gt)
            self.lq = self.lq.contiguous()  # for the warning: grad and param do not obey the gradient layout contract
        else:
            self.lq = data['lq'].to(self.device)
            if 'gt' in data:
                self.gt = data['gt'].to(self.device)
                self.gt_usm = self.usm_sharpener(self.gt)
    
    



    def get_component(self, imgs, locs, size=128):
        le_location = locs[:,0,:]
        re_location = locs[:,1,:]
        mo_location = locs[:,3,:]
      
        le_location = le_location.int()
        re_location = re_location.int()
        mo_location = mo_location.int()
        
        le = self.roi_align(imgs.clone(), le_location, size, 'le')
        re = self.roi_align(imgs.clone(), re_location, size, 're')
        mo = self.roi_align(imgs.clone(), mo_location, size, 'mo')

        return le, re, mo

    def roi_align_self(self, input, location, target_size):
        return torch.cat([F.interpolate(input[i:i+1,:,location[i,1]:location[i,3],location[i,0]:location[i,2]],(target_size,target_size),mode='bilinear',align_corners=False) for i in range(input.size(0))],0)

    def roi_align(self, input, location, target_size, name):
        locs = []
        for i in range(input.size(0)):
            x0 = location[i,1]
            x1 = location[i,3]
            y0 = location[i,0]
            y1 = location[i,2]
            l_x = x1 - x0
            l_y = y1 - y0

            #         x0 = 


            try:
                loc = F.interpolate(input[i:i+1,:,x0:x0+l_x,y0:y0+l_y],(target_size,target_size),mode='bilinear',align_corners=False) 
            except:
                if len(locs) > 0:
                    loc = locs[-1]
                else:
                    print([name, x0, y0, l_x, l_y])
                    k = i + 1
                    loc = F.interpolate(input[k:k+1,:,location[k,1]:location[k,3],location[k,0]:location[k,2]],(target_size,target_size),mode='bilinear',align_corners=False)

            locs.append(loc)
        return torch.cat(locs, dim=0)