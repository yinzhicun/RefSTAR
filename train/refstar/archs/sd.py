import torch
from torch import nn as nn
from torch.nn import functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from peft import LoraConfig
from diffusers import AutoencoderKL, UNet2DConditionModel
from .arc2face.text_encoder import CLIPTextModelWrapper
from insightface.app import FaceAnalysis
from transformers import AutoTokenizer
from .reference_attention import ReferenceAttentionControl as ReferenceAttentionControl_Dual

def get_lora_config(rank, use_dora, target_modules):
    base_config = {'r': rank, 'lora_alpha': rank, 'init_lora_weights': 'gaussian', 'target_modules': target_modules}
    return LoraConfig(**base_config)

@ARCH_REGISTRY.register()
class SD_Org(nn.Module):

    def __init__(self, rank=64):
        super(SD_Org, self).__init__()
        self.unet = UNet2DConditionModel.from_pretrained('./models', subfolder='arc2face')
        self.vae = AutoencoderKL.from_pretrained('stable-diffusion-v1-5/stable-diffusion-v1-5', subfolder='vae')
        self.text_encoder = CLIPTextModelWrapper.from_pretrained('./models', subfolder='encoder', torch_dtype=torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained('stable-diffusion-v1-5/stable-diffusion-v1-5', subfolder='tokenizer', use_fast=False)
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.timestep = torch.tensor(999.0, dtype=torch.float32)
        unet_target_modules = ['to_q', 'to_k', 'to_v', 'to_out.0', 'proj_in', 'proj_out', 'ff.net.0.proj', 'ff.net.2', 'conv1', 'conv2', 'conv_shortcut', 'downsamplers.0.conv', 'upsamplers.0.conv', 'time_emb_proj']
        unet_lora_config = get_lora_config(rank=rank, use_dora=False, target_modules=unet_target_modules)
        self.unet.add_adapter(unet_lora_config)
        vae_target_modules = ['to_k', 'to_q', 'to_v', 'to_out.0', 'conv1', 'conv2']
        vae_lora_config = get_lora_config(rank=rank, use_dora=False, target_modules=vae_target_modules)
        self.vae.add_adapter(vae_lora_config)
        self.app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(512, 512))

    def extract_id_feats(self, ref):
        batch_input_ids = []
        for i in range(len(ref)):
            ref_t = ref[i] / 2 + 0.5
            ref_np = ref_t.permute(1, 2, 0).flip(2).cpu().numpy() * 255.0
            faces = self.app.get(ref_np)
            if len(faces) == 0:
                print('No ID EMBEDING')
                id_emb = torch.zeros([1, 512], dtype=torch.float32)
            else:
                faces = sorted(faces, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
                id_emb = torch.tensor(faces['embedding'], dtype=torch.float32)[None]
                id_emb = id_emb / torch.norm(id_emb, dim=1, keepdim=True)
            batch_input_ids.append(id_emb)
        batch_input_ids = torch.stack(batch_input_ids, dim=0).to(ref.device)
        arcface_token_id = self.tokenizer.encode('id', add_special_tokens=False)[0]
        input_text_ids = self.tokenizer('photo of a id person', truncation=True, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt').input_ids
        input_text_ids = input_text_ids.repeat(len(batch_input_ids), 1).to(ref.device)
        batch_input_ids = batch_input_ids.squeeze(1)
        id_emb_padded = F.pad(batch_input_ids, (0, self.text_encoder.config.hidden_size - 512), 'constant', 0)
        token_embs = self.text_encoder(input_ids=input_text_ids, return_token_embs=True)
        token_embs[input_text_ids == arcface_token_id] = id_emb_padded
        encoder_hidden_states = self.text_encoder(input_ids=input_text_ids.to(ref.device), input_token_embs=token_embs.to(ref.device))[0]
        return encoder_hidden_states

    def forward(self, x, ref):
        latents = self.vae.encode(x).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        (batch, _, _, _) = latents.size()
        id_emb = self.extract_id_feats(ref)
        model_output = self.unet(latents, self.timestep.to(latents.device), encoder_hidden_states=id_emb.to(latents.device).to(torch.float32), timestep_cond=None, cross_attention_kwargs=None, return_dict=False)[0]
        predicted_original_sample = (latents - torch.tensor(0.9953).to(latents.device).sqrt() * model_output) / torch.tensor(0.0047).to(latents.device).sqrt()
        out = self.vae.decode(predicted_original_sample / self.vae.config.scaling_factor, return_dict=False)[0]
        return out

@ARCH_REGISTRY.register()
class SD_Ref_Dual(nn.Module):
    """Networks consisting of Residual in Residual Dense Block, which is used
    in ESRGAN.

    ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks.

    We extend ESRGAN for scale x2 and scale x1.
    Note: This is one option for scale 1, scale 2 in RRDBNet.
    We first employ the pixel-unshuffle (an inverse operation of pixelshuffle to reduce the spatial size
    and enlarge the channel size before feeding inputs into the main ESRGAN architecture.

    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64
        num_block (int): Block number in the trunk network. Defaults: 23
        num_grow_ch (int): Channels for each growth. Default: 32.
    """

    def __init__(self, rank=64):
        super(SD_Ref_Dual, self).__init__()
        self.unet = UNet2DConditionModel.from_pretrained('./models', subfolder='arc2face')
        self.vae = AutoencoderKL.from_pretrained('stable-diffusion-v1-5/stable-diffusion-v1-5', subfolder='vae')
        self.text_encoder = CLIPTextModelWrapper.from_pretrained('./models', subfolder='encoder', torch_dtype=torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained('stable-diffusion-v1-5/stable-diffusion-v1-5', subfolder='tokenizer', use_fast=False)
        self.text_encoder.requires_grad_(False)
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(512, 512))
        self.timestep = torch.tensor(999.0, dtype=torch.float32)
        unet_target_modules = ['to_q', 'to_k', 'to_v', 'to_out.0', 'proj_in', 'proj_out', 'ff.net.0.proj', 'ff.net.2', 'conv1', 'conv2', 'conv_shortcut', 'downsamplers.0.conv', 'upsamplers.0.conv', 'time_emb_proj']
        unet_lora_config = get_lora_config(rank=rank, use_dora=False, target_modules=unet_target_modules)
        self.unet.add_adapter(unet_lora_config)
        vae_target_modules = ['to_k', 'to_q', 'to_v', 'to_out.0', 'conv1', 'conv2']
        vae_lora_config = get_lora_config(rank=rank, use_dora=False, target_modules=vae_target_modules)
        self.vae.add_adapter(vae_lora_config)
        self.unet.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.vae_lora_ref = AutoencoderKL.from_pretrained('stable-diffusion-v1-5/stable-diffusion-v1-5', subfolder='vae')
        self.unet_lora_ref = UNet2DConditionModel.from_pretrained('./models', subfolder='arc2face')
        # self.unet_lora_ref.requires_grad_(False)
        # self.vae_lora_ref.requires_grad_(False)
        self.reference_control_writer = ReferenceAttentionControl_Dual(self.unet_lora_ref, do_classifier_free_guidance=False, mode='write', fusion_blocks='full')
        self.reference_control_reader = ReferenceAttentionControl_Dual(self.unet, do_classifier_free_guidance=False, mode='read', fusion_blocks='full')

    def extract_id_feats(self, ref):
        batch_input_ids = []
        for i in range(len(ref)):
            ref_t = ref[i] / 2 + 0.5
            ref_np = ref_t.permute(1, 2, 0).flip(2).cpu().numpy() * 255.0
            faces = self.app.get(ref_np)
            if len(faces) == 0:
                print('No ID EMBEDING')
                id_emb = torch.zeros([1, 512], dtype=torch.float32)
            else:
                faces = sorted(faces, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]))[-1]
                id_emb = torch.tensor(faces['embedding'], dtype=torch.float32)[None]
                id_emb = id_emb / torch.norm(id_emb, dim=1, keepdim=True)
            batch_input_ids.append(id_emb)
        batch_input_ids = torch.stack(batch_input_ids, dim=0).to(ref.device)
        arcface_token_id = self.tokenizer.encode('id', add_special_tokens=False)[0]
        input_text_ids = self.tokenizer('photo of a id person', truncation=True, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors='pt').input_ids
        input_text_ids = input_text_ids.repeat(len(batch_input_ids), 1).to(ref.device)
        batch_input_ids = batch_input_ids.squeeze(1)
        id_emb_padded = F.pad(batch_input_ids, (0, self.text_encoder.config.hidden_size - 512), 'constant', 0)
        token_embs = self.text_encoder(input_ids=input_text_ids, return_token_embs=True)
        token_embs[input_text_ids == arcface_token_id] = id_emb_padded
        encoder_hidden_states = self.text_encoder(input_ids=input_text_ids.to(ref.device), input_token_embs=token_embs.to(ref.device))[0]
        return encoder_hidden_states

    def forward(self, x, ref, ref1=None):
        id_emb = self.extract_id_feats(ref.detach())
        latents = self.vae.encode(x).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        if ref1 is not None:
            latents_ref = self.vae_lora_ref.encode(ref1).latent_dist.sample()
        else:
            latents_ref = self.vae_lora_ref.encode(ref).latent_dist.sample()
        latents_ref = latents_ref * self.vae_lora_ref.config.scaling_factor
        (batch, _, _, _) = latents.size()
        self.unet_lora_ref(latents_ref, self.timestep, encoder_hidden_states=id_emb, timestep_cond=None, cross_attention_kwargs=None, return_dict=False)
        self.reference_control_reader.update(self.reference_control_writer)
        model_output = self.unet(latents, self.timestep, encoder_hidden_states=id_emb, timestep_cond=None, cross_attention_kwargs=None, return_dict=False)[0]
        predicted_original_sample = (latents - torch.tensor(0.9953).to(latents.device).sqrt() * model_output) / torch.tensor(0.0047).to(latents.device).sqrt()
        out = self.vae.decode(predicted_original_sample / self.vae.config.scaling_factor, return_dict=False)[0]
        return out