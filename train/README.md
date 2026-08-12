# RefSTAR Training

This folder contains the training configurations for the RefSTAR two-stage pipeline (512×512).

## Training Data

The released configs use `RefDegradationDataset` and require two directory roots, pointed to by `dataroot_id` and `dataroot_lm_id` in `train/train_phase1_512.yml` and `train/train_phase2_512.yml`.  By default they are set to:

- `data/CelebHQRefForRelease` — high-quality face images
- `data/CelebRef-HQ512_landmark` — 68/106-point landmark text files

These paths are relative to the working directory from which you launch training (the repository root by default).  Either place or symlink your data under these names, or edit the two `dataroot_*` values directly.

You can download the training datasets (`CelebHQRefForRelease` and `CelebRef-HQ512_landmark`) from either of the following links:

- [Baidu Netdisk](https://pan.baidu.com/s/1ujwSxI2czqAYDf48AADShQ?pwd=jkpe) (extraction code: `jkpe`)
- [Google Drive](https://drive.google.com/drive/folders/1QZ2gD7vZ3f1kfYfQ75QyqAhtNpwOQZAj?usp=sharing)

### Directory structure

Both roots use the same identity-folder layout.  Every image in `CelebHQRefForRelease` must have a matching `.txt` landmark file (same base name) under the corresponding identity in `CelebRef-HQ512_landmark`:

```
data/CelebHQRefForRelease/
├── 00001/
│   ├── 0.png
│   ├── 1.png
│   └── ...
├── 00002/
│   └── ...

data/CelebRef-HQ512_landmark/
├── 00001/
│   ├── 0.txt
│   ├── 1.txt
│   └── ...
├── 00002/
│   └── ...
```

For each training sample, the dataloader randomly selects two images from the same identity folder: one as the ground-truth / target and the other as the reference.

## Two-Stage Training

RefSTAR training is split into two stages:

1. **Phase 1** — train the base Arc2Face restoration model.
   - Config: `train/train_phase1_512.yml`
   - Model: `SD_Org` (`SDModel`)
   - Dataset: `RefDegradationDataset`

2. **Phase 2** — train the full dual-reference network with warping, cycle, and parsing/mask losses.
   - Config: `train/train_phase2_512.yml`
   - Model: `SD_Ref_Dual` (`SDPhase2Model`)
   - Dataset: `RefDegradationDataset`
   - Before starting phase 2, set `pretrain_network_g` and `pretrain_network_d` in `train_phase2_512.yml` to the latest generator and discriminator checkpoints from phase 1, e.g.:
     ```yaml
     pretrain_network_g: experiments/RefSTAR_phase1_512/models/net_g_xxx.pth
     pretrain_network_d: experiments/RefSTAR_phase1_512/models/net_d_xxx.pth
     ```

## How to Train

The training code is self-contained under this `train/` directory:

- `train/train.py` — training entry point
- `train/refstar/` — model/data/arch/loss package (only the modules needed for these two configs are auto-loaded)
- `train/train_phase1_512.yml` and `train/train_phase2_512.yml` — released configs

From the repository root run:

```bash
# Phase 1 (single GPU example)
python train/train.py -opt train/train_phase1_512.yml --launcher none

# Phase 1 (multi-GPU example, 4 GPUs)
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 train/train.py -opt train/train_phase1_512.yml --launcher pytorch --auto_resume

# Phase 2 (single GPU)
python train/train.py -opt train/train_phase2_512.yml --launcher none

# Phase 2 (multi-GPU, 4 GPUs)
python -m torch.distributed.launch --nproc_per_node=4 --master_port=29500 train/train.py -opt train/train_phase2_512.yml --launcher pytorch --auto_resume
```

Alternatively, you can run `bash train/train.sh` from the repository root.

Adjust `CUDA_VISIBLE_DEVICES`, `--nproc_per_node`, and `master_port` to match your hardware.


## Pretrained Files

The training code expects the following directories relative to the **working directory from which you launch training** (the repository root by default).  Create `./models/` and place or symlink the contents there:

```
models/
├── arc2face/      # Arc2Face UNet (Diffusers format)
├── encoder/       # Arc2Face text encoder (Diffusers format)
└── antelopev2/    # InsightFace models for ID-embedding extraction
```

For this release the Arc2Face and encoder weights are already bundled under `test/pretrained_models/arc2face/models/`.  From the repository root you can set up the required layout with:

```bash
mkdir -p models
ln -s test/pretrained_models/arc2face/models/arc2face     models/arc2face
ln -s test/pretrained_models/arc2face/models/encoder      models/encoder
ln -s test/pretrained_models/arc2face/models/antelopev2   models/antelopev2
```

(`FaceAnalysis(name='antelopev2', root='./')` resolves its model directory under `./models/antelopev2/`.)

Other required weights are already bundled inside the training package:

- `train/refstar/models/weights/ParseNet-latest.pth` — used by `FaceParse` for face-parsing masks.
- `train/refstar/losses/weights/model_ir_se50.pth` — used by the `IDLoss` face-identity loss.

If you need to download these weights (`ParseNet-latest.pth` and `model_ir_se50.pth`) separately, they are also available at:

- [Baidu Netdisk](https://pan.baidu.com/s/1ujwSxI2czqAYDf48AADShQ?pwd=jkpe) (extraction code: `jkpe`)
- [Google Drive](https://drive.google.com/drive/folders/1QZ2gD7vZ3f1kfYfQ75QyqAhtNpwOQZAj?usp=sharing)

The VAE and tokenizer are loaded from the `stable-diffusion-v1-5/stable-diffusion-v1-5` Diffusers model.  The first run will download these from HuggingFace unless they are already cached in `~/.cache/huggingface/hub/models--stable-diffusion-v1-5--stable-diffusion-v1-5/`.
