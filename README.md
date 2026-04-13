# RefSTAR: Blind Facial Image Restoration with Reference Selection, Transfer, and Reconstruction
<a href="https://ojs.aaai.org/index.php/AAAI/article/view/38194"><img src="https://img.shields.io/badge/AAAI-2026-blue.svg" height=22.5></a>
<a href="https://arxiv.org/pdf/2507.10470"><img src="https://img.shields.io/badge/arXiv-2507.10470-b31b1b.svg" height=22.5></a>

>**Abstract**: <br>
> Blind facial image restoration is highly challenging due to unknown complex degradations and the sensitivity of humans to faces. Although existing methods introduce auxiliary information from generative priors or high-quality reference images, they still struggle with identity preservation problems, mainly due to improper feature introduction on detailed textures. In this paper, we focus on effectively incorporating appropriate features from high-quality reference images, presenting a novel blind facial image restoration method that considers reference selection, transfer, and reconstruction (RefSTAR). In terms of selection, we construct a reference selection (RefSel) module. For training the RefSel module, we construct a RefSel-HQ dataset through a mask generation pipeline, which contains annotating masks for 10,000 ground truth-reference pairs. As for the transfer, due to the trivial solution in vanilla crossattention operations, a feature fusion paradigm is designed to force the features from the reference to be integrated. Finally, we propose a reference image reconstruction mechanism that further ensures the presence of reference image features in the output image. The cycle consistency loss is also redesigned in conjunction with the mask. Extensive experiments on various backbone models demonstrate superior performance, showing better identity preservation ability and reference feature transfer quality.

![method](assets/method.png)


## Getting Started

### Environment Setup

```shell
git clone https://github.com/yinzhicun/RefSTAR.git
cd RefSTAR
conda create -n refstar python=3.9
conda activate refstar
pip install -r requirements.txt
```

### Pretrained Models

We provide the pretrained checkpoints in [BaiduDisk](https://pan.baidu.com/s/1lh8TizeXEP3Jq5nl_E5djA?pwd=m8gj) and [Google Drive](https://drive.google.com/drive/folders/1-PW53efDRvFpxFhnHyl20lEKVW1jAF59?usp=sharing). Download and place them into the `test/pretrained_models/` directory. The expected structure is:

```
test/pretrained_models/
├── net_g_latest.pth              # Main RefSTAR model weights
└── RefSel/
    ├── net_g_35000.pth           # RefSel model weights
└── arc2face/
    ├── models/
    │   ├── arc2face/             # Arc2Face UNet weights
    │   └── encoder/              # CLIP text encoder weights
    │   └── antelopev2/           # InsightFace face analysis model
└── mobilenet0.25_Final.pth       # MobileNet face detection model weights (For unaligned image restoration)
└── RetinaFace-R50.pth            # RetinaFace face detection model weights (For unaligned image restoration)
└── ParseNet-latest.pth           # Face parsing model weights (For unaligned image restoration)


```

> **Note**: The VAE and tokenizer are loaded from [stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5). Please ensure it is accessible (downloaded locally or available via HuggingFace cache).

### Testing

To test the method, run the following command inside the `test/` directory:

```shell
cd test
CUDA_VISIBLE_DEVICES=0 python test.py \
    --testsets /path/to/LQ_images \
    --testsets_ref /path/to/Reference_images \
    --output_path /path/to/output
```

#### __Parameters:__
```
--testsets          # Path to the input LQ (low-quality) images
--testsets_ref      # Path to the reference images (filename must match LQ images)
--output_path       # Path to save the restored results (default: output)
--FaceModel         # Model name (default: RefSTAR)
--ModelPath         # Custom path to model weights (default: ./pretrained_models/net_g_latest.pth)
--output_size       # Output face resolution (default: 512)
--scale_factor      # Scale factor for input (default: 1)
--detect_faces      # Enable face detection mode for full images (default: False)
--bit               # Precision mode: 16 or 32 (default: 32)
--norm              # Enable input normalization (default: True)
--checklandmark     # Visualize detected landmarks on output (default: False)
```

> **Note**: When `--detect_faces` is set to `False`, the input images are treated as pre-cropped face images. When set to `True`, faces will be automatically detected and restored within the full image.

### Preparing Dataset

TODO

### Training

Training code will be released soon.

## Citation

```
@inproceedings{yin2026refstar,
  title={RefSTAR: Blind Face Image Restoration with Reference Selection, Transfer, and Reconstruction},
  author={Yin, Zhicun and Chen, Junjie and Liu, Ming and Wang, Zhixin and Li, Fan and Pei, Renjing and Li, Xiaoming and Lau, Rynson WH and Zuo, Wangmeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={40},
  number={14},
  pages={12053--12062},
  year={2026}
}
```

## Acknowledgements

This code is built on [Arc2Face](https://github.com/foivospar/Arc2Face), [GPEN](https://github.com/yangxy/GPEN), [Stable Diffusion](https://github.com/CompVis/stable-diffusion), and [InsightFace](https://github.com/deepinsight/insightface). We thank the authors for sharing the codes.