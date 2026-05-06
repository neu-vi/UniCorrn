<h1 align="center">🦄 UniCorrn: Unified Correspondence Transformer Across 2D and 3D <br> CVPR 2026</h1>

<div align="center">

[Prajnan Goswami<sup>1*</sup>](https://prajnancv.github.io),
[Tianye Ding<sup>1*</sup>](https://jerrygcding.github.io/),
[Feng Liu<sup>2</sup>](https://scholar.google.com/citations?&user=uiqXutMAAAAJ),
[Huaizu Jiang<sup>1</sup>](https://jianghz.me/)\
<sup>1</sup> [Visual Intelligence Lab, Northeastern University](https://github.com/neu-vi), <sup>2</sup> [Adobe Research](https://research.adobe.com)\
<sup>*</sup> Equal Contribution

<a href="https://arxiv.org/abs/2605.04044"><img src="https://img.shields.io/badge/arXiv-2605.04044-b31b1b" alt="arXiv"></a>
<a href="https://neu-vi.github.io/UniCorrn/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Coming%20Soon-yellow?labelColor=grey" alt="Hugging Face Coming Soon">
</div>

___

# Setup

```bash
conda create --name unicorrn python=3.11
conda activate unicorrn
```
or

```bash
conda create --prefix /environment_path/unicorrn python=3.11
conda activate /environment_path/unicorrn
```

## Installing dependencies
```bash
pip install -r requirements.txt
pip install --no-build-isolation --no-deps git+https://github.com/Silverster98/pointops
pip install --no-build-isolation --no-deps git+https://github.com/qinzheng93/vision3d.git
```

__Install unicorrn__
```bash
pip install -e .
```

__OPTIONAL__

Install Cuda Rotary Position Embedding cuRoPE.

```bash
cd unicorrn/model/embedder/curope
python setup.py build_ext --inplace
cd ../../../../
```

For the visual localization benchmark on InLoc.
```bash
pip install -r requirements_optional.txt
```

# Download pre-trained weights
```bash
mkdir -p pretrained_models/
```

__UniCorrn Stage 1 and Stage 2 pre-trained weights__
```bash
wget https://huggingface.co/prajnan/unicorrn/resolve/main/UniCorrn_Large_Stage1.pth -P pretrained_models/
wget https://huggingface.co/prajnan/unicorrn/resolve/main/UniCorrn_Large_Stage2.pth -P pretrained_models/
```


# Usage

__2D2D__
```python
import torch
import numpy as np
from PIL import Image

from unicorrn.model import build_model
from unicorrn.utils import safe_load_weights, plot_correspondences
from unicorrn.utils.config import read_yaml_config
from unicorrn.inference import (
    init_query_points,
    coarse_to_fine,
    cycle_uniform_grid_inference,
)

MODEL_CONFIG_PATH = "/your_project_path/configs/models/unicorrn_large_stage2.yml"
CKPT_PATH = "/your_project_path/pretrained_models/UniCorrn_Large_Stage2.pth"
GRID_SIZE = 32
CONFIDENCE_THRESHOLD = 3.8
MATCHING_RADIUS_PX = 5.0

# ---- Shared setup: load model and input images ----
model_cfg = read_yaml_config(MODEL_CONFIG_PATH)
model = build_model(model_cfg.NAME, cfg=model_cfg)
weights = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
safe_load_weights(model, weights["model"])
model = model.eval().cuda()

img1_path = "assets/image_a.png"
img2_path = "assets/image_b.png"

img1 = np.array(Image.open(img1_path).convert("RGB"))
img2 = np.array(Image.open(img2_path).convert("RGB"))

H, W = img1.shape[:2]

# ---- Usage 1: User-specified keypoints with confidence filter ----
queries = init_query_points(H, W, grid_size=GRID_SIZE).view(-1, 2).numpy()

kpts1, kpts2, confidence, _ = coarse_to_fine(
    img1,
    img2,
    queries,
    model,
    unified_model=True,
)

mask = confidence.squeeze() >= CONFIDENCE_THRESHOLD
kpts1 = kpts1[mask]
kpts2 = kpts2[mask]

plot_correspondences(img1, img2, kpts1, kpts2, marker_size=1, plot_line=False, save_path="example_confidence.jpg")

# ---- Usage 2: Cycle-consistency correspondence extraction ----
# Builds a uniform grid on img1 and keeps only matches whose backward
# prediction lands within `MATCHING_RADIUS_PX` of the original query.
cycle_kpts1, cycle_kpts2, _, _ = cycle_uniform_grid_inference(
    img1,
    img2,
    model,
    grid_size=GRID_SIZE,
    matching_radius_px=MATCHING_RADIUS_PX,
    unified_model=True,
)

plot_correspondences(img1, img2, cycle_kpts1, cycle_kpts2, marker_size=1, plot_line=False, save_path="example_cycle.jpg")
```

__2D3D__ and __3D3D__ examples are included in `notebooks/2d3d.ipynb` and `notebooks/3d3d.ipynb`.


# Training

## Datasets

__2D-2D__

We use the datasets listed below, following DUSt3R's preprocessing step. See the [Datasets section in DUSt3R](https://github.com/naver/dust3r?tab=readme-ov-file#datasets) for details.

  - [ARKitScenes](https://github.com/apple/ARKitScenes) - [Creative Commons Attribution-NonCommercial-ShareAlike 4.0](https://github.com/apple/ARKitScenes/tree/main?tab=readme-ov-file#license)
  - [BlendedMVS](https://github.com/YoYo000/BlendedMVS) - [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/)
  - [CO3Dv2](https://github.com/facebookresearch/co3d) - [Creative Commons Attribution-NonCommercial 4.0 International](https://github.com/facebookresearch/co3d/blob/main/LICENSE)
  - [ScanNet++](https://kaldir.vc.in.tum.de/scannetpp/) - [non-commercial research and educational purposes](https://kaldir.vc.in.tum.de/scannetpp/static/scannetpp-terms-of-use.pdf)
  - [WayMo Open dataset](https://github.com/waymo-research/waymo-open-dataset) - [Non-Commercial Use](https://waymo.com/open/terms/)
  - [MegaDepth](https://www.cs.cornell.edu/projects/megadepth/)
  - [StaticThings3D](https://github.com/lmb-freiburg/robustmvd/blob/master/rmvd/data/README.md#staticthings3d)

Download the datasets into `data/Datasets/`
```text
data/Datasets/
├── blendedmvs_processed/
└── megadepth_processed/
.
.
└── waymo_processed/
```

__2D-3D__

We follow [2D3DMATR](https://github.com/minhaolee/2D3DMATR) to prepare the 2D-3D datasets.

The __7Scenes__ dataset can be downloaded from [BaiduYun](https://pan.baidu.com/s/1duymPG4dJte4Yx-qov5yeg) (extraction code: `m7mc`). Place it under `data/Datasets/` and organize as follows:

```text
data/Datasets/
└── 7Scenes/
    ├── metadata/
    └── data/
        ├── chess/
        ├── fire/
        ├── heads/
        ├── office/
        ├── pumpkin/
        ├── redkitchen/
        └── stairs/
```

The __RGBD-ScenesV2__ dataset can be downloaded from [BaiduYun](https://pan.baidu.com/s/14A2y8jghCdk6nAZa0_yEZA) (extraction code: `2dc7`). Place it under `data/Datasets/` and organize as follows:

```text
data/Datasets/
└── RGBDScenesV2/
    ├── metadata/
    └── data/
        ├── rgbd-scenes-v2-scene_01/
        ├── ...
        └── rgbd-scenes-v2-scene_14/
```

__3D-3D__

We use [3DMatch](https://3dmatch.cs.princeton.edu/) and [ModelNet](https://modelnet.cs.princeton.edu/) for 3D-3D. Download the datasets into `data/Datasets/` using:
```bash
bash scripts/download_3d3d_data.sh
```

The resulting layout should be:

```text
data/Datasets/
├── indoor/
└── modelnet40_ply_hdf5_2048/
```


## Training scripts

We provide download links for the pre-trained CroCoV2 weights from the original repository, as well as an additional decoder weights file aligned with our feature fusion module.
```bash
wget https://download.europe.naverlabs.com/ComputerVision/CroCo/CroCo_V2_ViTLarge_BaseDecoder.pth -P pretrained_models/
wget https://huggingface.co/prajnan/unicorrn/resolve/main/CroCoV2_Large_BaseDecoder.pth -P pretrained_models/
```

Stage 1:
```bash
bash scripts/train_stage1.sh
```
Stage 2:
```bash
bash scripts/train_stage2.sh
```


# Benchmarks

## 2D-2D

Download the precomputed query points using [RoMa](https://github.com/Parskatt/RoMa) for MegaDepth1500 and ScanNet1500:
```bash
wget https://huggingface.co/prajnan/unicorrn/resolve/main/megadepth1500_query_points.json -P benchmarks
wget https://huggingface.co/prajnan/unicorrn/resolve/main/scannet1500_query_points.json -P benchmarks
```
MegaDepth-1500:
```bash
bash scripts/benchmark_2d2d_megadepth1500.sh
```
ScanNet-1500:
```bash
bash scripts/benchmark_2d2d_scannet1500.sh
```

__InLoc Visual localization__

Prepare the InLoc dataset following the steps in the [DUSt3R visloc guide](https://github.com/naver/dust3r/tree/main/dust3r_visloc#inloc) and place it under `data/Datasets/InLoc/`.

Then run:
```bash
bash scripts/benchmark_2d2d_inloc.sh
```

After completion, upload the `results_filename_ltvl.txt` file to https://www.visuallocalization.net/.

## 2D-3D

7Scenes:
```bash
bash scripts/benchmark_2d3d_7Scenes.sh
```
RGBD-ScenesV2:
```bash
bash scripts/benchmark_2d3d_rgbdscenesv2.sh
```

## 3D-3D

3DMatch and 3DLoMatch:
```bash
bash scripts/benchmark_3d3d_3dmatch.sh
```
ModelNet and ModelLoNet:
```bash
bash scripts/benchmark_3d3d_modelnet.sh
```


# Citation
If you find this repository useful in your research, please consider giving a star ⭐ and a citation
```bibtex
@inproceddings{goswami2026unicorrn,
  title={UniCorrn: Unified Correspondence Transformer Across 2D and 3D},
  author={Goswami, Prajnan and Ding, Tianye and Liu, Feng and Jiang, Huaizu},
  booktitle={CVPR},
  year={2026}
}
```

# Acknowledgements

We would like to thank the authors of [MASt3R](https://github.com/naver/mast3r), [RoMa](https://github.com/Parskatt/RoMa), [PointTransformerV3](https://github.com/Pointcept/PointTransformerV3), [2D3DMATR](https://github.com/minhaolee/2D3DMATR), [PREDATOR](https://github.com/prs-eth/OverlapPredator), [Vision3D](https://github.com/qinzheng93/vision3d), and many other repositories for open-sourcing their code.
