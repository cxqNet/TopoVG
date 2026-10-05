# Zero-Shot Open-Vocabulary Visual Grounding via Diffusion-Based Spatial-Topological Routing in Remote Sensing Images


This is the offical repo for paper **"Zero-Shot Open-Vocabulary Visual Grounding via Diffusion-Based Spatial-Topological Routing in Remote Sensing Images"**. 

Lab: 智能感知与图像理解教育部重点实验室

Authors: Puhua Chen , Xuqiang Cao , Shasha Mao , Yuwei Guo , Jiadong Lin , Fang Liu , and Licheng Jiao



## Dataset Optimization
- We standardize the format of **RRSIS-D** and **RISBench** to facilitate data analysis for visual localization tasks. Both datasets support **xyxy** and **xywh** bounding box formats.
- We perform data cleaning on RISBench. This segmentation dataset originally contains invalid samples with all-black masks. After filtering these samples, we derive accurate bounding boxes from valid masks.

You can obtain the well-organized **RRSIS-D** and **RISBench** datasets from [[Dataset Link]](https://pan.baidu.com/s/1fUBm1JrR87bUVIvnMSrVRQ?pwd=v3zx).

## Overview

<p align="center">
<img src="Fig/pic6.png" width="1200">
</p>

Remote sensing visual grounding **(RSVG)** aims to localize target objects in complex remote sensing imagery according to natural-language referring expressions, and has emerged as a key frontier task for advancing remote sensing image interpretation and open-vocabulary interactive understanding. Existing fully supervised methods typically rely on large-scale annotated data for training, and often exhibit limited generalization when confronted with cross-dataset shifts, unseen categories, and complex scene distributions. This restricts their applicability to remote sensing scenarios characterized by diverse object categories, intricate spatial relationships, and cross-scene applications. To address these limitations, we propose **TopoVG**, a zero-shot and training-free framework for open-vocabulary remote sensing visual grounding, which exploits attention priors embedded in a frozen text-to-image diffusion model to localize remote sensing targets without annotated training data. 
Specifically, a Spatial-Topological Expression Parsing (STEP) module is first introduced to canonicalize free-form referring expressions into an executable spatial-topological representation that disentangles open-vocabulary object semantics, visual attributes, absolute spatial positions, and inter-object relations. Conditioned on this representation, a Relation-Aware Diffusion Attention (RADA) module is then designed to route diffusion attention through anchor-guided spatial projection and self-attention affinity propagation, transforming category-level activations into relation-aware and structure-consistent target responses. Training-Free Box Decoding (TFBD) is further developed to convert the refined response map into a stable bounding-box prediction through boundary-aligned refinement and affinity-guided region selection.
Extensive experiments and ablation studies on the RRSIS-D and RISBench remote sensing datasets demonstrate that our approach outperforms existing weakly supervised and zero-shot methods across multiple metrics, providing a new zero-shot and training-free solution for remote sensing visual grounding.

## Download and extract the features

To improve experimental efficiency, we provide precomputed features for RRSIS-D and RISBench. Follow the steps below to obtain localization results and feature heatmaps. Precomputed features: [[Features_LINK]](RRSIS_D_FEATURES_URL) 

Extract the corresponding `features/` folder into `./Data/RRSIS-D/` or `./Data/RISBench/`, alongside `split/`, `JPEGImages/`, and `JsonTree.json`:


```text
Data/
├── RRSIS-D/
│   ├── JPEGImages/
│   ├── split/
│   │   └── test.json
│   ├── features/
│   │   ├── 03600_000.pt
│   │   └── ...
│   └── JsonTree.json
└── RISBench/
    ├── JPEGImages/
    ├── split/
    │   └── test.json
    ├── features/
    │   ├── <sample_id>.pt
    │   └── ...
    └── JsonTree.json
```

## Run evaluation from the project root

Create Conda Environment and Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```

```bash
# RRSIS-D
python evaluate.py --data-root ./Data/RRSIS-D --workers 4 --save-images true
```

Each sample folder contains `localization.png` and `heatmap.png`; per-sample results and overall metrics are saved in `evaluation.txt`. Set `--save-images false` to save only the text report.
