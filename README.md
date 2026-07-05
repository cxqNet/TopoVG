# Zero-Shot Open-Vocabulary Visual Grounding via Diffusion-Based Spatial-Topological Routing in Remote Sensing Images



This is the offical repo for paper "Zero-Shot Open-Vocabulary Visual Grounding via Diffusion-Based Spatial-Topological Routing in Remote Sensing Images". The dataset and code are coming soon.

### Overview



Remote sensing visual grounding **(RSVG)** aims to localize target objects in complex remote sensing imagery according to natural-language referring expressions, and has emerged as a key frontier task for advancing remote sensing image interpretation and open-vocabulary interactive understanding. Existing fully supervised methods typically rely on large-scale annotated data for training, and often exhibit limited generalization when confronted with cross-dataset shifts, unseen categories, and complex scene distributions. This restricts their applicability to remote sensing scenarios characterized by diverse object categories, intricate spatial relationships, and cross-scene applications. To address these limitations, we propose **TopoVG**, a zero-shot and training-free framework for open-vocabulary remote sensing visual grounding, which exploits attention priors embedded in a frozen text-to-image diffusion model to localize remote sensing targets without annotated training data. 
Specifically, a Spatial-Topological Expression Parsing (STEP) module is first introduced to canonicalize free-form referring expressions into an executable spatial-topological representation that disentangles open-vocabulary object semantics, visual attributes, absolute spatial positions, and inter-object relations. Conditioned on this representation, a Relation-Aware Diffusion Attention (RADA) module is then designed to route diffusion attention through anchor-guided spatial projection and self-attention affinity propagation, transforming category-level activations into relation-aware and structure-consistent target responses. Training-Free Box Decoding (TFBD) is further developed to convert the refined response map into a stable bounding-box prediction through boundary-aligned refinement and affinity-guided region selection.
Extensive experiments and ablation studies on the RRSIS-D and RISBench remote sensing datasets demonstrate that our approach outperforms existing weakly supervised and zero-shot methods across multiple metrics, providing a new zero-shot and training-free solution for remote sensing visual grounding.
