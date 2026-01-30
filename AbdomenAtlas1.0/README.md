# LHU-Net Training & Inference on AbdomenAtlas1.0

This repository contains scripts to train and run inference with **LHU-Net** using the **AbdomenAtlas1.0** dataset.

## Contents
- [Dataset & Pretrained Weights](#dataset--pretrained-weights)
- [Training](#training)
- [Inference](#inference)
- [Notes](#notes)
- [Citations](#citations)

## Dataset & Pretrained Weights

- **Dataset (AbdomenAtlas1.0Mini)**: available on Hugging Face  
  [Download](https://huggingface.co/datasets/AbdomenAtlas/AbdomenAtlas1.0Mini)

- **Pretrained weights (used for Touchstone benchmark)**:  
  [Download](https://myfiles.uni-regensburg.de/filr/public-link/file-download/0447879f9c0971d0019c0f442858484f/150134/-6336273837647786685/lhunet.pth)

## Training

1. Update the dataset/output paths in the training config files.
2. Run:

```bash
cd train
./train.sh
```
## Inference

1. Update the dataset/output paths in the inference config files.
2. Run:

```bash
cd inference
./inference.sh
```
### Disable metric computation
If you only want predictions (no metric calculation), comment out the **last line** in `alex.sh`.

## Notes

- The **LHU-Net version used in the Touchstone benchmark** is a *weaker variant* than the one in the main LHU-Net repository.
- This repository is implemented with **MONAI**, not **nnUNetv2**.

## Citations

### Touchstone Benchmark
```bibtex
@article{bassi2024touchstone,
  title={Touchstone benchmark: Are we on the right way for evaluating ai algorithms for medical segmentation?},
  author={Bassi, Pedro RAS and Li, Wenxuan and Tang, Yucheng and Isensee, Fabian and Wang, Zifu and Chen, Jieneng and Chou, Yu-Cheng and Kirchhoff, Yannick and Rokuss, Maximilian R and Huang, Ziyan and others},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={15184--15201},
  year={2024}
}
```
**LHU-Net**:
```bibtex
@inproceedings{sadegheih2025lhu,
  title={LHU-Net: A lean hybrid u-net for cost-efficient, high-performance volumetric segmentation},
  author={Sadegheih, Yousef and Bozorgpour, Afshin and Kumari, Pratibha and Azad, Reza and Merhof, Dorit},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={326--336},
  year={2025},
  organization={Springer}
}
```