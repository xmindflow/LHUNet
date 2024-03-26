
# LHU-Net: A Light Hybrid U-Net for Cost-Efficient High-Performance Volumetric Medical Image Segmentation

This repository contains the official implementation of LHU-Net. Our paper, "LHU-Net: A Light Hybrid U-Net for Cost-Efficient High-Performance Volumetric Medical Image Segmentation," addresses the growing complexity in medical image segmentation models, focusing on balancing computational efficiency with segmentation accuracy.

## Abstract

The rise of Transformer architectures in medical image analysis has led to significant advancements in medical image segmentation. However, these improvements often come at the cost of increased model complexity. LHU-Net is a meticulously designed Light Hybrid U-Net optimized for volumetric medical image segmentation, effectively balancing computational efficiency and segmentation accuracy. Rigorous evaluation across benchmark datasets demonstrates LHU-Net's superior performance, setting new benchmarks in efficiency and accuracy.

## Key Contributions

- **Efficient Hybrid Attention Selection**: Introduces a strategic deployment of specialized attention mechanisms within Transformers, enabling nuanced feature extraction tailored to the demands of medical image segmentation.
- **Benchmark Setting Efficiency**: Achieves high-performance segmentation with significantly reduced computational resources, demonstrating an optimal balance between model complexity and computational efficiency.
- **Versatile Superiority**: Showcases unparalleled versatility and state-of-the-art performance across multiple datasets, highlighting its robustness and potential as a universal solution for medical image segmentation.

## Model Architecture

LHU-Net leverages a hierarchical U-Net encoder-decoder structure optimized for 3D medical image segmentation. The architecture integrates convolutional-based blocks with hybrid attention mechanisms, capturing both local features and non-local dependencies effectively.

![LHU-Net Architecture](/path/to/architecture_image.png)

*For a detailed explanation of each component, please refer to our paper.*

## Datasets

Our experiments were conducted on five benchmark datasets: Synapse, ACDC, LA, NIH Pancreas dataset (CT-82), and BraTS Challenge 2018. LHU-Net demonstrated exceptional performance across these datasets, significantly outperforming existing state-of-the-art models in terms of efficiency and accuracy.

## Getting Started

This section provides instructions on how to run LHU-Net for your segmentation tasks.

### Requirements

List of software requirements and how to install them.

### Training

Instructions on how to train LHU-Net on your dataset.

### Inference

Guide on how to perform inference with a trained LHU-Net model.

## Citation

If you find this work useful for your research, please cite:

```bibtex
@inproceedings{sadegheih2024lhunet,
  title={LHU-Net: A Light Hybrid U-Net for Cost-Efficient High-Performance Volumetric Medical Image Segmentation},
  author={Sadegheih, Yousef and Bozorgpour, Afshin and Kumari, Pratibha and Azad, Reza and Merhof, Dorit},
  booktitle={CVPR},
  year={2024}
}
```

## Acknowledgments

Acknowledgment section from the paper.
