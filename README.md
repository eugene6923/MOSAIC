# When Do Diffusion Models Learn to Generate Multiple Objects?

> **Yujin Jeong, Arnas Uselis, Iro Laina, Seong Joon Oh, Anna Rohrbach**  
> ICML 2026

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2605.00273)
[![Status](https://img.shields.io/badge/Status-Full%20Pipeline-blue)]()

---

## Overview

This repository contains the implementation of **MOSAIC** (Multi-Object Spatial relations, AttrIbution, Counting), a controlled diagnostic framework for analyzing multi-object generation failures in text-to-image diffusion models.

## Key Contributions

- **MOSAIC Framework**: Isolates three compositional factors — **Color Attribution**, **Counting**, and **Spatial Relations** — enabling causal analysis of data effects.
- **Comprehensive Dataset Generation**: Controlled dataset generation pipeline for multi-object scenarios using Blender.
- **Diagnostic Analysis**: Systematic evaluation of diffusion model failures on compositional tasks.

## Repository Structure

```
MOSAIC/
├── README.md (this file - paper overview)
├── mosaic/
│   ├── README.md (dataset generation code documentation)
│   ├── data_generation/
│   │   ├── generate_dataset.py
│   │   ├── dataset_generators.py
│   │   ├── utils.py
│   │   └── constants.py
│   ├── generation_configs/ (config examples)
│   ├── generate_dataset.sh
│   └── requirements.txt
```

## Getting Started

For **dataset generation**, navigate to the `mosaic` folder and follow the instructions in [mosaic/README.md](mosaic/README.md).

## TODO

- [x] MOSAIC dataset generation code
- [ ] Training code
- [ ] Evaluation code

## Citation

```bibtex
@inproceedings{jeong2026mosaic,
  title={When Do Diffusion Models Learn to Generate Multiple Objects?},
  author={Jeong, Yujin and Uselis, Arnas and Laina, Iro and Oh, Seong Joon and Rohrbach, Anna},
  booktitle={ICML},
  year={2026}
}
```