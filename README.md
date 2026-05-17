# MOSAIC: Multi-Object Spatial relations, AttrIbution, Counting

> **When Do Diffusion Models Learn to Generate Multiple Objects?**  
> Yujin Jeong, Arnas Uselis, Iro Laina, Seong Joon Oh, Anna Rohrbach  
> ICML 2026

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2605.00273)
[![Code](https://img.shields.io/badge/Code-coming_soon-lightgrey)]()
[![Dataset](https://img.shields.io/badge/Dataset-coming_soon-lightgrey)]()

---

## Overview

MOSAIC is a controlled diagnostic framework for analyzing multi-object generation failures in text-to-image diffusion models. It isolates three compositional factors — **Color Attribution**, **Counting**, and **Spatial Relations** — enabling causal analysis of data effects.

## Key Findings

- **Scene complexity > data imbalance.** More objects in the scene hurts performance more than concept frequency.
- **Counting undergoes training collapse.** Accuracy peaks early then degrades in low-data regimes — mitigated by reducing scene complexity.
- **Compositional generalization collapses.** Holding out concept pairs breaks recombination. Difficulty: Attribution < Counting < Spatial Relations.

## Code & Data

🚧 Coming soon!

## Citation

```bibtex
@inproceedings{jeong2026mosaic,
  title={When Do Diffusion Models Learn to Generate Multiple Objects?},
  author={Jeong, Yujin and Uselis, Arnas and Laina, Iro and Oh, Seong Joon and Rohrbach, Anna},
  booktitle={ICML},
  year={2026}
}
```
