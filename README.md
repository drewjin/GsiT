# GSIFN: A Graph-Structured and Interlaced-Masked Multimodal Transformer-based Fusion Network for Multimodal Sentiment Analysis

## News

- `2024-9-16`: Submitted to COLING 2025, currently under review. 

## Acknowledgement

The repository is based on [MMSA](https://github.com/thuiar/MMSA) and [xLSTM](https://github.com/NX-AI/xlstm).

**We strongly recommend integrating the core code directly into the MMSA framework. This repository is structured entirely in line with the MMSA setup. However, minor issues may still arise, and we suggest directly incorporating the files from this repository into the MMSA framework for seamless execution.**

## Requirements

```
python >=3.8

torch >= 1.9.1
transformers >= 4.4.0
numpy >= 1.20.3
pandas >= 1.2.5
tqdm >= 4.62.2
nvidia-ml-py3 >= 7.352.0
scikit-learn >= 0.24.2
easydict >= 1.9
pytorch_transformers >= 1.2.0
```

Make sure you install `xlstm` from scratch in `GSIFN/src/MMSA-GSIFN/models/custom/GSIFN/modules/xLSTM`.

## Main Components

The main model is in `GSIFN/src/MMSA-GSIFN/models/custom/GSIFN/`.

The main model trainer is in `GSIFN/src/MMSA-GSIFN/trains/custom/GSIFN.py`.

The main configuration is in `GSIFN/src/MMSA-GSIFN/config/config_regression.json`.

## Paper Citation

```
@misc{jin2024gsifngraphstructuredinterlacedmaskedmultimodal,
      title={GSIFN: A Graph-Structured and Interlaced-Masked Multimodal Transformer-based Fusion Network for Multimodal Sentiment Analysis}, 
      author={Yijie Jin},
      year={2024},
      eprint={2408.14809},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.14809}, 
}
```
