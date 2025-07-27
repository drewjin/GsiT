# Multimodal Transformers are Hierarchical Modal-wise Heterogeneous Graphs

## Acknowledgement

The repository is based on [MMSA](https://github.com/thuiar/MMSA).

**We strongly recommend integrating the core code directly into the MMSA framework. This repository is structured entirely in line with the MMSA setup. However, minor issues may still arise, and we suggest directly incorporating the files from this repository into the MMSA framework for seamless execution.**

## Main Components

The main model is in `GsiT/src/MMSA-GsiT/models/custom/GSIT/`.

The main model trainer is in `GsiT/src/MMSA-GsiT/trains/custom/GSIT.py`.

The main configuration is in `GsiT/src/MMSA-GsiT/config/config_regression.json`.

The Triton kernel is in `GsiT/src/MMSA-GsiT/models/custom/GSIT/modules/Kernel`.

## Future Works

New kernel implementation is as follows:

[mbs-attn](https://github.com/drewjin/mbs-attn)

We are planning to add a PR to MMSA framework.
