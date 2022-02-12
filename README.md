# Unified Distributional Robustness 

## Introduction 
Pytorch implementation for the paper "A Unified Wasserstein Distributional Robustness Framework for Adversarial Training" (accepted to ICLR 2022). [Link to the paper](https://openreview.net/forum?id=Dzpe9C1mpiv).

## Citation 
Please consider to cite our paper if you find any useful in this source code or our paper

```
@inproceedings{
bui2022a,
title={A Unified Wasserstein Distributional Robustness Framework for Adversarial Training},
author={Anh Tuan Bui and Trung Le and Quan Hung Tran and He Zhao and Dinh Phung},
booktitle={International Conference on Learning Representations},
year={2022},
url={https://openreview.net/forum?id=Dzpe9C1mpiv}
}
```
  
## Structure 
- Cifar10-WideResNet: an implementation with WideResNet architecture, which is based on AWP's implementation. To reproduce Table 4 in our paper, including: UDR-AWP-AT, AWP-AT.  
- Cifar10-Resnet18: an implementation with Resnet18 architecture (supports WideResNet as well). To reproduce Table 3 in our paper, including baseline methods: PGD-AT, TRADES, MART and our methods UDR-PGD, UDR-TRADES and UDR-MART. 
- Toy2D: an implementation to demonstrate the benefit of soft-ball projection. 

