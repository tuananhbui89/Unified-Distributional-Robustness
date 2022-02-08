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

## Requirements 
- Python 3.7
- Pytorch 
- Auto-Attack 0.1
- Foolbox 3.2.1
- Numba 0.52.0
  

## Robustness Evaluation 
We use several attackers to challenge the baselines and our method. 

(1) PGD Attack. We use the pytorch version of the Cifar10 Challenge with norm Linf, from the [implementation](https://github.com/yaodongyu/TRADES/blob/master/pgd_attack_cifar10.py). Attack setting for the Cifar10 dataset: `epsilon`=8/255, `eta`=2/255, `num_steps`=200, `random_init`=True <!-- and `soft_label`=True to use current prediction as the label for generating adversarial examples (to avoid label leaking). -->  

(2) Auto-Attack. The official implementation in the [link](). We test with the standard version with both Linf and L2 norm. 

(3) Brendel & Bethge Attack. The B&B attack is adapted from [the implementation](https://github.com/wielandbrendel/adaptive_attacks_paper/tree/master/07_ensemble_diversity) of the paper ["On Adaptive Attacks to Adversarial Example Defenses"](https://arxiv.org/abs/2002.08347). It has been initialized with PGD attack (20 steps, `eta`=`epsilon`/2) to increase the success rate.  

We use the full test-set (10k images) for the attack (1) and 1000 first test images for the attacks (2-3).



# Unified-Distributional-Robustness
