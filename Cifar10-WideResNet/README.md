# Unified Distributional Robustness

## Introduction 
Pytorch implementation for the paper "A Unified Wasserstein Distributional Robustness Framework for Adversarial Training" (accepted to ICLR 2022). [Link to the paper](https://openreview.net/forum?id=Dzpe9C1mpiv).

In this sub-repository, we provide an implementation of AWP-AT-UDR method, which is based on the implementation from AWP-AT ([link](https://github.com/csdongxian/AWP)). In addition, we provide an evaluation script that can run attacks: PGD attack, Auto-Attack, B&B attack and C&W attack. 

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
- Pytorch 1.7.1
- Auto-Attack 0.1
- Foolbox 3.2.1
- Numba 0.52.0
- torchattacks
  
## Robustness Evaluation 
We use several attackers to challenge the baselines and our method. 

(1) PGD Attack. We use the implementation from AWP's repository. Attack setting for the Cifar10 dataset: `epsilon`=8/255, `eta`=2/255, `num_steps`=100.  

(2) Auto-Attack. The official implementation in the [link](https://github.com/fra31/auto-attack). We test with the standard version with Linf.  

(3) Brendel & Bethge Attack. The B&B attack is adapted from [the implementation](https://github.com/wielandbrendel/adaptive_attacks_paper/tree/master/07_ensemble_diversity) of the paper ["On Adaptive Attacks to Adversarial Example Defenses"](https://arxiv.org/abs/2002.08347). It has been initialized with PGD attack (20 steps, `eta`=`epsilon`/2) to increase the success rate.  

(4) C&W Attack. We use the implementation from [link](https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/cw.py) 

## Running 

```
python train_cifar10_udr.py --model=WideResNet --batch_size=128
python eval_cifar10.py --preprocess='meanstd' --n_ex=10000 --model=WideResNet --batch_size=100
```