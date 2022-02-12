# python eval_cifar10.py --method=pgd --lamda_init=1.0 --lamda_lr=0.01 --preprocess='meanstd' --n_ex=1000 --model=WideResNet --batch_size=100
python eval_cifar10.py --method=udr_learnable --lamda_init=1.0 --lamda_lr=0.02 --preprocess='meanstd' --n_ex=1000 --model=WideResNet --batch_size=100
# python eval_cifar10.py --method=awp --lamda_init=1.0 --lamda_lr=0.01 --preprocess='meanstd' --n_ex=1000 --model=WideResNet --batch_size=100
# python eval_cifar10.py --method=awp_udr_learnable --lamda_init=1.0 --lamda_lr=0.001 --preprocess='meanstd' --n_ex=1000 --model=WideResNet --batch_size=100
# python eval_cifar10.py --method=awp_trades --lamda_init=1.0 --lamda_lr=0.01 --preprocess='meanstd' --n_ex=1000 --batch_size=100 --model=WideResNet 
# python eval_cifar10.py --method=awp_trades_udr_learnable --lamda_init=1.0 --lamda_lr=0.01 --preprocess='meanstd' --n_ex=1000 --batch_size=100 --model=WideResNet 