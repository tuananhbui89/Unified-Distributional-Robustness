import os
import argparse
import time
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
import numpy as np 

# import sys
# sys.path.insert(0, '..')

from preactresnet import *
from wideresnet import *
from utils import transpose, cifar10

device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.cuda.empty_cache()

def filter_state_dict(state_dict):
    from collections import OrderedDict

    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'sub_block' in k:
            continue
        if 'module' in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict

def test(model, data_loader, device): 
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    accuracy = correct / len(data_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))
    
    return accuracy

def writelog(data=None, logfile=None, printlog=True):
    fid = open(logfile,'a')
    fid.write('%s\n'%(data))
    fid.flush()
    fid.close()
    if printlog: 
        print(data)

# simple Module to normalize an image
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean, dtype=torch.float32).cuda()
        self.std = torch.tensor(std, dtype=torch.float32).cuda()

    def forward(self, x):
        return (x - self.mean.type_as(x)[None, :, None, None]) / self.std.type_as(x)[None, :, None, None]

class Batches():
    def __init__(self, dataset, batch_size, shuffle, set_random_choices=False, num_workers=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.set_random_choices = set_random_choices
        self.dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=shuffle, drop_last=drop_last
        )

    def __iter__(self):
        if self.set_random_choices:
            self.dataset.set_random_choices()
        return ({'input': x.to(device).float(), 'target': y.to(device).long()} for (x,y) in self.dataloader)

    def __len__(self):
        return len(self.dataloader)

# cifar10_mean = (0.4914, 0.4822, 0.4465)
# cifar10_std = (0.2471, 0.2435, 0.2616)

# mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
# std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()

# def normalize(X):
#     return (X - mu)/std

upper_limit, lower_limit = 1,0

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)
    
def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def attack_pgd(model, X, y, epsilon, alpha, attack_iters, restarts,
               norm, early_stop=False,
               mixup=False, y_a=None, y_b=None, lam=None):
    max_loss = torch.zeros(y.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        if norm == "l_inf":
            delta.uniform_(-epsilon, epsilon)
        elif norm == "l_2":
            delta.normal_()
            d_flat = delta.view(delta.size(0),-1)
            n = d_flat.norm(p=2,dim=1).view(delta.size(0),1,1,1)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r/n*epsilon
        else:
            raise ValueError
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            output = model(X + delta) # Remove normalize 
            if early_stop:
                index = torch.where(output.max(1)[1] == y)[0]
            else:
                index = slice(None,None,None)
            if not isinstance(index, slice) and len(index) == 0:
                break
            if mixup:
                criterion = nn.CrossEntropyLoss()
                loss = mixup_criterion(criterion, model(X+delta), y_a, y_b, lam) # Remove normalize 
            else:
                loss = F.cross_entropy(output, y)
            loss.backward()
            grad = delta.grad.detach()
            d = delta[index, :, :, :]
            g = grad[index, :, :, :]
            x = X[index, :, :, :]
            if norm == "l_inf":
                d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            elif norm == "l_2":
                g_norm = torch.norm(g.view(g.shape[0],-1),dim=1).view(-1,1,1,1)
                scaled_g = g/(g_norm + 1e-10)
                d = (d + scaled_g*alpha).view(d.size(0),-1).renorm(p=2,dim=0,maxnorm=epsilon).view_as(d)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data[index, :, :, :] = d
            delta.grad.zero_()
        if mixup:
            criterion = nn.CrossEntropyLoss(reduction='none')
            all_loss = mixup_criterion(criterion, model(X+delta), y_a, y_b, lam) # Remove normalize
        else:
            all_loss = F.cross_entropy(model(X+delta), y, reduction='none') # Remove normalize
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    return max_delta

def str2bool(v):
    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='WideResNet',
                        choices=['WideResNet', 'PreActResNet18'])
    # parser.add_argument('--checkpoint', type=str, default='./model_test.pt')
    parser.add_argument('--data', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100'],
                        help='Which dataset the eval is on')
    parser.add_argument('--preprocess', type=str, default='meanstd',
                        choices=['meanstd', '01', '+-1'], help='The preprocess for data')
    parser.add_argument('--norm', type=str, default='Linf', choices=['L2', 'Linf'])
    parser.add_argument('--epsilon', type=float, default=8./255.)
    parser.add_argument('--eta', type=float, default=2./255.)
    parser.add_argument('--num_steps', type=int, default=100)

    parser.add_argument('--n_ex', type=int, default=10000)
    
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--version', type=str, default='standard')

    parser.add_argument('--method', type=str, default='awp')
    parser.add_argument('--eval_best', default=True, type=str2bool)
    parser.add_argument('--lamda_init', default=1.0, type=float)
    parser.add_argument('--lamda_lr', default=1e-2, type=float)
    parser.add_argument('--restarts', default=1, type=int)

    args = parser.parse_args()
    num_classes = int(args.data[5:])

    if args.preprocess == 'meanstd':
        if args.data == 'CIFAR10':
            mean = (0.4914, 0.4822, 0.4465)
            std = (0.2471, 0.2435, 0.2616)
        elif args.data == 'CIFAR100':
            mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
            std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    elif args.preprocess == '01':
        mean = (0, 0, 0)
        std = (1, 1, 1)
    elif args.preprocess == '+-1':
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
    else:
        raise ValueError('Please use valid parameters for normalization.')

    # model = ResNet18()
    if args.model == 'WideResNet':
        net = WideResNet(depth=34, num_classes=num_classes, widen_factor=10)
    elif args.model == 'PreActResNet18':
        net = PreActResNet18(num_classes=num_classes)
    else:
        raise ValueError('Please use choose correct architectures.')

    model_name = 'ds=cifar10_model={}_method={}'.format(args.model, 'awp_at_udr')

    logfile = os.path.join(model_name, f'log_result.txt')

    if args.eval_best:
        checkpoint = os.path.join(model_name, f'model_best.pth')
    else:
        epoch = 199 
        checkpoint = os.path.join(model_name, f'model_{epoch}.pth')

    ckpt = filter_state_dict(torch.load(checkpoint, map_location=device))
    net.load_state_dict(ckpt)

    model = nn.Sequential(Normalize(mean=mean, std=std), net)

    model.to(device)
    model.eval()

    print('Load successfully, from model_name:', model_name)
    
    dataset = cifar10('../cifar_data')
    test_set = list(zip(transpose(dataset['test']['data']/255.), dataset['test']['labels']))
    # test_loader = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=1)
    test_loader = Batches(test_set, args.batch_size, shuffle=False, num_workers=1)


    l = [batch['input'] for batch in test_loader]
    x_test = torch.cat(l, 0)
    l = [batch['target'] for batch in test_loader]
    y_test = torch.cat(l, 0)
    
    x_test = x_test[:args.n_ex]
    y_test = y_test[:args.n_ex]

    print('---- EVAL AUTO ATTACK ----')    
    # # load attack    
    from autoattack import AutoAttack
    adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, log_path=logfile)

    # cheap version
    # example of custom version
    if args.version == 'custom':
        adversary.attacks_to_run = ['apgd-ce', 'apgd-t']

    # run attack and save images
    adv_complete = adversary.run_standard_evaluation(x_test, y_test,
                                                        bs=args.batch_size)

    torch.save({'adv_complete': adv_complete}, '{}/{}_{}_1_{}_eps_{:.5f}.pth'.format(
        model_name, 'aa', args.version, adv_complete.shape[0], args.epsilon))

    aa_data = TensorDataset(adv_complete, y_test)
    aa_loader = DataLoader(aa_data, batch_size=1000, shuffle=False, num_workers=0)
    adv_acc = test(model, aa_loader, device)
    writelog('Auto-Attack, Standard, Linf, eps={}, adv_acc={:.4f}'.format(args.epsilon, adv_acc), logfile)

    # print('---- EVAL B&B ATTACK ----')
    import foolbox as fb
    class init_attack(object):
        
        def __init__(self, attack):
            self.attack = attack
            
        def run(self, model, originals, criterion_):
            return self.attack(model, x_test, criterion=criterion_, epsilons=args.epsilon)[1]

    pdg_init_attack = fb.attacks.LinfPGD(steps=20, abs_stepsize=args.epsilon/2, random_start=True)
    bb_attack = fb.attacks.LinfinityBrendelBethgeAttack(init_attack(pdg_init_attack), steps=200)
    fmodel = fb.PyTorchModel(model, bounds=(0, 1))
    _, _, init_success = pdg_init_attack(fmodel, x_test, y_test, epsilons=args.epsilon)
    _, advs, success = bb_attack(fmodel, x_test, 
                                criterion=fb.criteria.Misclassification(y_test), 
                                epsilons=args.epsilon)
    init_acc = 1 - np.mean(init_success.cpu().detach().numpy())
    adv_acc = 1 - np.mean(success.cpu().detach().numpy())
    writelog('B&B Attack, init with PGD-20, eps={}, adv_acc of initial PGD attack={:.4f}'.format(args.epsilon, init_acc), logfile)
    writelog('B&B Attack, eps={}, adv_acc={:.4f}'.format(args.epsilon, adv_acc), logfile)

    print('---- EVAL PGD100 Attack ----')

    test_loss = 0
    test_acc = 0
    test_robust_loss = 0
    test_robust_acc = 0
    test_n = 0
    for i, batch in enumerate(test_loader):
        X, y = batch['input'], batch['target']

        delta = attack_pgd(model, X, y, args.epsilon, args.eta, args.num_steps, args.restarts, 'l_inf', early_stop=True)
        delta = delta.detach()

        robust_output = model(torch.clamp(X + delta[:X.size(0)], min=lower_limit, max=upper_limit)) # remove normalize
        output = model(X) # remove normalize

        test_robust_acc += (robust_output.max(1)[1] == y).sum().item()
        test_acc += (output.max(1)[1] == y).sum().item()
        test_n += y.size(0)

    writelog('PGD attack, test_acc={:.4f}, test_robust_acc={:.4f}'.format(test_acc/test_n, test_robust_acc/test_n), logfile)
    
    print('------- EVAL CW ATTACK ---------')
    import torchattacks
    """
        Ref: 
            https://github.com/Harry24k/adversarial-attacks-pytorch/blob/master/torchattacks/attacks/cw.py
        Arguments:
        model (nn.Module): model to attack.
        c (float): c in the paper. parameter for box-constraint. (Default: 1e-4)    
            :math:`minimize \Vert\frac{1}{2}(tanh(w)+1)-x\Vert^2_2+c\cdot f(\frac{1}{2}(tanh(w)+1))`
        kappa (float): kappa (also written as 'confidence') in the paper. (Default: 0)
            :math:`f(x')=max(max\{Z(x')_i:i\neq t\} -Z(x')_t, - \kappa)`
        steps (int): number of steps. (Default: 1000)
        lr (float): learning rate of the Adam optimizer. (Default: 0.01)
        .. warning:: With default c, you can't easily get adversarial images. Set higher c like 1.
    """
    # load attack    
    for c in [0., 0.1, 0.2, 0.3, 0.5, 0.7]:
        writelog('----------EVAL CWL2 Attack ----------------', logfile)
        writelog('model_dir:{}'.format(model_dir), logfile)
        writelog('input bound: 0-1', logfile)
        writelog('CWL2 setting: c={}, kappa=0, steps=1000, lr=0.01'.format(c), logfile)

        cwl2_attack = torchattacks.CW(model, c=c, kappa=0, steps=1000, lr=0.01)
        cwl2_attack.set_return_type(type='float') # Return adversarial images with float value (0-1).
        
        num_correct = 0 
        num_samples = 0 
        for i in range(0, args.n_ex, args.batch_size):
            X, y = x_test[i:i+args.batch_size], y_test[i:i+args.batch_size]
            X.to(device)
            y.to(device)

            adv_images = cwl2_attack(X, y)
            
            outputs = model(adv_images)
            _, pre = torch.max(outputs.data, 1)

            num_correct += (pre == y).sum()
            num_samples += X.shape[0]
            adv_acc = num_correct / num_samples
            print('num_correct:{}, num_samples:{}, adv_acc:{}'.format(num_correct, num_samples, adv_acc))
        writelog('CWL2 Attack, adv_acc={:.4f}'.format(adv_acc), logfile)