
from logging import debug
import os
import time
import math
from tqdm import tqdm
from config import get_args

args = get_args()
#Check dataset
if args.dset == 'ImageNet-C':
    args.data = os.path.join(args.data_root, 'ImageNet')
    args.data_corruption = os.path.join(args.data_root, args.dset)
elif args.dset == 'Waterbirds':
    args.data_corruption = os.path.join(args.data_root, args.dset)
    for file in os.listdir(args.data_corruption):
        if file.endswith('h5py'):
            h5py_file = file
            break
    args.data_corruption_file = os.path.join(args.data_root, args.dset, h5py_file)
    
elif args.dset == 'ColoredMNIST':
    args.data_corruption = os.path.join(args.data_root, args.dset)
    
elif args.dset == 'Cifar10-C':  # Added handling for CIFAR10-C
    args.data_corruption = os.path.join(args.data_root, 'CIFAR-10-C')
    if not hasattr(args, 'corruption') or not hasattr(args, 'level'):
        raise ValueError("CIFAR10-C requires 'corruption_type' and 'severity' to be specified.")
else:
    raise ValueError("Wrong dataset")

biased = (args.exp_type == 'spurious')
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import json
import random
if args.wandb_log:
    import wandb
from datetime import datetime

import numpy as np
from pycm import *
from utils.utils import get_logger
from dataset.selectedRotateImageFolder import prepare_test_data
from utils.cli_utils import *

import torch    

from methods import tent, eata, sam, sar, deyo, deyo_new
import timm

import models.Res as Resnet
import pickle
from dataset.waterbirds_dataset import WaterbirdsDataset
from dataset.ColoredMNIST_dataset import ColoredMNIST
from dataset.cifar_dataset import CIFAR10C

from torchvision import datasets, transforms

def validate(val_loader, model, criterion, args):
    print("Evaluating model")
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix = "Test: "
    )
    
    if biased:
        LL_AM = AverageMeter('LL Acc', ':6.2f') 
        LS_AM = AverageMeter('LS Acc', ':6.2f')
        SL_AM = AverageMeter('SL Acc', ':6.2f')
        SS_AM = AverageMeter('SS Acc', ':6.2f')
        progress = ProgressMeter(
            len(val_loader),
            [batch_time, LL_AM, LS_AM, SL_AM, SS_AM],
            prefix='Test: ')

    end = time.time()
    correct_count = [0,0,0,0]
    total_count = [1e-6,1e-6,1e-6,1e-6]
    
    for i, dl in enumerate(tqdm(val_loader)):
        images, target = dl[0], dl[1]
        if torch.cuda.is_available():
            images = images.cuda()
            target = target.cuda()
        
        if biased:
            if args.dset == "Waterbirds":
                place = dl[2]['place'].cuda()
            else:
                place = dl[2].cuda()
            
            group = 2 * target + place
        else:
            group = None
        if args.method == "deyo" or args.method == "deyo_new":
            output = model(images, i, target, flag = False, group = group)
        else:
            output = model(images)
        
        
        if biased:
            TFtensor = (output.argmax(dim=1) == target)
            for group_idx in range(4):
                correct_count[group_idx] += TFtensor[group==group_idx].sum().item()
                total_count[group_idx] += len(TFtensor[group==group_idx])
            
            acc1, acc5 = accuracy(output, target, topk=(1, 1))
        else:
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
        
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        
        if (i + 1) % args.wandb_interval == 0:
            if biased:
                LL = correct_count[0]/total_count[0]*100
                LS = correct_count[1]/total_count[1]*100
                SL = correct_count[2]/total_count[2]*100
                SS = correct_count[3]/total_count[3]*100
                
                LL_AM.update(LL, images.size(0))
                LS_AM.update(LS, images.size(0))
                SL_AM.update(SL, images.size(0))
                SS_AM.update(SS, images.size(0))
                
            if args.wandb_log:
                wandb.log({f'{args.corruption}/LL': LL,
                            f'{args.corruption}/LS': LS,
                            f'{args.corruption}/SL': SL,
                            f'{args.corruption}/SS': SS,
                        })
            progress.display(i)
        
        batch_time.update(time.time() - end) 
        end = time.time()
    
    if biased:
        logger.info(f"- Detailed result under {args.corruption}. LL: {LL:.5f}, LS: {LS:.5f}, SL: {SL:.5f}, SS: {SS:.5f}")
        if args.wandb_log:
            wandb.log({'final_avg/LL': LL,
                       'final_avg/LS': LS,
                       'final_avg/SL': SL,
                       'final_avg/SS': SS,
                       'final_avg/AVG': (LL+LS+SL+SS)/4,
                       'final_avg/WORST': min(LL,LS,SL,SS)
                      })
        avg = (LL+LS+SL+SS)/4
        logger.info(f"Result under {args.corruption}. The adaptation accuracy of {args.method} is  average: {avg:.5f}")
        
        LLs.append(LL)
        LSs.append(LS)
        SLs.append(SL)
        SSs.append(SS)
        acc1s.append(avg)
        acc5s.append(min(LL,LS,SL,SS))
        
        logger.info(f"The LL accuracy are {LLs}")
        logger.info(f"The LS accuracy are {LSs}")
        logger.info(f"The SL accuracy are {SLs}")
        logger.info(f"The SS accuracy are {SSs}")
        
        logger.info(f"The average accuracy are {acc1s}")
        logger.info(f"The worst accuracy are {acc5s}")
    
    else:
        logger.info(f"Result under {args.corruption}. The adaptation accuracy of {args.method} is top1: {top1.avg:.5f} and top5: {top5.avg:.5f}")
        
        acc1s.append(top1.avg.item())
        acc5s.append(top5.avg.item())
        
        logger.info(f"acc1s are {acc1s}")
        logger.info(f"acc5s are {acc5s}")
    return top1.avg, top5.avg

if __name__ == "__main__":
    if args.dset=='ImageNet-C':
        args.num_class = 1000
    elif args.dset=='Waterbirds' or args.dset=='ColoredMNIST':
        args.num_class = 2
    elif args.dset == "Cifar10-C":
        args.num_class = 10 
    print('The number of classes:', args.num_class)
    if args.dset == 'Waterbirds':
        assert biased
        assert args.data_corruption_file.endswith('h5py')
        assert args.model == 'resnet50_bn_torch'
    
    elif args.dset == 'ColoredMNIST':
        assert biased
        assert args.model == 'resnet18_bn'
    
    elif args.dset == "Cifar10-C":
        assert args.model == "resnet18_bn"

    if biased:
        assert (args.dset == 'Waterbirds' or args.dset == 'ColoredMNIST')
        assert args.lr_mul == 5.0
        
    now = datetime.now()
    date_time = now.strftime("%m-%d-%H-%M-%S")
    
    total_top1 = AverageMeter('Acc@1', ':6.2f')
    total_top5 = AverageMeter('Acc@5', ':6.2f')
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
        
    args.logger_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + "-{}-{}-level{}-seed{}".format(args.method, args.model, args.level, args.seed) 
    if args.method == "deyo_new":
        args.logger_name += "-{}-{}".format(args.num_sim, args.alpha_cap)
    args.logger_name += ".txt"
    logger = get_logger(name="project", output_directory=args.output, log_name=args.logger_name, debug=False) 
    
    common_corruptions = [
                            'gaussian_noise',
                            'shot_noise',
                            'impulse_noise',
                            'defocus_blur', 
                            'glass_blur', 
                            'motion_blur', 
                            'zoom_blur', 
                            'snow', 
                            'frost', 
                            'fog', 
                            'brightness', 
                            'contrast', 
                            'elastic_transform', 
                            'pixelate', 
                            'jpeg_compression'
                            ]
    if biased:
        common_corruptions = ['spurious correlation']
    
    if args.exp_type == 'mix_shifts' and args.dset == 'ImageNet-C':
        datasets = []
        for cpt in common_corruptions:
            args.corruption = cpt
            logger.info(args.corruption)
            
            val_dataset, _ = prepare_test_data(args)
            if args.method in ['tent', 'no_adapt', 'eata', 'sar', 'deyo']:
                val_dataset.switch_mode(True, False)
            else:
                assert False, NotImplementedError
            
            datasets.append(val_dataset)
        
        from torch.utils.data import ConcatDataset
        mixed_dataset = ConcatDataset(datasets)
        logger.info(f"length of mixed dataset us {len(mixed_dataset)}")
        val_loader = torch.utils.data.DataLoader(mixed_dataset,
                                                 batch_size=args.test_batch_size,
                                                 shuffle=args.if_shuffle,
                                                 num_workers=args.workers, pin_memory=True)
        common_corruptions = ['mix_shifts']
        
    if args.exp_type == 'bs1':
        args.test_batch_size = 1
        logger.info("modify batch size to 1, for exp of single sample adaptation")
    
    if args.exp_type == 'label_shifts':
        args.if_shuffle = False
        logger.info("this exp is for label shifts, no need to shuffle the dataloader, use our pre-defined sample order")

    if args.method == 'eata' and (args.eata_fishers == 0 or args.fisher_alpha == 0):
        run_name = f'eta_lrmul{args.lr_mul}_ethr{args.deyo_margin}_dthr{args.plpd_threshold}' \
                   f'_emar{args.deyo_margin_e0}_seed{args.seed}'
    else:
        run_name = f'{args.method}_lrmul{args.lr_mul}_ethr{args.deyo_margin}_dthr{args.plpd_threshold}' \
                   f'_emar{args.deyo_margin_e0}_seed{args.seed}'

    if args.continual:
        run_name = f'{args.score_type}_{args.plpd_threshold}_{args.method}_continual_{date_time}'
    
    if args.wandb_log:
        wandb.init(
            project=f"{args.dset}_level{args.level}_{args.model}_{args.exp_type}",
            tags=['ideation'],
            config=args,
        )
        wandb.run.name = run_name

    args.e_margin *= math.log(args.num_class)
    args.sar_margin_e0 *= math.log(args.num_class)
    args.deyo_margin *= math.log(args.num_class)  # for thresholding
    args.deyo_margin_e0 *= math.log(args.num_class)  # for reweighting tuning
    
    acc1s, acc5s = [], []
    LLs, LSs, SLs, SSs = [], [], [], []
    ir = args.imbalance_ratio
    # Loop through corruption type
    for corrupt_i, corrupt in enumerate(common_corruptions):
        args.corruption = corrupt
        bs = args.test_batch_size
        args.print_freq = 50000 // 20 // bs
        
        #Load dataset
        if args.method in ['tent', 'eata', 'sar', 'deyo', 'no_adapt', "deyo_new"]:
            if (args.corruption != 'mix_shifts'):
                if args.dset == 'ImageNet-C':
                    val_dataset, val_loader = prepare_test_data(args)
                    val_dataset.switch_mode(True, False)
                elif args.dset == 'Waterbirds':
                    import torchvision.transforms as transforms
                    from torch.utils.data import DataLoader
                    transform = transforms.Compose(
                        [transforms.ToTensor(),
                         transforms.Resize((224, 224)),
                         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                        ]
                    )
                    val_dataset = WaterbirdsDataset(file=args.data_corruption_file, split='test', transform=transform)
                    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, 
                                                             shuffle=args.if_shuffle, num_workers=args.workers,
                                                             pin_memory=True)
                elif args.dset == "ColoredMNIST":
                    import torchvision.transforms as transforms
                    kwargs = {'num_workers': args.workers, 'pin_memory': True}
                    val_dataset = ColoredMNIST(root=args.data_corruption, env='test',  # flip=True,
                                               transform=transforms.Compose([
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                                               ]))
                    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, 
                                                             shuffle=args.if_shuffle, **kwargs)
                elif args.dset == "Cifar10-C":
                    import torchvision.transforms as transforms
                    kwargs = {'num_workers': args.workers, 'pin_memory': True}
                    val_dataset = CIFAR10C(root="/mnt/disk1/nam2/FedDG/DSU/multi-domain-generalization/DATA/cifar10_c/", 
                                           corruption_type=args.corruption, 
                                           severity=args.level,
                                           transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]))
                    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size,
                                                             shuffle=args.if_shuffle, **kwargs)
                else:
                    assert False, NotImplementedError
        else:
            assert False, NotImplementedError
        
        #Load model
        if args.method in ['tent', 'eata', 'sar', 'deyo', 'no_adapt', "deyo_new"]:
            if args.model == "resnet50_gn_timm":
                net = timm.create_model('resnet50_gn', pretrained=True)
                setattr(net, 'fc', net.head)
                args.lr = (0.00025 / 64) * bs * 2 if bs < 32 else 0.00025
            elif args.model == "vitbase_timm":
                net = timm.create_model('vit_base_patch16_224', pretrained=True)
                setattr(net, 'fc', net.head)
                args.lr = (0.001 / 64) * bs
            elif args.model == "resnet50_bn_torch":
                if args.dset == 'Waterbirds':
                    with open(os.path.join(args.pretrained_path), 'rb') as f:
                        net = pickle.load(f)
                elif args.dset == 'ImageNet-C':
                    net = Resnet.__dict__['resnet50'](pretrained=True)
                
                args.lr = (0.00025 / 64) * bs * 2 if bs < 32 else 0.00025
                args.lr *= args.lr_mul
            
            elif args.model == 'resnet18_bn':
                if args.dset == 'ColoredMNIST':
                    with open(args.pretrained_path, 'rb') as f:
                        net = pickle.load(f)
                elif args.dset == 'Cifar10-C':  
                    with open(args.pretrained_path, 'rb') as f:
                        net = pickle.load(f)
                args.lr = (0.00025 / 64) * bs * 2 if bs < 32 else 0.00025
                args.lr *= args.lr_mul
            else:
                assert False, NotImplementedError
            net = net.cuda()
        else:
            assert False, NotImplementedError
        
        if args.exp_type == 'bs1' and args.method == 'sar':
            args.lr = 2 * args.lr
            logger.info("double lr for sar under bs=1")
        
        if args.exp_type == 'bs1' and args.method == 'deyo':
            args.lr = 2 * args.lr
            logger.info("double lr for DeYO under bs=1")
        
        logger.info(args)
        
        if args.method == "tent":
            net = tent.configure_model(net)
            params, param_names = tent.collect_params(net)
            logger.info(param_names)
            optimizer = torch.optim.SGD(params, args.lr, momentum=0.9) 
            tented_model = tent.Tent(net, optimizer)
            acc1, acc5 = validate(val_loader, tented_model, None, args)
        elif args.method == "no_adapt":
            acc1, acc5 = validate(val_loader, net, None, args)
        elif args.method == "eata":
            if args.eata_fishers:
                print('EATA!')
            
                if args.dset == 'Waterbirds':
                    fisher_dataset = WaterbirdsDataset(file=args.data_corruption_file, split='train', transform=transform)
                    fisher_loader = torch.utils.data.DataLoader(fisher_dataset, batch_size=args.test_batch_size,
                                                                shuffle=args.if_shuffle, num_workers=args.workers,
                                                                pin_memory=True)
                elif args.dset == 'ColoredMNIST':
                    fisher_dataset = ColoredMNIST(root=args.data_corruption, env='all_train', flip=True,
                                                  transform=transforms.Compose([
                                                      transforms.ToTensor(),
                                                      transforms.Normalize((0.1307, 0.1307, 0.), (0.3081, 0.3081, 0.3081))
                                                  ]))
                    fisher_loader = torch.utils.data.DataLoader(fisher_dataset, batch_size=args.test_batch_size,
                                                                shuffle=args.if_shuffle, **kwargs)
                elif args.dset == "Cifar10-C":
                    import torchvision.transforms as transforms
                    kwargs = {'num_workers': args.workers, 'pin_memory': True}
                    fisher_dataset = CIFAR10C(root="/mnt/disk1/nam2/FedDG/DSU/multi-domain-generalization/DATA/cifar10_c/", 
                                           corruption_type=args.corruption, 
                                           severity=args.level,
                                           transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]))
                    fisher_loader = torch.utils.data.DataLoader(fisher_dataset, batch_size=args.test_batch_size,
                                                             shuffle=args.if_shuffle, **kwargs)
                else:
                    fisher_dataset, fisher_loader = prepare_test_data(args)
                fisher_dataset.set_dataset_size(args.fisher_size)
                fisher_dataset.switch_mode(True, False)

                net = eata.configure_model(net)
                print("Check eata config model")
                params, param_names = eata.collect_params(net)
                ewc_optimizer = torch.optim.SGD(params, 0.001)
                fishers = {}
                train_loss_fn = nn.CrossEntropyLoss().cuda()
                for iter_, data in enumerate(fisher_loader, start=1):
                    images, targets = data[0], data[1]
                    if args.gpu is not None:
                        images = images.cuda(non_blocking=True)
                    if torch.cuda.is_available():
                        targets = targets.cuda(non_blocking=True)
                    outputs = net(images)
                    _, targets = outputs.max(1)
                    loss = train_loss_fn(outputs, targets)
                    loss.backward()
                    for name, param in net.named_parameters():
                        if param.grad is not None:
                            if iter_ > 1:
                                fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                            else:
                                fisher = param.grad.data.clone().detach() ** 2
                            if iter_ == len(fisher_loader):
                                fisher = fisher / iter_
                            fishers.update({name: [fisher, param.data.clone().detach()]})
                    ewc_optimizer.zero_grad()
                logger.info("compute fisher matrices finished")
                del ewc_optimizer
            else:
                net = eata.configure_model(net)
                params, param_names = eata.collect_params(net)
                print('ETA!')
                fishers = None
            
            args.corruption = corrupt
            optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
            adapt_model = eata.EATA(args, net, optimizer, fishers, args.fisher_alpha, e_margin=args.e_margin, d_margin=args.d_margin)

            acc1, acc5 = validate(val_loader, adapt_model, None, args)
        elif args.method in ['sar']:
            net = sar.configure_model(net)
            params, param_names = sar.collect_params(net)
            logger.info(param_names)

            base_optimizer = torch.optim.SGD
            optimizer = sam.SAM(params, base_optimizer, lr=args.lr, momentum=0.9)
            adapt_model = sar.SAR(net, optimizer, margin_e0=args.sar_margin_e0)
            
            acc1, acc5 = validate(val_loader, adapt_model, None, args)
        elif args.method in ['deyo']:
            net = deyo.configure_model(net)
            params, param_names = deyo.collect_params(net)
            logger.info(param_names)

            optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
            adapt_model = deyo.DeYO(net, args, optimizer, deyo_margin=args.deyo_margin, margin_e0=args.deyo_margin_e0)
            acc1, acc5 = validate(val_loader, adapt_model, None, args)
        elif args.method in ['deyo_new']:
            net = deyo_new.configure_model(net)
            params, param_names = deyo_new.collect_params(net)
            logger.info(param_names)

            optimizer = torch.optim.SGD(params, args.lr, momentum=0.9)
            adapt_model = deyo_new.DeYO_Custom(net, args, optimizer, deyo_margin=args.deyo_margin, margin_e0=args.deyo_margin_e0)
            acc1, acc5 = validate(val_loader, adapt_model, None, args)
        else:
            assert False, NotImplementedError
        
        total_top1.update(acc1, 1)
        total_top5.update(acc5, 1)
    if not biased:
        logger.info(f"The average of top1 accuracy is {total_top1.avg}")
        logger.info(f"The average of top5 accuracy is {total_top5.avg}")
        if args.wandb_log:
            wandb.log({'final_avg/top1': total_top1.avg,
                       'final_avg/top5': total_top5.avg})

            wandb.finish()