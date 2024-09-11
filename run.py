from main_new import main
from logging import debug
import os
import time
import math
from config import get_args

args = get_args()
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
elif args.dset == 'CIFAR10-C':  # Added handling for CIFAR10-C
    args.data_corruption = os.path.join(args.data_root, 'CIFAR-10-C')
    if not hasattr(args, 'corruption_type') or not hasattr(args, 'severity'):
        raise ValueError("CIFAR10-C requires 'corruption_type' and 'severity' to be specified.")

biased = (args.exp_type == 'spurious')

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if __name__ == '__main__':
#   num_sims = [i for i in range(3, 10)]
    num_sims = [i for i in range(6, 10)]
    alpha_caps = [
                # 0.001,
                # 0.002, 
                # 0.004, 
                # 0.008, 
                # 0.016, 
                # 0.03125, 
                # 0.05, 
                0.075, 
                0.1
                ]
    for num_sim in num_sims:
        for alpha_cap in alpha_caps:
            args.num_sim = num_sim
            args.alpha_cap = alpha_cap
            main(args)