import argparse
import numpy as np
import time
from distutils.util import strtobool

import torch
from torch import optim, nn
from torchvision import datasets, transforms

import util
import network
from data import dataloader
from optimizers import SGD, Momentum, Adam

parser = argparse.ArgumentParser(description="Simulator for Neuromorphic Computing")

# device parameters
parser.add_argument('--FormulaType', type=int, default=0, choices=[0, 1, 2],
                    help="type of conductance modeling formula (0: log modeling / 1: exponential modeling) / 2: symmetric modeling")
parser.add_argument('--MaxPulse', type=int, default=500, help="max pulse of device")
parser.add_argument('--Gmin', type=float, default=0.5, help="minimum conductance of device")
parser.add_argument('--Gmax', type=float, default=15.5, help="maximum conductance of device")
parser.add_argument('--NonlinearityLTP', type=float, default=1.0, help="nonlinearity of LTP formula")
parser.add_argument('--NonlinearityLTD', type=float, default=1.0, help="nonlinearity of LTD formula")
parser.add_argument('--c2c_vari', type=float, default=0.0, help="cycle-to-cycle variation")
parser.add_argument('--d2d_vari', type=float, default=0.0, help="device-to-device variation")

# setting network, weight, conductance, pulse, cuda
parser.add_argument('--network', type=str, default='plain',
                    choices=['plain', 'resnet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'], help="network type")
parser.add_argument('--w_init_factor', type=float, default=1.0, help="factor of kaiming normal initialization")
parser.add_argument('--fixed_normalization', type=lambda x:bool(strtobool(x)), default=False,
                    help="type of conductance normalization (True: fixed normalization, False: layer-wise normalization)")
parser.add_argument('--w_dist_scale', type=float, default=1.5, help="scale of weight distribution (using only in layer-wise normalization)")
parser.add_argument('--weight_representation', type=str, default='bi', choices=['uni', 'bi'],
                    help="choosing weight representation between uni-directional method and bi-directional method")
parser.add_argument('--pulse_update', type=str, default='multi', choices=['multi', 'one'], help="pulse update type")
parser.add_argument('--delta_pulse_representation', type=int, default=1, choices=[0, 1],
                    help="type of delta pulse representation (0: pusle update, 1: normalized grad pulse update)")
parser.add_argument('--cuda', type=str, default='cuda:0', help="cuda type")

# training hyperparameters
parser.add_argument('--seed', type=int, default=0, help="fixing random seed")
parser.add_argument('--batch_size', type=int, default=200, help="batch size of dataloader for training")
parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Momentum', 'Adam'], help="optimizer type")
parser.add_argument('--loss', type=str, default='SSE', choices=['SSE', 'MSE', 'CE'], help="loss function")
parser.add_argument('--epochs', type=int, default=50, help="training epochs")
parser.add_argument('--learning_rate', type=float, default=1e-2, help="learning rate for training")
parser.add_argument('--lr_decay', type=str, default="30,40", help="epochs at which learning rate is going to be reduced")
parser.add_argument('--lr_decay_scale', type=float, default=0.125, help="scale of learning rate decay")
parser.add_argument('--weight_decay', type=float, default=0.0, help="a parameter of L2-norm regularization")
parser.add_argument('--batch_normalization', type=lambda x:bool(strtobool(x)), default=False, help="Whether batch normalization is used")
# Mometum
parser.add_argument('--beta', type=float, default=0.9, help="scale of momentum vector")
parser.add_argument('--dampening', type=float, default=0.0, help="scale of gradient")
# Adam
parser.add_argument('--betas', type=str, default="0.9,0.999", help="scale of first, second momentum")
parser.add_argument('--eps', type=float, default=1e-8, help="preventing denominator is 0")

# setting data set
parser.add_argument('--data_root', type=str, default='/workspace/Simulator_codes/dataset', help="root in which data will be saved")
parser.add_argument('--data_type', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'mnist'], help="dataset for training")
parser.add_argument('--data_normalization', type=lambda x:bool(strtobool(x)), default=True, help="apply normalization to dataset")
parser.add_argument('--data_Grayscale', type=lambda x:bool(strtobool(x)), default=False, help="using grayscaled dataset")
parser.add_argument('--binary_mnist', type=lambda x:bool(strtobool(x)), default=False, help="using binarized MNIST")
parser.add_argument('--image_CenterCrop', type=str, default='None', help="size of center-croped images (None: no croping)")
parser.add_argument('--image_Resize', type=str, default='None', help="size of images resized with bilinear interpolation (None: no resizing)")

# data augmentation
parser.add_argument('--image_RandomCrop', type=str, default='None', help="size of random-croped images (None: no croping)")
parser.add_argument('--image_RandomHorizontalFlip', type=lambda x:bool(strtobool(x)), default=True, help="apply random horizontal flip to images")
parser.add_argument('--image_RandomVerticalFlip', type=lambda x:bool(strtobool(x)), default=False, help="apply random vertical flip to images")
parser.add_argument('--image_RandomRotation', type=str, default='None', help="degree of image rotation (None: no croping)")

args = parser.parse_args()

# seed fix
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

args.lr_decay = util.str_to_list(args.lr_decay, int)
args.betas = util.str_to_list(args.betas, float)
args.image_CenterCrop = util.str_to_int(args.image_CenterCrop)
args.image_Resize = util.str_to_int(args.image_Resize)
args.image_RandomCrop = util.str_to_int(args.image_RandomCrop)
args.image_RandomRotation = util.str_to_int(args.image_RandomRotation)

print("------------------- Device Parameters --------------------")
if args.FormulaType == 0:
    print("Formula Type: Log modeling")
elif args.FormulaType == 1:
    print("Formula Type: Exponential modeling")
elif args.FormulaType == 2:
    print("Formula Type: Symmetric modeling")
print(f"Max Pulse: {args.MaxPulse} \nG-min: {args.Gmin} \nG-max: {args.Gmax}")
print(f"Nonlinearity LTP / LTD: {args.NonlinearityLTP} / {args.NonlinearityLTD}")
print(f"cycle-to-cycle variation: {args.c2c_vari}")
print(f"device-to-device variation: {args.d2d_vari}")
print()

print("------------- Setting Network and Conductance ------------")
print(f"Network: {args.network}")
if args.weight_representation == 'bi':
    print("Weight representation type: Bi-directional method")
else:
    print("Weight representation type: Uni-directional method (single conductance method)")
print(f"Weight initialization factor: {args.w_init_factor}")
print(f"Fixed normalization: {args.fixed_normalization}")
if args.fixed_normalization == False:
    print(f"Weight distribution scale: {args.w_dist_scale}")
print(f"Pulse update: {args.pulse_update}")
if args.pulse_update == 'multi':
    if args.delta_pulse_representation == 0:
        print("Delta pulse representation: Pusle Update")
    elif args.delta_pulse_representation == 1:
        print("Delta pulse representation: Normalized Grad Pulse Update")
print(f"Cuda: {args.cuda}")
print()

print("--------------------- Hyperparameters --------------------")
print(f"seed: {args.seed} \nbatch size: {args.batch_size}")
print(f"optimizer: {args.optimizer} \nloss function: {args.loss} \nepochs: {args.epochs}")
print(f"learning rate: {args.learning_rate} \nlearning rate decay: {args.lr_decay}")
print(f"learning rate decay scale: {args.lr_decay_scale} \nweight decay: {args.weight_decay}")
if args.optimizer == 'Momentum':
    print(f"beta: {args.beta} \ndampening: {args.dampening}")
elif args.optimizer == 'Adam':
    print(f"betas: {args.betas} \nepsilon: {args.eps}")
print()

print("---------------------- Loading Data ----------------------")
print(f"Dataset: {args.data_type}")
print(f"root of data: {args.data_root}")
if args.data_type == 'mnist': 
    print(f"binarized MNIST: {args.binary_mnist}")
print(f"Dataset Normalization: {args.data_normalization}")
print(f"Dataset Grayscale: {args.data_Grayscale}")
print(f"Image CenterCrop (size): {args.image_CenterCrop}")
print(f"Image Resize (size): {args.image_Resize}")

print(f"Image RandomCrop (size): {args.image_RandomCrop}")
print(f"Image RandomHorizontalFlip: {args.image_RandomHorizontalFlip}")
print(f"Image RandomVerticalFlip: {args.image_RandomVerticalFlip}")
print(f"Image RandomRotation (degree): {args.image_RandomRotation}")
print()

train_loader, test_loader = dataloader(args=args, shuffle=True, num_workers=1)
print()

print("---------------------------------- Loading Network Model ----------------------------------")
model = network.select_network(args)
print(model)

device = args.cuda if torch.cuda.is_available() else 'cpu'
model.to(device)

for name in list(model.conductances.keys()):
    model.conductances[name] = model.conductances[name].to(device)
    model.nonlinearities_LTP[name] = model.nonlinearities_LTP[name].to(device)
    model.nonlinearities_LTD[name] = model.nonlinearities_LTD[name].to(device)
    model.normalization_scales[name] = model.normalization_scales[name].to(device)

if args.loss == 'SSE':
    criterion = util.SSE
elif args.loss == 'MSE':
    criterion = util.MSE
else:
    criterion = nn.CrossEntropyLoss()

if args.batch_normalization:
    bn_parameters = []
    for i in range(len(list(model.parameters()))-1):
        bn_parameters += list(model.parameters())[3*i + 1: 3*i + 3]

    if args.optimizer == 'SGD':
        bn_optimizer = optim.SGD(bn_parameters, lr=args.learning_rate)
    elif args.optimizer == 'Momentum':
        bn_optimizer = optim.SGD(bn_parameters, lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    else:
        bn_optimizer = optim.Adam(bn_parameters, lr=args.learning_rate, betas=[0.9, 0.999], weight_decay=args.weight_decay)

    bn_scheduler = optim.lr_scheduler.MultiStepLR(bn_optimizer, milestones=args.lr_decay, gamma=args.lr_decay_scale)

step = 1
if args.optimizer == 'Momentum':
    momentum = {}
    for i, weight in enumerate(list(model.parameters())[::-1]):
        momentum[i] = torch.zeros_like(weight)
elif args.optimizer == 'Adam':
    first_momentum = {}
    second_momentum = {}
    for i, weight in enumerate(list(model.parameters())[::-1]):
        first_momentum[i] = torch.zeros_like(weight)
        second_momentum[i] = torch.zeros_like(weight)

print()
print("---------------------- Training Start -----------------------")

if args.batch_normalization:
    n_param = 3
else:
    n_param = 1

best_acc = 0
start_time = time.time()

for epoch in range(args.epochs):
    if args.lr_decay:
        if epoch in args.lr_decay:
            args.learning_rate *= args.lr_decay_scale

    # train
    model.train()

    train_loss = 0
    for x_train, y_train in train_loader:
        x_train = x_train.to(device)
        y_train = y_train.to(device)
        
        model.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        
        if args.batch_normalization:
            bn_optimizer.step()

        for i, (w, (name, cdt), nl_LTP, nl_LTD, ns) in enumerate(zip(list(model.parameters())[::-n_param],
                                                                    list(model.conductances.items())[::-1],
                                                                    list(model.nonlinearities_LTP.values())[::-1],
                                                                    list(model.nonlinearities_LTD.values())[::-1],
                                                                    list(model.normalization_scales.values())[::-1])):
            if args.optimizer == 'SGD':
                w.data, model.conductances[name] = SGD(args, w.data, w.grad.data, cdt, nl_LTP, nl_LTD, ns)
            elif args.optimizer == 'Momentum':
                w.data, model.conductances[name], momentum[i] = Momentum(
                    args, w.data, w.grad.data, cdt, nl_LTP, nl_LTD, ns, momentum[i], step)
            else:
                w.data, model.conductances[name], first_momentum[i], second_momentum[i] = Adam(
                    args, w.data, w.grad.data, cdt, nl_LTP, nl_LTD, ns, first_momentum[i], second_momentum[i], step)

        step += 1
        train_loss += loss.item()
        model.train_loss_per_batch.append(loss.item())

    if args.batch_normalization:
        bn_scheduler.step()

    train_loss /= (step-1)/(epoch+1)
    model.train_loss_per_epoch.append(train_loss)

    # test
    model.eval()

    test_acc = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test = x_test.to(device)
            y_test = y_test.to(device)

            outputs = model(x_test)
            _, predicted = outputs.max(dim=1)
            total += y_test.size(dim=0)
            correct += predicted.eq(y_test).sum().item()

    model.test_acc_per_epoch.append(correct / total * 100)
    if best_acc < model.test_acc_per_epoch[-1]:
        best_acc = model.test_acc_per_epoch[-1]

    print("Epoch {})\tTrain loss: {:.3f}\tTest accuracy: {:.2f}%".format(
        epoch+1, round(model.train_loss_per_epoch[-1], 3), round(model.test_acc_per_epoch[-1], 3)))

elapsed_time = util.time_hms(time.time()-start_time)
print(f"Elapsed time: {elapsed_time}\tBest accuracy: {round(best_acc, 3)}")




