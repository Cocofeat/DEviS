import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import torch.optim as optim
from torch.autograd import Variable
import logging
import time
from torch.utils.data import DataLoader
from PIL import Image
from models.TrustworthySeg import TMSU
from data.transform import ISIC2018_transform,LiTS2017_transform,DRIVE2004_transform,TOAR2019_transform,HC_2018_transform
from data.BraTS2019 import BraTS
from data.ISIC2018 import ISIC
from data.COVID19 import Covid
from data.CHAOS20 import CHAOS
from data.LiTS17 import LiTS
from data.DRIVE04 import DRIVE
from data.TOAR19 import TOAR
from data.HC2018 import HC

import cv2
from thop import profile
from models.criterions import get_soft_label
from predict import tailor_and_concat,RandomMaskingGenerator,softmax_output_litsdice,softmax_output_litshd,softmax_assd_litsscore,softmax_mIOU_litsscore,Uentropy_our,cal_ueo,cal_ece_our,softmax_mIOU_score,softmax_output_dice,softmax_output_hd,dice_isic,iou_isic,HD_isic,Dice_isic,IOU_isic,softmax_assd_score
from binary import assd,hd95
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import argparse
import nibabel as nib
import imageio
from plot import loss_plot,metrics_plot

def getArgs():
    local_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    parser = argparse.ArgumentParser()
    # Basic Information
    parser.add_argument('--user', default='name of user', type=str)
    parser.add_argument('--experiment', default='UMIS', type=str) # BraTS ISIC COVID
    parser.add_argument('--date', default=local_time.split(' ')[0], type=str)
    parser.add_argument('--description',
                        default='Trustworthy medical image segmentation by coco,'
                                'training on train.txt!',
                        type=str)
    # training detalis
    parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--end_epochs', type=int, default=99, metavar='N',
                        help='number of epochs to train [default: 500]')
    parser.add_argument('--submission', default='./results', type=str)

    parser.add_argument('--lambda-epochs', type=int, default=50, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')
    parser.add_argument('--save_freq', default=5, type=int)
    parser.add_argument('--lr', type=float, default=0.002, metavar='LR',
                        help='learning rate')# BraTS: 0.0002 # ISIC: 0.0002
    # DataSet Information
    parser.add_argument('--savepath', default='./results/plot/output', type=str)
    parser.add_argument('--save_dir', default='./results', type=str)
    parser.add_argument("--mode", default="train&test", type=str, help="train/test/train&test")
    parser.add_argument('--dataset', default='LiTS', type=str, help="BraTS/ISIC/LiTS/DRIVE/HC") #
    parser.add_argument("--folder", default="folder0", type=str, help="folder0/folder1/folder2/folder3/folder4")
    parser.add_argument('--input_H', default=240, type=int)
    parser.add_argument('--input_W', default=240, type=int)
    parser.add_argument('--input_D', default=160, type=int)  # 155
    parser.add_argument('--crop_H', default=128, type=int)
    parser.add_argument('--crop_W', default=128, type=int)
    parser.add_argument('--crop_D', default=128, type=int)
    parser.add_argument('--output_D', default=155, type=int)
    parser.add_argument('--batch_size', default=8, type=int, help="2/4/8/16")
    parser.add_argument('--OOD_Condition', default='normal', type=str, help="normal/noise/mask/blur/")
    parser.add_argument('--OOD_Level', default=1, type=int, help="0: 'No',1:'Low', 2:'Upper Low', 3:'Mid', 4:'Upper Mid', 5:'High'")
    # parser.add_argument('--OOD_Variance', default=2, type=int)
    parser.add_argument('--snapshot', default=True, type=bool, help="True/False")  # visualization results
    parser.add_argument('--Uncertainty_Loss', default=False, type=bool, help="True/False")  # adding uncertainty_loss
    parser.add_argument('--input_modality', default='four', type=str, help="t1/t2/both/four")  # Single/multi-modal
    parser.add_argument('--model_name', default='U', type=str, help="U/V/AU/TransU/ViT/")  # multi-modal
    parser.add_argument('--test_epoch', type=int, default=199, metavar='N',
                        help='best epoch')
    # for ViT
    # parser.add_argument('--n_skip', type=int,
    #                     default=3, help='using number of skip-connect, default is num')
    # parser.add_argument('--vit_name', type=str,
    #                     default='R50-ViT-B_16', help='select one vit model')
    # parser.add_argument('--vit_patches_size', type=int,
    #                     default=8, help='vit_patches_size, default is 16')
    args = parser.parse_args()
    # args.dims = [[240,240,160], [240,240,160]]
    # args.modes = len(args.dims)
    return args
def getDataset(args):
    if args.dataset=='BraTS':
        exp_folder =  args.folder
        train_file='MICCAI_BraTS_2019_Data_Training/Ttrain_subject.txt'
        valid_file='MICCAI_BraTS_2019_Data_Training/Tval_subject.txt'
        test_file ='MICCAI_BraTS_2019_Data_Training/Ttest_subject.txt'
        root='E:/BraTSdata1/archive2019/'
        # root = 'E:/Coco_file/BraTSdata/archive2019/'
        train_dir='MICCAI_BraTS_2019_Data_TTraining'
        valid_dir='MICCAI_BraTS_2019_Data_TValidation'
        test_dir='MICCAI_BraTS_2019_Data_TTest'
        train_list = os.path.join(root, train_file)
        train_root = os.path.join(root, train_dir)
        train_set = BraTS(train_list, train_root, args.mode,args.input_modality,OOD_Condition=args.OOD_Condition, level=args.OOD_Level,
                       folder=exp_folder)
        valid_list = os.path.join(root, valid_file)
        valid_root = os.path.join(root, valid_dir)
        valid_set = BraTS(valid_list, valid_root,'valid',args.input_modality,OOD_Condition=args.OOD_Condition, level=args.OOD_Level,
                       folder=exp_folder)
        test_list = os.path.join(root, test_file)
        test_root = os.path.join(root, test_dir)
        test_set = BraTS(test_list, test_root,'test',args.input_modality,OOD_Condition=args.OOD_Condition, level=args.OOD_Level,
                       folder=exp_folder)
        train_loader = DataLoader(dataset=train_set, batch_size=args.batch_size)
        valid_loader = DataLoader(valid_set, batch_size=1)
        test_loader = DataLoader(test_set, batch_size=1)
        print('Samples for train = {}'.format(len(train_loader.dataset)))
        print('Samples for valid = {}'.format(len(valid_loader.dataset)))
        print('Samples for test = {}'.format(len(test_loader.dataset)))
    elif args.dataset == 'ISIC':
        root_path = 'E:/Coco_file/ISIC2018_Task1_npy_all/'
        exp_folder = args.folder

        trainset = ISIC(dataset_folder=root_path, folder=exp_folder, train_type='train',
                        transform=ISIC2018_transform)
        validset = ISIC(dataset_folder=root_path, folder=exp_folder,
                        train_type='val',
                        transform=ISIC2018_transform)
        train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        valid_loader = DataLoader(dataset=validset, batch_size=1, shuffle=True, pin_memory=True)
        testset = ISIC(OOD_Condition=args.OOD_Condition, Level=args.OOD_Level, dataset_folder=root_path,
                       folder=exp_folder, train_type='test',
                       transform=ISIC2018_transform)
        test_loader = DataLoader(dataset=testset, batch_size=1, shuffle=False, pin_memory=True)
        print('Samples for train = {}'.format(len(train_loader.dataset)))
        print('Samples for valid = {}'.format(len(valid_loader.dataset)))
        print('Samples for test = {}'.format(len(test_loader.dataset)))
    elif args.dataset == 'LiTS':
        root_path = 'E:/Coco_file/LiTS/'
        exp_folder = args.folder
        trainset = LiTS(dataset_folder=root_path, train_type='train', folder=exp_folder,
                         transform=LiTS2017_transform)
        validset = LiTS(dataset_folder=root_path, train_type='val', folder=exp_folder,
                         transform=LiTS2017_transform)
        train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        valid_loader = DataLoader(dataset=validset, batch_size=1, shuffle=True, pin_memory=True)
        testset = LiTS(OOD_Condition=args.OOD_Condition, Level=args.OOD_Level, dataset_folder=root_path, train_type='test',
                        folder=exp_folder,
                        transform=LiTS2017_transform)
        test_loader = DataLoader(dataset=testset, batch_size=1, shuffle=False, pin_memory=True)
        print('Samples for train = {}'.format(len(train_loader.dataset)))
        print('Samples for valid = {}'.format(len(valid_loader.dataset)))
        print('Samples for test = {}'.format(len(test_loader.dataset)))
    elif args.dataset == 'DRIVE':
        root_path = 'E:/Coco_file/DRIVE/'
        exp_folder = args.folder
        trainset = DRIVE(dataset_folder=root_path, train_type='train', folder=exp_folder,
                         transform=DRIVE2004_transform)
        validset = DRIVE(dataset_folder=root_path, train_type='val', folder=exp_folder,
                         transform=DRIVE2004_transform)
        train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        valid_loader = DataLoader(dataset=validset, batch_size=1, shuffle=True, pin_memory=True)
        testset = DRIVE(OOD_Condition=args.OOD_Condition, Level=args.OOD_Level, dataset_folder=root_path, train_type='test',
                        folder=exp_folder,
                        transform=DRIVE2004_transform)
        test_loader = DataLoader(dataset=testset, batch_size=1, shuffle=False, pin_memory=True)
        print('Samples for train = {}'.format(len(train_loader.dataset)))
        print('Samples for valid = {}'.format(len(valid_loader.dataset)))
        print('Samples for test = {}'.format(len(test_loader.dataset)))
    elif args.dataset == 'HC':
        root_path = 'E:/Coco_file/HC/'
        exp_folder = args.folder
        trainset = HC(dataset_folder=root_path, train_type='train', folder=exp_folder,
                         transform=HC_2018_transform)
        validset = HC(dataset_folder=root_path, train_type='val', folder=exp_folder,
                         transform=HC_2018_transform)
        train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        valid_loader = DataLoader(dataset=validset, batch_size=1, shuffle=True, pin_memory=True)
        testset = HC(OOD_Condition=args.OOD_Condition, Level=args.OOD_Level, dataset_folder=root_path, train_type='test',
                        folder=exp_folder,
                        transform=HC_2018_transform)
        test_loader = DataLoader(dataset=testset, batch_size=1, shuffle=False, pin_memory=True)
        print('Samples for train = {}'.format(len(train_loader.dataset)))
        print('Samples for valid = {}'.format(len(valid_loader.dataset)))
        print('Samples for test = {}'.format(len(test_loader.dataset)))
    else:
        train_loader=None
        valid_loader=None
        test_loader=None
        print('There is no this dataset')
        raise NameError
    return train_loader,valid_loader,test_loader

def log_args(log_file):

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s ===> %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    # args FileHandler to save log file
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # args StreamHandler to print log to console
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # add the two Handler
    logger.addHandler(ch)
    logger.addHandler(fh)

def adjust_learning_rate(optimizer, epoch, max_epoch, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1 - (epoch) / max_epoch, power), 8)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(args,model,optimizer,epoch,train_loader):
    model.train()
    loss_meter = AverageMeter()
    step = 0
    dt_size = len(train_loader.dataset)
    for i, data in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, args.end_epochs, args.lr)
        step += 1
        input, target = data
        x = input.cuda()  # for multi-modal combine train
        target = target.cuda()
        # refresh the optimizer

        args.mode = 'train'
        # m = torch.unique(x)
        # n = torch.unique(target)
        evidences, loss = model(x, target, epoch, args.mode,args.dataset)

        print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // train_loader.batch_size + 1, loss.item()))
        # compute gradients and take step
        optimizer.zero_grad()
        loss.requires_grad_(True).backward()
        optimizer.step()
        loss_meter.update(loss.item())
    return loss_meter.avg

def val(args, model, current_epoch, best_dice, valid_loader):
    print('===========>Validation begining!===========')
    model.eval()
    loss_meter = AverageMeter()
    num_classes = args.num_classes
    dice_total, iou_total = 0, 0
    step = 0
    # model.eval()
    for i, data in enumerate(valid_loader):
        step += 1
        input, target = data

        x = input.cuda()  # for multi-modal combine train
        target = target.cuda()
        gt = target.cuda()
        with torch.no_grad():
            args.mode = 'val'
            if args.dataset =='BraTS':
                evidence = model(x, target[:, :, :, :155], current_epoch, args.mode, args.dataset)  # two input_modality or four input_modality
            else:
                evidence = model(x, target, current_epoch, args.mode,  args.dataset)
            alpha = evidence + 1

            S = torch.sum(alpha, dim=1, keepdim=True)
            prob = alpha / S
            _, predicted = torch.max(prob, 1)

            output = torch.squeeze(predicted).cpu().detach().numpy()
            target = torch.squeeze(target).cpu().numpy()
            if args.dataset == 'BraTS':
                iou_res = softmax_mIOU_score(output, target[:, :, :155])
                dice_res = softmax_output_dice(output, target[:, :, :155])
                dice_total += dice_res[1]
                iou_total += iou_res[1]
            elif args.dataset == 'LiTS':
                dice_res = softmax_output_litsdice(output, target)
                iou_res = softmax_mIOU_litsscore(output, target)
                dice_total += dice_res[0]
                iou_total += iou_res[0]
            else:
                soft_gt = get_soft_label(gt, num_classes)
                soft_predicted = get_soft_label(predicted.unsqueeze(0), num_classes)
                iou_res = IOU_isic(soft_predicted, soft_gt,num_classes)
                dice_res = Dice_isic(soft_predicted, soft_gt,num_classes)
                dice_total += dice_res
                iou_total += iou_res
            print('current_iou:{} ; current_dice:{}'.format(iou_res, dice_res))
    aver_dice = dice_total / len(valid_loader)
    aver_iou = iou_total / len(valid_loader)
    if aver_dice > best_dice \
            or (current_epoch + 1) % int(args.end_epochs - 1) == 0 \
            or (current_epoch + 1) % int(args.end_epochs - 2) == 0 \
            or (current_epoch + 1) % int(args.end_epochs - 3) == 0:
        print('aver_dice:{} > best_dice:{}'.format(aver_dice, best_dice))
        best_dice = aver_dice
        print('===========>save best model!')
        if args.Uncertainty_Loss:
            file_name = os.path.join(args.save_dir, args.model_name +'_'+args.dataset +'_'+ args.folder + '_DUloss'+'_epoch_{}.pth'.format(current_epoch))
        else:
            file_name = os.path.join(args.save_dir,
                                     args.model_name + '_' + args.dataset +'_'+ args.folder + '_epoch_{}.pth'.format(current_epoch))
        torch.save({
            'epoch': current_epoch,
            'state_dict': model.state_dict(),
        },
            file_name)
    return loss_meter.avg, best_dice

def test(args,model,test_loader):
    Net_name = args.model_name
    snapshot = args.snapshot  # False
    print('===========>Test begining!===========')
    logging.info('===========>Test begining!===========')
    logging.info('--------------------------------------This is all argsurations----------------------------------')
    for arg in vars(args):
        logging.info('{}={}'.format(arg, getattr(args, arg)))
    logging.info('----------------------------------------This is a halving line----------------------------------')
    logging.info('{}'.format(args.description))
    if args.Uncertainty_Loss:
        savepath = args.submission + '/'+ str(Net_name) + 'eviloss'  + '/' + str(args.dataset) +'/' + str(args.OOD_Condition) +'/'+ str(args. OOD_Level)
    else:
        savepath = args.submission + '/' + str(Net_name)+ 'evi' + '/' + str(args.dataset) +'/' + str(args.OOD_Condition) + '/' + str(args.OOD_Level)
    dice_total = 0
    dice_total_WT = 0
    dice_total_TC = 0
    dice_total_ET = 0
    hd_total = 0
    hd95_total = 0
    assd_total = 0
    hd_total_WT = 0
    hd_total_TC = 0
    hd_total_ET = 0
    assd_total_WT = 0
    assd_total_TC = 0
    assd_total_ET = 0
    noise_dice_total = 0
    noise_dice_total_WT = 0
    noise_dice_total_TC = 0
    noise_dice_total_ET = 0
    iou_total = 0
    iou_total_WT = 0
    iou_total_TC = 0
    iou_total_ET = 0
    noise_iou_total = 0
    noise_iou_total_WT = 0
    noise_iou_total_TC = 0
    noise_iou_total_ET = 0
    noise_hd_total = 0
    noise_assd_total = 0
    noise_hd_total_WT = 0
    noise_hd_total_TC = 0
    noise_hd_total_ET = 0
    noise_assd_total_WT = 0
    noise_assd_total_TC = 0
    noise_assd_total_ET = 0
    runtimes = []
    certainty_total = 0
    noise_certainty_total = 0
    mne_total = 0
    noise_mne_total = 0
    ece_total = 0
    noise_ece_total = 0
    ueo_total = 0
    noise_ueo_total = 0
    step = 0


    dt_size = len(test_loader.dataset)
    if args.Uncertainty_Loss:
        load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                 args.save_dir,
                                 args.model_name + '_' + args.dataset +'_'+ args.folder + '_DUloss' + '_epoch_{}.pth'.format(args.test_epoch))
    else:
        load_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                 args.save_dir,
                                 args.model_name + '_' + args.dataset +'_'+ args.folder + '_epoch_{}.pth'.format(args.test_epoch))

    if os.path.exists(load_file):
        checkpoint = torch.load(load_file)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        print('Successfully load checkpoint {}'.format(
            os.path.join(args.save_dir + '/' + args.model_name +'_'+args.dataset+ '_epoch_' + str(args.test_epoch))))
    else:
        print('There is no resume file to load!')
    names = test_loader.dataset.image_list

    model.eval()
    for i, data in enumerate(test_loader):
        msg = 'Subject {}/{}, '.format(i + 1, len(test_loader))

        step += 1
        input, noised_input, target = data  # input ground truth

        if args.dataset == 'BraTS':
            num_classes = 4
        elif args.dataset == 'LiTS':
            num_classes = 3
        elif args.dataset == 'TOAR':
            num_classes = 7
        elif args.dataset == 'HC':
            num_classes = 9
        else:
            num_classes = 2
        x = input.cuda()
        noised_x = noised_input.cuda()
        target = target.cuda()
        torch.cuda.synchronize()  # add the code synchronize() to correctly count the runtime.
        start_time = time.time()
        with torch.no_grad():
            args.mode = 'test'
            if args.dataset =='BraTS':
                evidences = model(x, target[:, :, :, :155], 0, args.mode, args.dataset)
                noised_evidences = model(noised_x, target[:, :, :, :155], 0, args.mode, args.dataset)
            else:
                evidences = model(x, target, 0, args.mode, args.dataset)
                noised_evidences = model(noised_x, target, 0, args.mode, args.dataset)
            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            logging.info('Single sample test time consumption {:.2f} seconds!'.format(elapsed_time))
            runtimes.append(elapsed_time)
            alpha = evidences + 1
            uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)

            S = torch.sum(alpha, dim=1, keepdim=True)
            prob = alpha / S
            mne = Uentropy_our(prob, num_classes)
            # min_mne = torch.min(mne)
            # max_mne = torch.max(mne)

            _, predicted = torch.max(alpha / S, 1)
            # for noise_x
            noised_alpha = noised_evidences + 1
            noised_uncertainty = num_classes / torch.sum(noised_alpha, dim=1, keepdim=True)

            _, noised_predicted = torch.max(noised_evidences.data, 1)
            noised_prob = noised_alpha / torch.sum(noised_alpha, dim=1, keepdim=True)
            noised_mne = Uentropy_our(noised_prob, num_classes)
            noised_output = torch.squeeze(noised_predicted).cpu().detach().numpy()
            output = torch.squeeze(predicted).cpu().detach().numpy()

            if args.dataset == 'BraTS':
                ece = cal_ece_our(torch.squeeze(predicted), torch.squeeze(target[:, :, :, :155].cpu()))
                noise_ece = cal_ece_our(torch.squeeze(noised_predicted), torch.squeeze(target[:, :, :, :155].cpu()))
                H, W, T = 240, 240, 155
                Otarget = torch.squeeze(target[:, :, :, :155]).cpu().numpy()
                target = torch.squeeze(target).cpu().numpy()  # .cpu().numpy(dtype='float32')
                hd_res = softmax_output_hd(output, target[:, :, :155])
                assd_res = softmax_assd_score(output, target[:, :, :155])
                iou_res = softmax_mIOU_score(output, target[:, :, :155])
                dice_res = softmax_output_dice(output, target[:, :, :155])
                dice_total_WT += dice_res[0]
                dice_total_TC += dice_res[1]
                dice_total_ET += dice_res[2]
                iou_total_WT += iou_res[0]
                iou_total_TC += iou_res[1]
                iou_total_ET += iou_res[2]
                hd_total_WT += hd_res[0]
                hd_total_TC += hd_res[1]
                hd_total_ET += hd_res[2]
                assd_total_WT += assd_res[0]
                assd_total_TC += assd_res[1]
                assd_total_ET += assd_res[2]
                noised_assd_res = softmax_assd_score(noised_output, target[:, :, :155])
                noised_hd_res = softmax_output_hd(noised_output, target[:, :, :155])
                noised_dice_res = softmax_output_dice(noised_output, target[:, :, :155])
                noised_iou_res = softmax_mIOU_score(noised_output, target[:, :, :155])
                noise_dice_total_WT += noised_dice_res[0]
                noise_dice_total_TC += noised_dice_res[1]
                noise_dice_total_ET += noised_dice_res[2]
                noise_iou_total_WT += noised_iou_res[0]
                noise_iou_total_TC += noised_iou_res[1]
                noise_iou_total_ET += noised_iou_res[2]
                noise_hd_total_WT += noised_hd_res[0]
                noise_hd_total_TC += noised_hd_res[1]
                noise_hd_total_ET += noised_hd_res[2]
                noise_assd_total_WT += noised_assd_res[0]
                noise_assd_total_TC += noised_assd_res[1]
                noise_assd_total_ET += noised_assd_res[2]

                mean_uncertainty = torch.mean(uncertainty)
                noised_mean_uncertainty = torch.mean(noised_uncertainty)
                # mne & ece
                mne_total += torch.mean(mne)  # mix _uncertainty mean_uncertainty mean_uncertainty_succ
                noise_mne_total += torch.mean(noised_mne)
                # ece
                ece_total += ece
                noise_ece_total += noise_ece
                # U ece ueo
                certainty_total += mean_uncertainty  # mix _uncertainty mean_uncertainty mean_uncertainty_succ
                # sum_certainty_total += sum_uncertainty
                noise_certainty_total += noised_mean_uncertainty  # noised_mix_uncertainty noised_mean_uncertainty noised_mean_uncertainty_succ
                # noise_sum_certainty_total += noised_sum_uncertainty
                pc = output
                noised_pc = noised_output
                # ueo
                thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
                to_evaluate = dict()
                to_evaluate['target'] = target[:, :, :155]
                u = torch.squeeze(uncertainty)
                U = u.cpu().detach().numpy()
                to_evaluate['prediction'] = pc
                to_evaluate['uncertainty'] = U
                UEO = cal_ueo(to_evaluate, thresholds)
                ueo_total += UEO
                noise_to_evaluate = dict()
                noise_to_evaluate['target'] = target[:, :, :155]
                noise_u = torch.squeeze(noised_uncertainty)
                noise_U = noise_u.cpu().detach().numpy()

                noise_to_evaluate['prediction'] = noised_pc
                noise_to_evaluate['uncertainty'] = noise_U
                noise_UEO = cal_ueo(noise_to_evaluate, thresholds)
                print('current_UEO:{};current_noise_UEO:{}; current_num:{}'.format(UEO, noise_UEO, i))
                noise_ueo_total += noise_UEO
                # confidence map
                conf = 1-uncertainty
                noised_conf = 1-noised_uncertainty
                confe = torch.exp(-uncertainty)
                noised_confe = torch.exp(-noised_uncertainty)
                mean_conf = torch.mean(conf)
                noised_mean_conf = torch.mean(noised_conf)
                mean_confe = torch.mean(confe)
                noised_mean_confe = torch.mean(noised_confe)
                conf_output = torch.squeeze(conf).cpu().detach().numpy()
                Nconf_output = torch.squeeze(noised_conf).cpu().detach().numpy()
                confe_output = torch.squeeze(confe).cpu().detach().numpy()
                Nconfe_output = torch.squeeze(noised_confe).cpu().detach().numpy()
                print('current_U:{};current_noise_U:{};current_num:{}'.format(mean_uncertainty,
                                                                              noised_mean_uncertainty, i))
                print('current_conf:{};current_noise_conf:{};current_num:{}'.format(mean_conf, noised_mean_conf,
                                                                                    i))
                print('current_confe:{};current_noise_confe:{};current_num:{}'.format(mean_confe,
                                                                                      noised_mean_confe, i))
                logging.info('current_U:{};current_noise_U:{};current_num:{}'.format(mean_uncertainty,
                                                                              noised_mean_uncertainty, i))
                logging.info('current_conf:{};current_noise_conf:{};current_num:{}'.format(mean_conf, noised_mean_conf,
                                                                                    i))
                logging.info('current_confe:{};current_noise_confe:{};current_num:{}'.format(mean_confe,
                                                                                      noised_mean_confe, i))
                # uncertainty np
                Otarget = target
                Oinput = torch.squeeze(x,0).cpu().detach().numpy()
                NOinput = torch.squeeze(noised_x,0).cpu().detach().numpy()
                U_output = torch.squeeze(uncertainty).cpu().detach().numpy()
                NU_output = torch.squeeze(noised_uncertainty).cpu().detach().numpy()
                # U_output = torch.squeeze(mne).cpu().detach().numpy()
                # NU_output = torch.squeeze(noised_mne).cpu().detach().numpy()
                name = str(i)
                if names:
                    name = names[i]
                    msg += '{:>20}, '.format(name)

                if snapshot:
                    """ --- grey figure---"""
                    # Snapshot_img = np.zeros(shape=(H,W,T),dtype=np.uint8)
                    # Snapshot_img[np.where(output[1,:,:,:]==1)] = 64
                    # Snapshot_img[np.where(output[2,:,:,:]==1)] = 160
                    # Snapshot_img[np.where(output[3,:,:,:]==1)] = 255
                    """ --- colorful figure--- """
                    Snapshot_img = np.zeros(shape=(H, W, 3, T), dtype=np.float32)
                    # K = [np.where(output[0,:,:,:] == 1)]
                    Snapshot_img[:, :, 0, :][np.where(pc == 1)] = 255
                    Snapshot_img[:, :, 1, :][np.where(pc == 2)] = 255
                    Snapshot_img[:, :, 2, :][np.where(pc == 3)] = 255

                    Noise_Snapshot_img = np.zeros(shape=(H, W, 3, T), dtype=np.float32)
                    # K = [np.where(output[0,:,:,:] == 1)]
                    Noise_Snapshot_img[:, :, 0, :][np.where(noised_pc == 1)] = 255
                    Noise_Snapshot_img[:, :, 1, :][np.where(noised_pc == 2)] = 255
                    Noise_Snapshot_img[:, :, 2, :][np.where(noised_pc == 3)] = 255
                    target_img = np.zeros(shape=(H, W, 3, T), dtype=np.float32)
                    # K = [np.where(output[0,:,:,:] == 1)]
                    target_img[:, :, 0, :][np.where(Otarget == 1)] = 255
                    target_img[:, :, 1, :][np.where(Otarget == 2)] = 255
                    target_img[:, :, 2, :][np.where(Otarget == 3)] = 255

                    for frame in range(T):
                        if not os.path.exists(os.path.join(savepath, name)):
                            os.makedirs(os.path.join(savepath,  name))

                        # scipy.misc.imsave(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])
                        imageio.imwrite(os.path.join(savepath, name, str(frame) + '.png'),
                                        Snapshot_img[:, :, :, frame])
                        imageio.imwrite(
                            os.path.join(savepath, name, str(frame) + '_noised.png'),
                            Noise_Snapshot_img[:, :, :, frame])
                        imageio.imwrite(os.path.join(savepath, name, str(frame) + '_gt.png'),
                                        target_img[:, :, :, frame])
                        imageio.imwrite(os.path.join(savepath, name, str(frame) + '_input_FLR.png'),
                                        Oinput[0,:, :, frame])
                        imageio.imwrite(os.path.join(savepath, name, str(frame) + '_input_T1ce.png'),
                                        Oinput[1,:, :, frame])
                        imageio.imwrite(os.path.join(savepath, name, str(frame)  + '_input_T1.png'),
                                        Oinput[2,:, :, frame])
                        imageio.imwrite(os.path.join(savepath, name, str(frame) + '_input_T2.png'),
                                        Oinput[3,:, :, frame])
                        imageio.imwrite(os.path.join(savepath, name, str(frame) +'_noised_input_FLR.png'),
                                        NOinput[0,:, :, frame])
                        imageio.imwrite(os.path.join(savepath, name, str(frame) + '_noised_input_T1ce.png'),
                                        NOinput[1,:, :, frame])
                        imageio.imwrite(os.path.join(savepath, name, str(frame) + '_noised_input_T1.png'),
                                        NOinput[2,:, :, frame])
                        imageio.imwrite(os.path.join(savepath, name, str(frame) + '_noised_input_T2.png'),
                                        NOinput[3,:, :, frame])
                        imageio.imwrite(
                            os.path.join(savepath, name, str(frame) + '_uncertainty.png'),
                            U_output[:, :, frame])
                        imageio.imwrite(
                            os.path.join(savepath, name,
                                         str(frame) + '_noised_uncertainty.png'),
                            NU_output[:, :, frame])
                        imageio.imwrite(
                            os.path.join(savepath, name, str(frame) + '_conf.png'),
                            conf_output[:, :, frame])
                        imageio.imwrite(
                            os.path.join(savepath, name,
                                         str(frame) + '_noised_conf.png'),
                            Nconf_output[:, :, frame])
                        imageio.imwrite(
                            os.path.join(savepath, name, str(frame) + '_confe.png'),
                            confe_output[:, :, frame])
                        imageio.imwrite(
                            os.path.join(savepath, name,
                                         str(frame) + '_noised_confe.png'),
                            Nconfe_output[:, :, frame])
                        print('current_U:{};current_noise_U:{};current_frame:{}'.format(np.mean(U_output[:, :, frame]),
                                                                                      np.mean(NU_output[:, :, frame]), frame))
                        print('current_conf:{};current_noise_conf:{};current_frame:{}'.format(np.mean(conf_output[:, :, frame]), np.mean(Nconf_output[:, :, frame]),
                                                                                            frame))
                        print('current_confe:{};current_noise_confe:{};current_frame:{}'.format(np.mean(confe_output[:, :, frame]),
                                                                                              np.mean(Nconfe_output[:, :, frame]), frame))
                        logging.info('current_U:{};current_noise_U:{};current_frame:{}'.format(np.mean(U_output[:, :, frame]),
                                                                                      np.mean(NU_output[:, :, frame]), frame))
                        logging.info(
                            'current_conf:{};current_noise_conf:{};current_frame:{}'.format(np.mean(conf_output[:, :, frame]), np.mean(Nconf_output[:, :, frame]),
                                                                                            frame))
                        logging.info('current_confe:{};current_noise_confe:{};current_frame:{}'.format(np.mean(confe_output[:, :, frame]),
                                                                                              np.mean(Nconfe_output[:, :, frame]), frame))


                        U_img = cv2.imread(
                            os.path.join(savepath, name, str(frame) + '_uncertainty.png'))
                        U_heatmap = cv2.applyColorMap(U_img, cv2.COLORMAP_JET)
                        cv2.imwrite(
                            os.path.join(savepath, name,
                                         str(frame) + '_colormap_uncertainty.png'),
                            U_heatmap)
                        NU_img = cv2.imread(
                            os.path.join(savepath, name,
                                         str(frame) + '_noised_uncertainty.png'))
                        NU_heatmap = cv2.applyColorMap(NU_img, cv2.COLORMAP_JET)
                        cv2.imwrite(
                            os.path.join(savepath, name,
                                         str(frame) + '_colormap_noised_uncertainty.png'),
                            NU_heatmap)

                        conf_img = cv2.imread(
                            os.path.join(savepath, name, str(frame) + '_conf.png'))
                        conf_heatmap = cv2.applyColorMap(conf_img, cv2.COLORMAP_JET)
                        cv2.imwrite(
                            os.path.join(savepath, name,
                                         str(frame) + '_colormap_conf.png'),
                            conf_heatmap)
                        Nconf_img = cv2.imread(
                            os.path.join(savepath, name,
                                         str(frame) + '_noised_conf.png'))
                        Nconf_heatmap = cv2.applyColorMap(Nconf_img, cv2.COLORMAP_JET)
                        cv2.imwrite(
                            os.path.join(savepath, name,
                                         str(frame) + '_colormap_noised_conf.png'),
                            Nconf_heatmap)

                        confe_img = cv2.imread(
                            os.path.join(savepath, name, str(frame) + '_confe.png'))
                        confe_heatmap = cv2.applyColorMap(confe_img, cv2.COLORMAP_JET)
                        cv2.imwrite(
                            os.path.join(savepath, name,
                                         str(frame) + '_colormap_confe.png'),
                            confe_heatmap)
                        Nconfe_img = cv2.imread(
                            os.path.join(savepath, name,
                                         str(frame) + '_noised_confe.png'))
                        Nconfe_heatmap = cv2.applyColorMap(Nconfe_img, cv2.COLORMAP_JET)
                        cv2.imwrite(
                            os.path.join(savepath, name,
                                         str(frame) + '_colormap_noised_confe.png'),
                            Nconfe_heatmap)
            elif args.dataset == 'LiTS':
                ece = cal_ece_our(torch.squeeze(predicted), torch.squeeze(target.cpu()))
                noise_ece = cal_ece_our(torch.squeeze(noised_predicted), torch.squeeze(target.cpu()))
                # confidence map
                conf = 1 - uncertainty
                noised_conf = 1 - noised_uncertainty
                confe = torch.exp(-uncertainty)
                noised_confe = torch.exp(-noised_uncertainty)
                mean_conf = torch.mean(conf)
                noised_mean_conf = torch.mean(noised_conf)
                mean_confe = torch.mean(confe)
                noised_mean_confe = torch.mean(noised_confe)
                conf_output = torch.squeeze(conf).cpu().detach().numpy()
                Nconf_output = torch.squeeze(noised_conf).cpu().detach().numpy()
                confe_output = torch.squeeze(confe).cpu().detach().numpy()
                Nconfe_output = torch.squeeze(noised_confe).cpu().detach().numpy()

                H, W, T = 256, 256, 16
                Otarget = torch.squeeze(target).cpu().numpy()
                target = torch.squeeze(target).cpu().numpy()  # .cpu().numpy(dtype='float32')
                hd_res = softmax_output_litshd(output, target)
                assd_res = softmax_assd_litsscore(output, target)
                iou_res = softmax_mIOU_litsscore(output, target)
                dice_res = softmax_output_litsdice(output, target)
                dice_total_WT += dice_res[0]
                dice_total_TC += dice_res[1]
                iou_total_WT += iou_res[0]
                iou_total_TC += iou_res[1]
                hd_total_WT += hd_res[0]
                hd_total_TC += hd_res[1]
                assd_total_WT += assd_res[0]
                assd_total_TC += assd_res[1]
                noised_assd_res = softmax_assd_litsscore(noised_output, target)
                noised_hd_res = softmax_output_litshd(noised_output, target)
                noised_dice_res = softmax_output_litsdice(noised_output, target)
                noised_iou_res = softmax_mIOU_litsscore(noised_output, target)
                noise_dice_total_WT += noised_dice_res[0]
                noise_dice_total_TC += noised_dice_res[1]
                noise_iou_total_WT += noised_iou_res[0]
                noise_iou_total_TC += noised_iou_res[1]
                noise_hd_total_WT += noised_hd_res[0]
                noise_hd_total_TC += noised_hd_res[1]
                noise_assd_total_WT += noised_assd_res[0]
                noise_assd_total_TC += noised_assd_res[1]

                mean_uncertainty = torch.mean(uncertainty)
                noised_mean_uncertainty = torch.mean(noised_uncertainty)
                # mne & ece
                mne_total += torch.mean(mne)  # mix _uncertainty mean_uncertainty mean_uncertainty_succ
                noise_mne_total += torch.mean(noised_mne)
                # ece
                ece_total += ece
                noise_ece_total += noise_ece
                # U ece ueo
                certainty_total += mean_uncertainty  # mix _uncertainty mean_uncertainty mean_uncertainty_succ
                # sum_certainty_total += sum_uncertainty
                noise_certainty_total += noised_mean_uncertainty  # noised_mix_uncertainty noised_mean_uncertainty noised_mean_uncertainty_succ
                # noise_sum_certainty_total += noised_sum_uncertainty
                pc = output
                noised_pc = noised_output
                # ueo
                thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
                to_evaluate = dict()
                to_evaluate['target'] = target
                u = torch.squeeze(uncertainty)
                U = u.cpu().detach().numpy()
                to_evaluate['prediction'] = pc
                to_evaluate['uncertainty'] = U
                UEO = cal_ueo(to_evaluate, thresholds)
                ueo_total += UEO
                noise_to_evaluate = dict()
                noise_to_evaluate['target'] = target
                noise_u = torch.squeeze(noised_uncertainty)
                noise_U = noise_u.cpu().detach().numpy()
                noise_to_evaluate['prediction'] = noised_pc
                noise_to_evaluate['uncertainty'] = noise_U
                noise_UEO = cal_ueo(noise_to_evaluate, thresholds)
                print('current_UEO:{};current_noise_UEO:{}; current_num:{}'.format(UEO, noise_UEO, i))
                noise_ueo_total += noise_UEO
                # uncertainty np
                Oinput = torch.squeeze(x).cpu().detach().numpy()
                NOinput = torch.squeeze(noised_x).cpu().detach().numpy()
                U_output = torch.squeeze(uncertainty).cpu().detach().numpy()
                NU_output = torch.squeeze(noised_uncertainty).cpu().detach().numpy()
                # U_output = torch.squeeze(mne).cpu().detach().numpy()
                # NU_output = torch.squeeze(noised_mne).cpu().detach().numpy()
                name = str(i)
                if names:
                    name = names[i]
                    msg += '{:>20}, '.format(name)
                name = name.replace('.nii', '')
                if snapshot:
                    """ --- grey figure---"""
                    # Snapshot_img = np.zeros(shape=(H,W,T),dtype=np.uint8)
                    # Snapshot_img[np.where(output[1,:,:,:]==1)] = 64
                    # Snapshot_img[np.where(output[2,:,:,:]==1)] = 160
                    # Snapshot_img[np.where(output[3,:,:,:]==1)] = 255
                    """ --- colorful figure--- """
                    Snapshot_img = np.zeros(shape=(T, H, W, 3), dtype=np.float32)
                    # K = [np.where(output[0,:,:,:] == 1)]
                    # KK = [np.where(pc == 1)]
                    Snapshot_img[:, :, :, 0][np.where(pc == 1)] = 255
                    Snapshot_img[:, :, :, 1][np.where(pc == 2)] = 255
                    Snapshot_img[:, :, :, 2] = 0
                    Noise_Snapshot_img = np.zeros(shape=(T, H, W, 3), dtype=np.float32)
                    # K = [np.where(output[0,:,:,:] == 1)]
                    Noise_Snapshot_img[:, :, :, 0][np.where(noised_pc == 1)] = 255
                    Noise_Snapshot_img[:, :, :, 1][np.where(noised_pc == 2)] = 255
                    Noise_Snapshot_img[:, :, :, 2] = 0
                    target_img = np.zeros(shape=(T, H, W, 3), dtype=np.float32)
                    # K = [np.where(output[0,:,:,:] == 1)]
                    target_img[:, :, :, 0][np.where(Otarget == 1)] = 255
                    target_img[:, :, :, 1][np.where(Otarget == 2)] = 255
                    target_img[:, :, :, 2] = 0

                    for frame in range(T):
                        if not os.path.exists(os.path.join(savepath, name)):
                            os.makedirs(os.path.join(savepath, name))

                        # scipy.misc.imsave(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])
                        imageio.imwrite(os.path.join(savepath, name, str(frame) + '.png'),
                                        Snapshot_img[frame,:, :, :])
                        imageio.imwrite(
                            os.path.join(savepath, name, str(frame) + '_noised.png'),
                            Noise_Snapshot_img[frame,:, :, :])

                        imageio.imwrite(os.path.join(savepath, name, str(frame) + '_gt.png'),
                                        target_img[frame, :, :, :])
                        imageio.imwrite(
                            os.path.join(savepath, name, str(frame) + '_uncertainty.png'),
                            U_output[frame,:, :])
                        imageio.imwrite(
                            os.path.join(savepath, name,
                                         str(frame) + '_noised_uncertainty.png'),
                            NU_output[frame,:, :])
                        U_img = cv2.imread(
                            os.path.join(savepath, name, str(frame) + '_uncertainty.png'))
                        U_heatmap = cv2.applyColorMap(U_img, cv2.COLORMAP_JET)
                        cv2.imwrite(
                            os.path.join(savepath, name,
                                         str(frame) + '_colormap_uncertainty.png'),
                            U_heatmap)
                        NU_img = cv2.imread(
                            os.path.join(savepath, name,
                                         str(frame) + '_noised_uncertainty.png'))
                        NU_heatmap = cv2.applyColorMap(NU_img, cv2.COLORMAP_JET)
                        cv2.imwrite(
                            os.path.join(savepath, name,
                                         str(frame) + '_colormap_noised_uncertainty.png'),
                            NU_heatmap)
                        imageio.imwrite(os.path.join(savepath, name, str(frame) +'_input.png'),
                                        Oinput[frame,:, :])
                        imageio.imwrite(os.path.join(savepath, name, str(frame) +'_noised_input.png'),
                                        NOinput[frame,:, :])

                        imageio.imwrite(
                            os.path.join(savepath, name, str(frame) +'_conf.png'),
                            conf_output[frame,:, :])
                        imageio.imwrite(
                            os.path.join(savepath, name, str(frame) +
                                         '_noised_conf.png'),
                            Nconf_output[frame,:, :])
                        imageio.imwrite(
                            os.path.join(savepath, name, str(frame) + '_confe.png'),
                            confe_output[frame,:, :])
                        imageio.imwrite(
                            os.path.join(savepath, name, str(frame) +
                                         '_noised_confe.png'),
                            Nconfe_output[frame,:, :])

                        conf_img = cv2.imread(
                            os.path.join(savepath, name, str(frame) + '_conf.png'))
                        conf_heatmap = cv2.applyColorMap(conf_img, cv2.COLORMAP_JET)
                        cv2.imwrite(
                            os.path.join(savepath, name,
                                         str(frame) + '_colormap_conf.png'),
                            conf_heatmap)
                        Nconf_img = cv2.imread(
                            os.path.join(savepath, name,
                                         str(frame) + '_noised_conf.png'))
                        Nconf_heatmap = cv2.applyColorMap(Nconf_img, cv2.COLORMAP_JET)
                        cv2.imwrite(
                            os.path.join(savepath, name,
                                         str(frame) + '_colormap_noised_conf.png'),
                            Nconf_heatmap)

                        confe_img = cv2.imread(
                            os.path.join(savepath, name, str(frame) + '_confe.png'))
                        confe_heatmap = cv2.applyColorMap(confe_img, cv2.COLORMAP_JET)
                        cv2.imwrite(
                            os.path.join(savepath, name,
                                         str(frame) + '_colormap_confe.png'),
                            confe_heatmap)
                        Nconfe_img = cv2.imread(
                            os.path.join(savepath, name,
                                         str(frame) + '_noised_confe.png'))
                        Nconfe_heatmap = cv2.applyColorMap(Nconfe_img, cv2.COLORMAP_JET)
                        cv2.imwrite(
                            os.path.join(savepath, name,
                                         str(frame) + '_colormap_noised_confe.png'),
                            Nconfe_heatmap)
            else:
                ece = cal_ece_our(torch.squeeze(predicted), torch.squeeze(target.cpu()))
                noise_ece = cal_ece_our(torch.squeeze(noised_predicted), torch.squeeze(target.cpu()))
                soft_gt = get_soft_label(target, num_classes)
                soft_predicted = get_soft_label(predicted.unsqueeze(0), num_classes)
                iou_res = IOU_isic(soft_predicted, soft_gt,num_classes)
                hd_res = HD_isic(torch.squeeze(predicted).cpu().numpy(), torch.squeeze(target).cpu().numpy(), num_classes)
                assd_res = assd(torch.squeeze(predicted).cpu().byte().numpy().astype(np.uint8), torch.squeeze(target).cpu().numpy().astype(np.uint8))
                dice_res = Dice_isic(soft_predicted, soft_gt,num_classes)

                soft_noised_predicted = get_soft_label(noised_predicted.unsqueeze(0), num_classes)
                noised_iou_res = IOU_isic(soft_noised_predicted, soft_gt,num_classes)
                noised_hd_res = HD_isic(torch.squeeze(noised_predicted).cpu().numpy(), torch.squeeze(target).cpu().numpy(), num_classes)
                noised_dice_res = Dice_isic(soft_noised_predicted, soft_gt,num_classes)
                noised_assd_res = assd(torch.squeeze(noised_predicted).cpu().byte().numpy().astype(np.uint8),
                                torch.squeeze(target).cpu().numpy().astype(np.uint8))
                noised_hd95_res = hd95(torch.squeeze(noised_predicted).cpu().numpy(), torch.squeeze(target).cpu().numpy())

                dice_total += dice_res
                iou_total += iou_res
                hd_total += hd_res
                # hd95_total +=hd95_res
                assd_total += assd_res
                noise_dice_total += noised_dice_res
                noise_iou_total += noised_iou_res
                noise_hd_total += noised_hd_res
                # noise_hd95_total += noised_hd95_res
                noise_assd_total += noised_assd_res

                # U ece ueo
                H = x.shape[2]
                W = x.shape[3]
                pc = output
                noised_pc = noised_output
                mean_uncertainty = torch.mean(uncertainty)
                noised_mean_uncertainty = torch.mean(noised_uncertainty)

                mean_mne = torch.mean(mne)
                noised_mean_mne = torch.mean(noised_mne)
                certainty_total += mean_uncertainty  # mix _uncertainty mean_uncertainty mean_uncertainty_succ
                noise_certainty_total += noised_mean_uncertainty  # noised_mix_uncertainty noised_mean_uncertainty noised_mean_uncertainty_succ

                if torch.isnan(noised_mean_mne):
                    noised_mean_mne = 0
                mne_total += mean_mne  # mix _uncertainty mean_uncertainty mean_uncertainty_succ
                noise_mne_total += noised_mean_mne
                # confidence map
                conf = 1 - uncertainty
                noised_conf = 1 - noised_uncertainty
                confe = torch.exp(-uncertainty)
                noised_confe = torch.exp(-noised_uncertainty)
                mean_conf = torch.mean(conf)
                noised_mean_conf = torch.mean(noised_conf)
                mean_confe = torch.mean(confe)
                noised_mean_confe = torch.mean(noised_confe)
                conf_output = torch.squeeze(conf).cpu().detach().numpy()
                Nconf_output = torch.squeeze(noised_conf).cpu().detach().numpy()
                confe_output = torch.squeeze(confe).cpu().detach().numpy()
                Nconfe_output = torch.squeeze(noised_confe).cpu().detach().numpy()
                print('current_U:{};current_noise_U:{};current_num:{}'.format(mean_uncertainty,
                                                                              noised_mean_uncertainty, i))
                print('current_conf:{};current_noise_conf:{};current_num:{}'.format(mean_conf, noised_mean_conf,
                                                                                    i))
                print('current_confe:{};current_noise_confe:{};current_num:{}'.format(mean_confe,
                                                                                      noised_mean_confe, i))
                logging.info('current_U:{};current_noise_U:{};current_num:{}'.format(mean_uncertainty,
                                                                                     noised_mean_uncertainty, i))
                logging.info('current_conf:{};current_noise_conf:{};current_num:{}'.format(mean_conf, noised_mean_conf,
                                                                                           i))
                logging.info('current_confe:{};current_noise_confe:{};current_num:{}'.format(mean_confe,
                                                                                             noised_mean_confe, i))

                # ece
                ece_total += ece
                noise_ece_total += noise_ece
                # ueo
                thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
                to_evaluate = dict()
                to_evaluate['target'] = torch.squeeze(target).cpu().detach().numpy()
                u = torch.squeeze(uncertainty)
                U = u.cpu().detach().numpy()
                to_evaluate['prediction'] = pc
                to_evaluate['uncertainty'] = U
                UEO = cal_ueo(to_evaluate, thresholds)
                ueo_total += UEO
                noise_to_evaluate = dict()
                noise_to_evaluate['target'] = torch.squeeze(target).cpu().detach().numpy()
                noise_u = torch.squeeze(noised_uncertainty)
                noise_U = noise_u.cpu().detach().numpy()
                noise_to_evaluate['prediction'] = noised_pc
                noise_to_evaluate['uncertainty'] = noise_U
                noise_UEO = cal_ueo(noise_to_evaluate, thresholds)
                print('current_UEO:{};current_noise_UEO:{}; current_num:{}'.format(UEO, noise_UEO, i))
                noise_ueo_total += noise_UEO
                # uncertainty np
                Otarget = torch.squeeze(target).cpu().detach().numpy()
                Oinput = torch.squeeze(x,0).permute(1, 2, 0).cpu().detach().numpy()
                NOinput = torch.squeeze(noised_x,0).permute(1, 2, 0).cpu().detach().numpy()
                U_output = torch.squeeze(uncertainty).cpu().detach().numpy()
                NU_output = torch.squeeze(noised_uncertainty).cpu().detach().numpy()

                print('current_dice:{},current_noised_dice:{}'.format(dice_res, noised_dice_res))
                print('current_iou:{},current_noised_iou:{}'.format(iou_res, noised_iou_res))
                print('current_hd:{},current_noised_hd:{}'.format(hd_res, noised_hd_res))
                print('current_assd:{},current_noised_assd:{}'.format(assd_res, noised_assd_res))

                name = str(i)
                if names:
                    name = names[i]
                    msg += '{:>20}, '.format(name)
                if args.dataset == 'ISIC':
                    name = name.replace('.npy', '')
                elif args.dataset == 'HC':
                    name = name.replace('.npy', '')
                    name = name.replace('.png', '')
                else:
                    name = name.replace('.png', '')
                if not os.path.exists(os.path.join(savepath)):
                    os.makedirs(os.path.join(savepath))
                # np.savez(savepath+'/'+ name,U=U,noise_U=noise_U,output=output,Otarget=Otarget)
                np.savez(savepath+'/'+ name,U=U)

                if snapshot:
                    """ --- grey figure---"""
                    # Snapshot_img = np.zeros(shape=(H,W,T),dtype=np.uint8)
                    # Snapshot_img[np.where(output[1,:,:,:]==1)] = 64
                    # Snapshot_img[np.where(output[2,:,:,:]==1)] = 160
                    # Snapshot_img[np.where(output[3,:,:,:]==1)] = 255
                    """ --- colorful figure--- """
                    # Snapshot_img = np.zeros(shape=(H, W), dtype=np.float32)
                    # Snapshot_img[np.where(pc == 1)] = 255

                    Snapshot_img = np.zeros(shape=(H, W), dtype=np.float32)
                    Snapshot_img[np.where(pc == 1)] = 255
                    Snapshot_img[np.where(pc > 1)] = 225
                    Snapshot_img[np.where(pc > 2)] = 200
                    Snapshot_img[np.where(pc > 3)] = 175
                    Snapshot_img[np.where(pc > 4)] = 150
                    Snapshot_img[np.where(pc > 5)] = 125
                    Snapshot_img[np.where(pc > 6)] = 100
                    Snapshot_img[np.where(pc > 7)] = 75
                    Snapshot_img[np.where(pc > 8)] = 50

                    Noise_Snapshot_img = np.zeros(shape=(H, W), dtype=np.float32)
                    Noise_Snapshot_img[np.where(noised_pc == 1)] = 255

                    # target_img = np.zeros(shape=(H, W), dtype=np.float32)
                    # target_img[np.where(Otarget == 1)] = 255
                    # HC
                    target_img = np.zeros(shape=(H, W), dtype=np.float32)
                    target_img[np.where(Otarget == 1)] = 255
                    target_img[np.where(Otarget > 1)] = 225
                    target_img[np.where(Otarget > 2)] = 200
                    target_img[np.where(Otarget > 3)] = 175
                    target_img[np.where(Otarget > 4)] = 150
                    target_img[np.where(Otarget > 5)] = 125
                    target_img[np.where(Otarget > 6)] = 100
                    target_img[np.where(Otarget > 7)] = 75
                    target_img[np.where(Otarget > 8)] = 50
                    # if not os.path.exists(os.path.join(savepath, name)):
                    #     os.makedirs(os.path.join(savepath, name))

                        # scipy.misc.imsave(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])
                    imageio.imwrite(os.path.join(savepath, name+'_pred.png'),
                                    Snapshot_img)
                    imageio.imwrite(os.path.join(savepath, name+'_noised.png'),
                                    Noise_Snapshot_img)
                    imageio.imwrite(os.path.join(savepath, name+'_gt.png'),
                                    target_img)
                    imageio.imwrite(os.path.join(savepath, name+'_uncertainty.png'),
                                    U_output)
                    imageio.imwrite(
                        os.path.join(savepath, name+'_noised_uncertainty.png'),
                        NU_output)

                    imageio.imwrite(
                        os.path.join(savepath, name+'_conf.png'),
                        conf_output)
                    imageio.imwrite(
                        os.path.join(savepath, name+
                                     '_noised_conf.png'),
                        Nconf_output)
                    imageio.imwrite(
                        os.path.join(savepath, name+ '_confe.png'),
                        confe_output)
                    imageio.imwrite(
                        os.path.join(savepath, name+
                                     '_noised_confe.png'),
                        Nconfe_output)
                    print('current_U:{};current_noise_U:{};current_name:{}'.format(np.mean(U_output),
                                                                                    np.mean(NU_output),
                                                                                    name))
                    print('current_conf:{};current_noise_conf:{};current_name:{}'.format(
                        np.mean(conf_output), np.mean(Nconf_output),
                        name))
                    print('current_confe:{};current_noise_confe:{};current_name:{}'.format(
                        np.mean(confe_output),
                        np.mean(Nconfe_output), name))
                    logging.info(
                        'current_U:{};current_noise_U:{};current_name:{}'.format(np.mean(U_output),
                                                                                  np.mean(NU_output),
                                                                                  name))
                    logging.info(
                        'current_conf:{};current_noise_conf:{};current_name:{}'.format(
                            np.mean(conf_output), np.mean(Nconf_output),
                            name))
                    logging.info('current_confe:{};current_noise_confe:{};current_name:{}'.format(
                        np.mean(confe_output),
                        np.mean(Nconfe_output), name))

                    U_img = cv2.imread(os.path.join(savepath, name+'_uncertainty.png'))
                    U_heatmap = cv2.applyColorMap(U_img, cv2.COLORMAP_JET)
                    cv2.imwrite(
                        os.path.join(savepath, name+ '_colormap_uncertainty.png'),
                        U_heatmap)
                    NU_img = cv2.imread(
                        os.path.join(savepath, name+ '_noised_uncertainty.png'))
                    NU_heatmap = cv2.applyColorMap(NU_img, cv2.COLORMAP_JET)
                    cv2.imwrite(
                        os.path.join(savepath, name+ '_colormap_noised_uncertainty.png'),
                        NU_heatmap)
                    imageio.imwrite(os.path.join(savepath, name+ '_input.png'),
                                    Oinput)
                    imageio.imwrite(os.path.join(savepath, name+ '_noised_input.png'),
                                    NOinput)

                    conf_img = cv2.imread(
                        os.path.join(savepath, name+ '_conf.png'))
                    conf_heatmap = cv2.applyColorMap(conf_img, cv2.COLORMAP_JET)
                    cv2.imwrite(
                        os.path.join(savepath, name+
                                     '_colormap_conf.png'),
                        conf_heatmap)
                    Nconf_img = cv2.imread(
                        os.path.join(savepath, name+
                                     '_noised_conf.png'))
                    Nconf_heatmap = cv2.applyColorMap(Nconf_img, cv2.COLORMAP_JET)
                    cv2.imwrite(
                        os.path.join(savepath, name+
                                     '_colormap_noised_conf.png'),
                        Nconf_heatmap)

                    confe_img = cv2.imread(
                        os.path.join(savepath, name+ '_confe.png'))
                    confe_heatmap = cv2.applyColorMap(confe_img, cv2.COLORMAP_JET)
                    cv2.imwrite(
                        os.path.join(savepath, name+
                                     '_colormap_confe.png'),
                        confe_heatmap)
                    Nconfe_img = cv2.imread(
                        os.path.join(savepath, name+
                                     '_noised_confe.png'))
                    Nconfe_heatmap = cv2.applyColorMap(Nconfe_img, cv2.COLORMAP_JET)
                    cv2.imwrite(
                        os.path.join(savepath, name+
                                     '_colormap_noised_confe.png'),
                        Nconfe_heatmap)
            print('current_dice:{} ; current_noised_dice:{}'.format(dice_res, noised_dice_res))
            print('current_iou:{} ; current_noised_iou:{}'.format(iou_res, noised_iou_res))
            print('current_hd:{} ; current_noised_hd:{}'.format(hd_res, noised_hd_res))

    num = len(test_loader)
    if args.dataset == 'BraTS':
        aver_certainty = certainty_total / num
        aver_noise_certainty = noise_certainty_total / num
        aver_mne = mne_total / num
        aver_noise_mne = noise_mne_total / num
        aver_ece = ece_total / num
        aver_noise_ece = noise_ece_total / num
        aver_ueo = ueo_total / num
        aver_noise_ueo = noise_ueo_total / num
        aver_dice_WT = dice_total_WT / num
        aver_dice_TC = dice_total_TC / num
        aver_dice_ET = dice_total_ET / num
        aver_noise_dice_WT = noise_dice_total_WT / num
        aver_noise_dice_TC = noise_dice_total_TC / num
        aver_noise_dice_ET = noise_dice_total_ET / num
        aver_iou_WT = iou_total_WT / num
        aver_iou_TC = iou_total_TC / num
        aver_iou_ET = iou_total_ET / num
        aver_noise_iou_WT = noise_iou_total_WT / num
        aver_noise_iou_TC = noise_iou_total_TC / num
        aver_noise_iou_ET = noise_iou_total_ET / num
        aver_hd_WT = hd_total_WT / num
        aver_hd_TC = hd_total_TC / num
        aver_hd_ET = hd_total_ET / num
        aver_noise_hd_WT = noise_hd_total_WT / num
        aver_noise_hd_TC = noise_hd_total_TC / num
        aver_noise_hd_ET = noise_hd_total_ET / num
        aver_assd_WT = assd_total_WT / num
        aver_assd_TC = assd_total_TC / num
        aver_assd_ET = assd_total_ET / num
        aver_noise_assd_WT = noise_assd_total_WT / num
        aver_noise_assd_TC = noise_assd_total_TC / num
        aver_noise_assd_ET = noise_assd_total_ET / num
        print('aver_dice_WT=%f,aver_dice_TC = %f,aver_dice_ET = %f' % (
        aver_dice_WT * 100, aver_dice_TC * 100, aver_dice_ET * 100))
        print('aver_noise_dice_WT=%f,aver_noise_dice_TC = %f,aver_noise_dice_ET = %f' % (
            aver_noise_dice_WT * 100, aver_noise_dice_TC * 100, aver_noise_dice_ET * 100))
        print('aver_iou_WT=%f,aver_iou_TC = %f,aver_iou_ET = %f' % (
        aver_iou_WT * 100, aver_iou_TC * 100, aver_iou_ET * 100))
        print('aver_noise_iou_WT=%f,aver_noise_iou_TC = %f,aver_noise_iou_ET = %f' % (
            aver_noise_iou_WT * 100, aver_noise_iou_TC * 100, aver_noise_iou_ET * 100))
        print('aver_hd_WT=%f,aver_hd_TC = %f,aver_hd_ET = %f' % (aver_hd_WT, aver_hd_TC, aver_hd_ET))
        print('aver_noise_hd_WT=%f,aver_noise_hd_TC = %f,aver_noise_hd_ET = %f' % (
            aver_noise_hd_WT, aver_noise_hd_TC, aver_noise_hd_ET))
        print('aver_assd_WT=%f,aver_assd_TC = %f,aver_assd_ET = %f' % (aver_assd_WT, aver_assd_TC, aver_assd_ET))
        print('aver_noise_assd_WT=%f,aver_noise_assd_TC = %f,aver_noise_assd_ET = %f' % (
            aver_noise_assd_WT, aver_noise_assd_TC, aver_noise_assd_ET))
        print('aver_certainty=%f,aver_noise_certainty = %f' % (aver_certainty, aver_noise_certainty))
        print('aver_mne=%f,aver_noise_mne = %f' % (aver_mne, aver_noise_mne))
        print('aver_ece=%f,aver_noise_ece = %f' % (aver_ece, aver_noise_ece))
        print('aver_ueo=%f,aver_noise_ueo = %f' % (aver_ueo, aver_noise_ueo))
        logging.info('aver_dice_WT=%f,aver_dice_TC = %f,aver_dice_ET = %f' % (
        aver_dice_WT * 100, aver_dice_TC * 100, aver_dice_ET * 100))
        logging.info('aver_noise_dice_WT=%f,aver_noise_dice_TC = %f,aver_noise_dice_ET = %f' % (
            aver_noise_dice_WT * 100, aver_noise_dice_TC * 100, aver_noise_dice_ET * 100))
        logging.info('aver_iou_WT=%f,aver_iou_TC = %f,aver_iou_ET = %f' % (
        aver_iou_WT * 100, aver_iou_TC * 100, aver_iou_ET * 100))
        logging.info('aver_noise_iou_WT=%f,aver_noise_iou_TC = %f,aver_noise_iou_ET = %f' % (
            aver_noise_iou_WT, aver_noise_iou_TC, aver_noise_iou_ET))
        logging.info('aver_hd_WT=%f,aver_hd_TC = %f,aver_hd_ET = %f' % (aver_hd_WT, aver_hd_TC, aver_hd_ET))
        logging.info('aver_noise_hd_WT=%f,aver_noise_hd_TC = %f,aver_noise_hd_ET = %f' % (
            aver_noise_hd_WT, aver_noise_hd_TC, aver_noise_hd_ET))
        logging.info('aver_assd_WT=%f,aver_assd_TC = %f,aver_assd_ET = %f' % (aver_assd_WT, aver_assd_TC, aver_assd_ET))
        logging.info('aver_noise_assd_WT=%f,aver_noise_assd_TC = %f,aver_noise_assd_ET = %f' % (
            aver_noise_assd_WT, aver_noise_assd_TC, aver_noise_assd_ET))
        logging.info('aver_certainty=%f,aver_noise_certainty = %f' % (aver_certainty, aver_noise_certainty))
        logging.info('aver_mne=%f,aver_noise_mne = %f' % (aver_mne, aver_noise_mne))
        logging.info('aver_ece=%f,aver_noise_ece = %f' % (aver_ece, aver_noise_ece))
        logging.info('aver_ueo=%f,aver_noise_ueo = %f' % (aver_ueo, aver_noise_ueo))
        return [aver_dice_WT, aver_dice_TC, aver_dice_ET], [aver_noise_dice_WT, aver_noise_dice_TC,
                                                            aver_noise_dice_ET], [
                   aver_hd_WT, aver_hd_TC, aver_hd_ET], [aver_noise_hd_WT, aver_noise_hd_TC, aver_noise_hd_ET], [
                   aver_assd_WT, aver_assd_TC, aver_assd_ET], [aver_noise_assd_WT, aver_noise_assd_TC,
                                                               aver_noise_assd_ET]
    elif args.dataset =='LiTS':
        aver_certainty = certainty_total  / num
        aver_noise_certainty = noise_certainty_total / num
        aver_mne = mne_total / num
        aver_noise_mne = noise_mne_total/num
        aver_ece = ece_total / num
        aver_noise_ece = noise_ece_total / num
        aver_ueo = ueo_total / num
        aver_noise_ueo = noise_ueo_total/num
        aver_dice_WT = dice_total_WT / num
        aver_dice_TC = dice_total_TC / num
        aver_noise_dice_WT = noise_dice_total_WT / num
        aver_noise_dice_TC = noise_dice_total_TC / num
        aver_iou_WT = iou_total_WT / num
        aver_iou_TC = iou_total_TC / num
        aver_noise_iou_WT = noise_iou_total_WT / num
        aver_noise_iou_TC = noise_iou_total_TC / num
        aver_hd_WT = hd_total_WT / num
        aver_hd_TC = hd_total_TC / num
        aver_noise_hd_WT = noise_hd_total_WT / num
        aver_noise_hd_TC = noise_hd_total_TC / num
        aver_assd_WT = assd_total_WT / num
        aver_assd_TC = assd_total_TC / num
        aver_noise_assd_WT = noise_assd_total_WT / num
        aver_noise_assd_TC = noise_assd_total_TC / num
        print('aver_dice_WT=%f,aver_dice_TC = %f' % (aver_dice_WT*100, aver_dice_TC*100))
        print('aver_noise_dice_WT=%f,aver_noise_dice_TC = %f' % (
        aver_noise_dice_WT*100, aver_noise_dice_TC*100))
        print('aver_iou_WT=%f,aver_iou_TC = %f' % (aver_iou_WT*100, aver_iou_TC*100))
        print('aver_noise_iou_WT=%f,aver_noise_iou_TC = %f' % (
        aver_noise_iou_WT*100, aver_noise_iou_TC*100))
        print('aver_hd_WT=%f,aver_hd_TC = %f' % (aver_hd_WT, aver_hd_TC))
        print('aver_noise_hd_WT=%f,aver_noise_hd_TC = %f' % (
            aver_noise_hd_WT, aver_noise_hd_TC))
        print('aver_assd_WT=%f,aver_assd_TC = %f' % (aver_assd_WT, aver_assd_TC))
        print('aver_noise_assd_WT=%f,aver_noise_assd_TC = %f' % (
            aver_noise_assd_WT, aver_noise_assd_TC))
        print('aver_certainty=%f,aver_noise_certainty = %f' % (aver_certainty, aver_noise_certainty))
        print('aver_mne=%f,aver_noise_mne = %f' % (aver_mne, aver_noise_mne))
        print('aver_ece=%f,aver_noise_ece = %f' % (aver_ece, aver_noise_ece))
        print('aver_ueo=%f,aver_noise_ueo = %f' % (aver_ueo, aver_noise_ueo))
        logging.info(
        'aver_dice_WT=%f,aver_dice_TC = %f' % (aver_dice_WT, aver_dice_TC))
        logging.info('aver_noise_dice_WT=%f,aver_noise_dice_TC=%f' % (
        aver_noise_dice_WT, aver_noise_dice_TC))
        logging.info('aver_iou_WT=%f,aver_iou_TC = %f' % (
        aver_iou_WT, aver_iou_TC))
        logging.info('aver_noise_iou_WT=%f,aver_noise_iou_TC = %f' % (
        aver_noise_iou_WT, aver_noise_iou_TC))
        logging.info('aver_hd_WT=%f,aver_hd_TC = %f' % (
        aver_hd_WT, aver_hd_TC))
        logging.info('aver_noise_hd_WT=%f,aver_noise_hd_TC = %f' % (
        aver_noise_hd_WT, aver_noise_hd_TC))
        logging.info('aver_assd_WT=%f,aver_assd_TC = %f' % (aver_assd_WT, aver_assd_TC))
        logging.info('aver_noise_assd_WT=%f,aver_noise_assd_TC = %f' % (
            aver_noise_assd_WT, aver_noise_assd_TC))
        logging.info('aver_certainty=%f,aver_noise_certainty = %f' % (aver_certainty, aver_noise_certainty))
        logging.info('aver_mne=%f,aver_noise_mne = %f' % (aver_mne, aver_noise_mne))
        logging.info('aver_ece=%f,aver_noise_ece = %f' % (aver_ece, aver_noise_ece))
        logging.info('aver_ueo=%f,aver_noise_ueo = %f' % (aver_ueo, aver_noise_ueo))
        return [aver_dice_WT, aver_dice_TC], [aver_noise_dice_WT, aver_noise_dice_TC], [
                   aver_hd_WT, aver_hd_TC], [aver_noise_hd_WT, aver_noise_hd_TC], [
                   aver_assd_WT, aver_assd_TC], [aver_noise_assd_WT, aver_noise_assd_TC]
    else:
        aver_certainty = certainty_total  / num
        aver_noise_certainty = noise_certainty_total / num
        aver_mne = mne_total / num
        aver_noise_mne = noise_mne_total/num
        aver_ece = ece_total / num
        aver_noise_ece = noise_ece_total / num
        aver_ueo = ueo_total / num
        aver_noise_ueo = noise_ueo_total/num
        aver_dice = dice_total / num
        aver_noise_dice = noise_dice_total / num
        aver_iou = iou_total / num
        aver_noise_iou = noise_iou_total / num
        aver_hd = hd_total / num
        # aver_hd95 = hd95_total / num
        aver_noise_hd = noise_hd_total / num
        # aver_noise_hd95 = noise_hd95_total / num
        aver_assd = assd_total / num
        aver_noise_assd = noise_assd_total / num
        print('aver_dice=%f,aver_noise_dice = %f' % (aver_dice*100, aver_noise_dice*100))
        print('aver_iou=%f,aver_noise_iou = %f' % (aver_iou*100, aver_noise_iou*100))
        print('aver_hd=%f,aver_noise_hd = %f' % (aver_hd, aver_noise_hd))
        print('aver_assd=%f,aver_noise_assd = %f' % (aver_assd, aver_noise_assd))
        print('aver_certainty=%f,aver_noise_certainty = %f' % (aver_certainty, aver_noise_certainty))
        print('aver_mne=%f,aver_noise_mne = %f' % (aver_mne, aver_noise_mne))
        print('aver_ece=%f,aver_noise_ece = %f' % (aver_ece, aver_noise_ece))
        print('aver_ueo=%f,aver_noise_ueo = %f' % (aver_ueo, aver_noise_ueo))
        # print('aver_hd95=%f,aver_noise_hd95 = %f' % (aver_hd95, aver_noise_hd95))
        logging.info(
            'aver_dice=%f,aver_noise_dice = %f' % (aver_dice*100, aver_noise_dice*100))
        logging.info(
            'aver_iou=%f,aver_noise_iou = %f' % (aver_iou*100, aver_noise_iou*100))
        logging.info(
            'aver_hd=%f,aver_noise_hd = %f' % (aver_hd, aver_noise_hd))
        logging.info(
            'aver_assd=%f,aver_noise_assd = %f' % (aver_assd, aver_noise_assd))
        # logging.info(
        #     'aver_hd95=%f,aver_noise_hd95 = %f' % (aver_hd95, aver_noise_hd95))
        logging.info('aver_certainty=%f,aver_noise_certainty = %f' % (aver_certainty, aver_noise_certainty))
        logging.info('aver_mne=%f,aver_noise_mne = %f' % (aver_mne, aver_noise_mne))
        logging.info('aver_ece=%f,aver_noise_ece = %f' % (aver_ece, aver_noise_ece))
        logging.info('aver_ueo=%f,aver_noise_ueo = %f' % (aver_ueo, aver_noise_ueo))
        return aver_dice, aver_noise_dice, aver_hd, aver_noise_hd, aver_assd, aver_noise_assd

if __name__ == "__main__":
    args = getArgs()
    if args.dataset == 'BraTS':
        args.batch_size = 2
        args.num_classes = 4
        args.out_size = (240, 240,160)
        input_tensor = torch.randn(1, 4, args.out_size[0], args.out_size[1], args.out_size[2]).cuda()
    elif args.dataset == 'ISIC':
        args.num_classes = 2
        args.out_size = (224, 300)
        input_tensor = torch.randn(1, 3, args.out_size[0], args.out_size[1]).cuda()
    elif args.dataset == 'LiTS' :
        args.num_classes = 3
        args.out_size = (16, 256, 256)
        input_tensor = torch.randn(1, 1, args.out_size[0], args.out_size[1], args.out_size[2]).cuda()
    elif args.dataset == 'DRIVE' :
        args.num_classes = 2
        args.out_size = (584, 565)
        input_tensor = torch.randn(1, 3, args.out_size[0], args.out_size[1]).cuda()
    elif args.dataset == 'TOAR' :
        args.num_classes = 7
        args.out_size = (512, 512)
        input_tensor = torch.randn(1, 3, args.out_size[0], args.out_size[1]).cuda()
    elif args.dataset == 'HC' :
        args.num_classes = 10
        args.out_size = (128, 1024)
        input_tensor = torch.randn(1, 3, args.out_size[0], args.out_size[1]).cuda()
    else:
        print('There is no this dataset')
        raise NameError
    train_loader, valid_loader, test_loader = getDataset(args)

    model = TMSU(args)
    # calculate model's Params & Flops
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of model's Params: %.2fM" % (total / 1e6))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    model.cuda()

    log_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'log', args.experiment + args.date)
    log_file = log_dir + '.txt'
    log_args(log_file)
    epoch_loss = 0
    best_dice = 0
    loss_list = []
    dice_list = []
    OOD_Condition = ['noise','blur','mask']

    resume = ''
    if os.path.isfile(resume):
        logging.info('loading checkpoint {}'.format(resume))
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        logging.info('Successfully loading checkpoint {} and training from epoch: {}'
                     .format(resume, args.start_epoch))
    else:
        logging.info('re-training!!!')

    if args.mode =='train&test':
        for epoch in range(args.start_epoch, args.end_epochs + 1):
            print('===========Train begining!===========')
            print('Epoch {}/{}'.format(epoch, args.end_epochs - 1))
            epoch_loss = train(args,model,optimizer,epoch,train_loader)
            print("epoch %d avg_loss:%0.3f" % (epoch, epoch_loss))
            val_loss, best_dice = val(args, model, epoch, best_dice, valid_loader)
            loss_list.append(epoch_loss)
            dice_list.append(best_dice)
        loss_plot(args, loss_list)
        metrics_plot(args, 'dice', dice_list)
        test_dice,test_noise_dice,test_hd,test_noise_hd,test_assd,test_noise_assd = test(args,model,test_loader)
    elif args.mode =='train':
        for epoch in range(1, args.end_epochs + 1):
            print('===========Train begining!===========')
            print('Epoch {}/{}'.format(epoch, args.end_epochs - 1))
            epoch_loss = train(args,model,optimizer,epoch,train_loader)
            print("epoch %d avg_loss:%0.3f" % (epoch, epoch_loss))
            val_loss, best_dice = val(args, model, epoch, best_dice, valid_loader)
    elif args.mode =='test':
        for j in range(0,2):
            args.OOD_Condition = OOD_Condition[j]
            print("arg.OOD_Condition: %s" % (OOD_Condition[j]))
            start = 1
            end = 4
            for i in range(start,end):
                print("arg.OOD_Level: %d" % (i))
                args.OOD_Level = i
                train_loader, valid_loader, test_loader = getDataset(args)
                test_dice, test_noise_dice, test_hd, test_noise_hd, test_assd, test_noise_assd = test(args, model, test_loader)
    else:
        print('There is no this mode')
        raise NameError
