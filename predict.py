import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import numpy as np
from binary import assd
import nibabel as nib
import imageio
import cv2
from models.criterions import get_soft_label
from medpy.metric import binary
import onnxruntime as rt
import math
from PIL import Image
from sklearn.externals import joblib
from test_uncertainty import ece_binary,UncertaintyAndCorrectionEvalNumpy,Normalized_U
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def RandomMaskingGenerator(input_size, mask_ratio):
    if not isinstance(input_size, tuple):
        input_size = (input_size,) * 2
    if len(input_size)==2:
        height, width = input_size
        num_patches = height * width
        num_mask = int(mask_ratio * num_patches)
    else:
        height, width,t = input_size
        num_patches = height * width * t
        num_mask = int(mask_ratio * num_patches)
    repr_str = "Maks: total patches {}, mask patches {}".format(
            num_patches, num_mask
        )
    mask = np.hstack([
            # np.zeros(num_patches - num_mask),
            # np.ones(num_mask),
        np.zeros(num_mask),
        np.ones(num_patches - num_mask),
        ])
    np.random.shuffle(mask)
    if len(input_size) == 2:
        mask.resize(input_size)
    else:
        mask.resize(input_size)
    return mask,repr_str

def cal_ueo(to_evaluate,thresholds):
    UEO = []
    for threshold in thresholds:
        results = dict()
        metric = UncertaintyAndCorrectionEvalNumpy(threshold)
        metric(to_evaluate,results)
        ueo = results['corrected_add_dice']
        UEO.append(ueo)
    max_UEO = max(UEO)
    return max_UEO

def cal_ece(logits,targets):
    # ece_total = 0
    logit = logits
    target = targets
    pred = F.softmax(logit, dim=0)
    pc = pred.cpu().detach().numpy()
    pc = pc.argmax(0)
    ece = ece_binary(pc, target)
    return ece

def cal_ece_our(preds,targets):
    # ece_total = 0
    target = targets
    pc = preds.cpu().detach().numpy()
    ece = ece_binary(pc, target)
    return ece

def Uentropy(logits,c):
    # c = 4
    # logits = torch.randn(1, 4, 240, 240,155).cuda()
    pc = F.softmax(logits, dim=1)  # 1 4 240 240 155
    logpc = F.log_softmax(logits, dim=1)  # 1 4 240 240 155
    # u_all1 = -pc * logpc / c
    u_all = -pc * logpc / math.log(c)
    # max_u = torch.max(u_all)
    # min_u = torch.min(u_all)
    # NU1 = torch.sum(u_all, dim=1)
    # k = u_all.shape[1]
    # NU2 = torch.sum(u_all[:, 0:u_all.shape[1]-1, :, :], dim=1)
    NU = torch.sum(u_all[:,1:u_all.shape[1],:,:], dim=1)
    return NU

def Uentropy_our(logits,c):
    # c = 4
    # logits = torch.randn(1, 4, 240, 240,155).cuda()
    pc = logits  # 1 4 240 240 155
    logpc = torch.log(logits)  # 1 4 240 240 155
    # u_all1 = -pc * logpc / c
    u_all = -pc * logpc / math.log(c)
    # max_u = torch.max(u_all)
    # min_u = torch.min(u_all)
    # NU1 = torch.sum(u_all, dim=1)
    # k = u_all.shape[1]
    # NU2 = torch.sum(u_all[:, 0:u_all.shape[1]-1, :, :], dim=1)
    NU = torch.sum(u_all[:,1:u_all.shape[1],:,:], dim=1)
    return NU

def torch2onnx(model, save_path ):
    model.eval()
    batch_size = 2
    data = torch.rand(1, batch_size,128, 128, 128)
    input_names = ['input']
    output_names = ['out']
    torch.onnx.export(model,
                      data,
                      save_path,
                      export_params=True,
                      opset_version=11,
                      input_names=input_names,
                      output_names=output_names)
    print("torch2onnx finish")

def one_hot(ori, classes):

    batch, h, w, d = ori.size()
    new_gd = torch.zeros((batch, classes, h, w, d), dtype=ori.dtype).cuda()
    for j in range(classes):
        index_list = (ori == j).nonzero()

        for i in range(len(index_list)):
            batch, height, width, depth = index_list[i]
            new_gd[batch, j, height, width, depth] = 1

    return new_gd.float()

def one_hot_co(ori, classes):

    batch, h, w, d = ori.size()
    new_gd = torch.zeros((batch, classes, h, w, d), dtype=ori.dtype).cuda()
    for j in range(classes):
        temp = new_gd[:,j,:,:,:]
        INDEX=[ori == j]
        temp[[ori == j]]=1
        new_gd[:, j, :, :, :]=temp
    return new_gd.float()

def one_hot_co2D(ori, classes):

    batch,c, h, w = ori.size()
    new_gd = torch.zeros((batch, classes, h, w), dtype=ori.dtype).cuda()
    for j in range(classes):
        temp = new_gd[:,j,:,:].unsqueeze(1)
        # temp = new_gd[:, j, :, :]
        # INDEX=[ori == j]
        temp[[ori == j]]=1
        new_gd[:, j, :, :, ] = temp.squeeze(1)
        # new_gd[:, j, :, :,]=temp
    return new_gd.float()

def tailor_and_concat(x, model):
    temp = []

    temp.append(x[..., :128, :128, :128])
    temp.append(x[..., :128, 112:240, :128])
    temp.append(x[..., 112:240, :128, :128])
    temp.append(x[..., 112:240, 112:240, :128])
    temp.append(x[..., :128, :128, 27:155])
    temp.append(x[..., :128, 112:240, 27:155])
    temp.append(x[..., 112:240, :128, 27:155])
    temp.append(x[..., 112:240, 112:240, 27:155])

    if x.shape[1] == 1:
        y = torch.cat((x.clone(), x.clone(), x.clone(), x.clone()), 1)
    elif x.shape[1] == 4:
        y = x.clone()
    else:
        y = torch.cat((x.clone(), x.clone()), 1)

    for i in range(len(temp)):
        temp[i] = model(temp[i])
    # .squeeze(0)
    # l= temp[0].unsqueeze(0)
    y[..., :128, :128, :128] = temp[0]
    y[..., :128, 128:240, :128] = temp[1][..., :, 16:128, :]
    y[..., 128:240, :128, :128] = temp[2][..., 16:128, :, :]
    y[..., 128:240, 128:240, :128] = temp[3][..., 16:128, 16:128, :]
    y[..., :128, :128, 128:155] = temp[4][..., 96:123]
    y[..., :128, 128:240, 128:155] = temp[5][..., :, 16:128, 96:123]
    y[..., 128:240, :128, 128:155] = temp[6][..., 16:128, :, 96:123]
    y[..., 128:240, 128:240, 128:155] = temp[7][..., 16:128, 16:128, 96:123]

    return y[..., :155]

def model_PU(x, model):

    y = model.sample_m(x,m=8,testing=True)
    return y

def tailor_and_concat_PU(x, model):
    temp = []

    temp.append(x[..., :128, :128, :128])
    temp.append(x[..., :128, 112:240, :128])
    temp.append(x[..., 112:240, :128, :128])
    temp.append(x[..., 112:240, 112:240, :128])
    temp.append(x[..., :128, :128, 27:155])
    temp.append(x[..., :128, 112:240, 27:155])
    temp.append(x[..., 112:240, :128, 27:155])
    temp.append(x[..., 112:240, 112:240, 27:155])

    if x.shape[1] == 1:
        y = torch.cat((x.clone(), x.clone(), x.clone(), x.clone()), 1)
    elif x.shape[1] == 4:
        y = x.clone()
    else:
        y = torch.cat((x.clone(), x.clone()), 1)

    for i in range(len(temp)):
        temp[i] = model.sample_m(temp[i],m=16,testing=True)
        # temp[i] = model(temp[i], testing=True)
    # .squeeze(0)
    # l= temp[0].unsqueeze(0)
    y[..., :128, :128, :128] = temp[0]
    y[..., :128, 128:240, :128] = temp[1][..., :, 16:128, :]
    y[..., 128:240, :128, :128] = temp[2][..., 16:128, :, :]
    y[..., 128:240, 128:240, :128] = temp[3][..., 16:128, 16:128, :]
    y[..., :128, :128, 128:155] = temp[4][..., 96:123]
    y[..., :128, 128:240, 128:155] = temp[5][..., :, 16:128, 96:123]
    y[..., 128:240, :128, 128:155] = temp[6][..., 16:128, :, 96:123]
    y[..., 128:240, 128:240, 128:155] = temp[7][..., 16:128, 16:128, 96:123]

    return y[..., :155]

def tailor_and_concat_onnx(x, model):
    temp = []

    temp.append(x[..., :128, :128, :128])
    temp.append(x[..., :128, 112:240, :128])
    temp.append(x[..., 112:240, :128, :128])
    temp.append(x[..., 112:240, 112:240, :128])
    temp.append(x[..., :128, :128, 27:155])
    temp.append(x[..., :128, 112:240, 27:155])
    temp.append(x[..., 112:240, :128, 27:155])
    temp.append(x[..., 112:240, 112:240, 27:155])

    if x.shape[1] == 1:
        y = torch.cat((x.clone(), x.clone(), x.clone(), x.clone()), 1)
    else:
        y = torch.cat((x.clone(), x.clone()), 1)

    for i in range(len(temp)):
        torch2onnx(model, "tempnet.onnx")
        sess = rt.InferenceSession("tempnet.onnx")
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        temp[i] = sess.run([output_name], {input_name: np.array(temp[i])})
        # temp[i] = model(temp[i])
    # .squeeze(0)
    # l= temp[0].unsqueeze(0)
    y[..., :128, :128, :128] = temp[0]
    y[..., :128, 128:240, :128] = temp[1][..., :, 16:128, :]
    y[..., 128:240, :128, :128] = temp[2][..., 16:128, :, :]
    y[..., 128:240, 128:240, :128] = temp[3][..., 16:128, 16:128, :]
    y[..., :128, :128, 128:155] = temp[4][..., 96:123]
    y[..., :128, 128:240, 128:155] = temp[5][..., :, 16:128, 96:123]
    y[..., 128:240, :128, 128:155] = temp[6][..., 16:128, :, 96:123]
    y[..., 128:240, 128:240, 128:155] = temp[7][..., 16:128, 16:128, 96:123]

    return y[..., :155]

def hd_score2(o,t, eps_max=100,eps=0.0001):
    if (o.sum()==0):
        hd = eps_max
    elif (o.sum()==0) and (t.sum()==0):
        hd = eps
    else:
        #ret += hausdorff_distance(wt_mask, wt_pb),
        hd = binary.hd95(o, t, voxelspacing=None)

    return hd

def hd_score(o,t, eps_max=100,eps=1e-8):
    # o : Label
    # t: prediction
    if (o.sum()==0) and (t.sum()==0):
        hd = eps
    elif (o.sum()!=0) and (t.sum()==0):
        hd = eps_max
    elif (o.sum()==0) and (t.sum()!=0):
        hd = eps
    else:
        #ret += hausdorff_distance(wt_mask, wt_pb),
        hd = binary.hd95(o, t, voxelspacing=None)

    return hd

def mIOU(o, t, eps=1e-8):
    num = (o*t).sum() + eps
    den = (o | t).sum() + eps
    return num/den


def softmax_mIOU_score(output, target):
    mIOU_score = []
    mIOU_score.append(mIOU(o=(output==1),t=(target==1)))
    mIOU_score.append(mIOU(o=(output==2),t=(target==2)))
    mIOU_score.append(mIOU(o=(output==3),t=(target==3)))
    return mIOU_score

def softmax_mIOU_litsscore(output, target):
    mIOU_score = []
    mIOU_score.append(mIOU(o=(output==1),t=(target==1)))
    mIOU_score.append(mIOU(o=(output==2),t=(target==2)))
    return mIOU_score

def softmax_output_hd(output, target):
    ret = []

    # whole (label: 1 ,2 ,3)
    o = output > 0; t = target > 0 # ce
    ret += hd_score(o, t),
    # core (tumor core 1 and 3)
    o = (output == 1) | (output == 3)
    t = (target == 1) | (target == 3)
    ret += hd_score(o, t),
    # active (enhanccing tumor region 1 )# 3
    o = (output == 3);t = (target == 3)
    ret += hd_score(o, t),

    return ret

def softmax_output_litshd(output, target):
    ret = []

    # whole (label: 1 ,2 ,3)
    o = output > 0; t = target > 0 # ce
    ret += hd_score(o, t),
    # active (enhanccing tumor region 1 )# 3
    o = (output == 2);t = (target == 2)
    ret += hd_score(o, t),

    return ret
# def softmax_output_assd(output, target):
#     ret = []
#
#     # whole (label: 1 ,2 ,3)
#     o = output > 0; t = target > 0 # ce
#     ret += assd(o, t)
#     # core (tumor core 1 and 3)
#     o = (output == 1) | (output == 3)
#     t = (target == 1) | (target == 3)
#     ret += assd(o, t)
#     # active (enhanccing tumor region 1 )# 3
#     o = (output == 3);t = (target == 3)
#     ret += assd(o, t)
#
#     return return

def softmax_assd_score(output, target):
    ret = []

    # whole (label: 1 ,2 ,3)
    o = output > 0; t = target > 0 # ce
    ret += assd(o, t),
    # core (tumor core 1 and 3)
    o = (output == 1) | (output == 3)
    t = (target == 1) | (target == 3)
    ret += assd(o, t),
    # active (enhanccing tumor region 1 )# 3
    o = (output == 3);t = (target == 3)
    ret += assd(o, t),

    return ret

def softmax_assd_litsscore(output, target):
    ret = []

    # Liver (label: 1 ,2 )
    o = output > 0; t = target > 0 # ce
    ret.append( assd(o, t))
    # tumor (2)
    o = (output == 2);t = (target == 2)
    ret.append( assd(o, t))

    return ret

def dice_score(o, t, eps=1e-8):
    num = 2*(o*t).sum() + eps
    den = o.sum() + t.sum() + eps
    return num/den

def softmax_output_litsdice(output, target):
    ret = []

    # Liver (label: 1 ,2 )
    # o = output > 0; t = target > 0 # ce
    # ret += dice_score(o, t),
    o = (output == 1) | (output == 2)
    t = (target == 1) | (target == 2)
    ret += dice_score(o, t),
    # tumor (2)
    o = (output == 2);t = (target == 2)
    ret += dice_score(o, t),

    return ret

def softmax_output_dice(output, target):
    ret = []

    # whole (label: 1 ,2 ,3)
    o = output > 0; t = target > 0 # ce
    ret += dice_score(o, t),
    # core (tumor core 1 and 3)
    o = (output == 1) | (output == 3)
    t = (target == 1) | (target == 3)
    ret += dice_score(o, t),
    # active (enhanccing tumor region 1 )# 3
    o = (output == 3);t = (target == 3)
    ret += dice_score(o, t),

    return ret

def HD_isic(prediction, soft_ground_truth, num_class):
    o = prediction > 0; t = soft_ground_truth > 0 # ce
    ret = hd_score(o, t)
    return ret
def Dice_isic(prediction, soft_ground_truth, num_class):
    # predict = prediction.permute(0, 2, 3, 1)
    pred = prediction.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
    dice_score = 2.0 * intersect / (ref_vol + seg_vol + 1.0)
    dice_mean_score = torch.mean(dice_score.data[1:dice_score.shape[0]])

    return dice_mean_score
def Dice_lits(prediction, soft_ground_truth, num_class):
    # predict = prediction.permute(0, 2, 3, 1)
    pred = prediction.contiguous().view(-1, 1)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, 1)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
    dice_score = 2.0 * intersect / (ref_vol + seg_vol + 1.0)
    dice_mean_score = torch.mean(dice_score.data[1:dice_score.shape[0]])

    return dice_mean_score

def dice_isic(prediction, soft_ground_truth, num_class):
    ret = []

    # whole (label: 1 ,2 ,3)
    o = prediction > 0; t = soft_ground_truth > 0 # ce
    ret = dice_score(o, t)

    return ret


def IOU_isic(prediction, soft_ground_truth, num_class):
    # predict = prediction.permute(0, 2, 3, 1)
    pred = prediction.contiguous().view(-1, num_class)
    # pred = F.softmax(pred, dim=1)
    ground = soft_ground_truth.view(-1, num_class)
    ref_vol = torch.sum(ground, 0)
    intersect = torch.sum(ground * pred, 0)
    seg_vol = torch.sum(pred, 0)
    iou_score = intersect / (ref_vol + seg_vol - intersect + 1.0)
    iou_mean_score = torch.mean(iou_score.data[1:iou_score.shape[0]])

    return iou_mean_score

def iou_isic(output, target):
    mIOU_score=(mIOU(o=(output==1),t=(target==1)))
    return mIOU_score

keys = 'whole', 'core', 'enhancing', 'loss'

def validate_softmax(
        args,
        save_dir,
        best_dice,
        current_epoch,
        valid_loader,
        model,
        names=None,# The names of the patients orderly!
        ):


    save_freq = args.save_freq
    end_epoch =args.end_epochs
    multimodel = args.input_modality
    Net_name = args.model_name
    runtimes = []
    dice_total = 0
    iou_total = 0
    num = len(valid_loader)

    for i, data in enumerate(valid_loader):
        print('-------------------------------------------------------------------')
        msg = 'Subject {}/{}, '.format(i + 1, len(valid_loader))
        x, target = data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x = x.to(device)
        target = target.to(device)

        torch.cuda.synchronize()  # add the code synchronize() to correctly count the runtime.
        start_time = time.time()
        if args.dataset == 'BraTS':
            H, W, T = 240, 240, 160
            target = torch.squeeze(target).cpu().numpy()
            if Net_name =='PU':
                logit = tailor_and_concat_PU(x, model)
            else:
                logit = tailor_and_concat(x, model)
            output = F.softmax(logit, dim=1)
            output = output[0, :, :H, :W, :T].cpu().detach().numpy()
            output = output.argmax(0)
            iou_res = softmax_mIOU_score(output, target[:, :, :155])
            dice_res = softmax_output_dice(output, target[:, :, :155])
            dice_total += dice_res[1]
            iou_total += iou_res[1]
        elif args.dataset == 'LiTS':
            if Net_name =='PU':
                logit = model_PU(x,model)
                logit = logit.unsqueeze(0)
            else:
                logit = model(x)
            output = F.softmax(logit, dim=1)
            # output = output.cpu().detach().numpy()
            predicted = output.argmax(1)
            output = torch.squeeze(predicted).cpu().detach().numpy()
            target = torch.squeeze(target).cpu().numpy()

            dice_res = softmax_output_litsdice(output, target)
            iou_res = softmax_mIOU_litsscore(output, target)
            dice_total += dice_res[0]
            iou_total += iou_res[0]
        else:
            if Net_name =='PU':
                logit = model_PU(x,model)
                logit = logit.unsqueeze(0)
            else:
                logit = model(x)
            output = F.softmax(logit, dim=1)
            # output = output.cpu().detach().numpy()
            predicted = output.argmax(1)
            soft_gt = get_soft_label(target, args.num_classes)
            soft_predicted = get_soft_label(predicted.unsqueeze(0), args.num_classes)
            iou_res = IOU_isic(soft_predicted, soft_gt, args.num_classes)
            dice_res = Dice_isic(soft_predicted, soft_gt, args.num_classes)
            dice_total += dice_res
            iou_total += iou_res


        print('current_dice:{}'.format(dice_res))
        torch.cuda.synchronize()
        elapsed_time = time.time() - start_time
        logging.info('Single sample test time consumption {:.2f} minutes!'.format(elapsed_time / 60))
        runtimes.append(elapsed_time)

        if names:
            name = names[i]
            msg += '{:>20}, '.format(name)
        print(msg)

    aver_dice = dice_total / num
    aver_iou = iou_total / num
    print('current_aver_dice:{}'.format(aver_dice))
    if (current_epoch + 1) % int(save_freq) == 0:
        if aver_dice > best_dice or (current_epoch + 1) % int(end_epoch - 1) == 0 \
            or (current_epoch + 1) % int(end_epoch - 2) == 0 \
            or (current_epoch + 1) % int(end_epoch - 3) == 0:
            print('aver_dice:{} > best_dice:{}'.format(aver_dice, best_dice))
            logging.info('aver_dice:{} > best_dice:{}'.format(aver_dice, best_dice))
            logging.info('===========>save best model!')
            best_dice = aver_dice
            print('===========>save best model!')
            file_name = os.path.join(save_dir, args.model_name + '_' + args.dataset +'_'+ args.folder + '_epoch_{}.pth'.format(current_epoch))
            torch.save({
            'epoch': current_epoch,
            'state_dict': model.state_dict(),
                },
                file_name)
    print('runtimes:', sum(runtimes)/len(runtimes))

    return best_dice,aver_dice,aver_iou

def testensemblemax(
        test_loader,
        model,
        args,
        names=None,  # The names of the patients orderly!
        ):
    multimodel = args.input_modality
    Net_name = args.model_name
    savepath = args.submission + '/'+ str(Net_name) + '/' + str(args.dataset) + '/' + str(args.OOD_Condition) +'/'+ str(args. OOD_Level)
    # Variance = args.Variance
    save_format = args.save_format
    snapshot = args.snapshot
    en_time = args.en_time
    # model.eval()

    runtimes = []
    dice_total= 0
    dice_total_WT = 0
    dice_total_TC = 0
    dice_total_ET = 0
    hd_total=0
    assd_total = 0
    hd_total_WT = 0
    hd_total_TC = 0
    hd_total_ET = 0
    iou_total_WT = 0
    iou_total_TC = 0
    iou_total_ET = 0
    assd_total_WT = 0
    assd_total_TC = 0
    assd_total_ET = 0
    iou_total=0
    noise_dice_total=0
    noise_iou_total = 0
    noise_iou_total_WT = 0
    noise_iou_total_TC = 0
    noise_iou_total_ET = 0
    noise_dice_total_WT = 0
    noise_dice_total_TC = 0
    noise_dice_total_ET = 0
    noise_assd_total_WT = 0
    noise_assd_total_TC = 0
    noise_assd_total_ET = 0
    noise_assd_total = 0
    noise_hd_total = 0
    noise_hd_total_WT = 0
    noise_hd_total_TC = 0
    noise_hd_total_ET = 0
    mean_uncertainty_total=0
    noised_mean_uncertainty_total=0
    uncertainty_total = 0
    noised_uncertainty_total = 0
    ece_total = 0
    noise_ece_total = 0
    ece = 0
    noise_ece = 0
    ueo_total = 0
    noise_ueo_total = 0
    mne_total = 0
    noised_mne_total = 0
    num = len(test_loader)

    for i, data in enumerate(test_loader):
        print('-------------------------------------------------------------------')
        msg = 'Subject {}/{}, '.format(i + 1, len(test_loader))
        x, noised_x,target = data  # input ground truth
        # noise_m = torch.randn_like(x) * Variance
        # noise = torch.clamp(torch.randn_like(x) * Variance, -Variance * 2, Variance * 2)
        # noise = torch.clamp(torch.randn_like(x) * Variance, -Variance, Variance)
        # noise = torch.clamp(torch.randn_like(x) * Variance)
        # noised_x = x + noise_m
        if Net_name == 'PU' or Net_name=='ViT':
            x = x.cuda()
            noised_x = noised_x.cuda()
            target = target.cuda()
            # x = x.to(device)
            # noised_x = noised_x.to(device)
        else:
            x.cuda()
            noised_x.cuda()

        num_classes = args.num_classes
        torch.cuda.synchronize()  # add the code synchronize() to correctly count the runtime.
        start_time = time.time()
        if args.dataset=='BraTS':

            Oinput = torch.squeeze(x).cpu().detach().numpy()
            NOinput = torch.squeeze(noised_x).cpu().detach().numpy()
            H, W, T = 240, 240, 155
            # output = np.zeros((4, x.shape[2], x.shape[3], 155),dtype='float32')
            # noised_output = np.zeros((4, x.shape[2], x.shape[3], 155),dtype='float32')
            mean_uncertainty = torch.zeros(0)
            noised_mean_uncertainty = torch.zeros(0)
            # pc = torch.zeros(x.shape[2], x.shape[3], 155)
            # noised_pc = torch.zeros(x.shape[2], x.shape[3], 155)
            logit = torch.zeros(1,x.shape[1],x.shape[2], x.shape[3], 155)
            logit_noise = torch.zeros(1,x.shape[1],x.shape[2], x.shape[3], 155)
            # pc = np.zeros((x.shape[2], x.shape[3], 155), dtype='float32')
            # noised_pc = np.zeros((x.shape[2], x.shape[3], 155), dtype='float32')
            # # torch.cuda.synchronize()  # add the code synchronize() to correctly count the runtime.
            # # start_time = time.time()
            # uncertainty = torch.zeros(1, x.shape[2], x.shape[3], 155)
            # noised_uncertainty = torch.zeros(1, x.shape[2], x.shape[3], 155)
            model_times = en_time
            target = torch.squeeze(target).cpu().numpy()
            Otarget = target
            # load ensemble models
            for j in range(model_times):
                print('ensemble model:{}'.format(j))
                logit += tailor_and_concat(x, model[j])  # 1 4 240 240 155
                logit_noise += tailor_and_concat(noised_x, model[j])  # 1 4 240 240 155

            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time
            logging.info('Single sample test time consumption {:.2f} minutes!'.format(elapsed_time / 60))
            # calculate ensemble uncertainty by normalized entropy
            logit = logit / model_times
            logit_noise = logit_noise /model_times
            # u
            uncertainty = Uentropy(logit, num_classes)
            uncertainty = torch.squeeze(uncertainty)
            mean_uncertainty = torch.mean(uncertainty)
            noised_uncertainty = Uentropy(logit_noise, num_classes)
            noised_mean_uncertainty = torch.mean(noised_uncertainty)
            uncertainty_total += mean_uncertainty
            noised_uncertainty_total += noised_mean_uncertainty

            # ece
            ece = cal_ece(torch.squeeze(logit), target[:, :, :155])
            noise_ece = cal_ece(torch.squeeze(logit_noise), target[:, :, :155])
            ece_total += ece
            noise_ece_total += noise_ece
            # pc
            logit = F.softmax(logit, dim=1)
            output = logit[0, :, :H, :W, :T].cpu().detach().numpy()
            pc = output.argmax(0)
            logit_noise = F.softmax(logit_noise, dim=1)
            noised_output = logit_noise[0, :, :H, :W, :T].cpu().detach().numpy()
            noised_pc = noised_output.argmax(0)

            ece_total += ece
            noise_ece_total += noise_ece

            joblib.dump({'pc': pc,
                         'noised_pc': noised_pc, 'noised_uncertainty': noised_uncertainty,
                         'uncertainty': uncertainty}, 'Uensemble_uncertainty_{}.pkl'.format(i))

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
            noise_ueo_total += noise_UEO
            # # iou_res = softmax_mIOU_score(output, target[:, :, :155])
            # output = output[0, :, :H, :W, :T].cpu().detach().numpy()
            # output = output.argmax(0)
            U_output = torch.squeeze(uncertainty).cpu().detach().numpy()
            NU_output = torch.squeeze(noised_uncertainty).cpu().detach().numpy()
            # iou_res = softmax_mIOU_score(output, target[:, :, :155])
            hd_res = softmax_output_hd(pc, target[:, :, :155])
            dice_res = softmax_output_dice(pc, target[:, :, :155])
            dice_total_WT += dice_res[0]
            dice_total_TC += dice_res[1]
            dice_total_ET += dice_res[2]
            hd_total_WT += hd_res[0]
            hd_total_TC += hd_res[1]
            hd_total_ET += hd_res[2]
            assd_res = softmax_assd_score(pc,target[:, :, :155])
            assd_total_WT += assd_res[0]
            assd_total_TC += assd_res[1]
            assd_total_ET += assd_res[2]
            # for noise_x
            # noised_output = noised_output[0, :, :H, :W, :T].cpu().detach().numpy()
            # noised_output = noised_output.argmax(0)
            # noise_iou_res = softmax_mIOU_score(noised_output, target[:, :, :155])
            noise_hd_res = softmax_output_hd(noised_pc, target[:, :, :155])
            noise_dice_res = softmax_output_dice(noised_pc, target[:, :, :155])
            noise_assd_res = softmax_assd_score(noised_pc, target[:, :, :155])
            noise_dice_total_WT += noise_dice_res[0]
            noise_dice_total_TC += noise_dice_res[1]
            noise_dice_total_ET += noise_dice_res[2]
            noise_hd_total_WT += noise_hd_res[0]
            noise_hd_total_TC += noise_hd_res[1]
            noise_hd_total_ET += noise_hd_res[2]
            noise_assd_total_WT += noise_assd_res[0]
            noise_assd_total_TC += noise_assd_res[1]
            noise_assd_total_ET += noise_assd_res[2]
            print('current_dice:{},current_noised_dice:{}'.format(dice_res, noise_dice_res))
            print('current_hd:{},current_noised_hd:{}'.format(hd_res, noise_hd_res))
            print('current_assd:{},current_noised_assd:{}'.format(assd_res, noise_assd_res))
            print('current_UEO:{};current_noise_UEO:{}; current_num:{}'.format(UEO, noise_UEO, i))
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
                        os.makedirs(os.path.join(savepath, name))

                    # scipy.misc.imsave(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])
                    imageio.imwrite(os.path.join(savepath, name, str(frame) + '.png'),
                                    Snapshot_img[:, :, :, frame])
                    imageio.imwrite(
                        os.path.join(savepath, name, str(frame) + '_noised.png'),
                        Noise_Snapshot_img[:, :, :, frame])
                    imageio.imwrite(os.path.join(savepath, name, str(frame) + '_gt.png'),
                                    target_img[:, :, :, frame])
                    imageio.imwrite(os.path.join(savepath, name, str(frame) + '_input_FLR.png'),
                                    Oinput[0, :, :, frame])
                    imageio.imwrite(os.path.join(savepath, name, str(frame) + '_input_T1ce.png'),
                                    Oinput[1, :, :, frame])
                    imageio.imwrite(os.path.join(savepath, name, str(frame) + '_input_T1.png'),
                                    Oinput[2, :, :, frame])
                    imageio.imwrite(os.path.join(savepath, name, str(frame) + '_input_T2.png'),
                                    Oinput[3, :, :, frame])
                    imageio.imwrite(os.path.join(savepath, name, str(frame) + '_noised_input_FLR.png'),
                                    NOinput[0, :, :, frame])
                    imageio.imwrite(os.path.join(savepath, name, str(frame) + '_noised_input_T1ce.png'),
                                    NOinput[1, :, :, frame])
                    imageio.imwrite(os.path.join(savepath, name, str(frame) + '_noised_input_T1.png'),
                                    NOinput[2, :, :, frame])
                    imageio.imwrite(os.path.join(savepath, name, str(frame) + '_noised_input_T2.png'),
                                    NOinput[3, :, :, frame])
                    imageio.imwrite(
                        os.path.join(savepath, name, str(frame) + '_uncertainty.png'),
                        U_output[:, :, frame])
                    imageio.imwrite(
                        os.path.join(savepath, name,
                                     str(frame) + '_noised_uncertainty.png'),
                        NU_output[:, :, frame])
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
        elif args.dataset == 'LiTS':
            H, W, T = 256, 256, 16
            logit =  torch.zeros(x.shape[0],num_classes,x.shape[2], x.shape[3], x.shape[4])
            logit_noise =  torch.zeros(x.shape[0],num_classes,x.shape[2], x.shape[3], x.shape[4])
            output =  torch.zeros(x.shape[0],num_classes,x.shape[2], x.shape[3], x.shape[4])
            noised_output =  torch.zeros(x.shape[0],num_classes,x.shape[2], x.shape[3], x.shape[4])
            pc =  torch.zeros(x.shape[0],1,x.shape[2], x.shape[3], x.shape[4])
            noised_pc =  torch.zeros(x.shape[0],1,x.shape[2], x.shape[3], x.shape[4])
            uncertainty = torch.zeros(1,x.shape[2], x.shape[3], x.shape[4])
            noised_uncertainty = torch.zeros(1,x.shape[2], x.shape[3], x.shape[4])
            mean_uncertainty = torch.zeros(1,x.shape[2], x.shape[3], x.shape[4])
            noised_mean_uncertainty = torch.zeros(1,x.shape[2], x.shape[3], x.shape[4])
            ece = torch.zeros(1)
            noise_ece = torch.zeros(1)
            torch.cuda.synchronize()  # add the code synchronize() to correctly count the runtime.
            start_time = time.time()
            if args.model_name =='UE0':
                for j in range(en_time):
                    show_model = model[j]
                    print('ensemble model:{}'.format(j))
                    logit = show_model(x)
                    logit_noise = show_model(noised_x)
                    output += F.softmax(logit, dim=1)
                    noised_output += F.softmax(logit_noise, dim=1)
                    # U
                    uncertainty += Uentropy(logit, num_classes)
                    noised_uncertainty += Uentropy(logit_noise, num_classes)
                    # ece
                    ece += cal_ece(torch.squeeze(logit), torch.squeeze(target))
                    noise_ece += cal_ece(torch.squeeze(logit_noise), torch.squeeze(target))
                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                logging.info('Single sample test time consumption {:.2f} minutes!'.format(elapsed_time / 60))
                logging.info('Single sample test time consumption {:.2f} seconds!'.format(elapsed_time))
                # calculate ensemble uncertainty by normalized entropy
                uncertainty = uncertainty / en_time
                mean_uncertainty = torch.mean(uncertainty)
                noised_uncertainty = noised_uncertainty / en_time
                noised_mean_uncertainty = torch.mean(noised_uncertainty)
                output = output/en_time
                noised_output = noised_output/en_time
                # pc = (output/10).argmax(1)
                # noised_pc = (noised_output/10).argmax(1)
                # ece
                ece_total += ece/en_time
                noise_ece_total += noise_ece/en_time
                output = output[0, :, :, :, :, ].cpu().detach().numpy()
                output = output.argmax(0)
                target = torch.squeeze(target).cpu().numpy()  # .cpu().numpy(dtype='float32')
                Otarget = target
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
                noised_output = noised_output[0, :, :, :, :, ].cpu().detach().numpy()
                noised_output = noised_output.argmax(0)
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

                # ece
                ece_total += ece
                noise_ece_total += noise_ece
                # U ece ueo
                uncertainty_total += mean_uncertainty  # mix _uncertainty mean_uncertainty mean_uncertainty_succ
                # sum_certainty_total += sum_uncertainty
                noised_uncertainty_total += noised_mean_uncertainty  # noised_mix_uncertainty noised_mean_uncertainty noised_mean_uncertainty_succ
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
                print('current_dice:{},current_noised_dice:{}'.format(dice_res, noised_dice_res))
                print('current_iou:{},current_noised_iou:{}'.format(iou_res, noised_iou_res))
                print('current_hd:{},current_noised_hd:{}'.format(hd_res, noised_hd_res))
                print('current_assd:{},current_noised_assd:{}'.format(assd_res, noised_assd_res))
                name = str(i)
                if names:
                    name = names[i]
                    msg += '{:>20}, '.format(name)
                # uncertainty np
                Oinput = torch.squeeze(x).cpu().detach().numpy()
                NOinput = torch.squeeze(noised_x).cpu().detach().numpy()
                U_output = torch.squeeze(uncertainty).cpu().detach().numpy()
                NU_output = torch.squeeze(noised_uncertainty).cpu().detach().numpy()
                # U_output = torch.squeeze(mne).cpu().detach().numpy()
                # NU_output = torch.squeeze(noised_mne).cpu().detach().numpy()
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
                                        Snapshot_img[frame, :, :, :])
                        imageio.imwrite(
                            os.path.join(savepath, name, str(frame) + '_noised.png'),
                            Noise_Snapshot_img[frame, :, :, :])

                        imageio.imwrite(os.path.join(savepath, name, str(frame) + '_gt.png'),
                                        target_img[frame, :, :, :])
                        imageio.imwrite(
                            os.path.join(savepath, name, str(frame) + '_uncertainty.png'),
                            U_output[frame, :, :])
                        imageio.imwrite(
                            os.path.join(savepath, name,
                                         str(frame) + '_noised_uncertainty.png'),
                            NU_output[frame, :, :])
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
                        imageio.imwrite(os.path.join(savepath, name, str(frame) + '_input.png'),
                                        Oinput[frame, :, :])
                        imageio.imwrite(os.path.join(savepath, name, str(frame) + '_noised_input.png'),
                                        NOinput[frame, :, :])
        else:
            H, W = 224, 300
            logit =  torch.zeros(x.shape[0],num_classes,x.shape[2], x.shape[3])
            logit_noise =  torch.zeros(x.shape[0],num_classes,x.shape[2], x.shape[3])
            output =  torch.zeros(x.shape[0],num_classes,x.shape[2], x.shape[3])
            noised_output =  torch.zeros(x.shape[0],num_classes,x.shape[2], x.shape[3])
            pc =  torch.zeros(x.shape[0],1,x.shape[2], x.shape[3])
            noised_pc =  torch.zeros(x.shape[0],1,x.shape[2], x.shape[3])
            uncertainty = torch.zeros(1,x.shape[2], x.shape[3])
            noised_uncertainty = torch.zeros(1,x.shape[2], x.shape[3])
            mean_uncertainty = torch.zeros(1,x.shape[2], x.shape[3])
            noised_mean_uncertainty = torch.zeros(1,x.shape[2], x.shape[3])
            ece = torch.zeros(1)
            noise_ece = torch.zeros(1)
            torch.cuda.synchronize()  # add the code synchronize() to correctly count the runtime.
            start_time = time.time()
            if args.model_name =='UE0':
                for j in range(args.en_time):
                    show_model = model[j]
                    print('ensemble model:{}'.format(j))
                    logit = show_model(x)
                    logit_noise = show_model(noised_x)
                    output += F.softmax(logit, dim=1)
                    noised_output += F.softmax(logit_noise, dim=1)
                    # U
                    uncertainty += Uentropy(logit, num_classes)
                    noised_uncertainty += Uentropy(logit_noise, num_classes)
                    # ece
                    ece += cal_ece(torch.squeeze(logit), torch.squeeze(target))
                    noise_ece += cal_ece(torch.squeeze(logit_noise), torch.squeeze(target))
                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                logging.info('Single sample test time consumption {:.2f} minutes!'.format(elapsed_time / 60))
                # calculate ensemble uncertainty by normalized entropy
                uncertainty = uncertainty / args.en_time
                mean_uncertainty = torch.mean(uncertainty)
                noised_uncertainty = noised_uncertainty / args.en_time
                noised_mean_uncertainty = torch.mean(noised_uncertainty)
                pc = (output/args.en_time).argmax(1)
                noised_pc = (noised_output/args.en_time).argmax(1)
                # ece
                ece_total += ece/args.en_time
                noise_ece_total += noise_ece/args.en_time
            elif args.model_name =='UED':
                for j in range(args.en_time):
                    show_model = model[j]
                    print('ensemble model:{}'.format(j))
                    for ii in range(args.en_time):
                        print('dropout model:{}'.format(ii))
                        logit += show_model(x)
                        logit_noise += show_model(noised_x)
                        output += F.softmax(logit, dim=1)
                        noised_output += F.softmax(logit_noise, dim=1)
                        # U
                        uncertainty += Uentropy(logit, num_classes)
                        noised_uncertainty += Uentropy(logit_noise, num_classes)
                        # ece
                        ece += cal_ece(torch.squeeze(logit), torch.squeeze(target))
                        noise_ece += cal_ece(torch.squeeze(logit_noise), torch.squeeze(target))
                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                logging.info('Single sample test time consumption {:.2f} minutes!'.format(elapsed_time / 60))
                # calculate ensemble uncertainty by normalized entropy
                uncertainty = uncertainty / 100
                mean_uncertainty = torch.mean(uncertainty)
                noised_uncertainty = noised_uncertainty / 100
                noised_mean_uncertainty = torch.mean(noised_uncertainty)
                pc = (output/100).argmax(1)
                noised_pc = (noised_output/100).argmax(1)
                # ece
                ece_total += ece/100
                noise_ece_total += noise_ece/100

            # U ece ueo
            uncertainty_total += mean_uncertainty  # mix _uncertainty mean_uncertainty mean_uncertainty_succ
            # sum_certainty_total += sum_uncertainty
            noised_uncertainty_total += noised_mean_uncertainty  # noised_mix_uncertainty noised_mean_uncertainty noised_mean_uncertainty_succ
            # noise_sum_certainty_total += noised_sum_uncertainty

            # ueo
            thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
            to_evaluate = dict()
            to_evaluate['target'] = torch.squeeze(target).cpu().detach().numpy()
            u = torch.squeeze(uncertainty)
            U = u.cpu().detach().numpy()
            to_evaluate['prediction'] = torch.squeeze(pc).cpu().detach().numpy()
            to_evaluate['uncertainty'] = U
            UEO = cal_ueo(to_evaluate, thresholds)
            ueo_total += UEO
            noise_to_evaluate = dict()
            noise_to_evaluate['target'] = target.cpu().detach().numpy()
            noise_u = torch.squeeze(noised_uncertainty)
            noise_U = noise_u.cpu().detach().numpy()
            noise_to_evaluate['prediction'] = torch.squeeze(noised_pc).cpu().detach().numpy()
            noise_to_evaluate['uncertainty'] = noise_U
            noise_UEO = cal_ueo(noise_to_evaluate, thresholds)
            print('current_UEO:{};current_noise_UEO:{}; current_num:{}'.format(UEO,noise_UEO, i))
            noise_ueo_total += noise_UEO
            # uncertainty np
            Otarget = torch.squeeze(target).cpu().detach().numpy()
            Oinput = torch.squeeze(x,0).permute(1,2,0).cpu().detach().numpy()
            NOinput = torch.squeeze(noised_x,0).permute(1,2,0).cpu().detach().numpy()
            U_output = torch.squeeze(uncertainty).cpu().detach().numpy()
            NU_output = torch.squeeze(noised_uncertainty).cpu().detach().numpy()

            # dice hd assd
            soft_gt = get_soft_label(target, num_classes)
            soft_predicted = get_soft_label(pc.unsqueeze(0), num_classes)
            iou_res = IOU_isic(soft_predicted, soft_gt, num_classes)
            dice_res = Dice_isic(soft_predicted, soft_gt, num_classes)
            hd_res = HD_isic(torch.squeeze(pc).cpu().numpy(), torch.squeeze(target).cpu().numpy(), num_classes)
            assd_res = assd(torch.squeeze(pc).cpu().numpy(),
                            torch.squeeze(target).cpu().numpy())

            # del data,x,noised_x,noise_m
            noised_hd_res = HD_isic(torch.squeeze(noised_pc).cpu().numpy(), torch.squeeze(target).cpu().numpy(),
                                    num_classes)
            noised_assd_res = assd(torch.squeeze(noised_pc).cpu().numpy(), torch.squeeze(target).cpu().numpy())
            soft_noised_pc = get_soft_label(noised_pc.unsqueeze(0), num_classes)
            noised_iou_res = IOU_isic(soft_noised_pc, soft_gt, num_classes)
            noised_dice_res = Dice_isic(soft_noised_pc, soft_gt, num_classes)

            dice_total += dice_res
            iou_total += iou_res
            hd_total += hd_res
            assd_total += assd_res
            noise_dice_total += noised_dice_res
            noise_iou_total += noised_iou_res
            noise_hd_total += noised_hd_res
            noise_assd_total += noised_assd_res
            print('current_dice:{},current_noised_dice:{}'.format(dice_res, noised_dice_res))
            print('current_iou:{},current_noised_iou:{}'.format(iou_res, noised_iou_res))
            print('current_hd:{},current_noised_hd:{}'.format(hd_res, noised_hd_res))
            print('current_assd:{},current_noised_assd:{}'.format(assd_res, noised_assd_res))
            name = str(i)
            if names:
                name = names[i]
                msg += '{:>20}, '.format(name)
            name = name.replace('.npy', '')
            if snapshot:
                """ --- grey figure---"""
                # Snapshot_img = np.zeros(shape=(H,W,T),dtype=np.uint8)
                # Snapshot_img[np.where(output[1,:,:,:]==1)] = 64
                # Snapshot_img[np.where(output[2,:,:,:]==1)] = 160
                # Snapshot_img[np.where(output[3,:,:,:]==1)] = 255
                """ --- colorful figure--- """
                Snapshot_img = np.zeros(shape=(H, W), dtype=np.float32)
                Snapshot_img[np.where((torch.squeeze(pc).cpu().numpy()) == 1)] = 255


                Noise_Snapshot_img = np.zeros(shape=(H, W), dtype=np.float32)
                Noise_Snapshot_img[np.where((torch.squeeze(noised_pc).cpu().numpy()) == 1)] = 255

                # target_img = np.zeros(shape=(H, W), dtype=np.float32)
                # target_img[np.where(Otarget == 1)] = 255

                if not os.path.exists(os.path.join(savepath, name)):
                    os.makedirs(os.path.join(savepath, name))

                # scipy.misc.imsave(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])
                imageio.imwrite(os.path.join(savepath, name, '_pred.png'),
                                Snapshot_img)
                imageio.imwrite(os.path.join(savepath, name, '_noised.png'),
                                Noise_Snapshot_img)
            # imageio.imwrite(os.path.join(savepath, name, '_gt.png'),
            #                     target_img)
                imageio.imwrite(os.path.join(savepath, name, '_uncertainty.png'),
                                U_output)
                imageio.imwrite(
                    os.path.join(savepath, name, '_noised_uncertainty.png'),
                    NU_output)
                U_img = cv2.imread(os.path.join(savepath, name, '_uncertainty.png'))
                U_heatmap = cv2.applyColorMap(U_img, cv2.COLORMAP_JET)
                cv2.imwrite(
                    os.path.join(savepath, name, '_colormap_uncertainty.png'),
                    U_heatmap)
                NU_img = cv2.imread(
                    os.path.join(savepath, name, '_noised_uncertainty.png'))
                NU_heatmap = cv2.applyColorMap(NU_img, cv2.COLORMAP_JET)
                cv2.imwrite(
                    os.path.join(savepath, name, '_colormap_noised_uncertainty.png'),
                    NU_heatmap)
            # imageio.imwrite(os.path.join(savepath, name,  '_input.png'),
            #                 Oinput)
            # imageio.imwrite(os.path.join(savepath, name, '_noised_input.png'),
            #                 NOinput)
            print(msg)


    if args.dataset == 'BraTS':
        aver_certainty = uncertainty_total / num
        aver_noise_certainty = noised_uncertainty_total / num
        aver_ece = ece_total / num
        aver_noise_ece = noise_ece_total / num
        aver_ueo = ueo_total / num
        aver_noise_ueo = noise_ueo_total / num
        aver_mne = mne_total / num
        aver_noise_mne = noised_mne_total/num
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
        aver_certainty = uncertainty_total  / num
        aver_noise_certainty = noised_uncertainty_total / num
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
        print('aver_ece=%f,aver_noise_ece = %f' % (aver_ece, aver_noise_ece))
        print('aver_ueo=%f,aver_noise_ueo = %f' % (aver_ueo, aver_noise_ueo))
        logging.info(
        'aver_dice_WT=%f,aver_dice_TC = %f' % (aver_dice_WT*100, aver_dice_TC*100))
        logging.info('aver_noise_dice_WT=%f,aver_noise_dice_TC=%f' % (
        aver_noise_dice_WT*100, aver_noise_dice_TC*100))
        logging.info('aver_iou_WT=%f,aver_iou_TC = %f' % (
        aver_iou_WT*100, aver_iou_TC*100))
        logging.info('aver_noise_iou_WT=%f,aver_noise_iou_TC = %f' % (
        aver_noise_iou_WT*100, aver_noise_iou_TC*100))
        logging.info('aver_hd_WT=%f,aver_hd_TC = %f' % (
        aver_hd_WT, aver_hd_TC))
        logging.info('aver_noise_hd_WT=%f,aver_noise_hd_TC = %f' % (
        aver_noise_hd_WT, aver_noise_hd_TC))
        logging.info('aver_assd_WT=%f,aver_assd_TC = %f' % (aver_assd_WT, aver_assd_TC))
        logging.info('aver_noise_assd_WT=%f,aver_noise_assd_TC = %f' % (
            aver_noise_assd_WT, aver_noise_assd_TC))
        logging.info('aver_certainty=%f,aver_noise_certainty = %f' % (aver_certainty, aver_noise_certainty))
        logging.info('aver_ece=%f,aver_noise_ece = %f' % (aver_ece, aver_noise_ece))
        logging.info('aver_ueo=%f,aver_noise_ueo = %f' % (aver_ueo, aver_noise_ueo))
        return [aver_dice_WT, aver_dice_TC], [aver_noise_dice_WT, aver_noise_dice_TC], [
                   aver_hd_WT, aver_hd_TC], [aver_noise_hd_WT, aver_noise_hd_TC], [
                   aver_assd_WT, aver_assd_TC], [aver_noise_assd_WT, aver_noise_assd_TC]
    else:
        aver_certainty = uncertainty_total  / num
        aver_noise_certainty = noised_uncertainty_total / num
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
        logging.info('aver_ece=%f,aver_noise_ece = %f' % (aver_ece, aver_noise_ece))
        logging.info('aver_ueo=%f,aver_noise_ueo = %f' % (aver_ueo, aver_noise_ueo))
        return aver_dice, aver_noise_dice, aver_hd, aver_noise_hd, aver_assd, aver_noise_assd

def test_softmax(args,
        test_loader,
        model,
        load_file,
        names=None,  # The names of the patients orderly!
        verbose=False,
        # snapshot=False,  # for visualization. Default false. It is recommended to generate the visualized figures.
        # visual='',  # the path to save visualization
        ):
    multimodel = args.input_modality
    Net_name = args.model_name
    save_format = args.save_format
    snapshot = args.snapshot
    savepath = args.submission + '/'+ str(Net_name) + '/' + str(args.dataset) + '/' + str(args.OOD_Condition) +'/'+ str(args. OOD_Level)
    # model.eval()
    runtimes=[]
    all_target = []
    dice_total= 0
    dice_total_WT = 0
    dice_total_TC = 0
    dice_total_ET = 0
    hd_total=0
    assd_total = 0
    hd_total_WT = 0
    hd_total_TC = 0
    hd_total_ET = 0
    iou_total_WT = 0
    iou_total_TC = 0
    iou_total_ET = 0
    assd_total_WT = 0
    assd_total_TC = 0
    assd_total_ET = 0
    iou_total=0
    noise_dice_total=0
    noise_iou_total = 0
    noise_iou_total_WT = 0
    noise_iou_total_TC = 0
    noise_iou_total_ET = 0
    noise_dice_total_WT = 0
    noise_dice_total_TC = 0
    noise_dice_total_ET = 0
    noise_assd_total_WT = 0
    noise_assd_total_TC = 0
    noise_assd_total_ET = 0
    noise_assd_total = 0
    noise_hd_total = 0
    noise_hd_total_WT = 0
    noise_hd_total_TC = 0
    noise_hd_total_ET = 0
    uncertainty_total = 0
    noised_uncertainty_total = 0
    ece_total = 0
    noise_ece_total = 0
    ece = 0
    noise_ece = 0
    mne = 0
    noised_mne = 0
    mne_total = 0
    noised_mne_total = 0
    ueo_total = 0
    noise_ueo_total = 0
    num = len(test_loader)
    for i, data in enumerate(test_loader):
        print('-------------------------------------------------------------------')
        msg = 'Subject {}/{}, '.format(i + 1, len(test_loader))
        name = str(i)
        if names:
            name = names[i]
            msg += '{:>20}, '.format(name)
        x, noised_x,target = data  # input ground truth

        # choose OOD  condition & level
        # if args.OOD_Level == 'high':
        #     OOD_Variance = 2
        # elif args.OOD_Level == 'upmid':
        #     OOD_Variance = 1.5
        # elif args.OOD_Level == 'mid':
        #     OOD_Variance = 1
        # elif args.OOD_Level == 'low':
        #     OOD_Variance = 0.5
        # else:
        #     OOD_Variance = 0

        # add gaussian noise to input data
        # noise_i = torch.randn_like(x) * OOD_Variance
        # noised_x = x + noise_i

        # noise_m = torch.randn_like(x) * Variance
        # noised_x = x + noise_m

        # np_x = torch.squeeze(x).cpu().numpy()
        # blur_Variance = 20
        # kernel_size = (35, 35)
        # noised_x = cv2.GaussianBlur(np_x, kernel_size, blur_Variance)
        # noised_x = torch.unsqueeze(torch.from_numpy(noised_x),dim=0)
        # noised_x = noised_x.float()

        # np_x = torch.squeeze(x).cpu().numpy()
        # mask_ratio = 0.1
        # window_size = np_x.shape[0],np_x.shape[1],np_x.shape[2]
        # mask,_ = RandomMaskingGenerator(window_size, mask_ratio)
        # noised_x = np_x * mask
        # noised_x = torch.unsqueeze(torch.from_numpy(noised_x), dim=0)
        # noised_x = noised_x.float()

        # if args.dataset == 'BraTS':
        #     num_classes = 4
        #     noised_x[:, 0, ...] = x[:, 0, ...]
        # elif args.dataset == 'ISIC':
        #     num_classes = 2
        if Net_name == 'PU' or Net_name=='ViT':
        # if Net_name == 'PU':
            x = x.cuda()
            noised_x = noised_x.cuda()
            target = target.cuda()
            # x = x.to(device)
            # noised_x = noised_x.to(device)
        else:
            x.cuda()
            noised_x.cuda()

        # all_target.append(target)
        # torch.cuda.synchronize()  # add the code synchronize() to correctly count the runtime.
        # start_time = time.time()

        if args.dataset == 'BraTS':
            num_classes = 4
            if Net_name == 'PU':
                target = target.cuda()
                # target = torch.squeeze(target).cpu().numpy()
                logit = tailor_and_concat_PU(x, model)
                logit_noise = tailor_and_concat_PU(noised_x, model)

            elif Net_name == 'Udrop':
                T_drop = 1
                logit = torch.zeros(1, x.shape[1], x.shape[2], x.shape[3], 155)
                logit_noise = torch.zeros(1, x.shape[1], x.shape[2], x.shape[3], 155)
                torch.cuda.synchronize()  # add the code synchronize() to correctly count the runtime.
                start_time = time.time()
                for j in range(T_drop):
                    print('dropout time:{}'.format(j))
                    logit += tailor_and_concat(x, model)  # 1 4 240 240 155
                    logit_noise += tailor_and_concat(noised_x, model)  # 1 4 240 240 155

                logit = logit / T_drop
                logit_noise = logit_noise / T_drop
                torch.cuda.synchronize()
                elapsed_time = time.time() - start_time
                print('Single sample test time consumption {:.6f} minutes!'.format(elapsed_time / 60))
                logging.info('Single sample test time consumption {:.6f} minutes!'.format(elapsed_time / 60))
            else:
                logit = tailor_and_concat(x, model)
                # torch.cuda.synchronize()
                # elapsed_time = time.time() - start_time
                # logging.info('Single sample test time consumption {:.2f} minutes!'.format(elapsed_time / 60))
                # runtimes.append(elapsed_time)
                logit_noise = tailor_and_concat(noised_x, model)
            # torch.cuda.synchronize()
            # elapsed_time = time.time() - start_time
            # logging.info('Single sample test time consumption {:.6f} minutes!'.format(elapsed_time / 60))
            # runtimes.append(elapsed_time)
            output = F.softmax(logit, dim=1)
            noised_output = F.softmax(logit_noise, dim=1)
            # u
            uncertainty = Uentropy(logit, num_classes)
            noised_uncertainty = Uentropy(logit_noise, num_classes)
            # mne
            mne = Uentropy_our(logit, num_classes)
            noised_mne = Uentropy_our(logit_noise, num_classes)
            # # ece
            # ece = cal_ece(torch.squeeze(logit), torch.squeeze(target).cpu().detach().numpy())
            # noise_ece = cal_ece(torch.squeeze(logit_noise), torch.squeeze(target).cpu().detach().numpy())
        else: # args.dataset == 'ISIC' or args.dataset == 'CHAOS' or args.dataset == 'LiTS'
            num_classes = args.num_classes
            torch.cuda.synchronize()  # add the code synchronize() to correctly count the runtime.
            start_time = time.time()
            if Net_name == 'PU':
                target = target.cuda()
                logit = model_PU(x, model)
                # time
                logit = logit.unsqueeze(0)
                logit_noise = model_PU(noised_x, model)
                logit_noise = logit_noise.unsqueeze(0)
                output = F.softmax(logit, dim=1)
                noised_output = F.softmax(logit_noise, dim=1)
                # u
                uncertainty = Uentropy(logit, num_classes)
                noised_uncertainty = Uentropy(logit_noise, num_classes)
                # ece
                ece = cal_ece(torch.squeeze(logit), torch.squeeze(target).cpu().detach().numpy())
                noise_ece = cal_ece(torch.squeeze(logit_noise), torch.squeeze(target).cpu().detach().numpy())
                # mne
                mne = Uentropy_our(logit, num_classes)
                noised_mne = Uentropy_our(logit_noise, num_classes)
            elif Net_name =='Udrop':
                if args.dataset == 'LiTS':
                    output = torch.zeros(1, num_classes, x.shape[2], x.shape[3], x.shape[4])
                    noised_output = torch.zeros(1, num_classes, x.shape[2], x.shape[3], x.shape[4])
                    uncertainty = torch.zeros(1, x.shape[2], x.shape[3], x.shape[4])
                    noised_uncertainty = torch.zeros(1, x.shape[2], x.shape[3], x.shape[4])
                    ece = torch.zeros(1)
                    noise_ece = torch.zeros(1)
                    for j in range(10):
                        print('Drop time{}'.format(j))
                        logit = model(x)
                        logit_noise = model(noised_x)
                        output += F.softmax(logit, dim=1)
                        noised_output += F.softmax(logit_noise, dim=1)
                        # u
                        uncertainty += Uentropy(logit, num_classes)
                        noised_uncertainty += Uentropy(logit_noise, num_classes)
                        # ece
                        ece += cal_ece(torch.squeeze(logit), torch.squeeze(target))
                        noise_ece += cal_ece(torch.squeeze(logit_noise), torch.squeeze(target))
                        # mne
                        mne += Uentropy_our(logit, num_classes)
                        noised_mne += Uentropy_our(logit_noise, num_classes)
                    torch.cuda.synchronize()
                    elapsed_time = time.time() - start_time
                    print('Single sample test time consumption {:.6f} minutes!'.format(elapsed_time / 60))
                    logging.info('Single sample test time consumption {:.6f} minutes!'.format(elapsed_time / 60))
                else:
                    output = torch.zeros(1, num_classes, x.shape[2], x.shape[3])
                    noised_output = torch.zeros(1, num_classes, x.shape[2], x.shape[3])
                    uncertainty = torch.zeros(1, x.shape[2], x.shape[3])
                    noised_uncertainty = torch.zeros(1, x.shape[2], x.shape[3])
                    ece = torch.zeros(1)
                    noise_ece = torch.zeros(1)
                    for j in range(10):
                        print('Drop time{}'.format(j))
                        logit = model(x)
                        logit_noise = model(noised_x)
                        output += F.softmax(logit, dim=1)
                        noised_output += F.softmax(logit_noise, dim=1)
                        # u
                        uncertainty += Uentropy(logit, num_classes)
                        noised_uncertainty += Uentropy(logit_noise, num_classes)
                        # ece
                        ece += cal_ece(torch.squeeze(logit), torch.squeeze(target))
                        noise_ece += cal_ece(torch.squeeze(logit_noise), torch.squeeze(target))
                        # mne
                        mne += Uentropy_our(logit, num_classes)
                        noised_mne += Uentropy_our(logit_noise, num_classes)
                # time
                # torch.cuda.synchronize()
                # elapsed_time = time.time() - start_time
                # logging.info('Single sample test time consumption {:.2f} minutes!'.format(elapsed_time / 60))
                output = output/10
                noised_output = noised_output/10
                uncertainty = uncertainty / 10
                noised_uncertainty = noised_uncertainty / 10
                ece = ece/10
                noise_ece = noise_ece/10
                mne = mne/10
                noised_mne = noised_mne/10
            else:
                logit = model(x)
                # time
                # torch.cuda.synchronize()
                # elapsed_time = time.time() - start_time
                # logging.info('Single sample test time consumption {:.2f} minutes!'.format(elapsed_time / 60))
                logit_noise = model(noised_x)
                output = F.softmax(logit, dim=1)
                noised_output = F.softmax(logit_noise, dim=1)
                # u
                uncertainty = Uentropy(logit, num_classes)
                noised_uncertainty = Uentropy(logit_noise, num_classes)
                # ece
                ece = cal_ece(torch.squeeze(logit), torch.squeeze(target).cpu().numpy())
                noise_ece = cal_ece(torch.squeeze(logit_noise), torch.squeeze(target).cpu().numpy())
                # mne
                mne = Uentropy_our(logit, num_classes)
                noised_mne = Uentropy_our(logit_noise, num_classes)
            # torch.cuda.synchronize()
            # elapsed_time = time.time() - start_time
            # logging.info('Single sample test time consumption {:.2f} seconds!'.format(elapsed_time))
            # logging.info('Single sample test time consumption {:.2f} minutes!'.format(elapsed_time / 60))
        # uncertainty = Uentropy(logit, num_classes)
        # noised_uncertainty = Uentropy(logit_noise, num_classes)
        # norm_uncertainty = Normalized_U(uncertainty)
        # noised_norm_uncertainty = Normalized_U(noised_uncertainty)
        # # total_uncertainty = torch.sum(uncertainty, -1, keepdim=True)
        # # sum_uncertainty = torch.mean(total_uncertainty)

        # dice hd assd iou
        if args.dataset == 'BraTS':
            Oinput = torch.squeeze(x).cpu().detach().numpy()
            NOinput = torch.squeeze(noised_x).cpu().detach().numpy()
            H, W, T = 240, 240, 155
            output = output[0, :, :H, :W, :T].cpu().detach().numpy()
            output = output.argmax(0)
            target = torch.squeeze(target).cpu().numpy()  # .cpu().numpy(dtype='float32')
            Otarget = target
            iou_res = softmax_mIOU_score(output, target[:, :, :155])
            hd_res = softmax_output_hd(output, target[:, :, :155])
            dice_res = softmax_output_dice(output, target[:, :, :155])
            assd_res = softmax_assd_score(output, target[:, :, :155])
            dice_total_WT += dice_res[0]
            dice_total_TC += dice_res[1]
            dice_total_ET += dice_res[2]
            iou_total_WT += iou_res[0]
            iou_total_TC += iou_res[1]
            iou_total_ET += iou_res[2]
            assd_total_WT += assd_res[0]
            assd_total_TC += assd_res[1]
            assd_total_ET += assd_res[2]
            hd_total_WT += hd_res[0]
            hd_total_TC += hd_res[1]
            hd_total_ET += hd_res[2]
            # for noise_x
            noised_output = noised_output[0, :, :H, :W, :T].cpu().detach().numpy()
            noised_output = noised_output.argmax(0)
            noise_iou_res = softmax_mIOU_score(noised_output, target[:, :, :155])
            noise_hd_res = softmax_output_hd(noised_output, target[:, :, :155])
            noise_dice_res = softmax_output_dice(noised_output, target[:, :, :155])
            noise_assd_res = softmax_assd_score(noised_output, target[:, :, :155])
            noise_dice_total_WT += noise_dice_res[0]
            noise_dice_total_TC += noise_dice_res[1]
            noise_dice_total_ET += noise_dice_res[2]
            noise_iou_total_WT += noise_iou_res[0]
            noise_iou_total_TC += noise_iou_res[1]
            noise_iou_total_ET += noise_iou_res[2]
            noise_assd_total_WT += noise_assd_res[0]
            noise_assd_total_TC += noise_assd_res[1]
            noise_assd_total_ET += noise_assd_res[2]
            noise_hd_total_WT += noise_hd_res[0]
            noise_hd_total_TC += noise_hd_res[1]
            noise_hd_total_ET += noise_hd_res[2]
            mean_uncertainty = torch.mean(uncertainty)
            # # noised_total_uncertainty = torch.sum(noised_uncertainty, -1, keepdim=True)
            # # noised_sum_uncertainty = torch.mean(noised_total_uncertainty)
            noised_mean_uncertainty = torch.mean(noised_uncertainty)
            #
            uncertainty_total += mean_uncertainty  # mix _uncertainty mean_uncertainty mean_uncertainty_succ
            # sum_certainty_total += sum_uncertainty
            noised_uncertainty_total += noised_mean_uncertainty  # noised_mix_uncertainty noised_mean_uncertainty noised_mean_uncertainty_succ
            # noise_sum_certainty_total += noised_sum_uncertainty
            # ece
            ece_total += cal_ece(torch.squeeze(logit), target[:, :, :155])
            noise_ece_total += cal_ece(torch.squeeze(logit_noise), target[:, :, :155])
            # ueo
            thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
            to_evaluate = dict()
            to_evaluate['target'] = target[:, :, :155]
            pred = F.softmax(torch.squeeze(logit), dim=0)
            pc = pred.cpu().detach().numpy()
            pc = pc.argmax(0)
            u = torch.squeeze(uncertainty)
            U = u.cpu().detach().numpy()
            to_evaluate['prediction'] = pc
            to_evaluate['uncertainty'] = U
            UEO = cal_ueo(to_evaluate, thresholds)
            ueo_total += UEO
            noise_to_evaluate = dict()
            noise_to_evaluate['target'] = target[:, :, :155]
            noise_pred = F.softmax(torch.squeeze(logit_noise), dim=0)
            noised_pc = noise_pred.cpu().detach().numpy()
            noised_pc = noised_pc.argmax(0)
            noise_u = torch.squeeze(noised_uncertainty)
            noise_U = noise_u.cpu().detach().numpy()
            noise_to_evaluate['prediction'] = noised_pc
            noise_to_evaluate['uncertainty'] = noise_U
            noise_UEO = cal_ueo(noise_to_evaluate, thresholds)
            print('current_UEO:{};current_noise_UEO:{}; current_num:{}'.format(UEO, noise_UEO, i))
            noise_ueo_total += noise_UEO
            # uncertainty np
            U_output = torch.squeeze(uncertainty).cpu().detach().numpy()
            NU_output = torch.squeeze(noised_uncertainty).cpu().detach().numpy()
            print('current_dice:{},current_noised_dice:{}'.format(dice_res,noise_dice_res))
            print('current_iou:{},current_noised_iou:{}'.format(iou_res, noise_iou_res))
            print('current_hd:{},current_noised_hd:{}'.format(hd_res, noise_hd_res))
            print('current_assd:{},current_noised_assd:{}'.format(assd_res, noise_assd_res))

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
                        os.makedirs(os.path.join(savepath, name))

                    # scipy.misc.imsave(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])
                    imageio.imwrite(os.path.join(savepath, name, str(frame) + '.png'),
                                    Snapshot_img[:, :, :, frame])
                    imageio.imwrite(
                        os.path.join(savepath, name, str(frame) + '_noised.png'),
                        Noise_Snapshot_img[:, :, :, frame])
                    imageio.imwrite(os.path.join(savepath, name, str(frame) + '_gt.png'),
                                    target_img[:, :, :, frame])
                    imageio.imwrite(os.path.join(savepath, name, str(frame) + '_input_FLR.png'),
                                    Oinput[0, :, :, frame])
                    imageio.imwrite(os.path.join(savepath, name, str(frame) + '_input_T1ce.png'),
                                    Oinput[1, :, :, frame])
                    imageio.imwrite(os.path.join(savepath, name, str(frame) + '_input_T1.png'),
                                    Oinput[2, :, :, frame])
                    imageio.imwrite(os.path.join(savepath, name, str(frame) + '_input_T2.png'),
                                    Oinput[3, :, :, frame])
                    imageio.imwrite(os.path.join(savepath, name, str(frame) + '_noised_input_FLR.png'),
                                    NOinput[0, :, :, frame])
                    imageio.imwrite(os.path.join(savepath, name, str(frame) + '_noised_input_T1ce.png'),
                                    NOinput[1, :, :, frame])
                    imageio.imwrite(os.path.join(savepath, name, str(frame) + '_noised_input_T1.png'),
                                    NOinput[2, :, :, frame])
                    imageio.imwrite(os.path.join(savepath, name, str(frame) + '_noised_input_T2.png'),
                                    NOinput[3, :, :, frame])
                    imageio.imwrite(
                        os.path.join(savepath, name, str(frame) + '_uncertainty.png'),
                        U_output[:, :, frame])
                    imageio.imwrite(
                        os.path.join(savepath, name,
                                     str(frame) + '_noised_uncertainty.png'),
                        NU_output[:, :, frame])
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

        elif args.dataset == 'LiTS':
            # dice_total_WT means liver
            # dice_total_TC means tumor
            H, W, T = 256, 256, 16
            output = output[0, :, :, :,  :,].cpu().detach().numpy()
            output = output.argmax(0)
            target = torch.squeeze(target).cpu().numpy()  # .cpu().numpy(dtype='float32')
            Otarget = target
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
            noised_output = noised_output[0, :, :, :,  :,].cpu().detach().numpy()
            noised_output = noised_output.argmax(0)
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
            mean_mne = torch.mean(mne)
            noised_mean_mne = torch.mean(noised_mne)
            mne_total += mean_mne  # mix _uncertainty mean_uncertainty mean_uncertainty_succ
            noised_mne_total += noised_mean_mne
            # ece
            ece_total += ece
            noise_ece_total += noise_ece
            # U ece ueo
            uncertainty_total += mean_uncertainty  # mix _uncertainty mean_uncertainty mean_uncertainty_succ
            # sum_certainty_total += sum_uncertainty
            noised_uncertainty_total += noised_mean_uncertainty  # noised_mix_uncertainty noised_mean_uncertainty noised_mean_uncertainty_succ
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
            print('current_dice:{},current_noised_dice:{}'.format(dice_res,noised_dice_res))
            print('current_iou:{},current_noised_iou:{}'.format(iou_res, noised_iou_res))
            print('current_hd:{},current_noised_hd:{}'.format(hd_res, noised_hd_res))
            print('current_assd:{},current_noised_assd:{}'.format(assd_res, noised_assd_res))
            name = str(i)
            if names:
                name = names[i]
                msg += '{:>20}, '.format(name)
            # uncertainty np
            Oinput = torch.squeeze(x).cpu().detach().numpy()
            NOinput = torch.squeeze(noised_x).cpu().detach().numpy()
            U_output = torch.squeeze(uncertainty).cpu().detach().numpy()
            NU_output = torch.squeeze(noised_uncertainty).cpu().detach().numpy()
            # U_output = torch.squeeze(mne).cpu().detach().numpy()
            # NU_output = torch.squeeze(noised_mne).cpu().detach().numpy()
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
        else:
            H =  x.shape[2]
            W = x.shape[3]
            pc = (output).argmax(1)
            noised_pc = (noised_output).argmax(1)
            predicted = output.argmax(1)
            noised_output = noised_output.argmax(1)

            soft_gt = get_soft_label(target, num_classes)
            soft_predicted = get_soft_label(predicted.unsqueeze(0), num_classes)
            iou_res = IOU_isic(soft_predicted, soft_gt, num_classes)
            dice_res = Dice_isic(soft_predicted, soft_gt, num_classes)
            hd_res = HD_isic(torch.squeeze(predicted).cpu().numpy(), torch.squeeze(target).cpu().numpy(), num_classes)
            assd_res = assd(torch.squeeze(predicted).cpu().numpy(),
                            torch.squeeze(target).cpu().numpy())

            # del data,x,noised_x,noise_m
            noised_hd_res = HD_isic(torch.squeeze(noised_output).cpu().numpy(), torch.squeeze(target).cpu().numpy(),
                                    num_classes)
            noised_assd_res = assd(torch.squeeze(noised_output).cpu().numpy(), torch.squeeze(target).cpu().numpy())
            noised_output = get_soft_label(noised_output.unsqueeze(0), num_classes)
            noised_iou_res = IOU_isic(noised_output, soft_gt, num_classes)
            noised_dice_res = Dice_isic(noised_output, soft_gt, num_classes)

            dice_total += dice_res
            iou_total += iou_res
            hd_total += hd_res
            assd_total += assd_res
            noise_dice_total += noised_dice_res
            noise_iou_total += noised_iou_res
            noise_hd_total += noised_hd_res
            noise_assd_total += noised_assd_res
            # U ece ueo
            mean_uncertainty = torch.mean(uncertainty)
            noised_mean_uncertainty = torch.mean(noised_uncertainty)
            uncertainty_total += mean_uncertainty  # mix _uncertainty mean_uncertainty mean_uncertainty_succ
            noised_uncertainty_total += noised_mean_uncertainty  # noised_mix_uncertainty noised_mean_uncertainty noised_mean_uncertainty_succ
            # ece
            ece_total += ece
            noise_ece_total += noise_ece
            # ueo
            thresholds = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
            to_evaluate = dict()
            to_evaluate['target'] = torch.squeeze(target).cpu().detach().numpy()
            u = torch.squeeze(uncertainty)
            U = u.cpu().detach().numpy()
            to_evaluate['prediction'] = torch.squeeze(pc).cpu().detach().numpy()
            to_evaluate['uncertainty'] = U
            UEO = cal_ueo(to_evaluate, thresholds)
            ueo_total += UEO
            noise_to_evaluate = dict()
            noise_to_evaluate['target'] = target.cpu().detach().numpy()
            noise_u = torch.squeeze(noised_uncertainty)
            noise_U = noise_u.cpu().detach().numpy()
            noise_to_evaluate['prediction'] = torch.squeeze(noised_pc).cpu().detach().numpy()
            noise_to_evaluate['uncertainty'] = noise_U
            noise_UEO = cal_ueo(noise_to_evaluate, thresholds)
            print('current_UEO:{};current_noise_UEO:{}; current_num:{}'.format(UEO,noise_UEO, i))
            noise_ueo_total += noise_UEO
            # uncertainty np
            Otarget = torch.squeeze(target).cpu().detach().numpy()
            Oinput = torch.squeeze(x,0).permute(1,2,0).cpu().detach().numpy()
            NOinput = torch.squeeze(noised_x,0).permute(1,2,0).cpu().detach().numpy()

            U_output = torch.squeeze(uncertainty).cpu().detach().numpy()
            NU_output = torch.squeeze(noised_uncertainty).cpu().detach().numpy()
            # U_output = torch.squeeze(mne).cpu().detach().numpy()
            # NU_output = torch.squeeze(noised_mne).cpu().detach().numpy()
            print('current_dice:{},current_noised_dice:{}'.format(dice_res,noised_dice_res))
            print('current_iou:{},current_noised_iou:{}'.format(iou_res, noised_iou_res))
            print('current_hd:{},current_noised_hd:{}'.format(hd_res, noised_hd_res))
            print('current_assd:{},current_noised_assd:{}'.format(assd_res, noised_assd_res))
            name = str(i)
            if names:
                name = names[i]
                msg += '{:>20}, '.format(name)
            name = name.replace('.npy', '')
            if snapshot:
                """ --- grey figure---"""
                # Snapshot_img = np.zeros(shape=(H,W,T),dtype=np.uint8)
                # Snapshot_img[np.where(output[1,:,:,:]==1)] = 64
                # Snapshot_img[np.where(output[2,:,:,:]==1)] = 160
                # Snapshot_img[np.where(output[3,:,:,:]==1)] = 255
                """ --- colorful figure--- """
                Snapshot_img = np.zeros(shape=(H, W), dtype=np.float32)
                Snapshot_img[np.where((torch.squeeze(pc).cpu().numpy()) == 1)] = 255

                Noise_Snapshot_img = np.zeros(shape=(H, W), dtype=np.float32)
                Noise_Snapshot_img[np.where((torch.squeeze(noised_pc).cpu().numpy()) == 1)] = 255

                target_img = np.zeros(shape=(H, W), dtype=np.float32)
                target_img[np.where(Otarget == 1)] = 255

                if not os.path.exists(os.path.join(savepath, name)):
                    os.makedirs(os.path.join(savepath, name))

                    # scipy.misc.imsave(os.path.join(visual, name, str(frame)+'.png'), Snapshot_img[:, :, :, frame])
                imageio.imwrite(os.path.join(savepath, name, '_pred.png'),
                                Snapshot_img)
                imageio.imwrite(os.path.join(savepath, name, '_noised.png'),
                                Noise_Snapshot_img)
                imageio.imwrite(os.path.join(savepath, name, '_gt.png'),
                                target_img)
                imageio.imwrite(os.path.join(savepath, name, '_uncertainty.png'),
                                U_output)
                imageio.imwrite(
                    os.path.join(savepath, name, '_noised_uncertainty.png'),
                    NU_output)
                U_img = cv2.imread(os.path.join(savepath, name, '_uncertainty.png'))
                U_heatmap = cv2.applyColorMap(U_img, cv2.COLORMAP_JET)
                cv2.imwrite(
                    os.path.join(savepath, name, '_colormap_uncertainty.png'),
                    U_heatmap)
                NU_img = cv2.imread(
                    os.path.join(savepath, name, '_noised_uncertainty.png'))
                NU_heatmap = cv2.applyColorMap(NU_img, cv2.COLORMAP_JET)
                cv2.imwrite(
                    os.path.join(savepath, name, '_colormap_noised_uncertainty.png'),
                    NU_heatmap)
                imageio.imwrite(os.path.join(savepath, name, '_input.png'),
                                Oinput)
                imageio.imwrite(os.path.join(savepath, name, '_noised_input.png'),
                                NOinput)

        print(msg)


    if args.dataset == 'BraTS':
        aver_certainty = uncertainty_total / num
        aver_noise_certainty = noised_uncertainty_total / num
        aver_mne = mne_total / num
        aver_noise_mne = noised_mne_total / num
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
        aver_certainty = uncertainty_total  / num
        aver_noise_certainty = noised_uncertainty_total / num
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
        logging.info('aver_ece=%f,aver_noise_ece = %f' % (aver_ece, aver_noise_ece))
        logging.info('aver_ueo=%f,aver_noise_ueo = %f' % (aver_ueo, aver_noise_ueo))
        return [aver_dice_WT, aver_dice_TC], [aver_noise_dice_WT, aver_noise_dice_TC], [
                   aver_hd_WT, aver_hd_TC], [aver_noise_hd_WT, aver_noise_hd_TC], [
                   aver_assd_WT, aver_assd_TC], [aver_noise_assd_WT, aver_noise_assd_TC]
    else:
        aver_certainty = uncertainty_total  / num
        aver_noise_certainty = noised_uncertainty_total / num
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
        logging.info('aver_ece=%f,aver_noise_ece = %f' % (aver_ece, aver_noise_ece))
        logging.info('aver_ueo=%f,aver_noise_ueo = %f' % (aver_ueo, aver_noise_ueo))
        return aver_dice, aver_noise_dice, aver_hd, aver_noise_hd, aver_assd, aver_noise_assd
        # return aver_dice, aver_noise_dice, aver_iou, aver_noise_iou