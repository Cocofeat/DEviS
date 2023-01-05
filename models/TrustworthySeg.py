import torch
import argparse
import os
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from models.criterions import KL,ce_loss,mse_loss,dce_evidence_loss,dce_evidence_u_loss
from models.lib.NetZoo import Vnet_t12,Unet_t12,AUnet_t12,AUnet_t1,AUnet_t2,Vnet_t1,Vnet_t2
from torch.autograd import Variable
from predict import tailor_and_concat
from models.lib.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from models.lib.VNet3D import VNet
from models.lib.VNet2D import VNet2D
from models.lib.UNet3DZoo import Unet,AttUnet
from models.lib.UNet2DZoo import Unet2D,AttUnet2D,resnet34_unet,Unet2DDRIVE
# from models.lib.TransU_zoo import Transformer_U
# from models.lib.vit_seg_modeling import ViT
# from sklearn.preprocessing import MinMaxScaler

class TMSU(nn.Module):

    def __init__(self, args):
        """
        :param classes: Number of classification categories
        :param modes: Number of modes
        :param classifier_dims: Dimension of the classifier
        :param annealing_epoch: KL divergence annealing epoch during training
        """
        super(TMSU, self).__init__()
        # ---- Net Backbone ----
        num_classes = args.num_classes
        dataset = args.dataset
        # modes = args.modes
        model_name = args.model_name
        input_modality = args.input_modality
        total_epochs = args.end_epochs
        lambda_epochs = args.lambda_epochs
        if dataset == 'BraTS':
            if model_name == 'AU' and input_modality == 'four':
                self.backbone = AttUnet(in_channels=4, base_channels=16, num_classes=num_classes)
            elif model_name == 'V' and input_modality == 'four':
                self.backbone = VNet(n_channels=4, n_classes=num_classes, n_filters=16, normalization='gn',
                                     has_dropout=False)
            elif model_name == 'U' and input_modality == 'four':
                self.backbone = Unet(in_channels=4, base_channels=16, num_classes=num_classes)
            else:
                print('There is no this model')
                raise NameError
        elif dataset == 'ISIC':
            if model == 'AU' :
                self.backbone = AttUnet2D(3, classes)
            elif model == 'V':
                self.backbone = VNet2D(n_channels=3, n_classes=classes, n_filters=32, normalization='gn',
                                     has_dropout=False)
            elif model == 'ResU':
                self.backbone = resnet34_unet(classes, pretrained=False)
            elif model == 'U':
                self.backbone = Unet2D(3, classes)
            else:
                print('There is no this model')
                raise NameError
        elif dataset == 'LiTS':
            if model_name == 'AU' :
                self.backbone = AttUnet(in_channels=1, base_channels=16, num_classes=num_classes)
            elif model_name == 'V':
                self.backbone = VNet(n_channels=1, n_classes=num_classes, n_filters=16, normalization='gn',
                                     has_dropout=False)
            elif model_name == 'U':
                self.backbone = Unet(in_channels=1, base_channels=16, num_classes=num_classes)
            else:
                print('There is no this model')
                raise NameError
        elif dataset == 'DRIVE':
            if model_name == 'AU' :
                self.backbone = AttUnet2D(3, num_classes)
            elif model_name == 'V':
                self.backbone = VNet2D(n_channels=3, n_classes=num_classes, n_filters=32, normalization='gn',
                                     has_dropout=False)
            elif model_name == 'ResU':
                self.backbone = resnet34_unet(num_classes, pretrained=False)
            elif model_name == 'U':
                self.backbone = Unet2DDRIVE(3, num_classes)
            else:
                print('There is no this model')
                raise NameError
        elif dataset == 'TOAR':
            if model_name == 'AU' :
                self.backbone = AttUnet2D(3, num_classes)
            elif model_name == 'V':
                self.backbone = VNet2D(n_channels=3, n_classes=num_classes, n_filters=32, normalization='gn',
                                     has_dropout=False)
            elif model_name == 'ResU':
                self.backbone = resnet34_unet(num_classes, pretrained=False)
            elif model_name == 'U':
                self.backbone = Unet2D(1, num_classes)
            else:
                print('There is no this model')
                raise NameError
        elif dataset == 'HC':
            if model_name == 'AU' :
                self.backbone = AttUnet2D(3, num_classes)
            elif model_name == 'V':
                self.backbone = VNet2D(n_channels=3, n_classes=num_classes, n_filters=32, normalization='gn',
                                     has_dropout=False)
            elif model_name == 'ResU':
                self.backbone = resnet34_unet(num_classes, pretrained=False)
            elif model_name == 'U':
                self.backbone = Unet2D(1, num_classes)
            else:
                print('There is no this model')
                raise NameError
        else:
            print('There is no this dataset')
            raise NameError
        self.backbone.cuda()
        self.classes = num_classes
        self.disentangle = False
        self.eps = 1e-10
        self.lambda_epochs = lambda_epochs
        self.total_epochs = total_epochs + 1
        self.u_loss = args.Uncertainty_Loss

    def forward(self, X, y, global_step, mode,dataset):
        # X data
        # y target
        # global_step : epochs

        # step zero: backbone
        if mode == 'train':
            backbone_output = self.backbone(X)
        elif mode == 'val':
            if dataset == 'BraTS':
                backbone_output = tailor_and_concat(X, self.backbone)
            else:
                backbone_output = self.backbone(X)
        else:
            if dataset == 'BraTS':
                backbone_output = tailor_and_concat(X, self.backbone)
            else:
                backbone_output = self.backbone(X)

        # step one
        evidence = self.infer(backbone_output) # batch_size * class * image_size
        backbone_pred = F.softmax(backbone_output,1)  # batch_size * class * image_size

        # step two
        alpha = evidence + 1
        if mode == 'train':

            if self.u_loss:
                loss = dce_evidence_u_loss(y.to(torch.int64), alpha, self.classes, global_step, self.lambda_epochs,self.total_epochs,self.eps,self.disentangle,evidence,backbone_pred)
            else:
                loss = dce_evidence_loss(y.to(torch.int64), alpha, self.classes, global_step, self.lambda_epochs,self.total_epochs,self.eps,self.disentangle,evidence,backbone_pred)
            loss = torch.mean(loss)
            return evidence, loss
        else:
            return evidence

    def infer(self, input):
        """
        :param input: modal data
        :return: evidence of modal data
        """
        evidence = F.softplus(input)
        return evidence

