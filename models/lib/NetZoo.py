import torch
from models.lib.VNet3D import VNet
from models.lib.UNet3DZoo import Unet,AttUnet

__all__ = ['VNet3D', 'Unet', 'AttUnet']

def Vnet_t12(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s lib.
    Args:
        pretrained (bool): If True, returns a lib pre-trained on ImageNet
    """
    model = VNet(n_channels=2, n_classes=4, n_filters=16, normalization='gn', has_dropout=False)
    model.cuda()
    model.eval()
    if pretrained:
        model_state = torch.load('C:/Coco_file/TMS-main/checkpoint/TransBTS2022-01-08/V_both_epoch_199.pth')
        model.load_state_dict(model_state['state_dict'])
        # lib.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s']))
    return model

def Vnet_t2(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s lib.
    Args:
        pretrained (bool): If True, returns a lib pre-trained on ImageNet
    """
    model = VNet(n_channels=1, n_classes=4, n_filters=16, normalization='gn', has_dropout=False)
    model.cuda()
    model.eval()
    if pretrained:
        model_state = torch.load('C:/Coco_file/TMS-main/checkpoint/TransBTS2021-12-31/V_t2_epoch_198.pth')
        model.load_state_dict(model_state['state_dict'])
        # lib.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s']))
    return model

def Vnet_t1(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s lib.
    Args:
        pretrained (bool): If True, returns a lib pre-trained on ImageNet
    """
    model = VNet(n_channels=1, n_classes=4, n_filters=16, normalization='gn', has_dropout=False)
    model.cuda()
    model.eval()
    if pretrained:
        model_state = torch.load('C:/Coco_file/TMS-main/checkpoint/TransBTS2021-12-31/V_t1_epoch_198.pth')
        model.load_state_dict(model_state['state_dict'])
        # lib.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s']))
    return model

def Unet_t12(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s lib.
    Args:
        pretrained (bool): If True, returns a lib pre-trained on ImageNet
    """
    model = Unet(in_channels=2, base_channels=16, num_classes=4)
    model.cuda()
    if pretrained:
        model_state = torch.load('C:/Coco_file/SmmNet-master/SmmNet-master/res2net50_v1b_26w_4s-3cf99910.pth')
        model.load_state_dict(model_state['state_dict'])
        # lib.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s']))
    return model

def AUnet_t12(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s lib.
    Args:
        pretrained (bool): If True, returns a lib pre-trained on ImageNet
    """
    model = AttUnet(in_channels=2, base_channels=16, num_classes=4)
    model.cuda()
    if pretrained:
        model_state = torch.load('C:/Coco_file/TMS-main2/checkpoint/TransBTS2022-01-08/AU_both_epoch_44.pth')
        model.load_state_dict(model_state['state_dict'])
        # lib.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s']))
    return model

def AUnet_t1(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s lib.
    Args:
        pretrained (bool): If True, returns a lib pre-trained on ImageNet
    """
    model = AttUnet(in_channels=1, base_channels=16, num_classes=4)
    model.cuda()
    if pretrained:
        model_state = torch.load('C:/Coco_file/TMS-main/checkpoint/TransBTS2021-12-31/AU_t1_epoch_199.pth')
        model.load_state_dict(model_state['state_dict'])
        # lib.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s']))
    return model

def AUnet_t2(pretrained=False, **kwargs):
    """Constructs a Res2Net-50_v1b_26w_4s lib.
    Args:
        pretrained (bool): If True, returns a lib pre-trained on ImageNet
    """
    model = AttUnet(in_channels=1, base_channels=16, num_classes=4)
    model.cuda()
    if pretrained:
        model_state = torch.load('C:/Coco_file/TMS-main/checkpoint/TransBTS2021-12-31/AU_t2_epoch_199.pth')
        model.load_state_dict(model_state['state_dict'])
        # lib.load_state_dict(model_zoo.load_url(model_urls['res2net50_v1b_26w_4s']))
    return model
if __name__ == '__main__':
    images = torch.rand(1, 1, 240, 240, 160).cuda(0)
    model = Vnet_t12(pretrained=True)
    model = model.cuda(0)
    print(model(images).size())
