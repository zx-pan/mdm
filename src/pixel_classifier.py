import os
import torch
import torch.nn as nn
import numpy as np
from collections import Counter
from torch.distributions import Categorical
from src.utils import colorize_mask, oht_to_scalar
from src.data_util import get_palette, get_class_names
from PIL import Image

from src.models.unet import UNet
from src.models.deeplab import DeepLab
from src.models.uctransnet.UCTransNet import UCTransNet
from src.models.uctransnet import Config
from src.models.medt.axialnet import MedT
from monai.networks.nets import SwinUNETR, BasicUNetPlusPlus, AttentionUnet

from src.metrics import dice, hausdorff_distance_95, ObjDiceLoss, ObjHausLoss, get_fast_aji, fscore


# Adopted from https://github.com/nv-tlabs/datasetGAN_release/blob/d9564d4d2f338eaad78132192b865b6cc1e26cac/datasetGAN/train_interpreter.py#L68
class pixel_classifier(nn.Module):
    def __init__(self, numpy_class, dim):
        super(pixel_classifier, self).__init__()
        if numpy_class < 30:
            self.layers = nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=32),
                nn.Linear(32, numpy_class)
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, numpy_class)
            )

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, x):
        return self.layers(x)


def predict_labels(models, features, size):
    if isinstance(features, np.ndarray):
        features = torch.from_numpy(features)
    
    mean_seg = None
    all_seg = []
    all_entropy = []
    seg_mode_ensemble = []

    softmax_f = nn.Softmax(dim=1)
    with torch.no_grad():
        for MODEL_NUMBER in range(len(models)):
            preds = models[MODEL_NUMBER](features.cuda())
            entropy = Categorical(logits=preds).entropy()
            all_entropy.append(entropy)
            all_seg.append(preds)

            if mean_seg is None:
                mean_seg = softmax_f(preds)
            else:
                mean_seg += softmax_f(preds)

            img_seg = oht_to_scalar(preds)
            img_seg = img_seg.reshape(*size)
            img_seg = img_seg.cpu().detach()

            seg_mode_ensemble.append(img_seg)

        mean_seg = mean_seg / len(all_seg)

        full_entropy = Categorical(mean_seg).entropy()

        js = full_entropy - torch.mean(torch.stack(all_entropy), 0)
        top_k = js.sort()[0][- int(js.shape[0] / 10):].mean()

        img_seg_final = torch.stack(seg_mode_ensemble, dim=-1)
        img_seg_final = torch.mode(img_seg_final, 2)[0]
    return img_seg_final, top_k


def save_predictions(args, image_paths, preds):
    palette = get_palette(args['category'])
    os.makedirs(os.path.join(args['exp_dir'], 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(args['exp_dir'], 'visualizations'), exist_ok=True)

    for i, pred in enumerate(preds):
        filename = image_paths[i]['seg'].split('/')[-1].split('.')[0]
        pred = np.squeeze(pred)
        np.save(os.path.join(args['exp_dir'], 'predictions', filename + '.npy'), pred)

        mask = colorize_mask(pred, palette)
        Image.fromarray(mask).save(
            os.path.join(args['exp_dir'], 'visualizations', filename + '.png')
        )


def save_predictions_gts(args, image_paths, preds, gts):
    palette = get_palette(args['category'])
    # os.makedirs(os.path.join(args['exp_dir'], 'predictions'), exist_ok=True)
    os.makedirs(os.path.join(args['exp_dir'], 'visualizations'), exist_ok=True)

    for i, pred in enumerate(preds):
        if args['category'] == "brats_1":
            filename = image_paths[i]['seg'].split('/')[-1].split('.')[0]  # brats
        else:
            filename = image_paths[i].split('/')[-1].split('.')[0]  # bedroom28
        pred, gt = np.squeeze(pred), np.squeeze(gts[i])

        mask = colorize_mask(pred, palette)
        # gt = colorize_mask(gt, palette)
        Image.fromarray(mask).save(
            os.path.join(args['exp_dir'], 'visualizations', filename + '.png')
        )
        # Image.fromarray(gt).save(
        #     os.path.join(args['exp_dir'], 'visualizations', filename + '_gt.png')
        # )


def compute_f1(preds, gts):
    """""
    Compute F1 Score
    """""
    f1_all = []
    for pred, gt in zip(preds, gts):
        f1_all.append(fscore(pred.astype(np.float32), gt.astype(np.float32)))

    return np.array(f1_all).mean()


def compute_aji(preds, gts):
    """""
    Compute AJI distributed by MoNuSeg
    """""
    aji_all = []
    for pred, gt in zip(preds, gts):
        aji_all.append(get_fast_aji(pred.astype(np.int), gt.astype(np.int)))

    return np.array(aji_all).mean()


def compute_obj_hd(preds, gts):
    """""
    Compute object-level Hausdorff distance
    """""
    obj_hd_all = []
    cal_obj_hd = ObjHausLoss()
    for pred, gt in zip(preds, gts):
        obj_hd_all.append(cal_obj_hd(torch.from_numpy(pred), torch.from_numpy(gt)))

    return np.array(obj_hd_all).mean()


def compute_obj_dice(preds, gts):
    """""
    Compute object-level dice
    """""
    obj_dice_all = []
    cal_obj_dice = ObjDiceLoss()
    for pred, gt in zip(preds, gts):
        obj_dice_all.append(cal_obj_dice(torch.from_numpy(pred), torch.from_numpy(gt)))

    return np.array(obj_dice_all).mean()


def compute_dice(preds, gts):
    """""
    Compute dice
    """""
    dice_all = []
    for pred, gt in zip(preds, gts):
        dice_all.append(dice(pred.astype(np.float32), gt.astype(np.float32)))

    return np.array(dice_all).mean()


def compute_hd_95(preds, gts):
    """""
    Compute hd 95
    """""
    hd_95 = []
    for pred, gt in zip(preds, gts):
        hd_95.append(hausdorff_distance_95(pred.astype(np.float32), gt.astype(np.float32)))

    return np.array(hd_95).mean()


def load_ensemble(args, device='cpu', seg_type='ddpm', best_epoch=False):
    models = []
    for i in range(args['model_num']):
        if best_epoch:
            model_path = os.path.join(args['exp_dir'], f'best_model_{i}.pth')
        else:
            model_path = os.path.join(args['exp_dir'], f'model_{i}.pth')
        state_dict = torch.load(model_path)['model_state_dict']

        if seg_type in ['AttentionUnet', 'BasicUNetPlusPlus', 'MedT']:
            # solve 'module' in key problem
            weights_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace('module.', '') if 'module' in k else k
                weights_dict[new_k] = v
            state_dict = weights_dict

        if seg_type == 'AttentionUnet':
            weights_dict = {}
            for k, v in state_dict.items():
                new_k = k.replace('sub', 'submodule.') if 'sub' in k else k
                weights_dict[new_k] = v
            state_dict = weights_dict

        if seg_type == 'ddpm':
            model = nn.DataParallel(pixel_classifier(args["number_class"], len(args['model_type'])*args['dim'][-1]))
        elif seg_type == 'UNet':
            model = nn.DataParallel(UNet(args['in_channels'], args["number_class"]))
        elif seg_type == 'DeepLab':
            model = nn.DataParallel(DeepLab(num_classes=args['number_class'],
                        backbone="resnet",
                        output_stride=16,
                        sync_bn=False,
                        freeze_bn=False))
        elif seg_type == 'SwinUNETR':
            model = nn.DataParallel(SwinUNETR(img_size=(args['dim'][:-1]), in_channels=args['in_channels'], out_channels=args['number_class'], feature_size=24, use_checkpoint=True, spatial_dims=args['dims']))
        elif seg_type == 'UCTransNet':
            config_vit = Config.get_CTranS_config()
            model = nn.DataParallel(UCTransNet(config_vit, n_channels=args['in_channels'], n_classes=args['number_class'], img_size=args['image_size']))
        elif seg_type == 'BasicUNetPlusPlus':
            model = BasicUNetPlusPlus(spatial_dims=2, in_channels=args['in_channels'], out_channels=args['number_class'])
        elif seg_type == 'AttentionUnet':
            model = AttentionUnet(spatial_dims=2, in_channels=args['in_channels'], out_channels=args['number_class'], channels=[64,128,256,512,1024], strides=[2,2,2,2,2])
        elif seg_type == 'MedT':
            model = MedT(img_size=args['image_size'], imgchan=args['in_channels'])
        model.load_state_dict(state_dict)
        try:
            model = model.module.to(device)
        except:
            model = model.to(device)
        models.append(model.eval())
    return models
