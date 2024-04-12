import os
import torch
import torch.nn as nn
from tqdm import tqdm
import json
import gc
import numpy as np
from torch.utils.data import DataLoader
import argparse

from src.utils import setup_seed
from src.pixel_classifier import load_ensemble, save_predictions, compute_dice, save_predictions_gts, compute_hd_95, compute_aji, compute_obj_hd, compute_obj_dice
from src.datasets import make_transform, ImageLabelPNGDataset
from src.utils import oht_to_scalar

from guided_diffusion.guided_diffusion.dist_util import dev

from src.models.unet import UNet
from src.models.deeplab import DeepLab
from src.models.uctransnet.UCTransNet import UCTransNet
from src.models.uctransnet import Config
from src.models.medt.axialnet import MedT
from monai.networks.nets import SwinUNETR, BasicUNetPlusPlus, AttentionUnet

from monai.inferers import sliding_window_inference
from monai import data


def evaluation(args, model):
    val_ds = ImageLabelPNGDataset(
        data_dir=args['testing_path'],
        mode='test',
        model_type=args['model_type'],
        resolution=args['image_size'],
        num_images=args['testing_number'],
        transform=make_transform(
            args['model_type'],
            (args['image_size'],args['image_size'])
        ),
        category=args['category']
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

    preds, gts = [], []
    for idx, batch_data in enumerate(tqdm(val_loader)):
        img, label = batch_data[0], batch_data[1]
        img = img.to(dev())

        model.eval()

        with torch.no_grad():
            pred = sliding_window_inference(inputs=img, predictor=model.to(dev()), sw_device='cuda',
                                            device='cpu', roi_size=args['dim'][:-1], sw_batch_size=1, overlap=0.2, mode="gaussian")
            # print(pred.shape) # (b, c, h, w)
            if args['model_type'] == 'BasicUNetPlusPlus':
                pred = pred[0]  # monai unet++ returns a list
            img_seg = oht_to_scalar(pred)
            pred = img_seg.cpu().detach()
            label = torch.squeeze(label, 1)
            gts.append(label.numpy())
            preds.append(pred.numpy())

    if len(args['dim']) == 3:
        save_predictions_gts(args, val_ds.image_paths, preds, gts)  # not supported for 3d yet

    if args['category'] == 'glas_1':
        obj_dice = compute_obj_dice(preds, gts)
        obj_hd = compute_obj_hd(preds, gts)
        print(f'Overall obj dice: ', obj_dice)
        print(f'Overall obj hd: ', obj_hd)
        return obj_dice, obj_hd
    elif args['category'] == 'monuseg_1':
        aji = compute_aji(preds, gts)
        dice = compute_dice(preds, gts)
        print(f'Overall dice: ', dice)
        print(f'Overall aji: ', aji)
        return dice, aji


# Adopted from https://github.com/nv-tlabs/datasetGAN_release/blob/d9564d4d2f338eaad78132192b865b6cc1e26cac/datasetGAN/train_interpreter.py#L434
def train_batch(args):
    dataset = ImageLabelPNGDataset(
        data_dir=args['training_path'],
        mode='train',
        model_type=args['model_type'],
        resolution=args['image_size'],
        num_images=args['training_number'],
        transform=make_transform(
            args['model_type'],
            (args['image_size'],args['image_size'])
        ),
        category=args['category']
    )
    train_loader = DataLoader(dataset=dataset, batch_size=args['batch_size'], shuffle=True, drop_last=False)
    criterion = nn.CrossEntropyLoss()

    obj_dice_all, obj_hd_all, dice_all, aji_all = [], [], [], []

    print(" *********************** Current dataloader length " + str(len(train_loader)) + " ***********************")
    for MODEL_NUMBER in range(args['start_model_num'], args['model_num'], 1):
        gc.collect()
        if args['model_type'] == 'DeepLab':
            net = DeepLab(num_classes=args['number_class'],
                          backbone="resnet",
                          output_stride=16,
                          sync_bn=False,
                          freeze_bn=False)
        elif args['model_type'] == 'SwinUNETR': # batch size 8
            net = SwinUNETR(img_size=(args['dim'][:-1]), in_channels=args['in_channels'],
                            out_channels=args['number_class'], feature_size=24, use_checkpoint=True, spatial_dims=args['dims'])
        elif args['model_type'] == 'UNet':
            net = UNet(n_channels=args['in_channels'], n_classes=(args['number_class']))
        elif args['model_type'] == 'UCTransNet':  # batch size 8
            config_vit = Config.get_CTranS_config()
            net = UCTransNet(config_vit, n_channels=args['in_channels'], n_classes=args['number_class'],
                             img_size=args['image_size'])
        elif args['model_type'] == 'BasicUNetPlusPlus':  # batch size 8
            net = BasicUNetPlusPlus(spatial_dims=2, in_channels=args['in_channels'], out_channels=args['number_class'])
        elif args['model_type'] == 'AttentionUnet':  # batch size 8
            net = AttentionUnet(spatial_dims=2, in_channels=args['in_channels'], out_channels=args['number_class'], channels=[64,128,256,512,1024], strides=[2,2,2,2,2])
        elif args['model_type'] == 'MedT':  # batch size 4
            net = MedT(img_size=args['image_size'], imgchan=args['in_channels'])
        else:
            raise Exception(f"Wrong model type: {args['model_type']}")
        
        net = nn.DataParallel(net).cuda()

        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        net.train()

        iteration = 0
        break_count = 0
        best_loss = 10000000
        stop_sign = 0

        for epoch in range(5000):
            for x_batch, y_batch in train_loader:
                net.train()
                x_batch, y_batch = x_batch.to(dev()), y_batch.to(dev())
                y_batch = y_batch.type(torch.long)
                optimizer.zero_grad()
                y_pred = net(x_batch)
                if args['model_type'] == 'BasicUNetPlusPlus':
                    y_pred = y_pred[0]  # monai unet++ returns a list
                y_batch = torch.squeeze(y_batch, 1)
                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

                iteration += 1
                if iteration % (len(train_loader)) == 0 and iteration > args['start_epoch']:
                    print('Epoch : ', str(epoch), 'iteration', iteration, 'loss', loss.item())
                    if loss.item() < best_loss:
                        best_loss = loss.item()
                        break_count = 0
                    else:
                        break_count += 1

                    if break_count > args['max_break']:
                        stop_sign = 1
                        print("*************** Break, Total iters,", iteration, ", at epoch", str(epoch),
                              "***************")
                        break

            if stop_sign == 1:
                break

        model_path = os.path.join(args['exp_dir'],
                                  'best_model_' + str(MODEL_NUMBER) + '.pth')
        MODEL_NUMBER += 1
        print('save to:', model_path)
        torch.save({'model_state_dict': net.state_dict()},
                   model_path)

        if args['category'] == 'glas_1':
            obj_dice, obj_hd = evaluation(args, net)
            obj_dice_all.append(obj_dice)
            obj_hd_all.append(obj_hd)
        elif args['category'] == 'monuseg_1':
            dice, aji = evaluation(args, net)
            dice_all.append(dice)
            aji_all.append(aji)

    if args['category'] == 'glas_1':
        print('Obj Dice Mean: ', np.nanmean(obj_dice_all))
        print('Obj Dice Std: ', np.nanstd(obj_dice_all))
        print('Obj HD Mean: ', np.nanmean(obj_hd_all))
        print('Obj HD Std: ', np.nanstd(obj_hd_all))
    elif args['category'] == 'monuseg_1':
        print('Dice Mean: ', np.nanmean(dice_all))
        print('Dice Std: ', np.nanstd(dice_all))
        print('Aji Mean: ', np.nanmean(aji_all))
        print('Aji Std: ', np.nanstd(aji_all))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default="/afs/crc.nd.edu/user/z/zpan3/Models/ddpm-segmentation/experiments/monuseg_1/unet.json", type=str)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    setup_seed(args.seed)

    # Load the experiment config
    opts = json.load(open(args.exp, 'r'))
    opts.update(vars(args))
    opts['image_size'] = opts['dim'][0]

    path = opts['exp_dir']
    os.makedirs(path, exist_ok=True)
    print('Experiment folder: %s' % (path))
    os.system('cp %s %s' % (args.exp, opts['exp_dir']))

    # Check whether all models in ensemble are trained
    pretrained = [os.path.exists(os.path.join(opts['exp_dir'], f'best_model_{i}.pth'))
                  for i in range(opts['model_num'])]

    if not all(pretrained):
        # train all remaining models
        opts['start_model_num'] = sum(pretrained)
        train_batch(opts)

    # Evaluate all models and report mean and std
    print('Loading pretrained models...')
    models = load_ensemble(opts, device='cuda', seg_type=opts['model_type'], best_epoch=True)
    obj_dice_all, obj_hd_all, dice_all, aji_all = [], [], [], []
    for i in range(len(models)):
        if opts['category'] == 'glas_1':
            obj_dice, obj_hd = evaluation(opts, models[i])
            obj_dice_all.append(obj_dice)
            obj_hd_all.append(obj_hd)
        elif opts['category'] == 'monuseg_1':
            dice, aji = evaluation(opts, models[i])
            dice_all.append(dice)
            aji_all.append(aji)
    if opts['category'] == 'glas_1':
        print('Obj Dice Mean: ', np.nanmean(obj_dice_all))
        print('Obj Dice Std: ', np.nanstd(obj_dice_all))
        print('Obj HD Mean: ', np.nanmean(obj_hd_all))
        print('Obj HD Std: ', np.nanstd(obj_hd_all))
    elif opts['category'] == 'monuseg_1':
        print('Dice Mean: ', np.nanmean(dice_all))
        print('Dice Std: ', np.nanstd(dice_all))
        print('Aji Mean: ', np.nanmean(aji_all))
        print('Aji Std: ', np.nanstd(aji_all))
