import os
import torch
import torch.nn as nn
from tqdm import tqdm
import json
import gc

from torch.utils.data import DataLoader

import argparse
from src.utils import setup_seed
from src.pixel_classifier import load_ensemble, save_predictions, compute_dice, save_predictions_gts, compute_hd_95, pixel_classifier, \
    compute_aji, compute_obj_hd, compute_obj_dice
from src.datasets import make_transform, ImageLabelPNGDataset, FeatureDataset
from src.feature_extractors import create_feature_extractor, collect_features

# from guided_diffusion.guided_diffusion.script_util import model_and_diffusion_defaults, add_dict_to_argparser
# from guided_diffusion.guided_diffusion.dist_util import dev

from mask_diffusion.guided_diffusion.script_util import model_and_diffusion_defaults as mask_defaults, add_dict_to_argparser
from mask_diffusion.guided_diffusion.dist_util import dev


import numpy as np
from monai.inferers import sliding_window_inference
from src.utils import oht_to_scalar
from monai import data


def prepare_batch_data(args, x_batch, y_batch, feature_extractors):

    dataset = FeatureDataset(x_batch, y_batch)

    # print(f"Preparing the train batch for {args['category']}...")

    X = torch.zeros((len(dataset), *[args['dim'][-1]*len(feature_extractors), *args['dim'][:-1]]), dtype=torch.float)
    y = torch.zeros((len(dataset), *args['dim'][:-1]), dtype=torch.uint8)

    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=dev()).manual_seed(args['seed'])
        noise = torch.randn(1, args['in_channels'], *args['dim'][:-1],
                            generator=rnd_gen, device=dev())
    else:
        noise = None

    for row, (img, label) in enumerate(tqdm(dataset)):
        img = img[None].to(dev())

        fuse_features = torch.zeros((args['dim'][-1] * len(feature_extractors), *args['dim'][:-1]), dtype=torch.float)
        for i, feature_extractor in enumerate(feature_extractors):
            features = feature_extractor(img, noise=noise)
            features = collect_features(args, features)
            features = features/torch.norm(features)
            fuse_features[i*args['dim'][-1]:(i+1)*args['dim'][-1]] = features

        # fuse_features = torch.cat(fuse_features, dim=0)

        X[row] = fuse_features.cpu()
        y[row] = label

    d = X.shape[1]
    # print(f'Total dimension {d}')
    # 2d 0/1 glas
    X = X.permute(1, 0, 2, 3).reshape(d, -1).permute(1, 0)
    y = y.flatten()

    return X, y


class seg_inference(nn.Module):
    def __init__(self, feature_extractors, model, noise, args):
        super(seg_inference, self).__init__()
        self.feature_extractors = feature_extractors
        self.model = model
        self.args = args
        self.noise = noise

    @torch.no_grad()
    def forward(self, img):
        fuse_features = torch.zeros((self.args['dim'][-1]*len(self.feature_extractors), *self.args['dim'][:-1]), dtype=torch.float)
        for i, feature_extractor in enumerate(self.feature_extractors):
            features = feature_extractor(img, noise=self.noise)
            features = collect_features(self.args, features)
            features = features / torch.norm(features)
            fuse_features[i*self.args['dim'][-1]:(i+1)*self.args['dim'][-1]] = features

        fuse_features = fuse_features.view(fuse_features.shape[0], -1).permute(1, 0)
        if isinstance(fuse_features, np.ndarray):
            fuse_features = torch.from_numpy(fuse_features)
        preds = self.model(fuse_features.to(dev()))
        img_seg = preds.reshape(*self.args['dim'][:-1], preds.shape[1])
        img_seg = img_seg.permute(2, 0, 1)
        img_seg = img_seg[None, :, :, :]

        return img_seg


def evaluation(args, model):
    val_ds = ImageLabelPNGDataset(
        data_dir=args['testing_path'],
        mode='test',
        model_type=args['model_type'][0],
        resolution=args['image_size'],
        num_images=args['testing_number'],
        transform=make_transform(
            args['model_type'][0],
            (args['image_size'], args['image_size'])
        ),
        category=args['category'],
        robust=False
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

    args['share_noise'] = False
    if 'share_noise' in args and args['share_noise']:
        rnd_gen = torch.Generator(device=dev()).manual_seed(args['seed'])
        noise = torch.randn(1, args['in_channels'], *args['dim'][:-1],
                            generator=rnd_gen, device=dev())
    else:
        noise = None

    preds, gts = [], []

    feature_extractors = []
    model_types = args['model_type']
    for model_type in model_types:
        args['model_type'] = model_type
        feature_extractor = create_feature_extractor(**args)
        feature_extractors.append(feature_extractor)
    args['model_type'] = model_types

    # feature_extractor = create_feature_extractor(**args)

    for idx, batch_data in enumerate(tqdm(val_loader)):
        img, label = batch_data[0], batch_data[1]
        img = img.to(dev())

        model.eval()

        with torch.no_grad():
            segmentor = seg_inference(feature_extractors, model, noise, args)
            pred = sliding_window_inference(inputs=img, predictor=segmentor.to(dev()), sw_device='cuda',
                                            device='cpu', roi_size=args['dim'][:-1], sw_batch_size=1, overlap=0.2, mode="gaussian")
            # print(pred.shape) # (b, c, h, w)
            img_seg = oht_to_scalar(pred)
            pred = img_seg.cpu().detach()
            label = torch.squeeze(label, 1)
            gts.append(label.numpy())
            preds.append(pred.numpy())

    if len(args['dim']) == 3:
        save_predictions_gts(args, val_ds.image_paths, preds, gts)  # not supported for 3d yet

    if args['category'] == 'glas_1':
        obj_dice = compute_obj_dice(preds, gts)
        iou = obj_dice / (2 - obj_dice)
        print(f'Overall obj dice: ', obj_dice)
        print(f'Overall iou: ', iou)
        return obj_dice, iou
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
        model_type=args['model_type'][0],
        resolution=args['image_size'],
        num_images=args['training_number'],
        transform=make_transform(
            args['model_type'][0],
            (args['image_size'], args['image_size'])
        ),
        category=args['category']
    )
    train_loader = DataLoader(dataset=dataset, batch_size=args['batch_size'], shuffle=True, drop_last=False, num_workers=8)
    criterion = nn.CrossEntropyLoss()

    feature_extractors = []
    model_types = args['model_type']
    for model_type in model_types:
        args['model_type'] = model_type
        feature_extractor = create_feature_extractor(**args)
        feature_extractors.append(feature_extractor)
    args['model_type'] = model_types

    print(" *********************** Current dataloader length " + str(len(train_loader)) + " ***********************")

    obj_dice_all, iou_all, dice_all, aji_all = [], [], [], []

    for MODEL_NUMBER in range(args['start_model_num'], args['model_num'], 1):
        gc.collect()
        classifier = pixel_classifier(numpy_class=(args['number_class']), dim=len(feature_extractors)*args['dim'][-1])
        classifier.init_weights()

        classifier = nn.DataParallel(classifier).cuda()
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
        classifier.train()

        iteration = 0
        break_count = 0
        best_loss = 10000000
        stop_sign = 0

        for epoch in range(5000):
            for x_batch, y_batch in train_loader:
                X_batch, y_batch = prepare_batch_data(args, x_batch, y_batch, feature_extractors)
                X_batch, y_batch = X_batch.to(dev()), y_batch.to(dev())
                y_batch = y_batch.type(torch.long)
                optimizer.zero_grad()
                y_pred = classifier(X_batch)

                loss = criterion(y_pred, y_batch)
                loss.backward()
                optimizer.step()

                iteration += 1
                if iteration % (len(train_loader)) == 0 and epoch > args['start_epoch']:
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
        torch.save({'model_state_dict': classifier.state_dict()},
                   model_path)

        if args['category'] == 'glas_1':
            obj_dice, iou= evaluation(args, classifier)
            obj_dice_all.append(obj_dice)
            iou_all.append(iou)
        elif args['category'] == 'monuseg_1':
            dice, aji = evaluation(args, classifier)
            dice_all.append(dice)
            aji_all.append(aji)

    if args['category'] == 'glas_1':
        print('Obj Dice Mean: ', np.nanmean(obj_dice_all))
        print('Obj Dice Std: ', np.nanstd(obj_dice_all))
        print('IoU Mean: ', np.nanmean(iou_all))
        print('IoU Std: ', np.nanstd(iou_all))
    elif args['category'] == 'monuseg_1':
        print('Dice Mean: ', np.nanmean(dice_all))
        print('Dice Std: ', np.nanstd(dice_all))
        print('Aji Mean: ', np.nanmean(aji_all))
        print('Aji Std: ', np.nanstd(aji_all))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # add_dict_to_argparser(parser, model_and_diffusion_defaults())
    add_dict_to_argparser(parser, mask_defaults())

    parser.add_argument('--exp', type=str)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    setup_seed(args.seed)

    # Load the experiment config
    opts = json.load(open(args.exp, 'r'))
    opts.update(vars(args))
    opts['image_size'] = opts['dim'][0]

    # Prepare the experiment folder
    if len(opts['steps']) > 0:
        suffix = '_'.join([str(step) for step in opts['steps']])
        suffix += '_' + '_'.join([str(step) for step in opts['blocks']])
        opts['exp_dir'] = os.path.join(opts['exp_dir'], suffix)

    path = opts['exp_dir']
    os.makedirs(path, exist_ok=True)
    print('Experiment folder: %s' % (path))
    os.system('cp %s %s' % (args.exp, opts['exp_dir']))

    # Check whether all models are trained
    # We did not use model ensemble in our paper but you can try it out
    pretrained = [os.path.exists(os.path.join(opts['exp_dir'], f'best_model_{i}.pth'))
                  for i in range(opts['model_num'])]

    if not all(pretrained):
        # train all remaining models
        opts['start_model_num'] = sum(pretrained)
        train_batch(opts)
