import cv2
import torch
import numpy as np
from PIL import Image
import sys
sys.path.append("..")
sys.path.append(".")
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from guided_diffusion.guided_diffusion.image_datasets import _list_image_files_recursively
import os
import os.path
import nibabel
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import monai.transforms as m_transforms
import json
from scipy import ndimage
import random


def random_rot_flip(image, label):
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def make_transform(model_type: str, resolution: int):
    """ Define input transforms for pretrained models """
    if model_type in ['ddpm', 'mask_ddpm']:
        transform = transforms.Compose([
            # transforms.Resize(resolution), # If glas crop dataset, del this line
            transforms.ToTensor(),
            lambda x: 2 * x - 1
        ])
    elif model_type in ['mae', 'swav', 'swav_w2', 'DeepLab', 'UNet', 'SwinUNETR', 'UCTransNet', 'BasicUNetPlusPlus', 'AttentionUnet', 'MedT']:
        transform = transforms.Compose([
            # transforms.Resize(resolution), # If glas crop dataset, del this line
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        raise Exception(f"Wrong model type: {model_type}")
    return transform


class FeatureDataset(Dataset):
    '''
    Dataset of the pixel representations and their labels.

    :param X_data: pixel representations [num_pixels, feature_dim]
    :param y_data: pixel labels [num_pixels]
    '''
    def __init__(
        self,
        X_data: torch.Tensor,
        y_data: torch.Tensor
    ):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class ImageLabelPNGDataset(Dataset):
    '''
    :param data_dir: path to a folder with images and their annotations.
                     Annotations are supposed to be in *.npy format.
    :param resolution: image and mask output resolution.
    :param num_images: restrict a number of images in the dataset.
    :param transform: image transforms.
    '''
    def __init__(
        self,
        data_dir: str,
        mode: str,
        model_type: str,
        resolution: int,
        num_images=-1,
        transform=None,
        category='glas_1',
        robust=False
    ):
        super().__init__()
        self.resolution = resolution
        self.transform = transform
        self.image_paths = _list_image_files_recursively(data_dir)
        self.image_paths = sorted(self.image_paths)
        self.mode = mode
        self.model_type = model_type
        self.category = category
        self.robust = robust

        if num_images > 0:
            print(f"Take first {num_images} images...")
            self.image_paths = self.image_paths[:num_images]
            # print(self.image_paths)

        if self.category=='glas_1':
            if self.robust:
                self.label_paths = [
                    '/'.join(image_path.split('/')[:-4]) + '/labelcol/' + (image_path.split('/')[-1])[:-4] + '_anno.png'
                    for image_path in self.image_paths
                ]
            else:
                self.label_paths = [
                    '/'.join(image_path.split('/')[:-2]) + '/labelcol/' + (image_path.split('/')[-1])[:-4] + '_anno.png'
                    for image_path in self.image_paths
                ]
        elif self.category=='monuseg_1':
            self.label_paths = [
                '/'.join(image_path.split('/')[:-2]) + '/labelcol/' + (image_path.split('/')[-1])[:-4] + '.png'
                for image_path in self.image_paths
            ]
        # print(self.label_paths)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load an image
        image_path = self.image_paths[idx]
        pil_image = Image.open(image_path)
        pil_image = pil_image.convert("RGB")
        # assert pil_image.size[0] == pil_image.size[1], \
        #       f"Only square images are supported: ({pil_image.size[0]}, {pil_image.size[1]})"

        # Load a corresponding mask and resize it to (self.resolution, self.resolution)

        label_path = self.label_paths[idx]
        # print(tensor_image.shape)
        label = cv2.imread(label_path, 0).astype('uint8')
        # print(np.max(label), np.min(label))

        label[label <= 0] = 0
        label[label > 0] = 1

        # print(np.max(label), np.min(label))

        # print(self.resolution)
        if self.mode == 'train':
            i, j, h, w = T.RandomCrop.get_params(pil_image, output_size=(self.resolution, self.resolution))
            pil_image = TF.crop(pil_image, i, j, h, w)
            tensor_image = self.transform(pil_image)
            image = tensor_image.numpy()
            label = Image.fromarray(label.astype(np.uint8))
            label = TF.crop(label, i, j, h, w)
            tensor_image = torch.from_numpy(image)
        elif self.mode == 'test':
            tensor_image = self.transform(pil_image)

        label = np.expand_dims(label, axis=0)

        tensor_label = torch.from_numpy(label)
        # print(tensor_label.shape)
        return tensor_image, tensor_label


class ImageLabelDataset(Dataset):
    '''
    :param data_dir: path to a folder with images and their annotations.
                     Annotations are supposed to be in *.npy format.
    :param resolution: image and mask output resolution.
    :param num_images: restrict a number of images in the dataset.
    :param transform: image transforms.
    '''
    def __init__(
        self,
        data_dir: str,
        resolution: int,
        num_images= -1,
        transform=None,
    ):
        super().__init__()
        self.resolution = resolution
        self.transform = transform
        self.image_paths = _list_image_files_recursively(data_dir)
        self.image_paths = sorted(self.image_paths)

        if num_images > 0:
            print(f"Take first {num_images} images...")
            self.image_paths = self.image_paths[:num_images]

        self.label_paths = [
            '.'.join(image_path.split('.')[:-1] + ['npy'])
            for image_path in self.image_paths
        ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load an image
        image_path = self.image_paths[idx]
        pil_image = Image.open(image_path)
        pil_image = pil_image.convert("RGB")
        assert pil_image.size[0] == pil_image.size[1], \
               f"Only square images are supported: ({pil_image.size[0]}, {pil_image.size[1]})"

        tensor_image = self.transform(pil_image)
        # Load a corresponding mask and resize it to (self.resolution, self.resolution)
        label_path = self.label_paths[idx]
        label = np.load(label_path).astype('uint8')
        label = cv2.resize(
            label, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST
        )
        tensor_label = torch.from_numpy(label)
        return tensor_image, tensor_label


class InMemoryImageLabelDataset(Dataset):
    '''

    Same as ImageLabelDataset but images and labels are already loaded into RAM.
    It handles DDPM/GAN-produced datasets and is used to train DeepLabV3.

    :param images: np.array of image samples [num_images, H, W, 3].
    :param labels: np.array of correspoding masks [num_images, H, W].
    :param resolution: image and mask output resolusion.
    :param num_images: restrict a number of images in the dataset.
    :param transform: image transforms.
    '''

    def __init__(
            self,
            images: np.ndarray,
            labels: np.ndarray,
            resolution=256,
            transform=None
    ):
        super().__init__()
        assert  len(images) == len(labels)
        self.images = images
        self.labels = labels
        self.resolution = resolution
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        assert image.size[0] == image.size[1], \
               f"Only square images are supported: ({image.size[0]}, {image.size[1]})"

        tensor_image = self.transform(image)
        label = self.labels[idx]
        label = cv2.resize(
            label, (self.resolution, self.resolution), interpolation=cv2.INTER_NEAREST
        )
        tensor_label = torch.from_numpy(label)
        return tensor_image, tensor_label


# unit test
if __name__ == "__main__":
    dataset = ImageLabelPNGDataset(
        data_dir='/afs/crc.nd.edu/user/z/zpan3/Datasets/GlaS/test/img_corrupted',
        mode='train',
        model_type='SwinUNETR',
        resolution=256,
        num_images=80,
        transform=make_transform('SwinUNETR',(256, 256)),
        robust=True
    )
    loader = DataLoader(
        dataset, batch_size=2, shuffle=False, num_workers=1, drop_last=True
    )
    dataiter = iter(loader)
    x, y = next(dataiter)
    # save x as png
    x = x.numpy()
    x = np.transpose(x, (0, 2, 3, 1))
    x = x.astype(np.uint8)
    x =x[0]
    cv2.imwrite('test.png', x)
    print(x.shape)
    print(y.shape)


