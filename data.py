import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageEnhance
from torchvision.transforms import InterpolationMode

IMAGE_EXTS = ('.jpg', '.png')
MASK_EXTS = ('.jpg', '.png')
DEPTH_EXTS = ('.bmp', '.png', '.jpg')


def _normalize_dir(path: str) -> str:
    return str(Path(path)).rstrip('/') + '/'


def _require_env_dir(name: str) -> str:
    value = os.environ.get(name, '').strip()
    if not value:
        raise RuntimeError(f'Set {name} before using the built-in dataset presets.')
    return _normalize_dir(value)


def _join_env_dir(root_name: str, *parts: str) -> str:
    return _normalize_dir(Path(_require_env_dir(root_name)).joinpath(*parts))


def _stem(path: str) -> str:
    return Path(path).stem


def _list_files(root: str, exts):
    return [os.path.join(root, f) for f in os.listdir(root) if f.lower().endswith(exts)]


def _build_file_map(root: str, exts):
    file_map = {}
    for path in _list_files(root, exts):
        file_map[_stem(path)] = path
    return file_map


def _build_paired_samples(image_root: str, gt_root: str, depth_root: str, boundary_root: str = None, take_tail: int = None):
    image_map = _build_file_map(image_root, IMAGE_EXTS)
    gt_map = _build_file_map(gt_root, MASK_EXTS)
    depth_map = _build_file_map(depth_root, DEPTH_EXTS)
    common_stems = sorted(set(image_map) & set(gt_map) & set(depth_map))
    if take_tail is not None:
        common_stems = common_stems[-take_tail:]

    boundary_map = _build_file_map(boundary_root, MASK_EXTS) if boundary_root is not None else None
    samples = []
    for stem in common_stems:
        if boundary_map is not None and stem not in boundary_map:
            continue
        samples.append({
            'stem': stem,
            'image': image_map[stem],
            'gt': gt_map[stem],
            'depth': depth_map[stem],
            'boundary': boundary_map[stem] if boundary_map is not None else None,
        })
    return samples


def _train_all_specs():
    return [
        ('UVT20K', _join_env_dir('UMFNET_SOD_ROOT', 'UVT20K', 'Train', 'RGB'), _join_env_dir('UMFNET_SOD_ROOT', 'UVT20K', 'Train', 'GT'), _join_env_dir('UMFNET_SOD_ROOT', 'UVT20K', 'Train', 'T'), _join_env_dir('UMFNET_SOD_ROOT', 'UVT20K', 'Train', 'GT'), None),
        ('UNVT5000', _join_env_dir('UMFNET_SOD_ROOT', 'WeaklyAligned', 'VT5000-Train_unalign', 'RGB'), _join_env_dir('UMFNET_SOD_ROOT', 'WeaklyAligned', 'VT5000-Train_unalign', 'GT'), _join_env_dir('UMFNET_SOD_ROOT', 'WeaklyAligned', 'VT5000-Train_unalign', 'T'), _join_env_dir('UMFNET_SOD_ROOT', 'WeaklyAligned', 'VT5000-Train_unalign', 'GT'), None),
        ('VT5000', _join_env_dir('UMFNET_RGBTSOD_ROOT', 'VT5000', 'Train', 'RGB'), _join_env_dir('UMFNET_RGBTSOD_ROOT', 'VT5000', 'Train', 'GT'), _join_env_dir('UMFNET_RGBTSOD_ROOT', 'VT5000', 'Train', 'T'), _join_env_dir('UMFNET_RGBTSOD_ROOT', 'VT5000', 'Train', 'GT'), None),
    ]


def _train_rgbt_specs():
    return [
        ('VT5000', _join_env_dir('UMFNET_RGBTSOD_ROOT', 'VT5000', 'Train', 'RGB'), _join_env_dir('UMFNET_RGBTSOD_ROOT', 'VT5000', 'Train', 'GT'), _join_env_dir('UMFNET_RGBTSOD_ROOT', 'VT5000', 'Train', 'T'), _join_env_dir('UMFNET_RGBTSOD_ROOT', 'VT5000', 'Train', 'GT'), None),
    ]


def _train_lightfield_specs():
    return [
        ('DUTLFV2', _require_env_dir('UMFNET_DUTLF_RGB_ROOT'), _require_env_dir('UMFNET_DUTLF_GT_ROOT'), _require_env_dir('UMFNET_DUTLF_DEPTH_ROOT'), _require_env_dir('UMFNET_DUTLF_GT_ROOT'), None),
    ]


def mask_to_boundary_pil(mask_img: Image.Image, width: int = 3, mode: str = 'inner', out_mode: str = 'L') -> Image.Image:
    m = np.array(mask_img.convert('L'))
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)

    k = max(1, int(width))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    if mode == 'inner':
        eroded = cv2.erode(m, kernel, iterations=1)
        edge = cv2.subtract(m, eroded)
    elif mode == 'grad':
        dil = cv2.dilate(m, kernel, iterations=1)
        ero = cv2.erode(m, kernel, iterations=1)
        edge = cv2.subtract(dil, ero)
    else:
        raise ValueError("mode must be 'inner' or 'grad'")

    edge = (edge > 0).astype(np.uint8) * 255
    out = Image.fromarray(edge, mode='L')
    return out if out_mode == 'L' else out.convert('1')


def cv_random_flip(img, label, depth):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, depth


def cv_random_flip_with_boundary(img, label, depth, boundary):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        depth = depth.transpose(Image.FLIP_LEFT_RIGHT)
        boundary = boundary.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, depth, boundary


def randomCrop(image, label, depth):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1,
        (image_height - crop_win_height) >> 1,
        (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1,
    )
    return image.crop(random_region), label.crop(random_region), depth.crop(random_region)


def randomCrop_with_boundary(image, label, depth, boundary):
    border = 30
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1,
        (image_height - crop_win_height) >> 1,
        (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1,
    )
    return image.crop(random_region), label.crop(random_region), depth.crop(random_region), boundary.crop(random_region)


def randomRotation(image, label, depth):
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, Image.BICUBIC)
        label = label.rotate(random_angle, Image.NEAREST)
        depth = depth.rotate(random_angle, Image.BICUBIC)
    return image, label, depth


def randomRotation_with_boundary(image, label, depth, boundary):
    if random.random() > 0.8:
        random_angle = np.random.randint(-15, 15)
        image = image.rotate(random_angle, Image.BICUBIC)
        label = label.rotate(random_angle, Image.NEAREST)
        depth = depth.rotate(random_angle, Image.BICUBIC)
        boundary = boundary.rotate(random_angle, Image.NEAREST)
    return image, label, depth, boundary


def colorEnhance(image):
    bright_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Brightness(image).enhance(bright_intensity)
    contrast_intensity = random.randint(5, 15) / 10.0
    image = ImageEnhance.Contrast(image).enhance(contrast_intensity)
    color_intensity = random.randint(0, 20) / 10.0
    image = ImageEnhance.Color(image).enhance(color_intensity)
    sharp_intensity = random.randint(0, 30) / 10.0
    image = ImageEnhance.Sharpness(image).enhance(sharp_intensity)
    return image


def randomGaussian(image, mean=0.1, sigma=0.35):
    def gaussianNoisy(im, mean=mean, sigma=sigma):
        for i in range(len(im)):
            im[i] += random.gauss(mean, sigma)
        return im

    img = np.asarray(image)
    width, height = img.shape
    img = gaussianNoisy(img[:].flatten(), mean, sigma)
    img = img.reshape([width, height])
    return Image.fromarray(np.uint8(img))


def randomPeper(img):
    img = np.array(img)
    noiseNum = int(0.0015 * img.shape[0] * img.shape[1])
    for _ in range(noiseNum):
        randX = random.randint(0, img.shape[0] - 1)
        randY = random.randint(0, img.shape[1] - 1)
        img[randX, randY] = 0 if random.randint(0, 1) == 0 else 255
    return Image.fromarray(img)


class SalObjDataset(data.Dataset):
    def __init__(self, image_root, gt_root, depth_root, trainsize, boundary_flag=False):
        self.boundary_flag = boundary_flag
        self.trainsize = trainsize
        self.samples = self._build_samples(image_root, gt_root, depth_root)
        self.images = [sample['image'] for sample in self.samples]
        self.gts = [sample['gt'] for sample in self.samples]
        self.depths = [sample['depth'] for sample in self.samples]
        self.boundarys = [sample['boundary'] for sample in self.samples] if boundary_flag else []
        self.size = len(self.samples)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        self.depths_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

        print(f'[SalObjDataset] mode={image_root} paired_samples={self.size}')
        if self.size:
            preview = ', '.join(sample['stem'] for sample in self.samples[:5])
            print(f'[SalObjDataset] first_stems={preview}')

    def _build_samples(self, image_root, gt_root, depth_root):
        if image_root == 'all':
            datasets = _train_all_specs()
        elif image_root == 'RGBT':
            datasets = _train_rgbt_specs()
        elif image_root == 'LightField':
            datasets = _train_lightfield_specs()
        else:
            if not image_root or not gt_root or not depth_root:
                raise ValueError('Custom training requires image_root, gt_root, and depth_root.')
            boundary_root = gt_root if self.boundary_flag else None
            return _build_paired_samples(image_root, gt_root, depth_root, boundary_root)

        samples = []
        for name, img_root, gt_root, depth_root, boundary_root, take_tail in datasets:
            boundary_root = boundary_root if self.boundary_flag else None
            part = _build_paired_samples(img_root, gt_root, depth_root, boundary_root, take_tail=take_tail)
            print(f'[SalObjDataset] {name} paired={len(part)}')
            samples.extend(part)
        return samples

    def __getitem__(self, index):
        sample = self.samples[index]
        image = self.rgb_loader(sample['image'])
        gt = self.binary_loader(sample['gt'])
        depth = self.binary_loader(sample['depth'])

        w, h = depth.size
        image = image.resize((w, h), Image.BILINEAR)
        gt = gt.resize((w, h), Image.NEAREST)

        if self.boundary_flag:
            boundary = self.binary_loader(sample['boundary'])
            boundary = boundary.resize((w, h), Image.NEAREST)
            boundary = mask_to_boundary_pil(boundary, width=3, mode='inner', out_mode='L')

            image, gt, depth, boundary = cv_random_flip_with_boundary(image, gt, depth, boundary)
            image, gt, depth, boundary = randomCrop_with_boundary(image, gt, depth, boundary)
            image, gt, depth, boundary = randomRotation_with_boundary(image, gt, depth, boundary)
            image = colorEnhance(image)
            image = self.img_transform(image)
            gt = self.gt_transform(gt)
            depth = self.depths_transform(depth)
            boundary = self.gt_transform(boundary)
            return image, gt, depth, boundary

        image, gt, depth = cv_random_flip(image, gt, depth)
        image, gt, depth = randomCrop(image, gt, depth)
        image, gt, depth = randomRotation(image, gt, depth)
        image = colorEnhance(image)
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        depth = self.depths_transform(depth)
        return image, gt, depth

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def resize(self, img, gt, depth):
        assert img.size == gt.size and gt.size == depth.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST), depth.resize((w, h), Image.BILINEAR)
        return img, gt, depth

    def __len__(self):
        return self.size


def get_loader(image_root, gt_root, depth_root, batchsize, trainsize, shuffle=True, num_workers=8, pin_memory=True, boundary_flag=False):
    dataset = SalObjDataset(image_root, gt_root, depth_root, trainsize, boundary_flag)
    data_loader = data.DataLoader(
        dataset=dataset,
        batch_size=batchsize,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, depth_root, testsize, boundary_flag=False):
        self.boundary_flag = boundary_flag
        self.testsize = testsize
        boundary_root = gt_root if boundary_flag else None
        self.samples = _build_paired_samples(image_root, gt_root, depth_root, boundary_root)
        self.images = [sample['image'] for sample in self.samples]
        self.gts = [sample['gt'] for sample in self.samples]
        self.depths = [sample['depth'] for sample in self.samples]
        self.boundarys = [sample['boundary'] for sample in self.samples] if boundary_flag else []
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.gt_transform = transforms.ToTensor()
        self.depths_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ])

        self.size = len(self.samples)
        self.index = 0
        print(f'[test_dataset] paired_samples={self.size}')
        if self.size:
            preview = ', '.join(sample['stem'] for sample in self.samples[:5])
            print(f'[test_dataset] first_stems={preview}')

    def load_data(self):
        sample = self.samples[self.index]
        image = self.rgb_loader(sample['image'])
        gt = self.binary_loader(sample['gt'])
        depth = self.binary_loader(sample['depth'])

        w, h = depth.size
        image = image.resize((w, h), Image.BILINEAR)
        gt = gt.resize((w, h), Image.NEAREST)

        if self.boundary_flag:
            boundary = self.binary_loader(sample['boundary'])
            boundary = boundary.resize((w, h), Image.NEAREST)
            image = self.transform(image).unsqueeze(0)
            depth = self.depths_transform(depth).unsqueeze(0)
            name = os.path.basename(sample['gt'])
            image_for_post = self.rgb_loader(sample['image'])
            image_for_post = image_for_post.resize(gt.size)
            if name.endswith('.jpg'):
                name = name.split('.jpg')[0] + '.jpg'
            self.index = (self.index + 1) % self.size
            return image, gt, depth, boundary, name, np.array(image_for_post)

        image = self.transform(image).unsqueeze(0)
        depth = self.depths_transform(depth).unsqueeze(0)
        name = os.path.basename(sample['gt'])
        image_for_post = self.rgb_loader(sample['image'])
        image_for_post = image_for_post.resize(gt.size)
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.jpg'
        self.index = (self.index + 1) % self.size
        return image, gt, depth, name, np.array(image_for_post)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size
