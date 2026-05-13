import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from py_sod_metrics import Emeasure, Smeasure, WeightedFmeasure
from tqdm import tqdm

from data import test_dataset
from models.UMFNet import UMFNet


def _get_env_dir(name: str) -> str:
    value = os.environ.get(name, '').strip()
    if not value:
        raise RuntimeError(f'Set {name} before evaluating the built-in datasets.')
    return value


def _join_dir(root_name: str, *parts: str) -> str:
    return str(Path(_get_env_dir(root_name)).joinpath(*parts)).rstrip('/') + '/'


def resolve_dataset_roots(dataset: str):
    if dataset == 'UVT20K':
        return (
            _join_dir('UMFNET_SOD_ROOT', 'UVT20K', 'Test', 'RGB'),
            _join_dir('UMFNET_SOD_ROOT', 'UVT20K', 'Test', 'GT'),
            _join_dir('UMFNET_SOD_ROOT', 'UVT20K', 'Test', 'T'),
        )
    if dataset == 'UVT2000':
        return (
            _join_dir('UMFNET_SOD_ROOT', 'UVT2000', 'RGB'),
            _join_dir('UMFNET_SOD_ROOT', 'UVT2000', 'GT'),
            _join_dir('UMFNET_SOD_ROOT', 'UVT2000', 'T'),
        )
    if dataset == 'U-VT5000':
        return (
            _join_dir('UMFNET_SOD_ROOT', 'WeaklyAligned', 'VT5000-Test_unalign', 'RGB'),
            _join_dir('UMFNET_SOD_ROOT', 'WeaklyAligned', 'VT5000-Test_unalign', 'GT'),
            _join_dir('UMFNET_SOD_ROOT', 'WeaklyAligned', 'VT5000-Test_unalign', 'T'),
        )
    if dataset == 'U-VT1000':
        return (
            _join_dir('UMFNET_SOD_ROOT', 'WeaklyAligned', 'VT1000_unalign', 'RGB'),
            _join_dir('UMFNET_SOD_ROOT', 'WeaklyAligned', 'VT1000_unalign', 'GT'),
            _join_dir('UMFNET_SOD_ROOT', 'WeaklyAligned', 'VT1000_unalign', 'T'),
        )
    if dataset == 'U-VT821':
        return (
            _join_dir('UMFNET_SOD_ROOT', 'WeaklyAligned', 'VT821_unalign', 'RGB'),
            _join_dir('UMFNET_SOD_ROOT', 'WeaklyAligned', 'VT821_unalign', 'GT'),
            _join_dir('UMFNET_SOD_ROOT', 'WeaklyAligned', 'VT821_unalign', 'T'),
        )
    raise NotImplementedError(f'No dataset named {dataset}')


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--pth_path', type=str, default='./Results/Result_UMFNet/UMFNet_best.pth', help='checkpoint path')
parser.add_argument('--save_root', type=str, default='./test_maps', help='root directory for prediction maps')
parser.add_argument('--save_predictions', action='store_true', help='save prediction maps')
parser.add_argument('--datasets', nargs='+', default=['UVT20K'], help='dataset names to evaluate')
opt = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)


def normalize_prediction(pred_tensor):
    pred = pred_tensor.detach().cpu().numpy().astype(np.float32)
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    return (pred * 255).astype(np.uint8)


model = UMFNet()
checkpoint = torch.load(opt.pth_path, map_location='cpu')
state_dict = checkpoint.get('model_state_dict', checkpoint.get('model', checkpoint))
model.load_state_dict(state_dict, strict=True)
model.cuda()
model.eval()
save_result_path_name = Path(opt.pth_path).stem
test_datasets = opt.datasets

result_root = Path(opt.save_root) / save_result_path_name
result_root.mkdir(parents=True, exist_ok=True)
json_path = result_root / 'metrics_summary.json'
metrics_summary = {
    'checkpoint': opt.pth_path,
    'result_root': str(result_root),
    'datasets': {},
}

for dataset in test_datasets:
    save_path = result_root / dataset if opt.save_predictions else None
    if save_path is not None:
        save_path.mkdir(parents=True, exist_ok=True)
    image_root, gt_root, depth_root = resolve_dataset_roots(dataset)
    test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
    print('Testing on {} dataset...'.format(dataset), test_loader.size)
    sm = Smeasure()
    wfm = WeightedFmeasure()
    em = Emeasure()
    mae_sum = 0.0
    sample_count = 0
    for _ in tqdm(range(test_loader.size), total=test_loader.size, desc=f'Testing {dataset}'):
        image, gt, depth, name, image_for_post = test_loader.load_data()
        gt_arr = np.asarray(gt, np.float32)
        gt_eval = (gt_arr > 127.5).astype(np.uint8) * 255
        gt_mae = torch.from_numpy(gt_arr / 255.0).float()[None, None, ...]
        image = image.cuda()
        depth = depth.repeat(1, 3, 1, 1).cuda()
        out = model(image, depth)
        sal_logits = out['sal']
        res = sum(sal_logits)
        res = F.interpolate(res, size=gt_arr.shape, mode='bilinear', align_corners=False)
        pred_prob = res.sigmoid().detach().cpu()
        pred_eval = normalize_prediction(pred_prob.squeeze())

        sm.step(pred=pred_eval, gt=gt_eval)
        wfm.step(pred=pred_eval, gt=gt_eval)
        em.step(pred=pred_eval, gt=gt_eval)
        mae_sum += F.l1_loss(pred_prob, gt_mae, reduction='mean').item()
        sample_count += 1

        if save_path is not None:
            save_name = Path(name).stem + '.png'
            cv2.imwrite(str(save_path / save_name), pred_eval)

    em_results = em.get_results()['em']
    dataset_metrics = {
        'S_alpha': float(sm.get_results()['sm']),
        'F_beta_w': float(wfm.get_results()['wfm']),
        'E_S': float(np.max(em_results['curve'])),
        'MAE': float(mae_sum / max(sample_count, 1)),
        'num_samples': sample_count,
        'save_predictions': opt.save_predictions,
        'prediction_dir': str(save_path) if save_path is not None else '',
    }
    metrics_summary['datasets'][dataset] = dataset_metrics

    print(
        f"{dataset} "
        f"S_alpha: {dataset_metrics['S_alpha']:.6f}, "
        f"F_beta_w: {dataset_metrics['F_beta_w']:.6f}, "
        f"E_S: {dataset_metrics['E_S']:.6f}, "
        f"MAE: {dataset_metrics['MAE']:.6f}"
    )

with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(metrics_summary, f, indent=2, ensure_ascii=False)

print(f'Metrics JSON saved to {json_path}')
print('Test Done!')
