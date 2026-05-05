import logging
import os
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

from data import get_loader, test_dataset
from models.UMFNet import UMFNet
from options import opt
from utils import adjust_lr, clip_gradient


def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()


class SaliencyBoundaryLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=0.5, tolerance=1, pos_weight=None, reduction='mean', eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tolerance = tolerance
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction)

    def _dice_from_probs(self, probs, target):
        probs = probs.view(probs.size(0), -1)
        target = target.view(target.size(0), -1)
        inter = (probs * target).sum(dim=1)
        denom = probs.sum(dim=1) + target.sum(dim=1) + self.eps
        return (1.0 - (2.0 * inter + self.eps) / denom).mean()

    def _dice_from_logits(self, logits, target):
        return self._dice_from_probs(torch.sigmoid(logits), target)

    def _tolerant_dice(self, logits, target):
        probs = torch.sigmoid(logits)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        if probs.dim() == 3:
            probs = probs.unsqueeze(1)
        k = 2 * self.tolerance + 1
        target_dil = F.max_pool2d(target, kernel_size=k, stride=1, padding=self.tolerance)
        return self._dice_from_probs(probs, target_dil)

    def forward(self, b, gt_b):
        if gt_b.dtype not in (torch.float32, torch.float64):
            gt_b = gt_b.float()
        if b.dim() == 3:
            b = b.unsqueeze(1)
        if gt_b.dim() == 3:
            gt_b = gt_b.unsqueeze(1)
        bce = self.bce(b, gt_b)
        dice = self._dice_from_logits(b, gt_b)
        tol = self._tolerant_dice(b, gt_b) if self.gamma != 0 else bce.new_tensor(0.0)
        return self.alpha * bce + self.beta * dice + self.gamma * tol


def save_checkpoint(save_file, epoch, model, optimizer, best_mae, best_epoch, global_step):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_mae': best_mae,
        'best_epoch': best_epoch,
        'global_step': global_step,
    }, save_file)


def kl_weight(epoch):
    return LAMBDA_KL_MAX * min(1.0, epoch / max(1, KL_WARMUP_EPOCHS))


os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)
print(f'USE GPU {opt.gpu_id}')
cudnn.benchmark = True

save_path = opt.save_path
os.makedirs(save_path, exist_ok=True)
logging.basicConfig(
    filename=save_path + 'UMFNet.log',
    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
    level=logging.INFO,
    filemode='a',
    datefmt='%Y-%m-%d %I:%M:%S %p',
)
logging.info('UMFNet-Train')

model = UMFNet()
if opt.load_pre:
    model.load_pre(opt.load_pre)
    print('load model from', opt.load_pre)
model.cuda()

num_params = sum(p.numel() for p in model.parameters())
logging.info('Total Parameters (For Reference): {}'.format(num_params))
print('Total Parameters (For Reference): {}'.format(num_params))

optimizer = torch.optim.Adam(model.parameters(), opt.lr)

image_root = opt.image_root if opt.image_root else opt.rgb_root
gt_root = opt.gt_root
thermal_root = opt.depth_root

test_image_root = opt.test_rgb_root
test_gt_root = opt.test_gt_root
test_thermal_root = opt.test_depth_root

print('load data...')
train_loader = get_loader(image_root, gt_root, thermal_root, batchsize=opt.batchsize, trainsize=opt.trainsize, boundary_flag=True)
test_loader = test_dataset(test_image_root, test_gt_root, test_thermal_root, opt.trainsize)
total_step = len(train_loader)

logging.info('Config')
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};lr_sched:{};min_lr:{};warmup_lr:{};warmup_epochs:{};decay_rate:{};load:{};save_path:{};decay_epoch:{};image_root:{};test_start_epoch:{};resume:{};resume_mode:{}'.format(
        opt.epoch,
        opt.lr,
        opt.batchsize,
        opt.trainsize,
        opt.clip,
        opt.lr_sched,
        opt.min_lr,
        opt.warmup_lr,
        opt.warmup_epochs,
        opt.decay_rate,
        opt.load_pre,
        save_path,
        opt.decay_epoch,
        image_root,
        opt.test_start_epoch,
        opt.resume,
        opt.resume_mode,
    )
)

CE = nn.BCEWithLogitsLoss()
boundary_loss = SaliencyBoundaryLoss()

LAMBDA_SAL = 1.0
LAMBDA_BD = 1.0
LAMBDA_KL_MAX = 1e-3
KL_WARMUP_EPOCHS = 10

step = 0
best_mae = 1
best_epoch = 0
start_epoch = 1

if opt.resume:
    if not os.path.isfile(opt.resume):
        raise FileNotFoundError(f'Resume checkpoint not found: {opt.resume}')
    ckpt = torch.load(opt.resume, map_location='cpu')
    if opt.resume_mode == 'resume':
        model.load_state_dict(ckpt['model_state_dict'], strict=True)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_mae = ckpt.get('best_mae', 1)
        best_epoch = ckpt.get('best_epoch', 0)
        step = ckpt.get('global_step', 0)
        print(f'Resumed training from {opt.resume} at epoch {start_epoch}')
        logging.info(f'Resumed training from {opt.resume} at epoch {start_epoch}')
    elif opt.resume_mode == 'finetune':
        state_dict = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state_dict, strict=True)
        print(f'Loaded finetune weights from {opt.resume}')
        logging.info(f'Loaded finetune weights from {opt.resume}')


def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()

    loss_all = None
    sal_all = None
    bd_all = None
    kl_all = None
    kl_v_all = None
    kl_t_all = None
    epoch_step = 0
    cur_kl_w = kl_weight(epoch)

    try:
        for i, (images, gts, thermals, boundarys) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda(non_blocking=True)
            gts = gts.cuda(non_blocking=True)
            boundarys = boundarys.cuda(non_blocking=True)
            thermals = thermals.repeat(1, 3, 1, 1).cuda(non_blocking=True)

            out = model(images, thermals)
            sal_logits = out['sal']
            bd_logits = out['bd']
            kl_v = out['kl_v']
            kl_t = out['kl_t']

            L_sal = sum(CE(s, gts) + iou_loss(s, gts) for s in sal_logits)
            L_bd = sum(boundary_loss(b, boundarys) for b in bd_logits)
            L_kl = kl_v + kl_t
            loss = LAMBDA_SAL * L_sal + LAMBDA_BD * L_bd + cur_kl_w * L_kl

            loss.backward()
            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all = loss.detach() if loss_all is None else loss_all + loss.detach()
            sal_all = L_sal.detach() if sal_all is None else sal_all + L_sal.detach()
            bd_all = L_bd.detach() if bd_all is None else bd_all + L_bd.detach()
            kl_all = L_kl.detach() if kl_all is None else kl_all + L_kl.detach()
            kl_v_all = kl_v.detach() if kl_v_all is None else kl_v_all + kl_v.detach()
            kl_t_all = kl_t.detach() if kl_t_all is None else kl_t_all + kl_t.detach()

            if i % 500 == 0 or i == total_step or i == 1:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                lr_now = optimizer.state_dict()['param_groups'][0]['lr']
                msg = ('{ts} Epoch [{ep:03d}/{tot:03d}], Step [{st:04d}/{ts2:04d}], '
                       'LR:{lr:.7f} || total:{loss:.4f}  L_sal:{lsal:.4f}  '
                       'L_bd:{lbd:.4f}  L_kl:{lkl:.4f}  kl_v:{klv:.4f}  kl_t:{klt:.4f}  '
                       'weighted_kl:{wkl:.6f} (w={kw:.1e})').format(
                    ts=datetime.now(),
                    ep=epoch,
                    tot=opt.epoch,
                    st=i,
                    ts2=total_step,
                    lr=lr_now,
                    loss=loss.item(),
                    lsal=L_sal.item(),
                    lbd=L_bd.item(),
                    lkl=L_kl.item(),
                    klv=kl_v.item(),
                    klt=kl_t.item(),
                    wkl=cur_kl_w * L_kl.item(),
                    kw=cur_kl_w,
                )
                print(msg)
                logging.info('TRAIN ' + msg + ', mem_use:{:.0f}MB'.format(memory_used))

        loss_all = (loss_all / max(epoch_step, 1)).item()
        sal_all = (sal_all / max(epoch_step, 1)).item()
        bd_all = (bd_all / max(epoch_step, 1)).item()
        kl_all = (kl_all / max(epoch_step, 1)).item()
        kl_v_all = (kl_v_all / max(epoch_step, 1)).item()
        kl_t_all = (kl_t_all / max(epoch_step, 1)).item()
        logging.info(
            'TRAIN Epoch [{:03d}/{:03d}], Loss_AVG:{:.4f}, L_sal_AVG:{:.4f}, L_bd_AVG:{:.4f}, L_kl_AVG:{:.4f}, kl_v_AVG:{:.4f}, kl_t_AVG:{:.4f}, weighted_kl_AVG:{:.6f}'.format(
                epoch,
                opt.epoch,
                loss_all,
                sal_all,
                bd_all,
                kl_all,
                kl_v_all,
                kl_t_all,
                cur_kl_w * kl_all,
            )
        )

        save_checkpoint(save_path + 'UMFNet_last.pth', epoch, model, optimizer, best_mae, best_epoch, step)
        save_checkpoint(save_path + 'UMFNet_epoch_{:03d}.pth'.format(epoch), epoch, model, optimizer, best_mae, best_epoch, step)

    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        os.makedirs(save_path, exist_ok=True)
        save_checkpoint(save_path + 'UMFNet_epoch_{}.pth'.format(epoch + 1), epoch, model, optimizer, best_mae, best_epoch, step)
        print('save checkpoints successfully!')
        raise


def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for _ in range(test_loader.size):
            image, gt, thermal, name, img_for_post = test_loader.load_data()

            gt = torch.from_numpy(np.asarray(gt, np.float32) / 255.0).float().unsqueeze(0).unsqueeze(0).cuda()
            image = image.cuda()
            thermal = thermal.repeat(1, 3, 1, 1).cuda()

            out = model(image, thermal)
            sal_logits = out['sal']
            res = sum(sal_logits)
            res = F.interpolate(res, size=gt.shape[-2:], mode='bilinear', align_corners=False)
            res = res.sigmoid()
            mae_sum += F.l1_loss(res, gt, reduction='mean').item()

        mae = mae_sum / test_loader.size
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        elif mae < best_mae:
            best_mae = mae
            best_epoch = epoch
            save_checkpoint(save_path + 'UMFNet_best.pth', epoch, model, optimizer, best_mae, best_epoch, step)
            print('best epoch:{}'.format(epoch))
        logging.info('TEST Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print('Start train...')
    for epoch in range(start_epoch, opt.epoch + 1):
        cur_lr = adjust_lr(
            optimizer,
            opt.lr,
            epoch,
            decay_rate=opt.decay_rate,
            decay_epoch=opt.decay_epoch,
            lr_sched=opt.lr_sched,
            min_lr=opt.min_lr,
            warmup_lr=opt.warmup_lr,
            warmup_epochs=opt.warmup_epochs,
            total_epochs=opt.epoch,
        )
        logging.info('LR Epoch:{} LR:{:.7f}'.format(epoch, cur_lr))
        train(train_loader, model, optimizer, epoch, save_path)
        if epoch >= opt.test_start_epoch:
            test(test_loader, model, epoch, save_path)
