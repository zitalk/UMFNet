import math


def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30,
              lr_sched='cosine', min_lr=1e-6, warmup_lr=1e-6, warmup_epochs=5, total_epochs=100):
    if lr_sched == 'cosine':
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            progress = epoch / warmup_epochs
            lr = warmup_lr + (init_lr - warmup_lr) * progress
        else:
            effective_total = max(total_epochs - warmup_epochs, 1)
            progress = min(max((epoch - warmup_epochs) / effective_total, 0.0), 1.0)
            lr = min_lr + 0.5 * (init_lr - min_lr) * (1.0 + math.cos(math.pi * progress))
    else:
        decay = decay_rate ** (epoch // decay_epoch)
        lr = decay * init_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
