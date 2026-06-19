import math

def adjust_learning_rate(optimizer, epoch, config):
    if epoch < config['warmup_epochs']:
        lr = config['lr'] * epoch / config['warmup_epochs']
    else:
        lr = config['min_lr'] + (config['lr'] - config['min_lr']) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - config['warmup_epochs']) / (config['epochs'] - config['warmup_epochs'])))
    for param_group in optimizer.param_groups:
        if 'lr_scale' in param_group:
            param_group['lr'] = lr * param_group['lr_scale']
        else:
            param_group['lr'] = lr
    return lr