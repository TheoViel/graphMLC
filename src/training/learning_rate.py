def get_lr(epoch):
    if epoch < 20:
        return 5e-4
    else:
        return 5e-5


def get_lr(epoch):
    if epoch < 20:
        return 5e-4
    else:
        return 5e-5


def schedule_lr(optimizer, epoch, scheduler, scheduler_name='', avg_val_loss=0, epochs=100, 
                warmup_prop=0.1, lr_init=1e-3, min_lr=1e-6, verbose_eval=1):

    if epoch <= epochs * warmup_prop and warmup_prop > 0:
        lr = min_lr + (lr_init - min_lr) * epoch / (epochs * warmup_prop)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    elif epoch:
        if scheduler_name == 'cosine':
            scheduler.step()
        elif scheduler_name == 'reduce_lr':
            scheduler.step(avg_val_loss)
        else:
            lr = get_lr(epoch)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    lr = optimizer.param_groups[-1]['lr']

    return lr