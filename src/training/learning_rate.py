def schedule_lr(optimizer, epoch, scheduler, scheduler_name='', avg_val_loss=0, epochs=100, 
                warmup_prop=0.1, lr_init=1e-3, min_lr=1e-6):
    """
    Tool to apply scheduling to an optimizer. Initially made when warmup was not available in PyTorch.
    
    Arguments:
        optimizer {torch optimizer} -- Optimizer
        epoch {int} -- Current epoch
        scheduler {torch scheduler} -- Scheduler
    
    Keyword Arguments:
        scheduler_name {str} -- Name of the scheduler, expected in ['cosine', 'reduce_lr'] (default: {''})
        avg_val_loss {int} -- Current loss, only use if the scheduler is reduce_lr (default: {0})
        epochs {int} -- Total number of epochs (default: {100})
        warmup_prop {float} -- Proportion of epochs used for warmup (default: {0.1})
        lr_init {[type]} -- Learning rate to start from (default: {1e-3})
        min_lr {[type]} -- Learning rate to end to (default: {1e-6})
    
    Returns:
        int -- Learning rate at the current epoch
    """

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
            lr = 1e-3
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

    lr = optimizer.param_groups[-1]['lr']
    return lr
