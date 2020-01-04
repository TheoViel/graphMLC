from util import *
from metric import *
from imports import *

from training.learning_rate import *



def fit(model, train_dataset, val_dataset, epochs=50, batch_size=32, warmup_prop=0.1, lr=1e-3, min_lr=1e-5,
        verbose=1, verbose_eval=1, cp=False, model_name='model'):
    """
    Usual torch fit function
    
    Arguments:
        model {torch model} -- Model to train
        train_dataset {torch dataset} -- Dataset to train with
        val_dataset {torch dataset} -- Dataset to validate with
    
    Keyword Arguments:
        epochs {int} -- Number of epochs (default: {50})
        batch_size {int} -- Batch size (default: {32})
        warmup_prop {float} -- Warmup proportion (default: {0.1})
        lr {[type]} -- Start (or maximum) learning rate (default: {1e-3})
        min_lr {[type]} -- Minimum learning rate (default: {1e-5})
        verbose {int} -- Period of training information display (default: {1})
        cp {bool} -- Whether to save model weights with checkpointing (default: {False})
        model_name {str} -- Name of the model, for checkpoints saving (default: {'model'})
    """
       
    best_map = 0
    avg_val_loss = 0
    avg_loss = 0
    lr_init = lr

    optimizer = Adam(filter(lambda p : p.requires_grad,model.parameters()), lr=lr)

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs - ceil(epochs * warmup_prop) - 1, eta_min=min_lr)

    loss_fct = nn.BCEWithLogitsLoss(reduction='mean')

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=NUM_WORKERS)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
        
    for epoch in range(epochs):
        model.train()
        start_time = time.time()
        optimizer.zero_grad()

        lr = schedule_lr(optimizer, epoch, scheduler, scheduler_name="cosine", avg_val_loss=avg_val_loss,
            epochs=epochs, warmup_prop=warmup_prop, lr_init=lr_init, min_lr=min_lr)

        avg_loss = 0
        for step, (x, y_batch) in enumerate(train_loader):
            y_pred = model(x.cuda())
            loss = loss_fct(y_pred.double(), y_batch.double().cuda())

            loss.backward()
            avg_loss += loss.item() / len(train_loader)

            optimizer.step()
            optimizer.zero_grad()

        model.eval()
            
        avg_val_loss = 0.
        val_map = 0.


        with torch.no_grad():
            preds = np.empty((0, NUM_CLASSES))
            truths = np.empty((0, NUM_CLASSES))
            for x, y_batch in val_loader:
                y_pred = model(x.cuda()).detach()
                loss = loss_fct(y_pred.double(), y_batch.double().cuda())
                avg_val_loss += loss.item() / len(val_loader)

                preds = np.concatenate([preds, torch.sigmoid(y_pred).cpu().numpy()])
                truths = np.concatenate([truths, y_batch.numpy()])

        preds_voc = np.concatenate([preds, truths], axis=1)
        val_map = voc12_mAP(preds_voc)

        if val_map >= best_map and cp:
            save_model_weights(model, f"{model_name}_cp.pt", verbose=0)
            if use_ema:
                save_model_weights(model_shadow, f"{model_name}_shadow_cp.pt", verbose=0)
            best_map = val_map
        
        elapsed_time = time.time() - start_time

        if (epoch + 1) % verbose == 0:
            elapsed_time = elapsed_time * verbose
            print(f'Epoch {epoch + 1}/{epochs} \t lr={lr:.1e} \t t={elapsed_time:.0f}s \t loss={avg_loss:.3f} \t ', end='')
            print(f'mAP={val_map:.3f} \t val_loss={avg_val_loss:.3f} \t ', end='\n')

    torch.cuda.empty_cache()


def predict(model, dataset, batch_size=32):
    """
    Usual predict torch function
    
    Arguments:
        model {torch model} -- Model to predict with
        dataset {torch dataset} -- Dataset to get predictions from
    
    Keyword Arguments:
        batch_size {int} -- Batch size (default: {32})
    
    Returns:
        numpy array -- Predictions
    """

    model.eval()
    preds = np.empty((0, NUM_CLASSES))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)

    with torch.no_grad():
        for x, _ in loader:
            prob = torch.sigmoid(model(x.cuda()))
            probs = prob.detach().cpu().numpy()
            preds = np.concatenate([preds, probs])
            
    return preds
    

def predict_voc(model, dataset, batch_size=32):
    """
    Usual predict torch function but adapted to the format of the voc metric in metric.py
    
    Arguments:
        model {torch model} -- Model to predict with
        dataset {torch dataset} -- Dataset to get predictions from
    
    Keyword Arguments:
        batch_size {int} -- Batch size (default: {32})
    
    Returns:
        numpy array -- Predictions, but concatenated with the true value
    """
    model.eval()
    preds = np.empty((0, NUM_CLASSES * 2))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    with torch.no_grad():
        for x, target in loader:
            prob = torch.sigmoid(model(x.cuda()))
            probs = prob.detach().cpu().numpy()
            preds = np.concatenate([preds, np.concatenate([probs, target.numpy()], 1)])
            
    return preds
