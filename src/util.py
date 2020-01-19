from imports import *
from params import *
from training.train import predict

def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results
    
    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True #False


def save_model_weights(model, filename, verbose=1, cp_folder=CP_PATH):
    """
    Saves the weights of a PyTorch model
    
    Arguments:
        model {torch module} -- Model to save the weights of
        filename {str} -- Name of the checkpoint
    
    Keyword Arguments:
        verbose {int} -- Whether to display infos (default: {1})
        cp_folder {str} -- Folder to save to (default: {CP_PATH})
    """
    if verbose:
        print(f"\n -> Saving weights to {os.path.join(cp_folder, filename)}\n")
    torch.save(model.state_dict(), os.path.join(cp_folder, filename))


def load_model_weights(model, filename, verbose=1, cp_folder=CP_PATH):
    """
    Loads the weights of a PyTorch model. The exception handles cpu/gpu incompatibilities
    
    Arguments:
        model {torch module} -- Model to load the weights to
        filename {str} -- Name of the checkpoint
    
    Keyword Arguments:
        verbose {int} -- Whether to display infos (default: {1})
        cp_folder {str} -- Folder to load from (default: {CP_PATH})
    
    Returns:
        torch module -- Model with loaded weights
    """
    if verbose:
        print(f"\n -> Loading weights from {os.path.join(cp_folder,filename)}\n")
    try:
        model.load_state_dict(os.path.join(cp_folder, filename), strict=strict)
    except BaseException:
        model.load_state_dict(
            torch.load(os.path.join(cp_folder, filename), map_location="cpu"),
            strict=True,
        )
    return model


def count_parameters(model, all=False):
    """
    Count the parameters of a model
    
    Arguments:
        model {torch module} -- Model to count the parameters of
    
    Keyword Arguments:
        all {bool} -- Whether to include not trainable parameters in the sum (default: {False})
    
    Returns:
        int -- Number of parameters
    """
    if all:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_attention_ssgrl(model, dataset, idx):
    """
    Plots the attention maps of the Semantic Decoupling module for the detected classes of image dataset[idx]
    As the maps are smaller than the image, they are interpolated with bicubic resampling
    
    Arguments:
        model {SSGRLClassifier} -- Trained SSGRL model
        dataset {MLCDataset} -- Dataset to sample images from
        idx {int} -- Index of the image is the dataset
    """
    img = dataset[idx][0]
    img_tensor = torch.tensor(img).unsqueeze(0).cuda()

    out, att = model.get_attention(img_tensor)
    h = int(np.sqrt(att.size()[1]))
    att = att[0, :, :, 0].view(h, h, 20).detach().cpu().numpy()

    out = out[0].detach().cpu().numpy()

    img = np.clip(
        img * STD[:, np.newaxis, np.newaxis] + MEAN[:, np.newaxis, np.newaxis], 0, 1
    )
    img = img.transpose(1, 2, 0)

    for i in range(NUM_CLASSES):
        if out[i] > 0.5:
            plt.figure(figsize=(5, 5))

            att_map = att[:, :, i]
            att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min()) * 255

            # Resizing
            att_map = Image.fromarray(att_map)
            att_map = att_map.resize((IMG_SIZE, IMG_SIZE), resample=PIL.Image.BICUBIC)
            att_map = np.clip(np.array(att_map), 0, 255)

            # Weighting on original image
            alpha = 0.5
            att_map = (att_map[:, :, np.newaxis].repeat(3, 2) / 255).astype(np.float32)
            weighted = cv2.addWeighted(
                att_map, alpha, img.astype(np.float32), 1 - alpha, 0
            )

            plt.imshow(weighted)
            plt.axis(False)
            plt.title(f'Interpolated Attention Map for Class "{CLASSES[i]}"')
            plt.show()


def plot_coocurence(matrix, classes, cmap=plt.cm.Blues, title='Coocurence Matrix'):
    """
    Plots a matrix of coocurence or appearance conditional probabilities of classes
    
    Arguments:
        matrix {numpy array} -- Matrix to plot
        classes {list of strings} -- Class names, will be used for ticks
    
    Keyword Arguments:
        cmap {matplotlib colormap} -- Colormap (default: {plt.cm.Blues})
        title {str} -- Title of the figure (default: {'Coocurence Matrix'})
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.title(title, size=18)
    plt.colorbar()
    plt.grid(False)
    tick_marks = np.arange(-1, len(classes) + 1)
    plt.xticks(tick_marks, [''] + classes + [''], rotation=80, fontsize=14)
    plt.yticks(tick_marks, [''] + classes + [''], fontsize=14)

    plt.tight_layout()
    plt.ylabel('y', size=16)
    plt.xlabel('x', size=16)


def threshold_and_reweight_matrix(A, t=0.5, p=0.1):
    """
    Applies the matrix thresholding and reweighting such as described in :
    Zhao-Min Chen, Xiu-Shen Wei, Peng Wang, and Yanwen Guo. Multi-label image recognition with graph convolutional networks
    (http://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Multi-Label_Image_Recognition_With_Graph_Convolutional_Networks_CVPR_2019_paper.pdf)
    
    Arguments:
        A {numpy array} -- Matrix of appearance conditional probabilities
    
    Keyword Arguments:
        t {float} -- threshold (default: {0.5})
        p {float} -- reweighting p (default: {0.1})
    
    Returns:
        numpy array -- Reweighted and thesholded matrix
    """
    n_edges = A.shape[0]
    A = (A > t).astype(float)
    np.fill_diagonal(A, 0)
    
    A *= p / (A.sum(0, keepdims=True) + 1e-6)
    np.fill_diagonal(A, 1 - p)
    
    return A


def plot_results(model, dataset, n_plot=10, n_labels=5):
    """
    Plots an image, shows the true labels and the ones predicted with highest probability.
    Only plots images with several labels.
    
    Arguments:
        model {torch model} -- Model to get predictions from
        dataset {torch dataset} -- Dataset to predict on
    
    Keyword Arguments:
        n_plot {int} -- Number of images to plot (default: {10})
        n_labels {int} -- Number of labels to display (default: {3})
    """
    preds = predict(model, dataset)
    
    for i in range(n_plot):
        img, truth = dataset[i]

        if truth.sum() > 1:
        
            pred = preds[i].argsort()[::-1][:n_labels]
            
            img = np.clip(
                img * STD[:, np.newaxis, np.newaxis] + MEAN[:, np.newaxis, np.newaxis], 0, 1
            )
            img = img.transpose(1, 2, 0)
            
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            
            t = ', '.join([CLASSES[i] for i, v in enumerate(truth) if v])
            p = ', '.join([CLASSES[i] for i in pred])
            plt.title(f'Truth : "{t}""  --  Most likely predictions : "{p}""')
            plt.grid(False)
            plt.axis(False)
            plt.show()