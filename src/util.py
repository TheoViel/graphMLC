from imports import *
from params import *


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True #False


def save_model_weights(model, filename, verbose=1, cp_folder=CP_PATH):
    if verbose:
        print(f"\n -> Saving weights to {os.path.join(cp_folder,filename)}\n")
    torch.save(model.state_dict(), os.path.join(cp_folder, filename))


def load_model_weights(model, filename, verbose=1, cp_folder=CP_PATH, strict=True):
    if verbose:
        print(f"\n -> Loading weights from {os.path.join(cp_folder,filename)}\n")
    try:
        model.load_state_dict(os.path.join(cp_folder, filename), strict=strict)
    except BaseException:
        model.load_state_dict(
            torch.load(os.path.join(cp_folder, filename), map_location="cpu"),
            strict=strict,
        )
    return model


def count_parameters(model, all=False):
    if all:
        return sum(p.numel() for p in model.parameters())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def plot_attention_ssgrl(model, dataset, idx):
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
    fig, ax = plt.subplots(figsize=(12, 8))

    plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.title(title, size=15)
    plt.colorbar()
    plt.grid(False)
    tick_marks = np.arange(-1, len(classes) + 1)
    plt.xticks(tick_marks, [''] + classes + [''], rotation=80)
    plt.yticks(tick_marks, [''] + classes + [''])

    plt.tight_layout()
    plt.ylabel('True label', size=12)
    plt.xlabel('Predicted label', size=12)


def threshold_and_smooth_matrix(A, t=0.5, p=0.1):
    n_edges = A.shape[0]
    A = (A > t).astype(float)
    np.fill_diagonal(A, 0)
    
    A *= p / (A.sum(0, keepdims=True) + 1e-6)
    np.fill_diagonal(A, 1 - p)
    
    return A