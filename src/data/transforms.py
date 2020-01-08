from params import *
from imports import *


def to_tensor(x, **kwargs):
    """
    Swaps the channels and converts an image to float
    
    Arguments:
        x {numpy array} -- Image
    
    Returns:
        numpy array -- Transformed Image
    """
    return x.transpose(2, 0, 1).astype("float32")


def get_transfos(size=256, test=False, **kwargs):
    """
    Returns the transformations
    
    Keyword Arguments:
        size {int} -- Image size (default: {256})
        test {bool} -- Whether to return test time transforms (default: {False})  
    
    Returns:
        albumentations transforms -- Transforms to apply
    """
    if not test:
        transforms = albu.Compose(
            [
                albu.Resize(size, size, always_apply=True),
                albu.OneOf(
                    [
                        albu.RandomContrast(),
                        albu.RandomGamma(),
                        albu.RandomBrightness(),
                        albu.CLAHE(),
                    ],
                    p=0.3,
                ),
                albu.ShiftScaleRotate(
                    shift_limit=0, scale_limit=0, rotate_limit=45, p=0.5,
                ),
                albu.HorizontalFlip(p=0.5),
            ]
        )
    else:
        transforms = albu.Compose([albu.Resize(size, size, always_apply=True),])

    return transforms


def get_transfos_lssg(size=224, scale_size=640, test=False):
    """
    Transforms applied in the paper : 
    Tianshui Chen, Muxin Xu, Xiaolu Hui, Hefeng Wu, and Liang Lin. Learning semantic-specific graphrepresentation for multi-label image recognition
    (http://openaccess.thecvf.com/content_ICCV_2019/papers/Chen_Learning_Semantic-Specific_Graph_Representation_for_Multi-Label_Image_Recognition_ICCV_2019_paper.pdf)
    
    Keyword Arguments:
        size {int} -- Image size (default: {224})
        scale_size {int} -- Size to crop from (default: {640})
        test {bool} -- Whether to return test time transforms (default: {False})
    
    Returns:
        torch transforms -- Transforms to apply
    """

    if not test:
        transfos = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((scale_size, scale_size)),
                transforms.RandomChoice(
                    [
                        transforms.RandomCrop(640),
                        transforms.RandomCrop(576),
                        transforms.RandomCrop(512),
                        transforms.RandomCrop(384),
                        transforms.RandomCrop(320),
                    ]
                ),
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )
    else:
        transfos = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((scale_size, scale_size)),
                transforms.CenterCrop(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ]
        )
    return transfos
