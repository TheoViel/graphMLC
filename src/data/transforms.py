from params import *
from imports import *


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_transfos(test=False, size=256, **kwargs):
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
