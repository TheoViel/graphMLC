from imports import *
from data.transforms import *


def load_image(name, img_path=IMG_PATH):
    img = cv2.imread(IMG_PATH + name, 3)
    b, g, r = cv2.split(img)
    return cv2.merge([r, g, b])   


def get_labels(name, annotation_path=ANNOTATION_PATH):
    tree = ET.parse(annotation_path + name)
    root = tree.getroot()
    labels = []
    for child in root:
        if child.tag == "object":
            label = child[0].text
            labels.append(label)
            
            assert label in CLASSES, f"{label} not in class list"
    return labels


def encode_classes(classes):
    y = np.zeros(len(CLASSES))
    for c in classes:
        y[CLASSES.index(c)] = 1
    return y


class MLCDataset(Dataset):
    def __init__(self, img_names_path, transforms=None, img_path=IMG_PATH, annotation_path=ANNOTATION_PATH):
        super().__init__()

        self.img_path = img_path
        self.transforms = transforms

        self.img_names = []
        with open(img_names_path, 'r') as f:
            self.img_names = [l[:6] for l in f]
        f.close()

        self.classes = [get_labels(name + '.xml', annotation_path=ANNOTATION_PATH) for name in self.img_names]
        self.y = np.array([encode_classes(c) for c in self.classes])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img = load_image(self.img_names[idx] + '.jpg', img_path=self.img_path)

        if not self.transforms is None:
            try: # Albumentations
                img = self.transforms(image=img)['image']
                img = (img / 255 - MEAN) / STD
                img = to_tensor(img)
            except: # Torchvision
                img = self.transforms(img)

        
        return img, self.y[idx]
