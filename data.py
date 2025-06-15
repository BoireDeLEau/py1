import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# 1. 字符集定义（仅支持大写字母 + 数字，0 作为 blank）
CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
BLANK = '-' 

char2idx = {c: i + 1 for i, c in enumerate(CHARS)}
char2idx[BLANK] = 0

idx2char = {i: c for c, i in char2idx.items()}

def text_to_labels(text):
    return [char2idx[c] for c in text if c in char2idx]

def labels_to_text(labels):
    return ''.join([idx2char[i] for i in labels if i != 0])

# 2. 自定义 Dataset
class OCRDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((32, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.samples = []
        label_file = os.path.join(root_dir, "labels.txt")
        with open(label_file, "r", encoding='utf-8') as f:
            for line in f:
                filename, text = line.strip().split(maxsplit=1)
                self.samples.append((filename, text))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, text = self.samples[idx]
        img_path = os.path.join(self.root_dir, filename)
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)

        label = torch.tensor(text_to_labels(text), dtype=torch.long)
        return image, label, len(label)  
