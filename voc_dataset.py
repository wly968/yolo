import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor',
]
NUM_CLASSES = len(VOC_CLASSES) + 1


class YOLOFormatDatasetFRCNN(Dataset):
    def __init__(self, root, splits, transforms=None):
        """
        splits: list，如 ['train2007', 'train2012', 'val2007', 'val2012']
        """
        self.samples = []
        for split in splits:
            img_dir = os.path.join(root, 'images', split)
            label_dir = os.path.join(root, 'labels', split)
            if not os.path.isdir(img_dir):
                print(f"[WARN] {img_dir} 不存在，跳过")
                continue
            valid_exts = ('.jpg', '.jpeg', '.png')
            for name in os.listdir(img_dir):
                if name.lower().endswith(valid_exts):
                    txt = os.path.splitext(name)[0] + '.txt'
                    if os.path.exists(os.path.join(label_dir, txt)):
                        self.samples.append((img_dir, label_dir, name))
        assert len(self.samples) > 0, f"在 {root} 下未找到有效样本！"
        print(f"[Dataset] {splits} -> {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_dir, label_dir, img_name = self.samples[idx]
        img_path = os.path.join(img_dir, img_name)
        txt_path = os.path.join(label_dir, os.path.splitext(img_name)[0] + '.txt')

        img = Image.open(img_path).convert('RGB')
        W, H = img.size

        boxes, labels = [], []
        with open(txt_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                x1 = max(0.0, (cx - w / 2) * W)
                y1 = max(0.0, (cy - h / 2) * H)
                x2 = min(float(W), (cx + w / 2) * W)
                y2 = min(float(H), (cy + h / 2) * H)
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(cls_id + 1)

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            'labels': torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64),
        }
        img = F.pil_to_tensor(img).float() / 255.0
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))