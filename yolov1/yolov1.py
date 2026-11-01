import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch.nn as nn
import torchvision.models as models

class VOCYOLOTrainDataset(Dataset):
    def __init__(self,root,S=7,B=2,C=20):
        self.root = root

        self.S = S
        self.B = B
        self.C = C

        self.img_paths = []
        self.label_paths = []

        for split in ['train2007', 'train2012', 'val2007', 'val2012']:
            img_dir = os.path.join(root, 'images', split)
            label_dir = os.path.join(root, 'labels', split)
            
            for f in os.listdir(img_dir):
                if f.endswith('.jpg'):
                    self.img_paths.append(os.path.join(img_dir, f))
                    self.label_paths.append(os.path.join(label_dir, f.replace('.jpg', '.txt')))

        self.transform = T.Compose([
            T.Resize((448, 448)),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
            T.ToTensor()
        ])
        return
    
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self,idx):
        img_path = self.img_paths[idx]
        txt_path = self.label_paths[idx]

        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img)
        target = torch.zeros((self.S,self.S,self.B*5+self.C))

        if (os.path.exists(txt_path)):
            with open(txt_path,'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5:continue

                    cls_id,cx,cy,w,h = map(float,parts)
                    cls_id =  int(cls_id)

                    grid_x = int(cx * self.S)
                    grid_y = int(cy * self.S)
                    
                    #防止因为浮点数精度导致cx=1.0时越界！！！
                    grid_x = min(grid_x, self.S - 1)
                    grid_y = min(grid_y, self.S - 1)

                    x_offset = cx * self.S - grid_x
                    y_offset = cy * self.S - grid_y

                    if target[grid_y, grid_x, 4] == 0:
                        target[grid_y, grid_x, 0] = x_offset
                        target[grid_y, grid_x, 1] = y_offset
                        target[grid_y, grid_x, 2] = w
                        target[grid_y, grid_x, 3] = h
                        target[grid_y, grid_x, 4] = 1.0
                        
                        # 坐标和真实物体一模一样
                        target[grid_y, grid_x, 5] = x_offset
                        target[grid_y, grid_x, 6] = y_offset
                        target[grid_y, grid_x, 7] = w
                        target[grid_y, grid_x, 8] = h
                        target[grid_y, grid_x, 9] = 1.0
                        
                        target[grid_y, grid_x, 10 + cls_id] = 1.0

        return img_tensor, target

class VOCYOLOEvalDataset(Dataset):
    def __init__(self, root, split='test2007'):
        self.img_dir = os.path.join(root, 'images', split)
        self.label_dir = os.path.join(root, 'labels', split)
        
        self.img_names = [f for f in os.listdir(self.img_dir) if f.endswith('.jpg')]
        self.img_names.sort()
        
        self.transform = T.Compose([
            T.Resize((448, 448)),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, img_name)
        txt_path = os.path.join(self.label_dir, os.path.splitext(img_name)[0] + '.txt')

        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        img_tensor = self.transform(img)

        boxes = []
        labels = []

        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) != 5: continue
                    
                    cls_id, cx, cy, w, h = map(float, parts)
                    
                    # 转回绝对的 x1, y1, x2, y2 (基于原图尺寸)
                    x1 = cx - w / 2.0
                    y1 = cy - h / 2.0
                    x2 = cx + w / 2.0
                    y2 = cy + h / 2.0
                    
                    boxes.append([x1, y1, x2, y2])
                    labels.append(int(cls_id))

        # 如果没有框，返回空张量
        if len(boxes) == 0:
            boxes_tensor = torch.zeros((0, 4))
            labels_tensor = torch.zeros((0,), dtype=torch.long)
        else:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.tensor(labels, dtype=torch.long)

        return img_tensor, boxes_tensor, labels_tensor

class YOLOv1(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YOLOv1, self).__init__()
        self.S = S
        self.B = B
        self.C = C
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, S * S * (B * 5 + C))
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        x = x.view(-1, self.S, self.S, self.B * 5 + self.C)

        xy1 = torch.clamp(x[..., 0:2], min=-1.0, max=1.0)
        wh1 = torch.clamp(x[..., 2:4], min=0.0, max=1.0)
        conf1 = torch.sigmoid(x[..., 4:5])
        
        xy2 = torch.clamp(x[..., 5:7], min=-1.0, max=1.0)
        wh2 = torch.clamp(x[..., 7:9], min=0.0, max=1.0)
        conf2 = torch.sigmoid(x[..., 9:10])
        
        classes = torch.sigmoid(x[..., 10:])
        
        return torch.cat([xy1, wh1, conf1, xy2, wh2, conf2, classes], dim=-1)

class YOLOv1Loss(nn.Module):
    def __init__(self,S=7,B=2,C=20):
        super(YOLOv1Loss,self).__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5
        self.mse = nn.MSELoss(reduction="sum")

    def forward(self,predictions,target):
        pred_boxes = predictions[...,:10].reshape(-1,self.S,self.S,2,5)
        pred_box1 = pred_boxes[...,0,:4]
        pred_box2 = pred_boxes[...,1,:4]

        pred_conf1 = predictions[...,4]
        pred_conf2 = predictions[...,9]

        pred_classes = predictions[...,10:]

        target_box = target[...,:4]
        target_cls = target[...,10:]

        obj_mask = target[...,4]>0
        noobj_mask = ~obj_mask

        iou_b1 = self.calculate_iou(pred_box1, target_box)
        iou_b2 = self.calculate_iou(pred_box2, target_box)

        #选择iou更大的框
        ious = torch.stack([iou_b1, iou_b2], dim=-1)
        best_box_idx = torch.argmax(ious, dim=-1)

        resp_mask1 = obj_mask & (best_box_idx == 0)
        resp_mask2 = obj_mask & (best_box_idx == 1)

        loss_xy1 = self.mse(pred_box1[resp_mask1][..., :2], target_box[resp_mask1][..., :2])
        loss_wh1 = self.mse(torch.sqrt(torch.clamp(pred_box1[resp_mask1][..., 2:], min=0) + 1e-6),torch.sqrt(target_box[resp_mask1][..., 2:] + 1e-6))
        loss_xy2 = self.mse(pred_box2[resp_mask2][..., :2], target_box[resp_mask2][..., :2])
        loss_wh2 = self.mse(torch.sqrt(torch.clamp(pred_box2[resp_mask2][..., 2:], min=0) + 1e-6),torch.sqrt(target_box[resp_mask2][..., 2:] + 1e-6))
        
        loss_coord = self.lambda_coord * (loss_xy1 + loss_wh1 + loss_xy2 + loss_wh2)
  
        loss_conf_obj = self.mse(pred_conf1[resp_mask1], target[..., 4][resp_mask1]) + \
                        self.mse(pred_conf2[resp_mask2], target[..., 4][resp_mask2])
        not_resp_mask = ~(resp_mask1 | resp_mask2)
        loss_conf_noobj = self.mse(pred_conf1[not_resp_mask], torch.zeros_like(pred_conf1[not_resp_mask])) + \
                          self.mse(pred_conf2[not_resp_mask], torch.zeros_like(pred_conf2[not_resp_mask]))
        
        loss_conf = loss_conf_obj + self.lambda_noobj * loss_conf_noobj
        #分类损失
        loss_class = self.mse(pred_classes[obj_mask], target_cls[obj_mask])
        #总损失
        total_loss = loss_coord + loss_conf + loss_class
        
        return total_loss, loss_coord, loss_conf, loss_class


    def calculate_iou(self, box1, box2):
        b1_x1 = box1[..., 0] - box1[..., 2] / 2
        b1_y1 = box1[..., 1] - box1[..., 3] / 2
        b1_x2 = box1[..., 0] + box1[..., 2] / 2
        b1_y2 = box1[..., 1] + box1[..., 3] / 2
        
        b2_x1 = box2[..., 0] - box2[..., 2] / 2
        b2_y1 = box2[..., 1] - box2[..., 3] / 2
        b2_x2 = box2[..., 0] + box2[..., 2] / 2
        b2_y2 = box2[..., 1] + box2[..., 3] / 2

        inter_x1 = torch.max(b1_x1, b2_x1)
        inter_y1 = torch.max(b1_y1, b2_y1)
        inter_x2 = torch.min(b1_x2, b2_x2)
        inter_y2 = torch.min(b1_y2, b2_y2)

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        b1_area = torch.clamp(b1_x2 - b1_x1, min=0) * torch.clamp(b1_y2 - b1_y1, min=0)
        b2_area = torch.clamp(b2_x2 - b2_x1, min=0) * torch.clamp(b2_y2 - b2_y1, min=0)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-6)
        return iou

def eval_collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch], dim=0) # 图片可以拼
    boxes = [item[1] for item in batch]                    # 框保持列表
    labels = [item[2] for item in batch]                   # 标签保持列表
    return imgs, boxes, labels