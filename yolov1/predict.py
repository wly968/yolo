import torch
import torchvision.transforms as T
from PIL import Image,ImageDraw
from torchvision.ops import nms

from yolov1 import YOLOv1

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

def decode_predictions(preds,S=7,B=2,C=20,conf_thresh=0.1):
    boxes_list = []
    scores_list = []
    labels_list = []

    for grid_y in range(S):
        for grid_x in range(S):
            cell = preds[grid_y, grid_x]
            
            # 寻找这个格子里的 2 个预测框
            for b in range(B):
                start_idx = b * 5
                x, y = cell[start_idx], cell[start_idx + 1]
                w, h = cell[start_idx + 2], cell[start_idx + 3]
                box_conf = cell[start_idx + 4]
                
                if box_conf < conf_thresh:
                    continue
                
                # 找类别（取 20 个类别中概率最大的）
                cls_probs = cell[10:]
                cls_id = torch.argmax(cls_probs)
                cls_conf = cls_probs[cls_id]
                score = box_conf * cls_conf  # 最终得分
                
                # 转换为绝对坐标 (相对整张图 0~1)
                abs_x = (grid_x + x) / S
                abs_y = (grid_y + y) / S
                abs_w = torch.clamp(w, min=0.0,max=1.0) # 防止负的宽高
                abs_h = torch.clamp(h, min=0.0,max=1.0)
                
                # 转为 [x1, y1, x2, y2]
                x1 = abs_x - abs_w / 2
                y1 = abs_y - abs_h / 2
                x2 = abs_x + abs_w / 2
                y2 = abs_y + abs_h / 2
                
                boxes_list.append([x1, y1, x2, y2])
                scores_list.append(score)
                labels_list.append(cls_id)

    if len(boxes_list) == 0:
        return torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,), dtype=torch.long)
        
    return torch.tensor(boxes_list), torch.tensor(scores_list), torch.tensor(labels_list)

def apply_nms(boxes, scores, labels, iou_threshold=0.5):
    if len(boxes) == 0:
        # 确保返回正确的 shape
        return torch.empty((0, 4)), torch.empty((0,)), torch.empty((0,), dtype=torch.long)

    keep_indices = nms(boxes, scores, iou_threshold)
    return boxes[keep_indices], scores[keep_indices], labels[keep_indices]

def draw_boxes(img_pil, boxes, scores, labels):
    '''
    将结果绘制到图片上
    '''
    draw = ImageDraw.Draw(img_pil)
    W, H = img_pil.size

    for box, score, label in zip(boxes, scores, labels):
        # 还原到原图的绝对像素坐标
        x1, y1, x2, y2 = box[0].item() * W, box[1].item() * H, box[2].item() * W, box[3].item() * H
        cls_name = VOC_CLASSES[label.item()]
        
        # 画框
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        # 画标签
        text = f"{cls_name} {score.item():.2f}"
        draw.text((x1, max(0, y1 - 10)), text, fill="red")
    
    return img_pil

def predict_image(img_path, model, device='cuda'):
    '''
    预测一张图片的全流程
    '''
    img_pil = Image.open(img_path).convert('RGB')
    transform = T.Compose([T.Resize((448, 448)), T.ToTensor()])
    img_tensor = transform(img_pil).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        preds = model(img_tensor)
    
    preds = preds[0]

    # 3. 解码与 NMS
    boxes, scores, labels = decode_predictions(preds, conf_thresh=0.3)
    boxes, scores, labels = apply_nms(boxes, scores, labels, iou_threshold=0.5)

    # 4. 画图并展示
    result_img = draw_boxes(img_pil, boxes, scores, labels)
    result_img.show()
    print(f"检测到 {len(boxes)} 个目标！")

if __name__ == '__main__':
    import os
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = YOLOv1().to(device)
    
    weights_path = './saves/yolov1.pth' 
    if os.path.exists(weights_path):
        # 忽略一些无关紧要的键匹配警告
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
        print(f"成功加载模型权重: {weights_path}")
    else:
        print(f"找不到 {weights_path}，将使用随机初始化的模型瞎猜！")
        
    test_img = 'datasets/VOC/images/test2007/000004.jpg' 
    
    if os.path.exists(test_img):
        print(f"正在预测图片: {test_img}")
        predict_image(test_img, model, device)
    else:
        print(f"❌ 找不到图片 {test_img}，请检查路径！")