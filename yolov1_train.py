import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from yolov1.yolov1 import *
from yolov1.predict import decode_predictions, apply_nms
from torchmetrics.detection import MeanAveragePrecision
import os
import re
import time

pattern = r"Epoch \[(\d+)\] Average Loss: ([\d.]+) mAP50: ([\d.]+) mAP50-95: ([\d.]+)"

def main():
    best_map = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"训练设备: {device}")

    start_epoch = 0
    epochs = 135
    
    metric = MeanAveragePrecision(
        box_format='xyxy', 
        iou_type='bbox',
        backend="faster_coco_eval"
    ).to(device)

    #准备数据
    train_ds = VOCYOLOTrainDataset('datasets/VOC') 
    test_ds = VOCYOLOEvalDataset('datasets/VOC')
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=6,persistent_workers=True,prefetch_factor=2)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=6,pin_memory=True,collate_fn=eval_collate_fn,persistent_workers=True)

    #实例化模型和 Loss
    model = YOLOv1().to(device)
    criterion = YOLOv1Loss().to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    print(f"学习率调度器: AdamW")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # 开启混合精度（拯救 8GB 显存的神器）
    scaler = torch.amp.GradScaler('cuda')
    print(f"使用混合精度")
    
    print(f"共训练: {epochs}轮")

    os.makedirs('./checkpoint',exist_ok=True)
    checkpoint_path = './checkpoint/yolov1.pth'
    if os.path.exists(checkpoint_path):
        print("发现完整存档，正在完美恢复...")
        checkpoint = torch.load(checkpoint_path)
        
        #依次恢复所有状态
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        start_epoch = checkpoint['epoch'] + 1
        print(f"状态恢复完毕，学习率已复原，从 Epoch {start_epoch} 继续。")
        
    for epoch in range(start_epoch,epochs):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for imgs, targets in pbar:
            imgs, targets = imgs.to(device), targets.to(device)

            optimizer.zero_grad()
            
            #混合精度前向
            with torch.amp.autocast('cuda'):
                preds = model(imgs)
                loss, loss_coord, loss_conf, loss_cls = criterion(preds, targets)
            
            #反向与更新
            scaler.scale(loss).backward()
            
            #梯度裁剪
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            pbar.set_postfix({
                'loss': f"{loss.item():.1f}",
                'coord': f"{loss_coord.item():.1f}",
                'conf': f"{loss_conf.item():.1f}"
            })

        model.eval()
        metric.reset()
        with torch.no_grad():
            for imgs, targets_boxes, targets_labels in test_loader:
                imgs = imgs.to(device)
                
                #前向传播
                preds = model(imgs)
                
                batch_preds = []
                #逐张图解码并做 NMS
                for i in range(imgs.size(0)):
                    b, s, l = decode_predictions(preds[i], conf_thresh=0.25)
                    b, s, l = apply_nms(b, s, l, iou_threshold=0.5)
                    
                    batch_preds.append({
                        'boxes': b.to(device), 
                        'scores': s.to(device), 
                        'labels': l.to(device)
                    })
                    
                batch_targets = []
                #组装真实的Targets
                for i in range(imgs.size(0)):
                    batch_targets.append({
                        'boxes': targets_boxes[i].to(device), 
                        'labels': targets_labels[i].to(device)
                    })
                    
                #更新 mAP 统计器
                metric.update(batch_preds, batch_targets)

        #计算并打印当前epoch的mAP
        results = metric.compute()
        map50 = results['map_50'].item()
        map50_95 = results['map'].item()

        scheduler.step()

        if (map50 > best_map) :
            torch.save(model.state_dict(), './saves/yolov1.pth')
            best_map = map50

        print(f"Epoch [{epoch+1}] Average Loss: {epoch_loss/len(train_loader):.2f} mAP50: {map50:.4f} mAP50-95: {map50_95:.4f}")
        with open('./saves/yolov1/log.txt','a') as f :
            f.write(f"Epoch [{epoch+1}] Average Loss: {epoch_loss/len(train_loader):.2f} mAP50: {map50:.4f} mAP50-95: {map50_95:.4f}\n")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
        }, './checkpoint/yolov1.pth')

if __name__ == '__main__':
    main()