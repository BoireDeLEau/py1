import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from data import OCRDataset, char2idx, idx2char
from ocr import OCRModel
from data import labels_to_text
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  

# 参数
batch_size = 64
epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(char2idx)  

def labels_to_text(labels):
    return ''.join([idx2char[int(i)] for i in labels if int(i) != 0])

def collate_fn(batch):
    images, labels, label_lengths = zip(*batch)
    images = torch.stack(images)
    label_lengths = torch.tensor(label_lengths, dtype=torch.long)
    labels = pad_sequence(labels, batch_first=True, padding_value=0) 
    return images, labels, label_lengths

# 解码函数
def greedy_decode(log_probs):
    pred = log_probs.argmax(2).permute(1, 0) 
    results = []
    for p in pred:
        out = []
        prev = 0
        for ch in p:
            ch = ch.item()
            if ch != prev and ch != 0:
                out.append(ch)
            prev = ch
        results.append(labels_to_text(out))
    return results

# 数据加载
train_dataset = OCRDataset("dataset")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# 初始化模型、损失、优化器、调度器
model = OCRModel(num_classes=num_classes).to(device)
criterion = nn.CTCLoss(blank=0, zero_infinity=True)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

# 最佳损失
best_loss = float('inf')

# 训练循环
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for images, labels, label_lengths in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        label_lengths = label_lengths.to(device)

        optimizer.zero_grad()
        logits = model(images) 
        log_probs = logits.log_softmax(2)

        input_lengths = torch.full(size=(log_probs.size(1),), fill_value=log_probs.size(0), dtype=torch.long).to(device)
        
        if torch.any(input_lengths < label_lengths):
            print("[Warning] Skipping a batch due to input_len < label_len")
            continue  
        
        loss = criterion(log_probs, labels, input_lengths, label_lengths)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()
        total_loss += loss.item()

    scheduler.step()
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

    # 保存最优模型
    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "ocr_best.pt")

