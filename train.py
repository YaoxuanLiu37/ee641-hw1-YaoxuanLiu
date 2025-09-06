import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import json

from pathlib import Path
from dataset import ShapeDetectionDataset
from model import MultiScaleDetector
from loss import DetectionLoss
from utils import generate_anchors

ANCHORS = None

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total = 0.0
    count = 0
    for imgs, targets in dataloader:
        imgs = imgs.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss, _ = criterion(preds, targets, ANCHORS)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
        count += 1
    return total / max(1, count)

def validate(model, dataloader, criterion, device):
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for imgs, targets in dataloader:
            imgs = imgs.to(device)
            preds = model(imgs)
            loss, _ = criterion(preds, targets, ANCHORS)
            total += float(loss.item())
            count += 1
    return total / max(1, count)

def main():
    # Configuration
    batch_size = 16
    learning_rate = 0.001
    num_epochs = 50
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Datasets & loaders 
    data_root = Path("/content/datasets/detection")
    images_train = data_root / "train"
    ann_train = data_root / "train_annotations.json"
    images_val = data_root / "val"
    ann_val = data_root / "val_annotations.json"

    train_set = ShapeDetectionDataset(str(images_train), str(ann_train))
    val_set = ShapeDetectionDataset(str(images_val), str(ann_val))
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: (torch.stack([x[0] for x in b], dim=0), [x[1] for x in b]),
        num_workers=0
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: (torch.stack([x[0] for x in b], dim=0), [x[1] for x in b]),
        num_workers=0
    )

    # Model, loss, optimizer
    model = MultiScaleDetector(num_classes=3, num_anchors=3).to(device)
    criterion = DetectionLoss(num_classes=3)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # Build anchors for 3 scales using the model's actual feature-map sizes
    dummy = torch.zeros(1, 3, 224, 224).to(device)
    with torch.no_grad():
        outs = model(dummy)
    feat_sizes = [(o.shape[2], o.shape[3]) for o in outs]  
    anchor_scales = [[16, 24, 32], [48, 64, 96], [96, 128, 192]]
    global ANCHORS
    ANCHORS = generate_anchors(feat_sizes, anchor_scales, image_size=224)

    # Results paths
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / "training_log.json"
    best_model_path = results_dir / "best_model.pth"

    # Train
    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    for epoch in range(1, num_epochs + 1):
        tr = train_epoch(model, train_loader, criterion, optimizer, device)
        va = validate(model, val_loader, criterion, device)

        history["train_loss"].append(tr)
        history["val_loss"].append(va)
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

        if va < best_val:
            best_val = va
            torch.save(model.state_dict(), best_model_path)

        print(f"Epoch {epoch:03d}/{num_epochs} - train {tr:.4f} - val {va:.4f} - best {best_val:.4f}")

if __name__ == '__main__':
    main()
