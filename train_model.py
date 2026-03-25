"""
ASD Model Training Script
==========================
Fine-tunes MobileNetV2 for ASD visual behavioral classification.

Supported datasets:
- Custom dataset (images organized as asd/ and non_asd/ folders)
- Kaggle ASD Screening Image Dataset
- Autism-specific behavioral image datasets

Target accuracy: 85-90% (aligned with paper benchmarks)

Usage:
    python train_model.py --data_dir ./data/images --epochs 30 --batch_size 16
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, accuracy_score)
from sklearn.model_selection import StratifiedKFold
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.mobilenet_asd import ASDVisualFeatureExtractor


class ASDImageDataset(Dataset):
    """
    Dataset for ASD image classification.

    Expected structure:
        data_dir/
            asd/
                image1.jpg
                image2.jpg
                ...
            non_asd/
                image1.jpg
                ...

    Or alternatively flat with CSV labels.
    """

    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    def __init__(self, data_dir: str, transform=None, augment: bool = False):
        self.data_dir = data_dir
        self.transform = transform
        self.augment = augment
        self.samples = []
        self.labels = []
        self.class_names = ['non_asd', 'asd']

        # Try folder structure first
        asd_dir = os.path.join(data_dir, 'asd')
        non_asd_dir = os.path.join(data_dir, 'non_asd')

        if os.path.exists(asd_dir) and os.path.exists(non_asd_dir):
            self._load_from_folders(asd_dir, non_asd_dir)
        else:
            # Try flat structure with labels.json or labels.csv
            self._load_from_labels_file(data_dir)

        print(f"Dataset loaded: {len(self.samples)} samples "
              f"({sum(self.labels)} ASD, {len(self.labels) - sum(self.labels)} non-ASD)")

    def _load_from_folders(self, asd_dir: str, non_asd_dir: str):
        """Load from asd/ and non_asd/ folder structure."""
        for img_path in self._get_images(asd_dir):
            self.samples.append(img_path)
            self.labels.append(1)
        for img_path in self._get_images(non_asd_dir):
            self.samples.append(img_path)
            self.labels.append(0)

    def _load_from_labels_file(self, data_dir: str):
        """Load from flat folder with labels file."""
        labels_file = os.path.join(data_dir, 'labels.json')
        if os.path.exists(labels_file):
            with open(labels_file) as f:
                labels_data = json.load(f)
            for item in labels_data:
                path = os.path.join(data_dir, item['filename'])
                if os.path.exists(path):
                    self.samples.append(path)
                    self.labels.append(item['label'])

    def _get_images(self, folder: str):
        """Recursively get all image files from folder."""
        images = []
        for root, _, files in os.walk(folder):
            for f in files:
                if os.path.splitext(f)[1].lower() in self.EXTENSIONS:
                    images.append(os.path.join(root, f))
        return sorted(images)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced datasets."""
        labels = np.array(self.labels)
        counts = np.bincount(labels)
        weights = 1.0 / counts
        weights = weights / weights.sum()
        return torch.FloatTensor(weights)

    def get_sample_weights(self) -> list:
        """Get per-sample weights for WeightedRandomSampler."""
        class_weights = self.get_class_weights().numpy()
        return [class_weights[label] for label in self.labels]


def get_transforms(augment: bool = False):
    """Get image transforms for training/validation."""

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


class Trainer:
    """
    Training loop for ASD MobileNetV2 model.
    Implements:
    - Stratified train/val split
    - Learning rate scheduling
    - Early stopping
    - Class-weighted loss
    - Comprehensive metrics logging
    """

    def __init__(self, model, device, save_dir: str = './models/checkpoints'):
        self.model = model
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.best_val_acc = 0.0
        self.best_model_path = None

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30,
        lr: float = 1e-3,
        class_weights: torch.Tensor = None,
        patience: int = 8
    ):
        """Main training loop."""

        # Loss with class weighting for imbalanced data
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()

        # Optimizer: different LR for backbone vs head
        backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]
        head_params = list(self.model.feature_head.parameters()) + \
                      list(self.model.classifier.parameters())

        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': lr * 0.1},
            {'params': head_params, 'lr': lr}
        ], weight_decay=1e-4)

        # Cosine annealing LR scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        patience_counter = 0

        print(f"\n{'='*60}")
        print(f"Training MobileNetV2-ASD Classifier")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {epochs}")
        print(f"  LR: {lr}")
        print(f"  Train samples: {len(train_loader.dataset)}")
        print(f"  Val samples: {len(val_loader.dataset)}")
        print(f"{'='*60}\n")

        for epoch in range(epochs):
            t0 = time.time()

            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader, optimizer, criterion)

            # Validation phase
            val_loss, val_acc, val_preds, val_targets = self._val_epoch(val_loader, criterion)

            scheduler.step()

            # Record history
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)

            epoch_time = time.time() - t0

            print(f"Epoch [{epoch+1:3d}/{epochs}] | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} | "
                  f"Time: {epoch_time:.1f}s")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_path = os.path.join(
                    self.save_dir, f'best_model_acc{val_acc:.3f}.pth'
                )
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"  ✓ New best model saved: {self.best_model_path}")
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

            # Detailed metrics every 5 epochs
            if (epoch + 1) % 5 == 0:
                print(f"\n  Classification Report (Val):")
                print(classification_report(val_targets, val_preds,
                                            target_names=['non_ASD', 'ASD'], zero_division=0))

        print(f"\n{'='*60}")
        print(f"Training complete. Best Val Accuracy: {self.best_val_acc:.4f}")
        print(f"Best model: {self.best_model_path}")

        return self.best_model_path

    def _train_epoch(self, loader, optimizer, criterion):
        """Single training epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            optimizer.zero_grad()
            logits, _ = self.model(images)
            loss = criterion(logits, labels)
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return total_loss / len(loader), correct / total

    def _val_epoch(self, loader, criterion):
        """Single validation epoch."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits, _ = self.model(images)
                loss = criterion(logits, labels)

                total_loss += loss.item()
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        acc = accuracy_score(all_targets, all_preds)
        return total_loss / len(loader), acc, all_preds, all_targets

    def plot_history(self, save_path: str = None):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_title('Loss History')
        ax1.set_xlabel('Epoch')
        ax1.legend()

        ax2.plot(self.train_accs, label='Train Acc')
        ax2.plot(self.val_accs, label='Val Acc')
        ax2.axhline(0.85, color='g', linestyle='--', label='Target 85%')
        ax2.axhline(0.90, color='r', linestyle='--', label='Target 90%')
        ax2.set_title('Accuracy History')
        ax2.set_xlabel('Epoch')
        ax2.legend()

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"History plot saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def evaluate_final(self, test_loader: DataLoader):
        """Final evaluation with full metrics."""
        self.model.eval()
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                logits, _ = self.model(images)
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())

        acc = accuracy_score(all_targets, all_preds)
        try:
            auc = roc_auc_score(all_targets, all_probs)
        except Exception:
            auc = 0.0

        print(f"\nFinal Test Results:")
        print(f"  Accuracy: {acc:.4f} ({acc*100:.1f}%)")
        print(f"  AUC-ROC:  {auc:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(all_targets, all_preds,
                                    target_names=['non_ASD', 'ASD']))
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(all_targets, all_preds)
        print(cm)

        return {'accuracy': acc, 'auc': auc}


def main():
    parser = argparse.ArgumentParser(description='Train ASD MobileNetV2 Model')
    parser.add_argument('--data_dir', type=str, default='./data/images',
                        help='Path to image dataset (asd/ and non_asd/ folders)')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--val_split', type=float, default=0.2)
    parser.add_argument('--save_dir', type=str, default='./models/checkpoints')
    parser.add_argument('--feature_dim', type=int, default=128)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Check dataset exists
    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory not found: {args.data_dir}")
        print("\nExpected structure:")
        print("  data/images/")
        print("    asd/       <- ASD behavioral images")
        print("    non_asd/   <- Non-ASD images")
        print("\nSee SETUP_GUIDE.md for dataset acquisition instructions.")
        return

    # Load dataset
    train_transform, val_transform = get_transforms(augment=True)
    full_dataset = ASDImageDataset(args.data_dir, transform=train_transform)

    if len(full_dataset) == 0:
        print("ERROR: No images found in dataset directory")
        return

    # Train/val/test split
    n = len(full_dataset)
    n_val = int(n * args.val_split)
    n_test = int(n * 0.1)
    n_train = n - n_val - n_test

    train_set, val_set, test_set = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    # Override transforms for val/test (no augmentation)
    val_set.dataset.transform = val_transform

    print(f"Split: {n_train} train / {n_val} val / {n_test} test")

    # Weighted sampler for imbalanced data
    sample_weights = full_dataset.get_sample_weights()
    train_weights = [sample_weights[i] for i in train_set.indices]
    sampler = WeightedRandomSampler(train_weights, len(train_weights), replacement=True)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                               sampler=sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                             shuffle=False, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=args.batch_size,
                              shuffle=False, num_workers=2)

    # Initialize model
    model = ASDVisualFeatureExtractor(
        num_classes=2,
        feature_dim=args.feature_dim,
        pretrained=True
    ).to(device)

    # Train
    trainer = Trainer(model, device, save_dir=args.save_dir)
    class_weights = full_dataset.get_class_weights()
    best_path = trainer.train(
        train_loader, val_loader,
        epochs=args.epochs,
        lr=args.lr,
        class_weights=class_weights,
        patience=8
    )

    # Final evaluation
    if best_path and os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    trainer.evaluate_final(test_loader)

    # Plot history
    trainer.plot_history(save_path=os.path.join(args.save_dir, 'training_history.png'))

    print(f"\nDone. Model saved to {best_path}")


if __name__ == "__main__":
    main()
