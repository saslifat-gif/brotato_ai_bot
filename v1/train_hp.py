import argparse
import csv
import glob
import os

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class HPRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 8 * 40, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


class HPDataset(Dataset):
    def __init__(self, data_dir: str, labels_file: str = "labels.csv"):
        self.data_dir = data_dir
        self.samples = self._load_samples(data_dir, labels_file)
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((32, 160)),
                transforms.ToTensor(),
            ]
        )

    @staticmethod
    def _parse_label_from_name(path: str) -> float:
        filename = os.path.basename(path)
        parts = filename.split("_")
        if len(parts) >= 2:
            try:
                return float(parts[1])
            except ValueError:
                return 0.0
        return 0.0

    def _load_samples(self, data_dir: str, labels_file: str):
        labels_path = os.path.join(data_dir, labels_file)
        samples = []

        if os.path.exists(labels_path):
            with open(labels_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    image_name = (row.get("image") or "").strip()
                    if not image_name:
                        continue
                    ratio_raw = (row.get("ratio") or "0").strip()
                    try:
                        ratio = float(ratio_raw)
                    except ValueError:
                        continue
                    image_path = os.path.join(data_dir, image_name)
                    if os.path.exists(image_path):
                        samples.append((image_path, ratio))

        if not samples:
            for p in glob.glob(os.path.join(data_dir, "*.jpg")):
                samples.append((p, self._parse_label_from_name(p)))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, label = self.samples[idx]
        img = cv2.imread(image_path)
        img_tensor = self.transform(img)
        return img_tensor, torch.tensor([label], dtype=torch.float32)


def train(data_dir: str, epochs: int, batch_size: int, lr: float):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = HPDataset(data_dir)
    if len(dataset) == 0:
        print(f"[error] Dataset is empty: {data_dir}")
        print("Expected: .jpg files and/or labels.csv with columns: image,ratio")
        return

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = HPRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"Start training with {len(dataset)} samples...")
    best_loss = float("inf")
    for epoch in range(epochs):
        total_loss = 0.0
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / max(1, len(dataloader))
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "hp_model_best.pth")
            
    print(f"Training done. Best Loss: {best_loss:.6f}. Saved to hp_model_best.pth")


def parse_args():
    parser = argparse.ArgumentParser(description="Train HP regression model.")
    parser.add_argument("--data-dir", default="visual_regression_dataset", help="Dataset directory")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
