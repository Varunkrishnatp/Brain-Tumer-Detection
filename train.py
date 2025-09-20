import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import ResNet34_Weights
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(image_size: int) -> Tuple[transforms.Compose, transforms.Compose]:
	train_tfms = transforms.Compose([
		transforms.Resize((image_size, image_size)),
		transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET),
		transforms.ToTensor(),
		transforms.Normalize(mean=IMAGENET_MEAN,
							 std=IMAGENET_STD),
	])
	eval_tfms = transforms.Compose([
		transforms.Resize((image_size, image_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=IMAGENET_MEAN,
							 std=IMAGENET_STD),
	])
	return train_tfms, eval_tfms


def create_dataloaders(train_dir: str, val_dir: str, image_size: int, batch_size: int, num_workers: int) -> Tuple[DataLoader, DataLoader, Dict[int, str]]:
	train_tfms, eval_tfms = build_transforms(image_size)
	train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
	val_ds = datasets.ImageFolder(val_dir, transform=eval_tfms)
	idx_to_class = {idx: cls for cls, idx in train_ds.class_to_idx.items()}
	train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
	val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
	return train_loader, val_loader, idx_to_class


def build_model(num_classes: int) -> nn.Module:
	weights = ResNet34_Weights.DEFAULT
	model = models.resnet34(weights=weights)
	in_features = model.fc.in_features
	model.fc = nn.Sequential(
		nn.Dropout(0.3),
		nn.Linear(in_features, num_classes)
	)
	return model


def train_one_epoch(model, loader, criterion, optimizer, device):
	model.train()
	running_loss = 0.0
	running_correct = 0
	total = 0
	for images, labels in tqdm(loader, desc="Train", leave=False):
		images = images.to(device, non_blocking=True)
		labels = labels.to(device, non_blocking=True)
		optimizer.zero_grad(set_to_none=True)
		outputs = model(images)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		running_loss += loss.item() * images.size(0)
		_, preds = torch.max(outputs, 1)
		running_correct += (preds == labels).sum().item()
		total += images.size(0)
	return running_loss / total, running_correct / total


def evaluate(model, loader, criterion, device):
	model.eval()
	running_loss = 0.0
	running_correct = 0
	total = 0
	@torch.no_grad()
	def _eval_batch(images, labels):
		outputs = model(images)
		loss = criterion(outputs, labels)
		_, preds = torch.max(outputs, 1)
		return loss, preds
	with torch.no_grad():
		for images, labels in tqdm(loader, desc="Val", leave=False):
			images = images.to(device, non_blocking=True)
			labels = labels.to(device, non_blocking=True)
			loss, preds = _eval_batch(images, labels)
			running_loss += loss.item() * images.size(0)
			running_correct += (preds == labels).sum().item()
			total += images.size(0)
	return running_loss / total, running_correct / total


def save_checkpoint(model, idx_to_class: Dict[int, str], out_path: str):
	to_save = {
		"state_dict": model.state_dict(),
		"idx_to_class": idx_to_class,
		"architecture": "resnet34",
	}
	Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
	torch.save(to_save, out_path)


def main():
	parser = argparse.ArgumentParser(description="Train brain tumor classifier")
	parser.add_argument("--train_dir", default="Training", type=str, help="Path to Training/ directory")
	parser.add_argument("--val_dir", default="Testing", type=str, help="Path to Testing/ directory")
	parser.add_argument("--epochs", default=10, type=int)
	parser.add_argument("--batch_size", default=32, type=int)
	parser.add_argument("--lr", default=3e-4, type=float)
	parser.add_argument("--image_size", default=224, type=int)
	parser.add_argument("--num_workers", default=2, type=int)
	parser.add_argument("--output", default=os.path.join("models", "brain_tumor_resnet34.pt"), type=str)
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Using device: {device}")

	train_loader, val_loader, idx_to_class = create_dataloaders(
		args.train_dir, args.val_dir, args.image_size, args.batch_size, args.num_workers
	)
	model = build_model(num_classes=len(idx_to_class))
	model.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.AdamW(model.parameters(), lr=args.lr)
	scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

	best_val_acc = 0.0
	for epoch in range(1, args.epochs + 1):
		print(f"\nEpoch {epoch}/{args.epochs}")
		train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
		val_loss, val_acc = evaluate(model, val_loader, criterion, device)
		scheduler.step()
		print(f"Train loss: {train_loss:.4f} acc: {train_acc:.4f}")
		print(f"Val   loss: {val_loss:.4f} acc: {val_acc:.4f}")
		if val_acc > best_val_acc:
			best_val_acc = val_acc
			save_checkpoint(model, idx_to_class, args.output)
			print(f"Saved best model to {args.output}")

	print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
	main()
