import argparse
import os
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet34_Weights

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_model(checkpoint_path: str, num_classes: int):
	ckpt = torch.load(checkpoint_path, map_location="cpu")
	arch = ckpt.get("architecture", "resnet34")
	if arch == "resnet34":
		model = models.resnet34()
	elif arch == "resnet18":
		model = models.resnet18()
	else:
		raise ValueError(f"Unsupported architecture in checkpoint: {arch}")
	in_features = model.fc.in_features
	model.fc = torch.nn.Sequential(
		torch.nn.Dropout(0.3),
		torch.nn.Linear(in_features, num_classes)
	)
	model.load_state_dict(ckpt["state_dict"], strict=True)
	model.eval()
	return model


def build_loader(data_dir: str, image_size: int, batch_size: int, num_workers: int):
	tfms = transforms.Compose([
		transforms.Resize((image_size, image_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
	])
	ds = datasets.ImageFolder(data_dir, transform=tfms)
	loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
	return loader, ds.class_to_idx


def main():
	parser = argparse.ArgumentParser(description="Evaluate model on Testing set")
	parser.add_argument("--data_dir", default="Testing", type=str)
	parser.add_argument("--checkpoint", default=str(Path("models") / "brain_tumor_resnet34.pt"))
	parser.add_argument("--image_size", type=int, default=224)
	parser.add_argument("--batch_size", type=int, default=64)
	parser.add_argument("--num_workers", type=int, default=2)
	parser.add_argument("--out", type=str, default="evaluation_report.txt")
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	loader, class_to_idx = build_loader(args.data_dir, args.image_size, args.batch_size, args.num_workers)
	idx_to_class = {v: k for k, v in class_to_idx.items()}
	model = load_model(args.checkpoint, num_classes=len(idx_to_class))
	model.to(device)

	all_preds = []
	all_labels = []
	with torch.no_grad():
		for images, labels in loader:
			images = images.to(device)
			logits = model(images)
			preds = torch.argmax(logits, dim=1).cpu().numpy()
			all_preds.append(preds)
			all_labels.append(labels.numpy())

	all_preds = np.concatenate(all_preds)
	all_labels = np.concatenate(all_labels)

	report = classification_report(all_labels, all_preds, target_names=[idx_to_class[i] for i in range(len(idx_to_class))])
	cm = confusion_matrix(all_labels, all_preds)

	with open(args.out, "w", encoding="utf-8") as f:
		f.write("Classification Report\n")
		f.write(report + "\n\n")
		f.write("Confusion Matrix\n")
		for row in cm:
			f.write("\t".join(map(str, row)) + "\n")
	print(f"Saved evaluation to {args.out}")


if __name__ == "__main__":
	main()
