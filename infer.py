import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms, models

SUGGESTIONS = {
	"glioma_tumor": "Consult a neuro-oncologist. MRI with contrast and biopsy evaluation may be indicated.",
	"meningioma_tumor": "Consult neurosurgery. Often benign and slow-growing; MRI follow-up vs resection per size/location.",
	"pituitary_tumor": "Consult endocrinology and neurosurgery. Assess hormone levels and visual fields; MRI sellar protocol.",
	"no_tumor": "No tumor detected. If symptoms persist, follow up with a neurologist and consider repeat imaging.",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_model(checkpoint_path: str):
	ckpt = torch.load(checkpoint_path, map_location="cpu")
	idx_to_class = ckpt["idx_to_class"]
	arch = ckpt.get("architecture", "resnet34")
	num_classes = len(idx_to_class)
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
	return model, idx_to_class


def build_tfms(image_size: int):
	return transforms.Compose([
		transforms.Resize((image_size, image_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
	])


def predict_image(model, image_path: str, tfms, device):
	img = Image.open(image_path).convert("RGB")
	x = tfms(img).unsqueeze(0).to(device)
	with torch.no_grad():
		logits = model(x)
		probs = torch.softmax(logits, dim=1).cpu().squeeze(0)
	conf, pred_idx = torch.max(probs, dim=0)
	return pred_idx.item(), conf.item(), probs.tolist()


def main():
	parser = argparse.ArgumentParser(description="Infer brain tumor class and suggestions")
	parser.add_argument("image", type=str, help="Path to image file")
	parser.add_argument("--checkpoint", default=str(Path("models") / "brain_tumor_resnet34.pt"))
	parser.add_argument("--image_size", type=int, default=224)
	parser.add_argument("--json", action="store_true", help="Output JSON")
	args = parser.parse_args()

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model, idx_to_class = load_model(args.checkpoint)
	model.to(device)
	tfms = build_tfms(args.image_size)

	pred_idx, conf, probs = predict_image(model, args.image, tfms, device)
	label = idx_to_class[pred_idx]
	suggestion = SUGGESTIONS.get(label, "Consult a specialist for further evaluation.")
	result = {
		"label": label,
		"confidence": conf,
		"suggestion": suggestion,
		"probabilities": {idx_to_class[i]: p for i, p in enumerate(probs)}
	}
	if args.json:
		print(json.dumps(result, indent=2))
	else:
		print(f"Prediction: {label} (confidence {conf:.2%})")
		print(f"Suggestion: {suggestion}")


if __name__ == "__main__":
	main()
