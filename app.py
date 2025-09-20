import io
import os
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import models, transforms
import pandas as pd

# Constants
DEFAULT_CHECKPOINT = str(Path("models") / "brain_tumor_resnet34.pt")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SUGGESTIONS = {
	"glioma_tumor": "Consult a neuro-oncologist. MRI with contrast and biopsy evaluation may be indicated.",
	"meningioma_tumor": "Consult neurosurgery. Often benign; consider MRI follow-up vs resection.",
	"pituitary_tumor": "Consult endocrinology/neurosurgery. Assess hormones, visual fields; MRI sellar protocol.",
	"no_tumor": "No tumor detected. If symptoms persist, follow up with neurology and consider repeat imaging.",
}


def build_tfms(image_size: int):
	return transforms.Compose([
		transforms.Resize((image_size, image_size)),
		transforms.ToTensor(),
		transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
	])


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


def predict(model: torch.nn.Module, image: Image.Image, image_size: int):
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model.to(device)
	tfms = build_tfms(image_size)
	x = tfms(image.convert("RGB")).unsqueeze(0).to(device)
	with torch.no_grad():
		logits = model(x)
		probs = torch.softmax(logits, dim=1).cpu().squeeze(0)
	conf, pred_idx = torch.max(probs, dim=0)
	return pred_idx.item(), conf.item(), probs.tolist()


st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠", layout="centered")
st.title("🧠 Brain Tumor Classifier")
st.write("Upload a brain MRI image to predict: glioma, meningioma, pituitary, or no tumor.")

with st.sidebar:
	st.header("Settings")
	ckpt_path = st.text_input("Checkpoint path", value=DEFAULT_CHECKPOINT)
	image_size = st.slider("Image size", min_value=160, max_value=384, value=224, step=32)
	st.caption("Ensure the checkpoint exists and was trained with this project.")

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"]) 

col1, col2 = st.columns(2)

with col1:
	if uploaded:
		img = Image.open(uploaded)
		st.image(img, caption="Input Image", use_container_width=True)
	else:
		st.info("Please upload an image to get a prediction.")

with col2:
	if uploaded:
		if not os.path.exists(ckpt_path):
			st.error(f"Checkpoint not found: {ckpt_path}")
		else:
			try:
				model, idx_to_class = load_model(ckpt_path)
				pred_idx, conf, probs = predict(model, img, image_size)
				label = idx_to_class[pred_idx]
				st.subheader(f"Prediction: {label}")
				st.metric("Confidence", f"{conf:.2%}")
				st.write("Suggestion:")
				st.success(SUGGESTIONS.get(label, "Consult a specialist for further evaluation."))
				# Probabilities chart using DataFrame
				labels = [idx_to_class[i] for i in range(len(probs))]
				df = pd.DataFrame({"label": labels, "probability": probs}).set_index("label")
				st.bar_chart(df)
			except Exception as e:
				st.error(f"Failed to predict: {e}")
