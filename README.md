## Brain Tumor Classifier (Glioma, Meningioma, Pituitary, No Tumor)

This project trains a transfer-learning CNN (ResNet18) on your dataset organized as:

- `Training/`: four subfolders: `glioma_tumor`, `meningioma_tumor`, `pituitary_tumor`, `no_tumor`
- `Testing/`: same subfolders for evaluation

### 1) Setup (Windows PowerShell)

```powershell
# From C:\subject\bc
python -m venv .venv
. .venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

### 2) Train

```powershell
python train.py --train_dir Training --val_dir Testing --epochs 10 --batch_size 32 --lr 3e-4 --image_size 224 --num_workers 2 --output models\brain_tumor_resnet18.pt
```

- The best model checkpoint is saved to `models/brain_tumor_resnet18.pt`.

### 3) Evaluate on Testing/

```powershell
python eval.py --data_dir Testing --checkpoint models\brain_tumor_resnet18.pt --out evaluation_report.txt
```

- Outputs `evaluation_report.txt` with classification report and confusion matrix.

### 4) Inference on a single image

```powershell
python infer.py Testing\glioma_tumor\image.jpg --checkpoint models\brain_tumor_resnet18.pt
```

- Prints predicted label, confidence, and a clinical-style suggestion.
- Add `--json` to get machine-readable output.

### Notes
- Uses ImageNet normalization and augmentation.
- If GPU is available, PyTorch will use it automatically.
- Adjust `--epochs`, `--batch_size` as needed for accuracy vs speed.
