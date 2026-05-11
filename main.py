"""
main.py — 
Stanford Dogs Dataset (120 razas) · EfficientNetB3 Transfer Learning
FastAPI + PyTorch + torchvision

Endpoints:
  GET  /health          → estado del servidor y modelo
  POST /predict         → clasificar imagen → top-5 razas + confianza
  GET  /classes         → lista completa de 120 razas
  GET  /metrics         → métricas del modelo (si existen guardadas)

Evaluación que realizamos:
  - Matriz de confusión
  - Precision, Recall, F1 por clase
  - Curvas de entrenamiento
  - Análisis de errores
  - Visualización de activaciones (Grad-CAM)
"""

import os, json, time, io, base64
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support
)
import seaborn as sns

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

# ─────────────────────────────────────────────
# CONFIGURACION
# ─────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
DATASET_DIR   = BASE_DIR / "dataset" / "Images"      # Stanford Dogs: carpeta Images/
MODEL_PATH    = BASE_DIR / "model" / "dog_classifier.pth"
METRICS_PATH  = BASE_DIR / "model" / "metrics.json"
CURVES_PATH   = BASE_DIR / "model" / "training_curves.png"
CONFUSION_PATH= BASE_DIR / "model" / "confusion_matrix.png"
ACTIVATIONS_DIR = BASE_DIR / "activations"
CLASSES_PATH  = BASE_DIR / "model" / "classes.json"

IMG_SIZE      = 300          # EfficientNetB3 input
BATCH_SIZE    = 32
EPOCHS        = 25
LR            = 1e-4
NUM_WORKERS   = 0            # Windows: siempre 0
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES   = 120          # Stanford Dogs

# ─────────────────────────────────────────────
# CREAR DIRECTORIOS
# ─────────────────────────────────────────────
for d in [MODEL_PATH.parent, ACTIVATIONS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# TRANSFORMS
# ─────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE + 20, IMG_SIZE + 20)),
    transforms.RandomCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# MODELO: EfficientNetB3 (Transfer Learning)
# ─────────────────────────────────────────────
def build_model(num_classes: int) -> nn.Module:
    """
    EfficientNetB3 con clasificador personalizado.
    - Capas base congeladas inicialmente (feature extraction)
    - Fine-tuning desbloquea las últimas 30 capas
    """
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    
    # Congelar todas las capas base
    for param in model.parameters():
        param.requires_grad = False

    # Reemplazar clasificador final
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(512, num_classes),
    )

    # Descongelar últimas capas para fine-tuning
    layers = list(model.features.children())
    for layer in layers[-3:]:
        for param in layer.parameters():
            param.requires_grad = True

    return model.to(DEVICE)


# ─────────────────────────────────────────────
# ENTRENAMIENTO
# ─────────────────────────────────────────────
def train_model():
    """
    Entrena la CNN con el dataset de Stanford Dogs.
    Genera: modelo .pth, curvas de entrenamiento, métricas completas.
    
    ANTES de llamar esta función, asegúrate de que:
      dataset/Images/  contiene las 120 carpetas de razas
      (descarga: http://vision.stanford.edu/aditya86/ImageNetDogs/)
    """
    print(f"[TRAIN] Usando dispositivo: {DEVICE}")
    print(f"[TRAIN] Dataset: {DATASET_DIR}")

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en: {DATASET_DIR}\n"
            "Descarga Stanford Dogs Dataset y extráelo como:\n"
            "  dataset/Images/n02085620-Chihuahua/\n"
            "  dataset/Images/n02085782-Japanese_spaniel/\n  ..."
        )

    # Dataset completo
    full_dataset = ImageFolder(DATASET_DIR, transform=train_transform)
    classes = full_dataset.classes
    num_classes = len(classes)
    print(f"[TRAIN] Razas encontradas: {num_classes}")

    # Guardar lista de clases
    with open(CLASSES_PATH, "w") as f:
        json.dump(classes, f, indent=2)

    # Split 80/10/10
    n = len(full_dataset)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)
    n_test  = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    # Aplicar transform de validación al split val/test
    val_ds.dataset.transform = val_transform
    test_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model     = build_model(num_classes)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── LOOP DE ENTRENAMIENTO ──────────────────
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # — Train —
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            t_loss    += loss.item() * imgs.size(0)
            _, preds   = outputs.max(1)
            t_correct += preds.eq(labels).sum().item()
            t_total   += imgs.size(0)

        # — Validation —
        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss    = criterion(outputs, labels)
                v_loss    += loss.item() * imgs.size(0)
                _, preds   = outputs.max(1)
                v_correct += preds.eq(labels).sum().item()
                v_total   += imgs.size(0)

        train_loss = t_loss / t_total
        val_loss   = v_loss / v_total
        train_acc  = t_correct / t_total
        val_acc    = v_correct / v_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        scheduler.step()

        print(f"[Epoch {epoch:02d}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")

        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
                "classes": classes,
                "num_classes": num_classes,
            }, MODEL_PATH)
            print(f"  ✓ Modelo guardado (val_acc={val_acc:.4f})")

    # ── EVALUACIÓN FINAL (TEST SET) ────────────
    print("\n[EVAL] Evaluando en test set...")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Precision / Recall / F1
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted"
    )
    report = classification_report(all_labels, all_preds,
                                   target_names=classes, output_dict=True)

    # Guardar métricas JSON
    metrics = {
        "best_val_acc": round(best_val_acc, 4),
        "test_precision": round(float(precision), 4),
        "test_recall":    round(float(recall),    4),
        "test_f1":        round(float(f1),        4),
        "epochs": EPOCHS,
        "num_classes": num_classes,
        "device": str(DEVICE),
        "trained_at": datetime.now().isoformat(),
        "history": history,
        "per_class_report": report,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[EVAL] Precision={precision:.4f}  Recall={recall:.4f}  F1={f1:.4f}")

    # ── CURVAS DE ENTRENAMIENTO ────────────────
    _plot_training_curves(history)

    # ── MATRIZ DE CONFUSIÓN (top-20 razas) ────
    _plot_confusion_matrix(all_labels, all_preds, classes)

    print("\n[TRAIN] ¡Entrenamiento completo!")
    print(f"  Modelo:     {MODEL_PATH}")
    print(f"  Métricas:   {METRICS_PATH}")
    print(f"  Curvas:     {CURVES_PATH}")
    print(f"  Confusión:  {CONFUSION_PATH}")
    return metrics


def _plot_training_curves(history: dict):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#12121a")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0a0a0f")
        ax.tick_params(colors="#6b6b80")
        for spine in ax.spines.values():
            spine.set_edgecolor("#2a2a3a")

    e = range(1, len(history["train_loss"]) + 1)
    ax1.plot(e, history["train_loss"], color="#f5c842", lw=2, label="Train Loss")
    ax1.plot(e, history["val_loss"],   color="#ff6b35", lw=2, label="Val Loss")
    ax1.set_title("Pérdida (Loss)", color="#e8e8f0", fontsize=13)
    ax1.set_xlabel("Época", color="#6b6b80")
    ax1.legend(facecolor="#12121a", labelcolor="#e8e8f0", edgecolor="#2a2a3a")
    ax1.grid(color="#2a2a3a", linewidth=0.5)

    ax2.plot(e, [a*100 for a in history["train_acc"]], color="#f5c842", lw=2, label="Train Acc")
    ax2.plot(e, [a*100 for a in history["val_acc"]],   color="#ff6b35", lw=2, label="Val Acc")
    ax2.set_title("Precisión (%)", color="#e8e8f0", fontsize=13)
    ax2.set_xlabel("Época", color="#6b6b80")
    ax2.legend(facecolor="#12121a", labelcolor="#e8e8f0", edgecolor="#2a2a3a")
    ax2.grid(color="#2a2a3a", linewidth=0.5)

    plt.tight_layout()
    plt.savefig(CURVES_PATH, dpi=120, bbox_inches="tight",
                facecolor="#12121a")
    plt.close()
    print(f"[PLOT] Curvas guardadas: {CURVES_PATH}")


def _plot_confusion_matrix(y_true, y_pred, classes, top_n=20):
    """Muestra la matriz de confusión de las top_n clases más frecuentes."""
    from collections import Counter
    top_idx = [i for i, _ in Counter(y_true).most_common(top_n)]
    mask = [i for i, y in enumerate(y_true) if y in top_idx]
    y_t = [y_true[i] for i in mask]
    y_p = [y_pred[i] for i in mask]
    label_map = {v: k for k, v in enumerate(top_idx)}
    y_t2 = [label_map[y] for y in y_t]
    y_p2 = [label_map.get(y, -1) for y in y_p]
    y_p2 = [y if y != -1 else 0 for y in y_p2]
    names = [classes[i].split("-")[-1].replace("_", " ")[:15] for i in top_idx]

    cm = confusion_matrix(y_t2, y_p2, labels=list(range(top_n)))
    fig, ax = plt.subplots(figsize=(16, 14))
    fig.patch.set_facecolor("#12121a")
    ax.set_facecolor("#12121a")
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd",
                xticklabels=names, yticklabels=names,
                ax=ax, linewidths=0.5, linecolor="#2a2a3a",
                annot_kws={"size": 8})
    ax.set_title(f"Matriz de Confusión — Top {top_n} razas",
                 color="#e8e8f0", fontsize=14, pad=15)
    ax.set_xlabel("Predicción", color="#6b6b80", fontsize=11)
    ax.set_ylabel("Real",       color="#6b6b80", fontsize=11)
    ax.tick_params(colors="#e8e8f0", labelsize=8)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(CONFUSION_PATH, dpi=100, bbox_inches="tight",
                facecolor="#12121a")
    plt.close()
    print(f"[PLOT] Matriz de confusión guardada: {CONFUSION_PATH}")


# ─────────────────────────────────────────────
# GRAD-CAM (Visualización de Activaciones)
# ─────────────────────────────────────────────
class GradCAM:
    """Genera heatmap de atención sobre qué zona activó la predicción."""

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activations)
        target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        output = self.model(input_tensor)
        output[0, class_idx].backward()

        weights    = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam        = (weights * self.activations).sum(dim=1, keepdim=True)
        cam        = torch.relu(cam)
        cam        = cam.squeeze().cpu().numpy()
        cam        -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def generate_gradcam(model: nn.Module, img_tensor: torch.Tensor,
                     class_idx: int, orig_img: Image.Image) -> str:
    """Retorna imagen Grad-CAM como base64 PNG."""
    # Obtener última capa convolucional de EfficientNetB3
    target_layer = model.features[-1]
    grad_cam = GradCAM(model, target_layer)
    cam = grad_cam.generate(img_tensor.unsqueeze(0).to(DEVICE), class_idx)

    # Overlay sobre imagen original
    orig_arr = np.array(orig_img.resize((IMG_SIZE, IMG_SIZE)))
    heatmap  = cm.jet(cam)[:, :, :3]
    heatmap  = (heatmap * 255).astype(np.uint8)
    heatmap  = Image.fromarray(heatmap).resize(orig_img.size)
    heatmap_arr = np.array(heatmap)
    overlay  = (0.55 * orig_arr + 0.45 * heatmap_arr).clip(0, 255).astype(np.uint8)

    # Codificar a base64
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ─────────────────────────────────────────────
# CARGA DEL MODELO EN MEMORIA
# ─────────────────────────────────────────────
_model: nn.Module = None
_classes: list    = []

def load_model():
    global _model, _classes
    if not MODEL_PATH.exists():
        print("[WARN] Modelo no encontrado. Ejecuta primero el entrenamiento.")
        return False
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    _classes = ckpt["classes"]
    _model   = build_model(len(_classes))
    _model.load_state_dict(ckpt["model_state"])
    _model.eval()
    print(f"[MODEL] Cargado: {len(_classes)} razas · {DEVICE}")
    return True


def preprocess_image(image_bytes: bytes) -> tuple:
    """Retorna (tensor normalizado, imagen PIL original)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    tensor = val_transform(img)
    return tensor, img


@torch.no_grad()
def predict_image(img_tensor: torch.Tensor, top_k: int = 5) -> list:
    """Retorna lista de {breed, confidence} ordenada por confianza."""
    outputs = _model(img_tensor.unsqueeze(0).to(DEVICE))
    probs   = torch.softmax(outputs, dim=1)[0]
    top_p, top_i = probs.topk(top_k)
    results = []
    for prob, idx in zip(top_p.cpu().numpy(), top_i.cpu().numpy()):
        breed = _classes[idx]
        # Limpiar nombre: "n02085620-Chihuahua" → "Chihuahua"
        if "-" in breed:
            breed = breed.split("-", 1)[1]
        results.append({"breed": breed, "confidence": round(float(prob), 6)})
    return results


# ─────────────────────────────────────────────
# FASTAPI APP
# ─────────────────────────────────────────────
app = FastAPI(
    title="DogBreed Classifier API",
    description="Clasificación de razas de perros con EfficientNetB3 · Stanford Dogs Dataset",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup():
    load_model()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "num_classes": len(_classes),
        "device": str(DEVICE),
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Recibe imagen, retorna top-5 predicciones + Grad-CAM.
    
    Respuesta JSON:
    {
      "predictions": [{"breed": "Chihuahua", "confidence": 0.9412}, ...],
      "top_breed": "Chihuahua",
      "top_confidence": 0.9412,
      "gradcam_base64": "<base64 PNG>",
      "inference_ms": 123
    }
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no cargado. Entrena primero el modelo ejecutando: "
                   "python main.py --train"
        )

    # Validar tipo
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(400, "Solo se aceptan imágenes JPG, PNG o WEBP.")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(400, "Imagen demasiado grande (máx 10 MB).")

    t0 = time.perf_counter()
    try:
        img_tensor, orig_img = preprocess_image(contents)
        predictions          = predict_image(img_tensor, top_k=5)
    except Exception as e:
        raise HTTPException(500, f"Error procesando imagen: {e}")

    # Grad-CAM (puede fallar sin romper la predicción)
    gradcam_b64 = None
    try:
        top_idx    = _classes.index(
            [c for c in _classes if predictions[0]["breed"] in c][0]
        )
        gradcam_b64 = generate_gradcam(_model, img_tensor, top_idx, orig_img)
    except Exception:
        pass

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "predictions":       predictions,
        "top_breed":         predictions[0]["breed"],
        "top_confidence":    predictions[0]["confidence"],
        "gradcam_base64":    gradcam_b64,
        "inference_ms":      elapsed_ms,
    }


@app.get("/classes")
def get_classes():
    """Retorna la lista completa de razas en el modelo."""
    if not _classes:
        if CLASSES_PATH.exists():
            with open(CLASSES_PATH) as f:
                return {"classes": json.load(f), "count": 0}
        raise HTTPException(404, "No hay modelo cargado.")
    return {"classes": _classes, "count": len(_classes)}


@app.get("/metrics")
def get_metrics():
    """Retorna métricas del entrenamiento (si existen)."""
    if not METRICS_PATH.exists():
        raise HTTPException(404, "No se encontraron métricas. Entrena el modelo primero.")
    with open(METRICS_PATH) as f:
        return json.load(f)


@app.get("/curves")
def get_curves():
    """Retorna curvas de entrenamiento como base64 PNG."""
    if not CURVES_PATH.exists():
        raise HTTPException(404, "Curvas no encontradas.")
    with open(CURVES_PATH, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return {"image_base64": data, "format": "png"}


@app.get("/confusion")
def get_confusion():
    """Retorna matriz de confusión como base64 PNG."""
    if not CONFUSION_PATH.exists():
        raise HTTPException(404, "Matriz de confusión no encontrada.")
    with open(CONFUSION_PATH, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return {"image_base64": data, "format": "png"}


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DogBreed Classifier")
    parser.add_argument("--train",  action="store_true",
                        help="Entrenar el modelo desde cero")
    parser.add_argument("--host",   default="0.0.0.0")
    parser.add_argument("--port",   type=int, default=8000)
    args = parser.parse_args()

    if args.train:
        print("=" * 55)
        print("  INICIANDO ENTRENAMIENTO")
        print("=" * 55)
        train_model()
        print("\n[OK] Entrenamiento completo. Iniciando servidor...")

    print(f"\n[SERVER] Iniciando en http://localhost:{args.port}")
    print(f"[SERVER] Docs API: http://localhost:{args.port}/docs")
    uvicorn.run("main:app", host=args.host, port=args.port, reload=False)