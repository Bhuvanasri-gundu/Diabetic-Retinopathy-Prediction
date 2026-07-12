"""
DR Vision AI — Backend Engine
=============================
Central backend module extracted from project notebooks.
Contains all model definitions, preprocessing, Grad-CAM++, and inference logic.
"""

import os
import io
import random
import datetime
import numpy as np
import pandas as pd
import cv2
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


# ==============================================================
# Configuration
# ==============================================================
class Config:
    SEED = 42
    NUM_CLASSES = 5
    IMAGE_SIZE = 224
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths (relative to project root)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, "Models")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    PROCESSED_IMAGE_DIR = os.path.join(BASE_DIR, "processed_images")

    IMAGE_MODEL_PATH = os.path.join(MODELS_DIR, "best_model.pth")
    MULTIMODAL_MODEL_PATH = os.path.join(MODELS_DIR, "best_multimodal_model.pth")
    CLINICAL_CSV_PATH = os.path.join(DATA_DIR, "clinical_data_processed.csv")

    CLASS_NAMES = [
        "No DR",
        "Mild",
        "Moderate",
        "Severe",
        "Proliferative",
    ]

    CLASS_DESCRIPTIONS = [
        "Healthy retinal appearance.",
        "Microaneurysms only.",
        "Hemorrhages / exudates.",
        "Extensive bleeding.",
        "Neovascularization.",
    ]

    CLASS_COLORS = [
        "#22c55e",  # green
        "#eab308",  # yellow
        "#f97316",  # orange
        "#ef4444",  # red
        "#991b1b",  # dark red
    ]

    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]


# ==============================================================
# Validation transforms — exactly as in notebook
# ==============================================================
val_transforms = Compose([
    ToTensor(),
    Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD),
])

# For image-only model (needs resize + center crop since images aren't pre-processed)
val_transforms_image_only = Compose([
    Resize(Config.IMAGE_SIZE),
    CenterCrop(Config.IMAGE_SIZE),
    ToTensor(),
    Normalize(mean=Config.NORMALIZE_MEAN, std=Config.NORMALIZE_STD),
])


# ==============================================================
# Image Preprocessing — exact copy from notebook cell 6
# ==============================================================
def remove_black_borders(img, tolerance=10):
    """
    Crops the image to the bounding box of the active circular retinal scan.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, tolerance, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return img[y:y+h, x:x+w]
    return img


def preprocess_image_array(img, image_size=224):
    """
    Core image preprocessing steps: border removal, center-focused crop,
    circular mask, LAB-space CLAHE enhancement, and resizing.
    """
    # 1. Remove black borders
    img = remove_black_borders(img)
    h, w, _ = img.shape

    # 2. Center-focused crop (constant scale = 0.75 for validation / preprocessing)
    crop_scale = 0.75
    new_h, new_w = int(h * crop_scale), int(w * crop_scale)

    center_x, center_y = w // 2, h // 2
    x1 = max(0, center_x - new_w // 2)
    y1 = max(0, center_y - new_h // 2)
    x2 = min(w, x1 + new_w)
    y2 = min(h, y1 + new_h)
    img = img[y1:y2, x1:x2]

    # 3. Circular mask
    ch, cw, _ = img.shape
    mask = np.zeros((ch, cw), dtype=np.uint8)
    cv2.circle(mask, (cw // 2, ch // 2), int(0.49 * min(ch, cw)), 255, -1)
    img = cv2.bitwise_and(img, img, mask=mask)

    # 4. CLAHE in LAB space
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.merge((l, a, b))
    img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)

    # 5. Resize
    img_resized = cv2.resize(img, (image_size, image_size))
    return img_resized


def preprocess_uploaded_image(uploaded_file):
    """
    Takes a Streamlit UploadedFile, returns (raw_bgr, preprocessed_pil, img_tensor).
    """
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    uploaded_file.seek(0)
    raw_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if raw_bgr is None:
        raise ValueError("Could not decode the uploaded image.")

    # Preprocess
    processed_bgr = preprocess_image_array(raw_bgr, Config.IMAGE_SIZE)
    processed_rgb = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGB)
    processed_pil = Image.fromarray(processed_rgb)

    # Create tensor
    img_tensor = val_transforms(processed_pil).unsqueeze(0).to(Config.DEVICE)

    # Raw image for display
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
    raw_pil = Image.fromarray(raw_rgb)

    return raw_pil, processed_pil, img_tensor


# ==============================================================
# Image-Only Model — from classification notebook
# ==============================================================
def get_image_only_model(num_classes=5, pretrained=False):
    """
    Builds EfficientNet-B4 with the custom classifier head
    exactly as defined in diabetic_retinopathy_classification.ipynb.
    """
    if pretrained:
        weights = models.EfficientNet_B4_Weights.DEFAULT
        model = models.efficientnet_b4(weights=weights)
    else:
        model = models.efficientnet_b4(weights=None)

    in_features = model.classifier[1].in_features

    # Custom classifier block as specified
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=False),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=False),
        nn.Dropout(p=0.3, inplace=False),
        nn.Linear(512, num_classes),
    )

    # Disable ALL inplace operations
    for module in model.modules():
        if isinstance(module, (nn.ReLU, nn.SiLU)):
            module.inplace = False

    return model


# ==============================================================
# Multimodal Model — exact copy from notebook cell 16
# ==============================================================
class MultimodalDRClassifier(nn.Module):
    def __init__(self, num_clinical_features, num_classes=5, image_checkpoint=None):
        super(MultimodalDRClassifier, self).__init__()

        # 1. Image Feature Extractor - EfficientNet-B4
        self.image_model = models.efficientnet_b4(weights=None)
        in_features = self.image_model.classifier[1].in_features

        # Create same classifier structure used by image model checkpoint
        self.image_model.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=False),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.3, inplace=False),
            nn.Linear(512, num_classes),
        )

        # Load trained EfficientNet-B4 checkpoint
        if image_checkpoint and os.path.exists(image_checkpoint):
            state_dict = torch.load(image_checkpoint, map_location="cpu")
            self.image_model.load_state_dict(state_dict)

        # Remove final DR classification layer → 512-dim image features
        self.image_model.classifier = nn.Sequential(
            *list(self.image_model.classifier[:-1])
        )

        # Disable inplace activations
        for module in self.image_model.modules():
            if isinstance(module, (nn.ReLU, nn.SiLU)):
                module.inplace = False

        # Freeze complete EfficientNet first
        for param in self.image_model.parameters():
            param.requires_grad = False

        # Unfreeze ONLY final feature block
        for param in self.image_model.features[-1].parameters():
            param.requires_grad = True

        # Keep 512-dim projection head trainable
        for param in self.image_model.classifier.parameters():
            param.requires_grad = True

        # 2. Clinical Data MLP Encoder
        self.clinical_encoder = nn.Sequential(
            nn.Linear(num_clinical_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.2),
            nn.Linear(64, 32),
            nn.ReLU(inplace=False),
        )

        # 3. Multimodal Fusion Classification Head
        self.fusion_classifier = nn.Sequential(
            nn.Linear(512 + 32, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, image_tensor, clinical_tensor):
        img_feats = self.image_model(image_tensor)
        clin_feats = self.clinical_encoder(clinical_tensor)
        fused = torch.cat((img_feats, clin_feats), dim=1)
        output = self.fusion_classifier(fused)
        return output


# ==============================================================
# Grad-CAM++ — exact copy from notebook cell 26
# ==============================================================
class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handlers = []

        self.handlers.append(
            target_layer.register_forward_hook(self.save_activation)
        )
        self.handlers.append(
            target_layer.register_full_backward_hook(self.save_gradient)
        )

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, image_tensor, clinical_tensor=None, target_class=None):
        self.model.eval()
        with torch.enable_grad():
            if clinical_tensor is not None:
                output = self.model(image_tensor, clinical_tensor)
            else:
                output = self.model(image_tensor)

            if target_class is None:
                target_class = output.argmax(dim=1).item()

            self.model.zero_grad()
            score = output[0, target_class]
            score.backward()

        activations = self.activations.detach()
        gradients = self.gradients.detach()

        pos_gradients = torch.clamp(gradients, min=0.0)
        gradients_sq = pos_gradients ** 2
        gradients_cb = pos_gradients ** 3
        sum_activations = activations.sum(dim=(2, 3), keepdim=True)

        eps = 1e-10
        alpha_denom = 2.0 * gradients_sq + sum_activations * gradients_cb + eps
        alpha = gradients_sq / alpha_denom

        weights = (alpha * pos_gradients).sum(dim=(2, 3), keepdim=True)
        heatmap = (weights * activations).sum(dim=1, keepdim=True)
        heatmap = torch.clamp(heatmap, min=0.0)

        heatmap_min = heatmap.min()
        heatmap_max = heatmap.max()
        if heatmap_max > heatmap_min:
            heatmap = (heatmap - heatmap_min) / (heatmap_max - heatmap_min)
        else:
            heatmap = torch.zeros_like(heatmap)

        return heatmap.squeeze().cpu().numpy()

    def remove_hooks(self):
        for handler in self.handlers:
            handler.remove()


def overlay_heatmap(img_pil, heatmap, alpha=0.45):
    """Overlays a Grad-CAM heatmap onto a PIL image."""
    heatmap_resized = cv2.resize(heatmap, (img_pil.width, img_pil.height))
    heatmap_color = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    img_np = np.array(img_pil)
    blended = cv2.addWeighted(img_np, 1 - alpha, heatmap_color, alpha, 0)
    return Image.fromarray(blended)


# ==============================================================
# Model Loading
# ==============================================================
def load_image_only_model():
    """Loads the image-only EfficientNet-B4 model."""
    model = get_image_only_model(num_classes=Config.NUM_CLASSES, pretrained=False)

    if os.path.exists(Config.IMAGE_MODEL_PATH):
        state_dict = torch.load(Config.IMAGE_MODEL_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"✓ Loaded image-only model from {Config.IMAGE_MODEL_PATH}")
    else:
        print(f"✗ Image model not found at {Config.IMAGE_MODEL_PATH}")

    model = model.to(Config.DEVICE)
    model.eval()
    return model


def load_multimodal_model(num_clinical_features):
    """Loads the multimodal model."""
    model = MultimodalDRClassifier(
        num_clinical_features=num_clinical_features,
        num_classes=Config.NUM_CLASSES,
        image_checkpoint=Config.IMAGE_MODEL_PATH,
    )

    if os.path.exists(Config.MULTIMODAL_MODEL_PATH):
        state_dict = torch.load(Config.MULTIMODAL_MODEL_PATH, map_location="cpu")
        model.load_state_dict(state_dict)
        print(f"✓ Loaded multimodal model from {Config.MULTIMODAL_MODEL_PATH}")
    else:
        print(f"✗ Multimodal model not found at {Config.MULTIMODAL_MODEL_PATH}")

    model = model.to(Config.DEVICE)
    model.eval()
    return model


# ==============================================================
# Clinical Data Preprocessing — from notebook cell 10
# ==============================================================
CATEGORICAL_COLS = ["Gender", "Smoking_Status", "Hypertension"]
NUMERICAL_COLS = [
    "Age", "Duration_DM_Years", "HbA1c", "Fasting_Glucose_mg_dL",
    "BMI", "Systolic_BP", "Diastolic_BP",
    "Total_Cholesterol_mg_dL", "Serum_Creatinine_mg_dL",
]
CLINICAL_FEATURE_COLS = CATEGORICAL_COLS + NUMERICAL_COLS


def load_clinical_preprocessors():
    """
    Fits LabelEncoders and StandardScaler on the training split
    of clinical data — exactly as done in the notebook.
    Returns (scaler, encoders, train_df).
    """
    if not os.path.exists(Config.CLINICAL_CSV_PATH):
        print(f"✗ Clinical CSV not found at {Config.CLINICAL_CSV_PATH}")
        return None, None, None

    clinical_df = pd.read_csv(Config.CLINICAL_CSV_PATH)

    # Check which columns actually exist
    cat_cols = [c for c in CATEGORICAL_COLS if c in clinical_df.columns]
    num_cols = [c for c in NUMERICAL_COLS if c in clinical_df.columns]

    train_df, _ = train_test_split(
        clinical_df,
        test_size=0.2,
        random_state=Config.SEED,
        stratify=clinical_df["DR_Grade"],
    )
    train_df = train_df.copy()

    # Imputation
    for col in cat_cols:
        train_mode = train_df[col].mode()[0] if not train_df[col].empty else "Never"
        train_df[col] = train_df[col].fillna(train_mode)

    for col in num_cols:
        train_median = train_df[col].median() if not train_df[col].empty else 0.0
        train_df[col] = train_df[col].fillna(train_median)

    # Encoding
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col].astype(str))
        encoders[col] = le

    # Scaling
    scaler = StandardScaler()
    if len(num_cols) > 0:
        scaler.fit(train_df[num_cols])

    print(f"✓ Clinical preprocessors fitted on {len(train_df)} training samples")
    return scaler, encoders, train_df


def prepare_clinical_tensor(clinical_dict, scaler, encoders, train_df):
    """
    Converts a dict of clinical values into a normalized tensor.
    Matches the exact logic from notebook predict_patient() cell 28.
    """
    cat_cols = [c for c in CATEGORICAL_COLS if c in (encoders or {})]
    num_cols = NUMERICAL_COLS
    feature_cols = cat_cols + num_cols

    patient_df = pd.DataFrame([clinical_dict])

    # Impute missing categorical
    for col in cat_cols:
        if col not in patient_df.columns or pd.isna(patient_df[col].iloc[0]):
            train_mode = (
                train_df[col].mode()[0]
                if (train_df is not None and col in train_df.columns and not train_df[col].empty)
                else "Never"
            )
            patient_df[col] = train_mode

    # Impute missing numerical
    for col in num_cols:
        if col not in patient_df.columns or pd.isna(patient_df[col].iloc[0]):
            train_median = (
                train_df[col].median()
                if (train_df is not None and col in train_df.columns and not train_df[col].empty)
                else 0.0
            )
            patient_df[col] = train_median

    # Encode categorical
    for col in cat_cols:
        if encoders and col in encoders:
            try:
                val_str = str(patient_df[col].iloc[0])
                if val_str in encoders[col].classes_:
                    patient_df[col] = encoders[col].transform([val_str])
                else:
                    patient_df[col] = 0
            except Exception:
                patient_df[col] = 0
        else:
            patient_df[col] = 0

    # Normalize numerical
    if len(num_cols) > 0 and scaler is not None:
        # Ensure all numerical columns exist
        for col in num_cols:
            if col not in patient_df.columns:
                patient_df[col] = 0.0
        patient_df[num_cols] = scaler.transform(patient_df[num_cols])

    # Build tensor in the correct column order
    for col in feature_cols:
        if col not in patient_df.columns:
            patient_df[col] = 0.0

    clinical_vals = patient_df[feature_cols].values.astype(np.float32)
    clinical_tensor = torch.tensor(clinical_vals, dtype=torch.float32).to(Config.DEVICE)
    return clinical_tensor


# ==============================================================
# Inference Functions
# ==============================================================
def predict_image_only(model, img_tensor):
    """
    Runs image-only inference. Returns (predicted_class, confidence, probabilities).
    """
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)

    return (
        predicted_idx.item(),
        confidence.item(),
        probs.squeeze().cpu().numpy(),
    )


def predict_multimodal(model, img_tensor, clinical_tensor):
    """
    Runs multimodal inference. Returns (predicted_class, confidence, probabilities).
    """
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor, clinical_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probs, 1)

    return (
        predicted_idx.item(),
        confidence.item(),
        probs.squeeze().cpu().numpy(),
    )


def generate_gradcam_image_only(model, img_tensor, processed_pil, target_class=None):
    """Generates Grad-CAM++ for the image-only model."""
    img_tensor_grad = img_tensor.clone().detach().requires_grad_(True)
    target_layer = model.features[-1]
    gcam = GradCAMPlusPlus(model, target_layer)
    heatmap = gcam.generate_heatmap(
        img_tensor_grad, clinical_tensor=None, target_class=target_class
    )
    gcam.remove_hooks()
    overlay = overlay_heatmap(processed_pil, heatmap, alpha=0.45)
    return overlay, heatmap


def generate_gradcam_multimodal(model, img_tensor, clinical_tensor, processed_pil, target_class=None):
    """Generates Grad-CAM++ for the multimodal model."""
    img_tensor_grad = img_tensor.clone().detach().requires_grad_(True)
    target_layer = model.image_model.features[-1]
    gcam = GradCAMPlusPlus(model, target_layer)
    heatmap = gcam.generate_heatmap(
        img_tensor_grad, clinical_tensor=clinical_tensor, target_class=target_class
    )
    gcam.remove_hooks()
    overlay = overlay_heatmap(processed_pil, heatmap, alpha=0.45)
    return overlay, heatmap


# ==============================================================
# PDF Report Generation
# ==============================================================
def generate_pdf_report(
    patient_info,
    predicted_class,
    confidence,
    probabilities,
    raw_pil,
    processed_pil,
    gradcam_pil,
    model_type="Multimodal",
):
    """
    Generates a PDF diagnostic report. Returns bytes.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    class DRReport(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 14)
            self.set_text_color(13, 148, 136)
            self.cell(0, 10, "DR Vision AI - Retinal Analysis Report", ln=True, align="C")
            self.set_font("Helvetica", "", 8)
            self.set_text_color(100, 100, 100)
            self.cell(
                0, 5,
                f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ln=True, align="C",
            )
            self.ln(5)
            self.set_draw_color(13, 148, 136)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(5)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 7)
            self.set_text_color(150, 150, 150)
            self.cell(
                0, 10,
                "DR Vision AI | For Research Use Only | Not for diagnostic procedures without clinical oversight.",
                align="C",
            )

    pdf = DRReport()
    pdf.add_page()

    # Diagnosis Section
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(0, 8, "Diagnostic Summary", ln=True)
    pdf.ln(3)

    grade_name = Config.CLASS_NAMES[predicted_class]
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(50, 7, "Prediction:", ln=False)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"{grade_name} (Grade {predicted_class})", ln=True)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(50, 7, "Confidence:", ln=False)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, f"{confidence * 100:.1f}%", ln=True)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(50, 7, "Model Type:", ln=False)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, model_type, ln=True)

    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(50, 7, "Description:", ln=False)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, Config.CLASS_DESCRIPTIONS[predicted_class], ln=True)
    pdf.ln(5)

    # Patient Info
    if patient_info:
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Patient Information", ln=True)
        pdf.ln(3)
        pdf.set_font("Helvetica", "", 10)
        for key, val in patient_info.items():
            if val is not None and str(val).strip():
                pdf.cell(60, 6, f"{key}:", ln=False)
                pdf.cell(0, 6, str(val), ln=True)
        pdf.ln(5)

    # Class Probabilities
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Class Probabilities", ln=True)
    pdf.ln(3)
    pdf.set_font("Helvetica", "", 10)
    for i, prob in enumerate(probabilities):
        pdf.cell(60, 6, f"{Config.CLASS_NAMES[i]}:", ln=False)
        pdf.cell(0, 6, f"{prob * 100:.2f}%", ln=True)
    pdf.ln(5)

    # Save images and embed
    def embed_pil_image(pil_img, label, w=60):
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        buf.seek(0)
        img_path = os.path.join(
            Config.BASE_DIR,
            f"_temp_report_{label}.png",
        )
        with open(img_path, "wb") as f:
            f.write(buf.read())
        return img_path

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Retinal Image Analysis", ln=True)
    pdf.ln(3)

    temp_files = []
    try:
        x_start = pdf.get_x()
        y_pos = pdf.get_y()

        for label, pil_img, title in [
            ("raw", raw_pil, "Raw Scan"),
            ("processed", processed_pil, "Processed"),
            ("gradcam", gradcam_pil, "Grad-CAM++"),
        ]:
            img_path = embed_pil_image(pil_img, label)
            temp_files.append(img_path)

        # Place images side by side
        pdf.set_font("Helvetica", "I", 8)
        for idx, (label, title) in enumerate(
            [("raw", "Raw Scan"), ("processed", "Processed"), ("gradcam", "Grad-CAM++")]
        ):
            x_offset = 10 + idx * 65
            pdf.set_xy(x_offset, y_pos)
            pdf.cell(60, 5, title, align="C")

        y_img = y_pos + 6
        for idx, label in enumerate(["raw", "processed", "gradcam"]):
            x_offset = 10 + idx * 65
            img_path = os.path.join(Config.BASE_DIR, f"_temp_report_{label}.png")
            if os.path.exists(img_path):
                pdf.image(img_path, x=x_offset, y=y_img, w=55)

    except Exception:
        pass

    # Generate PDF bytes
    pdf_bytes = pdf.output()

    # Cleanup temp files
    for f in temp_files:
        try:
            os.remove(f)
        except OSError:
            pass

    return bytes(pdf_bytes)
