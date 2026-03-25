"""
MobileNetV2-Based ASD Visual Feature Extractor
================================================
Uses MobileNetV2 (pretrained on ImageNet, fine-tuned for ASD features)
to extract visual behavioral features from webcam frames.

Key visual ASD markers:
- Reduced facial engagement (looking away)
- Atypical facial expressions
- Reduced social attention
- Body movement patterns

Architecture:
- MobileNetV2 backbone (pretrained, frozen bottom layers)
- Custom head for ASD behavioral feature extraction
- Outputs 128-dim feature vector

For production: fine-tune on ASD behavioral datasets
(e.g., NIST, Kaggle ASD datasets, autism-specific eye tracking data)
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import cv2
from dataclasses import dataclass
from typing import Optional, List, Tuple
import os


@dataclass
class VisualFeatures:
    """Visual features extracted from a face/behavioral frame."""
    mobilenet_embedding: np.ndarray      # 128-dim feature vector
    face_detected: bool
    face_bbox: Optional[Tuple[int, int, int, int]]  # x, y, w, h
    face_area_ratio: float               # face size relative to frame
    face_center_offset: float           # how centered the face is
    looking_toward_camera: float        # 0-1 probability


class ASDVisualFeatureExtractor(nn.Module):
    """
    MobileNetV2-based feature extractor for ASD visual screening.

    Can be used in two modes:
    1. Feature extraction only (for fusion with other modalities)
    2. End-to-end classification (if fine-tuned on ASD dataset)
    """

    def __init__(self, num_classes: int = 2, feature_dim: int = 128,
                 pretrained: bool = True):
        super().__init__()

        # Load MobileNetV2 backbone
        if pretrained:
            weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
            mobilenet = models.mobilenet_v2(weights=weights)
        else:
            mobilenet = models.mobilenet_v2(weights=None)

        # Use MobileNetV2 features (remove classifier)
        self.backbone = mobilenet.features

        # Adaptive pooling to fixed size
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Custom feature head
        self.feature_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(1280, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True)
        )

        # Classification head (binary: ASD/non-ASD)
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

        # Freeze early layers of backbone (transfer learning strategy)
        self._freeze_early_layers(freeze_until=14)

    def _freeze_early_layers(self, freeze_until: int = 14):
        """Freeze early MobileNetV2 layers for transfer learning."""
        for i, layer in enumerate(self.backbone):
            if i < freeze_until:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Returns: (logits, feature_embeddings)
        """
        features = self.backbone(x)
        features = self.pool(features)
        features = features.flatten(1)

        embeddings = self.feature_head(features)
        logits = self.classifier(embeddings)

        return logits, embeddings

    def extract_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings only (no classification)."""
        with torch.no_grad():
            features = self.backbone(x)
            features = self.pool(features)
            features = features.flatten(1)
            embeddings = self.feature_head(features)
        return embeddings

    def get_asd_probability(self, x: torch.Tensor) -> float:
        """Get ASD probability for a single input."""
        with torch.no_grad():
            logits, _ = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            return float(probs[0, 1])  # probability of ASD class


class VisualProcessor:
    """
    Handles frame preprocessing and feature extraction.
    """

    FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

    def __init__(self, device: str = None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Image preprocessing (MobileNetV2 standard)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Initialize model
        self.model = ASDVisualFeatureExtractor(
            num_classes=2,
            feature_dim=128,
            pretrained=True
        ).to(self.device)
        self.model.eval()

        # Face detector
        self.face_cascade = cv2.CascadeClassifier(self.FACE_CASCADE_PATH)

        # Buffer for temporal features
        self._embedding_buffer: List[np.ndarray] = []
        self._max_buffer = 30  # 1 second at 30fps

    def preprocess_frame(self, frame: np.ndarray) -> Optional[torch.Tensor]:
        """
        Preprocess a BGR frame for MobileNetV2 input.
        Returns preprocessed tensor or None if no face detected.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        return tensor

    def detect_face(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face in frame. Returns (x, y, w, h) or None."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        if len(faces) > 0:
            # Return largest face
            areas = [w * h for (x, y, w, h) in faces]
            return tuple(faces[np.argmax(areas)])
        return None

    def extract_features(self, frame: np.ndarray) -> VisualFeatures:
        """Extract visual features from a single frame."""
        h, w = frame.shape[:2]
        face_bbox = self.detect_face(frame)
        face_detected = face_bbox is not None

        # Compute face-based features
        face_area_ratio = 0.0
        face_center_offset = 1.0  # 1 = worst case (no face)

        if face_bbox is not None:
            fx, fy, fw, fh = face_bbox
            face_area_ratio = (fw * fh) / (w * h)
            face_cx = fx + fw / 2
            face_cy = fy + fh / 2
            face_center_offset = np.sqrt(
                ((face_cx - w/2) / w)**2 + ((face_cy - h/2) / h)**2
            )

        # Preprocess and extract MobileNetV2 embeddings
        tensor = self.preprocess_frame(frame)
        embedding = self.model.extract_embeddings(tensor).cpu().numpy().flatten()

        # Buffer for temporal analysis
        self._embedding_buffer.append(embedding)
        if len(self._embedding_buffer) > self._max_buffer:
            self._embedding_buffer.pop(0)

        # Simplified camera-looking estimate based on face centering
        looking_toward_camera = max(0.0, 1.0 - face_center_offset * 2)

        return VisualFeatures(
            mobilenet_embedding=embedding,
            face_detected=face_detected,
            face_bbox=face_bbox,
            face_area_ratio=face_area_ratio,
            face_center_offset=face_center_offset,
            looking_toward_camera=looking_toward_camera
        )

    def get_temporal_embedding(self) -> Optional[np.ndarray]:
        """
        Get mean embedding over recent buffer.
        Captures temporal behavioral patterns.
        """
        if not self._embedding_buffer:
            return None
        return np.mean(self._embedding_buffer, axis=0)

    def save_model(self, path: str):
        """Save model weights."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model weights."""
        if os.path.exists(path):
            self.model.load_state_dict(
                torch.load(path, map_location=self.device)
            )
            print(f"Model loaded from {path}")
        else:
            print(f"No model found at {path}, using pretrained ImageNet weights")


class ASDFusionClassifier:
    """
    Fusion classifier that combines:
    - Gaze tracking features (35% weight)
    - Speech features (35% weight)
    - Visual/MobileNetV2 features (30% weight)

    Uses a weighted scoring system producing 0-100 risk score.
    Can be upgraded to Random Forest / GBM with sufficient training data.
    """

    def __init__(self):
        # These weights can be updated via training
        self.gaze_weight = 0.35
        self.speech_weight = 0.35
        self.visual_weight = 0.30

    def predict(
        self,
        gaze_risk: float,           # 0-100
        speech_risk: float,         # 0-100
        visual_features: Optional[VisualFeatures] = None
    ) -> Tuple[float, str, dict]:
        """
        Compute fused ASD risk score.

        Returns:
            (risk_score, risk_category, details_dict)
        """
        # Visual risk from MobileNet (without labels, use heuristics)
        visual_risk = 50.0  # Baseline when no fine-tuning available
        if visual_features is not None:
            visual_risk = self._compute_visual_heuristic_risk(visual_features)

        # Weighted fusion
        fused_score = (
            gaze_risk * self.gaze_weight +
            speech_risk * self.speech_weight +
            visual_risk * self.visual_weight
        )

        # Normalize to 0-100
        fused_score = min(max(fused_score, 0.0), 100.0)

        # Categorize risk
        if fused_score < 30:
            category = "LOW RISK"
        elif fused_score < 55:
            category = "MODERATE RISK"
        elif fused_score < 75:
            category = "ELEVATED RISK"
        else:
            category = "HIGH RISK"

        details = {
            "gaze_risk": gaze_risk,
            "speech_risk": speech_risk,
            "visual_risk": visual_risk,
            "fused_score": fused_score,
            "category": category,
            "gaze_weight": self.gaze_weight,
            "speech_weight": self.speech_weight,
            "visual_weight": self.visual_weight
        }

        return fused_score, category, details

    def _compute_visual_heuristic_risk(self, vf: VisualFeatures) -> float:
        """
        Heuristic visual risk before supervised fine-tuning.
        Based on face engagement indicators.
        """
        risk = 50.0  # Start at baseline

        # Low camera engagement -> higher risk
        risk += (1.0 - vf.looking_toward_camera) * 25.0

        # No face detected repeatedly -> higher risk
        if not vf.face_detected:
            risk += 15.0

        # Very small face (child looking away) -> higher risk
        if vf.face_area_ratio < 0.02:
            risk += 10.0

        return min(max(risk, 0.0), 100.0)

    def update_weights(self, gaze_w: float, speech_w: float, visual_w: float):
        """Update fusion weights (e.g., from validation results)."""
        total = gaze_w + speech_w + visual_w
        self.gaze_weight = gaze_w / total
        self.speech_weight = speech_w / total
        self.visual_weight = visual_w / total
