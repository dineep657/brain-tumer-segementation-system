import os
import time
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

from train_2d_classifier import LightweightClassifier2D, CLASSES

_trained_classifier_instance = None
CLASSIFIER_CHECKPOINT_PATH = os.path.join("models", "brain_tumor_classifier_2d.pth")

def load_trained_classifier() -> LightweightClassifier2D:
    """
    Loads the trained PyTorch 2D Multi-Class Classifier checkpoint from disk ('models/brain_tumor_classifier_2d.pth').
    Sets the classifier strictly to evaluation mode (eval()) for real inference.
    """
    global _trained_classifier_instance
    if _trained_classifier_instance is None:
        if not os.path.exists(CLASSIFIER_CHECKPOINT_PATH):
            raise FileNotFoundError(f"Trained classifier checkpoint missing at: {CLASSIFIER_CHECKPOINT_PATH}")
            
        model = LightweightClassifier2D(num_classes=len(CLASSES))
        checkpoint = torch.load(CLASSIFIER_CHECKPOINT_PATH, map_location=torch.device('cpu'))
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval()
        _trained_classifier_instance = model
        print(f"Loaded trained 2D Classifier checkpoint from '{CLASSIFIER_CHECKPOINT_PATH}'.")
    return _trained_classifier_instance

def classify_tumor_2d(tensor: torch.Tensor, tumor_detected: bool, min_confidence_threshold: float = 0.35) -> Dict[str, Any]:
    """
    Executes multi-class tumor type classification using the loaded trained classifier.
    Pipeline Scoping:
    - If NO tumor is detected by UNet segmentation: returns 'No Tumor'.
    - If tumor IS detected: passes tensor through classifier, evaluates softmax probabilities over ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor'].
    - If model confidence < min_confidence_threshold (0.35): returns 'Unknown / Uncertain'.
    """
    start_time = time.perf_counter()
    
    if not tumor_detected:
        end_time = time.perf_counter()
        return {
            "tumor_type": "No Tumor",
            "confidence": 1.0,
            "raw_probabilities": {c: (1.0 if c == "No Tumor" else 0.0) for c in CLASSES},
            "classifier_called": False,
            "execution_time_ms": round((end_time - start_time) * 1000.0, 2),
            "uncertain": False
        }
        
    classifier = load_trained_classifier()
    
    with torch.no_grad():
        logits = classifier(tensor) # (1, 4)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        
        top_idx = int(np.argmax(probabilities))
        top_class = CLASSES[top_idx]
        top_confidence = float(probabilities[top_idx])
        
        prob_dict = {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}
        
        if top_confidence < min_confidence_threshold:
            final_type = "Unknown / Uncertain"
            is_uncertain = True
        else:
            final_type = top_class
            is_uncertain = False
            
    end_time = time.perf_counter()
    execution_time_ms = round((end_time - start_time) * 1000.0, 2)
    
    return {
        "tumor_type": final_type,
        "confidence": top_confidence,
        "raw_probabilities": prob_dict,
        "classifier_called": True,
        "execution_time_ms": execution_time_ms,
        "uncertain": is_uncertain
    }
