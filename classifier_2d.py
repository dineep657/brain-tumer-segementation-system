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
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        _trained_classifier_instance = model
        print(f"Loaded trained 2D Classifier checkpoint from '{CLASSIFIER_CHECKPOINT_PATH}'.")
    return _trained_classifier_instance

def classify_tumor_2d(tensor: torch.Tensor, tumor_detected: bool, min_confidence_threshold: float = 0.50) -> Dict[str, Any]:
    """
    Executes multi-class tumor type classification using the loaded trained classifier.
    Pipeline Scoping:
    - If NO tumor is detected by UNet segmentation: returns 'No Tumor' without unnecessary computation.
    - If tumor IS detected: passes tensor through classifier, evaluates softmax probabilities over ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor'].
    - If model confidence < 50%: returns 'Unknown / Uncertain'.
    """
    start_time = time.perf_counter()
    
    if not tumor_detected:
        end_time = time.perf_counter()
        return {
            "predicted_class": "No Tumor",
            "classifier_confidence": None,
            "confidence_display": "N/A (No Tumor)",
            "all_class_probabilities": {c: 0.0 for c in CLASSES},
            "classifier_executed": False,
            "classifier_time_ms": round((end_time - start_time) * 1000.0, 2)
        }
        
    classifier = load_trained_classifier()
    
    with torch.no_grad():
        logits = classifier(tensor) # (1, 4)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
        
        max_idx = int(np.argmax(probabilities))
        max_prob = float(probabilities[max_idx])
        pred_label = CLASSES[max_idx]
        
        # Uncertainty handling: If classifier is not confident (prob < 0.50) or predicts No Tumor when UNet detected a tumor
        if max_prob < min_confidence_threshold or pred_label == 'No Tumor':
            final_class_label = "Unknown / Uncertain"
        else:
            final_class_label = pred_label
            
    end_time = time.perf_counter()
    classifier_time_ms = round((end_time - start_time) * 1000.0, 2)
    
    prob_dict = {CLASSES[i]: float(probabilities[i]) for i in range(len(CLASSES))}
    
    return {
        "predicted_class": final_class_label,
        "classifier_confidence": max_prob,
        "confidence_display": f"{max_prob * 100.0:.1f}%",
        "all_class_probabilities": prob_dict,
        "classifier_executed": True,
        "classifier_time_ms": classifier_time_ms
    }

if __name__ == "__main__":
    from inference_2d import preprocess_2d
    print("--- Testing Classifier Inference Module ---")
    t, r, dims = preprocess_2d("data/sample_glioma.png")
    res = classify_tumor_2d(t, tumor_detected=True)
    print(f"Classified Tumor Type: {res['predicted_class']} | Confidence: {res['confidence_display']} | Time: {res['classifier_time_ms']} ms")
