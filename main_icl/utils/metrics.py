"""
Metrics computation utilities for accuracy and entropy-based evaluations.
"""

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from collections import defaultdict
import numpy as np


def compute_accuracy(model_outputs, test_data, args):
    """Compute accuracy from model outputs and ground truth"""
    correct = 0
    total = len(model_outputs)
    
    id_to_answer = {item["id"]: item[args.answer] for item in test_data}
    
    for idx, output in enumerate(model_outputs):
        pred = output["predicted_answers"]
        example_id = output["id"]
        true_answer = id_to_answer[example_id]
        if isinstance(true_answer, list):
            true_answer = [str(x) for x in true_answer]
        else:
            true_answer = [str(true_answer)]
        
        if set(pred) == set(true_answer):
            correct += 1
            model_outputs[idx]["correct"] = True
        else:
            model_outputs[idx]["correct"] = False
        
    accuracy = correct / total
    
    return model_outputs, accuracy


def compute_auroc_aupr_from_entropy(model_outputs):
    """Compute AUROC/AUPR using model's correctness and entropy"""
    y_true = []
    entropy_scores = []

    for output in model_outputs:
        correct = output.get("correct", None)
        entropy = output.get("entropy", None)

        if correct is None or entropy is None:
            raise ValueError("Must run compute_accuracy and compute_entropy first.")

        y_true.append(1 if not correct else 0)  # Label 1 = incorrect
        entropy_scores.append(entropy)

    auroc = roc_auc_score(y_true, entropy_scores)
    aupr = average_precision_score(y_true, entropy_scores)
    return auroc, aupr


def compute_auroc_aupr_from_option_entropy(model_outputs):
    """Compute AUROC/AUPR using model's correctness and option_entropy"""
    y_true = []
    entropy_scores = []

    for output in model_outputs:
        correct = output.get("correct", None)
        entropy = output.get("option_entropy", None)

        if correct is None or entropy is None:
            raise ValueError("Must run compute_accuracy and compute_entropy first.")

        y_true.append(1 if not correct else 0)  # Label 1 = incorrect
        entropy_scores.append(entropy)

    auroc = roc_auc_score(y_true, entropy_scores)
    aupr = average_precision_score(y_true, entropy_scores)
    return auroc, aupr


def compute_cross_auroc_aupr(model_outputs, self_fv_model_outputs):
    """
    Compute AUROC/AUPR using model_outputs' correctness as labels 
    and self_fv_model_outputs' entropy as scores
    """
    if len(model_outputs) != len(self_fv_model_outputs):
        raise ValueError("model_outputs and self_fv_model_outputs must have the same length")
    
    y_true = []
    entropy_scores = []

    for i, (base_output, interv_output) in enumerate(zip(model_outputs, self_fv_model_outputs)):
        # Use base model's correctness as label
        correct = base_output.get("correct", None)
        # Use intervened model's entropy as score
        entropy = interv_output.get("entropy", None)

        if correct is None:
            raise ValueError("Must run compute_accuracy on model_outputs first.")
        if entropy is None:
            raise ValueError("self_fv_model_outputs must have entropy computed.")

        y_true.append(1 if not correct else 0)  # Label 1 = incorrect
        entropy_scores.append(entropy)

    auroc = roc_auc_score(y_true, entropy_scores)
    aupr = average_precision_score(y_true, entropy_scores)
    return auroc, aupr


def compute_cross_auroc_aupr_from_option_entropy(model_outputs, self_fv_model_outputs):
    """
    Compute AUROC/AUPR using model_outputs' correctness as labels 
    and self_fv_model_outputs' option_entropy as scores
    """
    if len(model_outputs) != len(self_fv_model_outputs):
        raise ValueError("model_outputs and self_fv_model_outputs must have the same length")
    
    y_true = []
    entropy_scores = []

    for i, (base_output, interv_output) in enumerate(zip(model_outputs, self_fv_model_outputs)):
        # Use base model's correctness as label
        correct = base_output.get("correct", None)
        # Use intervened model's option_entropy as score
        entropy = interv_output.get("option_entropy", None)

        if correct is None:
            raise ValueError("Must run compute_accuracy on model_outputs first.")
        if entropy is None:
            raise ValueError("self_fv_model_outputs must have option_entropy computed.")

        y_true.append(1 if not correct else 0)  # Label 1 = incorrect
        entropy_scores.append(entropy)

    auroc = roc_auc_score(y_true, entropy_scores)
    aupr = average_precision_score(y_true, entropy_scores)
    return auroc, aupr


def compute_macro_f1(model_outputs, test_data, args):
    """Compute macro-F1 score from model outputs and ground truth"""
    id_to_answer = {item["id"]: item[args.answer] for item in test_data}
    
    y_true = []
    y_pred = []
    
    for output in model_outputs:
        pred = output["predicted_answers"]
        example_id = output["id"]
        true_answer = id_to_answer[example_id]
        
        # Handle both single and multiple answers
        if isinstance(true_answer, list):
            true_answer = [str(x) for x in true_answer]
        else:
            true_answer = [str(true_answer)]
            
        # For multi-label cases, we need to handle this differently
        if len(true_answer) > 1:
            # Multi-label case: convert to binary vectors
            # This requires knowing all possible labels in advance
            all_labels = set()
            for item in test_data:
                labels = item[args.answer]
                if isinstance(labels, list):
                    all_labels.update([str(x) for x in labels])
                else:
                    all_labels.add(str(labels))
            
            all_labels = sorted(list(all_labels))
            
            # Convert to binary vectors
            true_vec = [1 if label in true_answer else 0 for label in all_labels]
            pred_vec = [1 if label in pred else 0 for label in all_labels]
            
            y_true.append(true_vec)
            y_pred.append(pred_vec)
        else:
            # Single-label case
            y_true.append(true_answer[0])
            y_pred.append(pred[0] if pred else "")
    
    # Compute macro-F1
    if len(y_true) > 0 and isinstance(y_true[0], list):
        # Multi-label case
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    else:
        # Single-label case
        macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    return macro_f1
