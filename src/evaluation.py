"""
Unified Model Evaluation & Experiment Analysis Module (Phase 11).
Computes evaluation metrics:
- Accuracy, Precision, Recall, F1-Score, Matthews Correlation Coefficient (MCC)
- Confusion Matrices (TN, FP, FN, TP)
- Prediction distributions (Actual UP/DOWN vs Predicted UP/DOWN)
- Naive Baselines (Majority Class & Uniform Random Baseline)
- Visualizations & Comparative Artifacts
"""

import os
import logging
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("evaluation")

# Set matplotlib non-interactive backend
plt.switch_backend('Agg')


def evaluate_baselines(y_test: pd.Series) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluates naive benchmark baselines on the test set:
    1. Majority Class Baseline (Always predicts the most frequent class in y_test).
    2. Random Baseline (Uniform random guessing with seed 42).

    Parameters:
    -----------
    y_test : pd.Series
        Testing target labels.

    Returns:
    --------
    Tuple[Dict[str, Any], Dict[str, Any]]
        (majority_results, random_results)
    """
    y_test_arr = np.array(y_test)
    n_samples = len(y_test_arr)

    # 1. Majority Class Baseline
    majority_class = int(pd.Series(y_test_arr).mode()[0])
    y_pred_maj = np.full(n_samples, fill_value=majority_class)

    acc_maj = float(accuracy_score(y_test_arr, y_pred_maj))
    prec_maj = float(precision_score(y_test_arr, y_pred_maj, zero_division=0))
    rec_maj = float(recall_score(y_test_arr, y_pred_maj, zero_division=0))
    f1_maj = float(f1_score(y_test_arr, y_pred_maj, zero_division=0))
    mcc_maj = float(matthews_corrcoef(y_test_arr, y_pred_maj))

    cm_maj = confusion_matrix(y_test_arr, y_pred_maj)

    majority_res = {
        'model_name': f'Majority Class ({majority_class})',
        'accuracy': acc_maj,
        'precision': prec_maj,
        'recall': rec_maj,
        'f1_score': f1_maj,
        'mcc': mcc_maj,
        'confusion_matrix': cm_maj,
        'n_pred_up': int((y_pred_maj == 1).sum()),
        'n_pred_down': int((y_pred_maj == 0).sum()),
        'n_actual_up': int((y_test_arr == 1).sum()),
        'n_actual_down': int((y_test_arr == 0).sum())
    }

    # 2. Uniform Random Baseline (50/50 seed 42)
    np.random.seed(42)
    y_pred_rnd = np.random.choice([0, 1], size=n_samples)

    acc_rnd = float(accuracy_score(y_test_arr, y_pred_rnd))
    prec_rnd = float(precision_score(y_test_arr, y_pred_rnd, zero_division=0))
    rec_rnd = float(recall_score(y_test_arr, y_pred_rnd, zero_division=0))
    f1_rnd = float(f1_score(y_test_arr, y_pred_rnd, zero_division=0))
    mcc_rnd = float(matthews_corrcoef(y_test_arr, y_pred_rnd))

    cm_rnd = confusion_matrix(y_test_arr, y_pred_rnd)

    random_res = {
        'model_name': 'Random Baseline',
        'accuracy': acc_rnd,
        'precision': prec_rnd,
        'recall': rec_rnd,
        'f1_score': f1_rnd,
        'mcc': mcc_rnd,
        'confusion_matrix': cm_rnd,
        'n_pred_up': int((y_pred_rnd == 1).sum()),
        'n_pred_down': int((y_pred_rnd == 0).sum()),
        'n_actual_up': int((y_test_arr == 1).sum()),
        'n_actual_down': int((y_test_arr == 0).sum())
    }

    return majority_res, random_res


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: pd.Series,
    model_name: str
) -> Dict[str, Any]:
    """
    Evaluates a trained classifier on the test dataset.
    """
    logger.info(f"Evaluating {model_name}...")
    y_pred = model.predict(X_test)
    y_test_arr = np.array(y_test)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    acc = float(accuracy_score(y_test_arr, y_pred))
    prec = float(precision_score(y_test_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_test_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_test_arr, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_test_arr, y_pred))

    cm = confusion_matrix(y_test_arr, y_pred)
    tn, fp, fn, tp = cm.ravel()

    report = classification_report(y_test_arr, y_pred, zero_division=0)

    results = {
        'model_name': model_name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'mcc': mcc,
        'confusion_matrix': cm,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'n_pred_up': int((y_pred == 1).sum()),
        'n_pred_down': int((y_pred == 0).sum()),
        'n_actual_up': int((y_test_arr == 1).sum()),
        'n_actual_down': int((y_test_arr == 0).sum()),
        'y_pred': y_pred,
        'y_prob': y_prob,
        'classification_report': report
    }

    logger.info(f"{model_name} Metrics -> Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f}")
    return results


def evaluate_quantum_model(
    model_params: Dict[str, Any],
    X_test_q_scaled: np.ndarray,
    y_test: pd.Series,
    model_name: str = "Quantum VQC"
) -> Dict[str, Any]:
    """
    Evaluates trained Quantum VQC model on the test dataset.
    """
    from src.quantum_model import predict_quantum

    logger.info(f"Evaluating {model_name}...")
    y_pred, y_prob = predict_quantum(model_params, X_test_q_scaled)
    y_test_arr = np.array(y_test)

    acc = float(accuracy_score(y_test_arr, y_pred))
    prec = float(precision_score(y_test_arr, y_pred, zero_division=0))
    rec = float(recall_score(y_test_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_test_arr, y_pred, zero_division=0))
    mcc = float(matthews_corrcoef(y_test_arr, y_pred))

    cm = confusion_matrix(y_test_arr, y_pred)
    tn, fp, fn, tp = cm.ravel()

    report = classification_report(y_test_arr, y_pred, zero_division=0)

    results = {
        'model_name': model_name,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'mcc': mcc,
        'confusion_matrix': cm,
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
        'n_pred_up': int((y_pred == 1).sum()),
        'n_pred_down': int((y_pred == 0).sum()),
        'n_actual_up': int((y_test_arr == 1).sum()),
        'n_actual_down': int((y_test_arr == 0).sum()),
        'y_pred': y_pred,
        'y_prob': y_prob,
        'classification_report': report
    }

    logger.info(f"{model_name} Metrics -> Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f} | MCC: {mcc:.4f}")
    return results


def compare_models(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Generates a unified model comparison summary table including MCC and prediction counts.
    """
    rows = []
    for model_name, res in results_dict.items():
        rows.append({
            'Model': model_name,
            'Accuracy': round(res['accuracy'], 4),
            'Precision': round(res['precision'], 4),
            'Recall': round(res['recall'], 4),
            'F1 Score': round(res['f1_score'], 4),
            'MCC': round(res['mcc'], 4),
            'Pred UP': res['n_pred_up'],
            'Pred DOWN': res['n_pred_down']
        })

    df_comp = pd.DataFrame(rows)
    return df_comp


def get_feature_importance(model: Any, feature_names: List[str]) -> pd.DataFrame:
    """
    Extracts and ranks feature importances for tree-based models.
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError("Provided model does not contain 'feature_importances_' attribute.")

    importances = model.feature_importances_
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

    return df_imp


def save_evaluation_plots(
    results_dict: Dict[str, Dict[str, Any]],
    rf_imp: Optional[pd.DataFrame] = None,
    xgb_imp: Optional[pd.DataFrame] = None,
    results_dir: str = "models/classical/results"
):
    """
    Saves confusion matrices, feature importance charts, and comparison plots.
    """
    os.makedirs(results_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Confusion Matrix Plots
    for model_name, res in results_dict.items():
        cm = res['confusion_matrix']
        clean_name = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Purples" if "quantum" in clean_name else "Blues",
            xticklabels=["DOWN (0)", "UP (1)"],
            yticklabels=["DOWN (0)", "UP (1)"]
        )
        plt.title(f"Confusion Matrix — {model_name}", fontsize=14, pad=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"cm_{clean_name}.png"), dpi=300)
        plt.close()

    # 2. Model Comparison Bar Chart
    df_comp = compare_models(results_dict)
    plot_metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "MCC"]
    df_melt = df_comp.melt(id_vars=["Model"], value_vars=[m for m in plot_metrics if m in df_comp.columns], var_name="Metric", value_name="Score")

    plt.figure(figsize=(12, 6))
    chart = sns.barplot(data=df_melt, x="Model", y="Score", hue="Metric", palette="Set2")
    plt.title("Model Performance Comparison (Baselines vs. Classical ML vs. Quantum VQC)", fontsize=15, pad=15)
    plt.ylim(-0.2, 1.1)
    plt.ylabel("Score / Index", fontsize=12)

    for p in chart.patches:
        height = p.get_height()
        if not np.isnan(height) and abs(height) > 0.001:
            chart.annotate(
                f"{height:.2f}",
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom' if height >= 0 else 'top', fontsize=8, xytext=(0, 3 if height >= 0 else -8),
                textcoords='offset points'
            )

    plt.legend(title="Metric", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "model_comparison.png"), dpi=300)
    plt.close()

    # 3. Top 10 Feature Importance Plots
    for imp_df, m_label in [(rf_imp, "Random Forest"), (xgb_imp, "XGBoost")]:
        if imp_df is not None and not imp_df.empty:
            clean_name = m_label.lower().replace(" ", "_")
            top10 = imp_df.head(10).sort_values(by="Importance", ascending=True)

            plt.figure(figsize=(8, 6))
            plt.barh(top10['Feature'], top10['Importance'], color="#2b5c8f")
            plt.title(f"Top 10 Feature Importances — {m_label}", fontsize=14, pad=12)
            plt.xlabel("Importance Score", fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, f"feature_importance_{clean_name}.png"), dpi=300)
            plt.close()

    logger.info(f"Saved all evaluation charts to {results_dir}")
