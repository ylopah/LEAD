import numpy as np
import logging
from cdt.metrics import SID
from utils import adj_mat_to_edge_list, is_dag
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss, confusion_matrix

logger = logging.getLogger("Metrics")

def accuracy(B_true, B_est):
  B_true = set(B_true)
  B_est = set(B_est)
  return len(B_true.intersection(B_est)) / len(B_true.union(B_est))

def precision(B_true, B_est):
  B_true = set(B_true)
  B_est = set(B_est)
  if len(B_est) == 0:
    return 0.0
  else:
    return len(B_true.intersection(B_est)) / len(B_est)

def recall(B_true, B_est):
  B_true = set(B_true)
  B_est = set(B_est)
  return len(B_true.intersection(B_est)) / len(B_true)

def F_score(B_true, B_est):
  """Edge-list based F-score calculation."""
  # Note: Keep this if other parts of your code rely on edge-list inputs
  p = precision(B_true, B_est)
  r = recall(B_true, B_est)
  if p + r == 0:
    return 0.0
  return 2 * p * r / (p + r)


def normalized_hamming_distance(prediction, target, nodes):
  """
  Calculate the normalized hamming distance and reference values.
  """
  prediction_set  = set(prediction)
  target_set  = set(target)

  # Use the length of nodes list as the base
  num_nodes = len(nodes) if isinstance(nodes, list) else 0
  if num_nodes == 0:
    # Fallback to unique nodes in target if list is empty
    unique_nodes = set()
    for i, j in target_set:
      unique_nodes.add(i); unique_nodes.add(j)
    num_nodes = len(unique_nodes)

  no_overlap = len(prediction_set.union(target_set)) - len(prediction_set.intersection(target_set))
  nhd = no_overlap / (num_nodes ** 2) if num_nodes > 0 else 0
  
  # Reference NHD represents a baseline error
  reference_nhd = min((len(prediction_set) + len(target_set)), (num_nodes ** 2)) / (num_nodes ** 2) if num_nodes > 0 else 1.0
  
  return nhd, reference_nhd, (nhd / reference_nhd if reference_nhd > 0 else 0)

def compute_metrics(B_true_matrix, B_est_matrix, nodes):
  """
  Main evaluation function for causal discovery.
  """
  # Ensure inputs are numpy arrays
  B_true_matrix = np.array(B_true_matrix)
  B_est_matrix = np.array(B_est_matrix)

  # 1. Structural Checks
  is_dag_true = bool(is_dag(B_true_matrix))
  is_dag_est = bool(is_dag(B_est_matrix))

  # 2. Convert to edge lists for custom NHD logic
  B_true_edges = adj_mat_to_edge_list(B_true_matrix)
  B_est_edges = adj_mat_to_edge_list(B_est_matrix)

  # 3. Custom NHD Calculations
  nhd_stats = normalized_hamming_distance(B_est_edges, B_true_edges, nodes)

  # 4. Binary Classification Metrics (Flattened Matrix)
  y_true = B_true_matrix.flatten()
  y_est = B_est_matrix.flatten()
  
  try:
    tn, fp, fn, tp = confusion_matrix(y_true, y_est, labels=[0, 1]).ravel()
  except ValueError:
    tn, fp, fn, tp = 0, 0, 0, 0

  # 5. SID Calculation (requires matrix input)
  try:
    sid_val = float(SID(B_true_matrix, B_est_matrix))
  except Exception:
    sid_val = -1.0 # Indicator for Non-DAG graphs

  # 6. SHD (Structural Hamming Distance)
  shd = int(fp + fn)
  
  return {
    'Precision': float(precision_score(y_true, y_est, zero_division=0)),
    'Recall': float(recall_score(y_true, y_est, zero_division=0)),
    'F1-score': float(f1_score(y_true, y_est, zero_division=0)),
    'Accuracy': float(accuracy_score(y_true, y_est)),
    'SHD': shd,
    'SID': sid_val,
    'TP': int(tp),
    'FP': int(fp),
    'FN': int(fn),
    'TN': int(tn),
    'Is_True_DAG': is_dag_true,
    'Is_Estimated_DAG': is_dag_est,
    'Predicted_Edges': len(B_est_edges),
    'True_Edges': len(B_true_edges),
    'NHD': float(hamming_loss(y_true, y_est)),
    'Reference_NHD': float(nhd_stats[1]),
    'NHD_Ratio': float(nhd_stats[2])
}