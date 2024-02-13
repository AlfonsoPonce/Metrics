import torch
from torch import Tensor

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

class ClassificationMetrics:
    
    def __init__(self) -> None:
        pass

    
    def accuracy(self, y_preds: Tensor, y_targets: Tensor):
        assert len(y_preds) == len(y_targets)
        return torch.sum(torch.eq(y_preds, y_targets)) / len(y_preds)
    
    def confusion_matrix(self, y_preds: Tensor, y_targets: Tensor):
        return confusion_matrix(y_targets.numpy(), y_preds.numpy())
        
    
    def precision(self, y_preds: Tensor, y_targets: Tensor):
        return precision_score(y_targets.numpy(), y_preds.numpy())
    
    def recall(self, y_preds: Tensor, y_targets: Tensor):
        return recall_score(y_targets.numpy(), y_preds.numpy())
    
    def F1Score(self, y_preds: Tensor, y_targets: Tensor):
        return f1_score(y_targets.numpy(), y_preds.numpy())

    def ROC_AUC(self, y_preds: Tensor, y_targets: Tensor):
        return roc_auc_score(y_targets.numpy(), y_preds.numpy())