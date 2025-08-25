import torch
from torch_geometric.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from ..utils.logging import get_logger

class Evaluator:
    """处理模型评估。"""

    def __init__(self, model):
        self.model = model
        self.logger = get_logger(self.__class__.__name__)

    def evaluate(self, dataloader: DataLoader):
        """在给定的数据集上评估模型。"""
        self.model.eval()  # 将模型设置为评估模式
        all_preds = []
        all_labels = []

        # 在没有梯度计算的上下文中进行评估
        with torch.no_grad():
            for batch in dataloader:
                output = self.model(batch)
                # 使用 sigmoid 将 logits 转换为概率，然后以 0.5 为阈值进行分类
                preds = torch.sigmoid(output) > 0.5
                all_preds.append(preds.cpu())
                all_labels.append(batch.y.cpu())

        # 将所有批次的预测和标签连接起来
        all_preds = torch.cat(all_preds).numpy()
        all_labels = torch.cat(all_labels).numpy()

        # 计算各种评估指标
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_preds)

        self.logger.info(f"评估结果:")
        self.logger.info(f"  准确率 (Accuracy): {accuracy:.4f}")
        self.logger.info(f"  精确率 (Precision): {precision:.4f}")
        self.logger.info(f"  召回率 (Recall): {recall:.4f}")
        self.logger.info(f"  F1 分数 (F1-Score): {f1:.4f}")
        self.logger.info(f"  AUC-ROC: {auc:.4f}")

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1, "auc": auc}
