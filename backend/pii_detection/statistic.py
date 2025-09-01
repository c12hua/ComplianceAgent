from typing import List, Tuple, Dict, Any
import numpy as np
from finalResult import judge_pii_with_deepseek
import json

"""
    总体实验数据统计函数
    
    输入:
    text(原始文本)  --数组？缺少实际输入案例
    actual_result(正确的归类信息) --数组？缺少实际输入案例
    
    返回:
    包含准确率、精确率、召回率和F1值的字典
"""


def overall_experiment_statistics(
    text, 
    actual_result
):
    pred = [] # 存入预测信息    
    for item in text:
        result = judge_pii_with_deepseek(item)
        data = json.loads(result)
        pred.append(data["is_sensitive"])


    # 转换为numpy数组以便计算
    true_array = np.array(actual_result)
    pred_array = np.array(pred)
    
    # 计算混淆矩阵元素
    TP = np.sum((true_array == 1) & (pred_array == 1))  # 真正例
    FP = np.sum((true_array == 0) & (pred_array == 1))  # 假正例
    TN = np.sum((true_array == 0) & (pred_array == 0))  # 真负例
    FN = np.sum((true_array == 1) & (pred_array == 0))  # 假负例
    
    # 计算各项指标
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 返回结果字典
    return {
        'accuracy': round(accuracy, 4),
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1_score, 4),
        'confusion_matrix': {
            'TP': TP,
            'FP': FP,
            'TN': TN,
            'FN': FN
        }
    }


