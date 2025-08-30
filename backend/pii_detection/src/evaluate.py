import os 
import json
import argparse

def normalize_entity_text(entity_text):
    """标准化实体文本，例如将'76岁'转换为'76'以便比较"""
    if entity_text.endswith('岁'):
        return entity_text[:-1]
    return entity_text

def load_entities(path):
    """读取一个 JSON 文件，返回 (entity, type) 集合"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entities = set()
    for e in data.get("entities", []):
        entity = e.get("entity")
        etype = e.get("type")
        if entity and etype:
            normalized_entity = normalize_entity_text(entity)
            entities.add((normalized_entity, etype))
    return entities

def evaluate(gold_dir, pred_dir):
    gold_files = [f for f in os.listdir(gold_dir) if f.endswith(".json")]
    total_tp, total_fp, total_fn = 0, 0, 0

    for fname in gold_files:
        gold_path = os.path.join(gold_dir, fname)
        pred_path = os.path.join(pred_dir, fname)
        if not os.path.exists(pred_path):
            print(f"⚠️ pred 缺少文件: {fname}")
            continue

        gold_entities = load_entities(gold_path)
        pred_entities = load_entities(pred_path)

        tp = len(gold_entities & pred_entities)
        fp = len(pred_entities - gold_entities)
        fn = len(gold_entities - pred_entities)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0.0

    print("\n====== Evaluation Summary ======")
    print(f"Gold dir: {gold_dir}")
    print(f"Pred dir: {pred_dir}")
    print(f"Total TP: {total_tp}, FP: {total_fp}, FN: {total_fn}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 score:  {f1:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, help="gold 文件夹路径")
    parser.add_argument("--pred", required=True, help="pred 文件夹路径")
    args = parser.parse_args()
    evaluate(args.gold, args.pred)
