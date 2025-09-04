from django.db import models



class PiiDetectionRecord(models.Model):
    text = models.TextField()
    detected_entities = models.JSONField()
    risk_level = models.CharField(max_length=20)
    # 新增：逐实体的 LLM 风险评估结果，格式为 {"实体文本": {"risk_level": "高|中|低|未知", "reason": "..."}, ...}
    entity_assessments = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"PII Detection at {self.created_at} (Risk: {self.risk_level})"
