from rest_framework import serializers
from .models import PiiDetectionRecord

class PiiDetectionRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = PiiDetectionRecord
    fields = '__all__'
