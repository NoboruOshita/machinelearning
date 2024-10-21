from django.db import models
# Create your models here.

class auth_user(models.Model):
    id=models.BigAutoField(primary_key=True)
    password=models.CharField(max_length=128, null=False, blank=True)
    last_login=models.DateTimeField(null=True, blank=True)
    username=models.CharField(max_length=150, null=False, blank=True)
    email=models.CharField(max_length=150, null=False, blank=True)
    class Meta:
        managed = False
        db_table = 'auth_user'

class blacklist(models.Model):
    id_blacklist=models.BigAutoField(primary_key=True)
    IP=models.CharField(max_length=100, null=False, blank=True)
    ransomware_type=models.CharField(max_length=100, null=False, blank=True)
    class Meta:
        managed = False
        db_table = 'blacklist'

class data_ransomware(models.Model):
    id_data_ransomware=models.BigAutoField(primary_key=True)
    id_user=models.ForeignKey('auth_user', on_delete=models.CASCADE)
    id_blacklist=models.ForeignKey('blacklist', on_delete=models.CASCADE)
    timestamp_s=models.BigIntegerField(null=True, blank=True)
    timestamp_ms=models.BigIntegerField(null=True, blank=True)
    lba=models.FloatField(null=True, blank=True)
    block_size=models.FloatField(null=True, blank=True)
    entropy_shannon=models.FloatField(null=True, blank=True)
    class Meta:
        managed = False
        db_table = 'data_ransomware'

class detection(models.Model):
    id_detection=models.BigAutoField(primary_key=True)
    id_data_ransomware=models.ForeignKey('data_ransomware', on_delete=models.CASCADE)
    ransomware_detected=models.BooleanField(default=False)
    ransomware_type=models.CharField(max_length=100, null=False, blank=True)
    percentage_reliability=models.FloatField(null=True, blank=True)
    detection_date=models.DateField(null=True, blank=True)
    class Meta:
        managed = False
        db_table = 'detection'

class detector_rf(models.Model):
    id_detector_rf=models.BigAutoField(primary_key=True)
    id_detection=models.ForeignKey('detection', on_delete=models.CASCADE)
    precision=models.FloatField(null=True, blank=True)
    accuracy=models.FloatField(null=True, blank=True)
    recall=models.FloatField(null=True, blank=True)
    f1_score=models.FloatField(null=True, blank=True)
    class Meta:
        managed = False
        db_table = 'detector_rf'

class detector_xgboost(models.Model):
    id_detector_xgb=models.BigAutoField(primary_key=True)
    id_detection=models.ForeignKey('detection', on_delete=models.CASCADE)
    precision=models.FloatField(null=True, blank=True)
    accuracy=models.FloatField(null=True, blank=True)
    recall=models.FloatField(null=True, blank=True)
    f1_score=models.FloatField(null=True, blank=True)
    class Meta:
        managed = False
        db_table = 'detector_xgboost'

class response(models.Model):
    id_response=models.BigAutoField(primary_key=True)
    id_detection=models.ForeignKey('detection', on_delete=models.CASCADE)
    action=models.CharField(max_length=500, null=False, blank=True)
    detail=models.CharField(max_length=800, null=False, blank=True)
    response_date=models.DateField(null=True, blank=True)
    class Meta:
        managed = False
        db_table = 'response'

class logs(models.Model):
    id_logs=models.BigAutoField(primary_key=True)
    id_detection=models.ForeignKey('detection', on_delete=models.CASCADE)
    id_response=models.ForeignKey('response', on_delete=models.CASCADE)
    class Meta:
        managed = False
        db_table = 'logs'