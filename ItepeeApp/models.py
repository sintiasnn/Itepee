from django.db import models

# Create your models here.
class TweetModel(models.Model):
    id = models.AutoField(primary_key=True)
    tweet = models.CharField(max_length=250)
    is_hate_speech = models.FloatField()
    target_hs_individu = models.FloatField()
    target_hs_group = models.FloatField()
    type_hs_religion = models.FloatField()
    type_hs_ras = models.FloatField()
    type_hs_physical = models.FloatField()
    type_hs_gender = models.FloatField()
    type_hs_other = models.FloatField()
    level_hs_weak = models.FloatField()
    level_hs_moderate = models.FloatField()
    level_hs_strong = models.FloatField()