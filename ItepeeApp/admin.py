from django.contrib import admin
from .models import TweetModel

@admin.register(TweetModel)
class TweetModelAdmin(admin.ModelAdmin):
    list_display = ('id', 'tweet', 'is_hate_speech')
