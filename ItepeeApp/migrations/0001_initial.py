# Generated by Django 4.1.7 on 2023-03-13 02:05

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='TweetModel',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('tweet', models.CharField(max_length=250)),
                ('is_hate_speech', models.IntegerField()),
                ('target_hs_individu', models.IntegerField()),
                ('target_hs_group', models.IntegerField()),
                ('type_hs_religion', models.IntegerField()),
                ('type_hs_ras', models.IntegerField()),
                ('type_hs_physical', models.IntegerField()),
                ('type_hs_gender', models.IntegerField()),
                ('type_hs_other', models.IntegerField()),
                ('level_hs_weak', models.IntegerField()),
                ('level_hs_moderate', models.IntegerField()),
                ('level_hs_strong', models.IntegerField()),
            ],
        ),
    ]
