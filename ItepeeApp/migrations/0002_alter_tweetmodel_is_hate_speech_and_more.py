# Generated by Django 4.1.7 on 2023-03-13 02:13

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ItepeeApp', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='tweetmodel',
            name='is_hate_speech',
            field=models.FloatField(),
        ),
        migrations.AlterField(
            model_name='tweetmodel',
            name='level_hs_moderate',
            field=models.FloatField(),
        ),
        migrations.AlterField(
            model_name='tweetmodel',
            name='level_hs_strong',
            field=models.FloatField(),
        ),
        migrations.AlterField(
            model_name='tweetmodel',
            name='level_hs_weak',
            field=models.FloatField(),
        ),
        migrations.AlterField(
            model_name='tweetmodel',
            name='target_hs_group',
            field=models.FloatField(),
        ),
        migrations.AlterField(
            model_name='tweetmodel',
            name='target_hs_individu',
            field=models.FloatField(),
        ),
        migrations.AlterField(
            model_name='tweetmodel',
            name='type_hs_gender',
            field=models.FloatField(),
        ),
        migrations.AlterField(
            model_name='tweetmodel',
            name='type_hs_other',
            field=models.FloatField(),
        ),
        migrations.AlterField(
            model_name='tweetmodel',
            name='type_hs_physical',
            field=models.FloatField(),
        ),
        migrations.AlterField(
            model_name='tweetmodel',
            name='type_hs_ras',
            field=models.FloatField(),
        ),
        migrations.AlterField(
            model_name='tweetmodel',
            name='type_hs_religion',
            field=models.FloatField(),
        ),
    ]
