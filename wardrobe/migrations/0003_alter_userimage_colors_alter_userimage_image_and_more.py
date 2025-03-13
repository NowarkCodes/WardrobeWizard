# Generated by Django 5.1.7 on 2025-03-13 09:34

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('wardrobe', '0002_userimage_colors_userimage_labels'),
    ]

    operations = [
        migrations.AlterField(
            model_name='userimage',
            name='colors',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
        migrations.AlterField(
            model_name='userimage',
            name='image',
            field=models.ImageField(upload_to='uploads/'),
        ),
        migrations.AlterField(
            model_name='userimage',
            name='labels',
            field=models.CharField(blank=True, max_length=255, null=True),
        ),
    ]
