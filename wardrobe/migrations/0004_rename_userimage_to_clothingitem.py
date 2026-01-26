# Generated manually for migrating UserImage to ClothingItem

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('wardrobe', '0003_alter_userimage_colors_alter_userimage_image_and_more'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        # First rename UserImage to ClothingItem
        migrations.RenameModel(
            old_name='UserImage',
            new_name='ClothingItem',
        ),
        
        # Rename 'labels' field to 'category'
        migrations.RenameField(
            model_name='clothingitem',
            old_name='labels',
            new_name='category',
        ),
        
        # Remove the 'colors' field (not in new model)
        migrations.RemoveField(
            model_name='clothingitem',
            name='colors',
        ),
        
        # Add new fields to ClothingItem
        migrations.AddField(
            model_name='clothingitem',
            name='classification_confidence',
            field=models.FloatField(default=0.0),
        ),
        migrations.AddField(
            model_name='clothingitem',
            name='notes',
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='clothingitem',
            name='wear_count',
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name='clothingitem',
            name='last_worn_date',
            field=models.DateField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='clothingitem',
            name='is_active',
            field=models.BooleanField(default=True),
        ),
        migrations.AddField(
            model_name='clothingitem',
            name='updated_at',
            field=models.DateTimeField(auto_now=True),
        ),
        
        # Alter category field to use choices
        migrations.AlterField(
            model_name='clothingitem',
            name='category',
            field=models.CharField(
                blank=True, 
                choices=[
                    ('T-Shirt', 'T-Shirt'), ('Longsleeve', 'Longsleeve'), ('Pants', 'Pants'),
                    ('Shoes', 'Shoes'), ('Shirt', 'Shirt'), ('Dress', 'Dress'), ('Outwear', 'Outwear'),
                    ('Shorts', 'Shorts'), ('Hat', 'Hat'), ('Skirt', 'Skirt'), ('Polo', 'Polo'),
                    ('Undershirt', 'Undershirt'), ('Blazer', 'Blazer'), ('Hoodie', 'Hoodie'),
                    ('Body', 'Body'), ('Top', 'Top'), ('Blouse', 'Blouse'), ('Other', 'Other'),
                ], 
                max_length=50, 
                null=True
            ),
        ),
        
        # Add model ordering
        migrations.AlterModelOptions(
            name='clothingitem',
            options={'ordering': ['-uploaded_at']},
        ),
        
        # Create Outfit model
        migrations.CreateModel(
            name='Outfit',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('occasion', models.CharField(blank=True, choices=[('casual', 'Casual'), ('work', 'Work'), ('formal', 'Formal'), ('athletic', 'Athletic')], max_length=20, null=True)),
                ('season', models.CharField(blank=True, choices=[('spring', 'Spring'), ('summer', 'Summer'), ('fall', 'Fall'), ('winter', 'Winter')], max_length=20, null=True)),
                ('is_saved', models.BooleanField(default=True)),
                ('wear_count', models.IntegerField(default=0)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('last_worn_date', models.DateField(blank=True, null=True)),
                ('items', models.ManyToManyField(related_name='outfits', to='wardrobe.clothingitem')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['-created_at'],
            },
        ),
        
        # Create WearLog model
        migrations.CreateModel(
            name='WearLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('date_worn', models.DateField()),
                ('notes', models.TextField(blank=True, null=True)),
                ('items', models.ManyToManyField(related_name='wear_logs', to='wardrobe.clothingitem')),
                ('outfit', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, to='wardrobe.outfit')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['-date_worn'],
            },
        ),
    ]
