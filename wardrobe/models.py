from django.db import models
from django.contrib.auth.models import User


class ClothingItem(models.Model):
    """
    Represents a clothing item in a user's wardrobe.
    Matches PRD section 7.2 Data Models - ClothingItem Model.
    """
    CATEGORY_CHOICES = [
        ('T-Shirt', 'T-Shirt'),
        ('Longsleeve', 'Longsleeve'),
        ('Pants', 'Pants'),
        ('Shoes', 'Shoes'),
        ('Shirt', 'Shirt'),
        ('Dress', 'Dress'),
        ('Outwear', 'Outwear'),
        ('Shorts', 'Shorts'),
        ('Hat', 'Hat'),
        ('Skirt', 'Skirt'),
        ('Polo', 'Polo'),
        ('Undershirt', 'Undershirt'),
        ('Blazer', 'Blazer'),
        ('Hoodie', 'Hoodie'),
        ('Body', 'Body'),
        ('Top', 'Top'),
        ('Blouse', 'Blouse'),
        ('Other', 'Other'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    image = models.ImageField(upload_to='uploads/')
    category = models.CharField(max_length=50, choices=CATEGORY_CHOICES, blank=True, null=True)
    classification_confidence = models.FloatField(default=0.0)
    notes = models.TextField(blank=True, null=True)
    wear_count = models.IntegerField(default=0)
    last_worn_date = models.DateField(blank=True, null=True)
    is_active = models.BooleanField(default=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-uploaded_at']

    def __str__(self):
        return f"{self.user.username}'s {self.category or 'item'}"


class Outfit(models.Model):
    """
    Represents a saved outfit combination.
    Matches PRD section 7.2 Data Models - Outfit Model.
    """
    OCCASION_CHOICES = [
        ('casual', 'Casual'),
        ('work', 'Work'),
        ('formal', 'Formal'),
        ('athletic', 'Athletic'),
    ]

    SEASON_CHOICES = [
        ('spring', 'Spring'),
        ('summer', 'Summer'),
        ('fall', 'Fall'),
        ('winter', 'Winter'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    name = models.CharField(max_length=100)
    items = models.ManyToManyField(ClothingItem, related_name='outfits')
    occasion = models.CharField(max_length=20, choices=OCCASION_CHOICES, blank=True, null=True)
    season = models.CharField(max_length=20, choices=SEASON_CHOICES, blank=True, null=True)
    is_saved = models.BooleanField(default=True)
    wear_count = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    last_worn_date = models.DateField(blank=True, null=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username}'s outfit: {self.name}"


class WearLog(models.Model):
    """
    Tracks when items or outfits are worn.
    Matches PRD section 7.2 Data Models - WearLog Model.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    outfit = models.ForeignKey(Outfit, on_delete=models.SET_NULL, blank=True, null=True)
    items = models.ManyToManyField(ClothingItem, related_name='wear_logs')
    date_worn = models.DateField()
    notes = models.TextField(blank=True, null=True)

    class Meta:
        ordering = ['-date_worn']

    def __str__(self):
        return f"{self.user.username}'s wear log on {self.date_worn}"


# Keep UserImage as an alias for backward compatibility during migration
UserImage = ClothingItem