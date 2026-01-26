from django.contrib import admin
from .models import ClothingItem, Outfit, WearLog


@admin.register(ClothingItem)
class ClothingItemAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'category', 'classification_confidence', 'wear_count', 'is_active', 'uploaded_at']
    list_filter = ['category', 'is_active', 'uploaded_at']
    search_fields = ['user__username', 'category', 'notes']
    readonly_fields = ['uploaded_at', 'updated_at']


@admin.register(Outfit)
class OutfitAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'name', 'occasion', 'season', 'wear_count', 'is_saved', 'created_at']
    list_filter = ['occasion', 'season', 'is_saved']
    search_fields = ['user__username', 'name']
    readonly_fields = ['created_at']


@admin.register(WearLog)
class WearLogAdmin(admin.ModelAdmin):
    list_display = ['id', 'user', 'date_worn', 'outfit']
    list_filter = ['date_worn']
    search_fields = ['user__username', 'notes']
