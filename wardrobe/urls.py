from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('menu/', views.menu, name='menu'),
    path('upload/', views.upload_image, name='upload'),
    path('history/', views.history, name='history'),
    path('delete/<int:image_id>/', views.delete_image, name='delete_image'),
    path('edit/<int:image_id>/', views.edit_item, name='edit_item'),
    path('worn/<int:image_id>/', views.mark_worn, name='mark_worn'),
    path('profile/', views.profile, name='profile'),
    path('result/<int:image_id>/', views.image_result, name='image_result'),
    path('stats/', views.wardrobe_stats, name='wardrobe_stats'),
    path('outfits/save/', views.save_outfit, name='save_outfit'),
    path('outfits/', views.saved_outfits, name='saved_outfits'),
    path('outfits/delete/<int:outfit_id>/', views.delete_outfit, name='delete_outfit'),
]