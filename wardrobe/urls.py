from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('menu/', views.menu, name='menu'),
    path('upload/', views.upload_image, name='upload'),
    path('history/', views.history, name='history'),
    path('delete/<int:image_id>/', views.delete_image, name='delete_image'),
    path('profile/', views.profile, name='profile'),  # Add this line
    path('result/<int:image_id>/', views.image_result, name='image_result'),
]