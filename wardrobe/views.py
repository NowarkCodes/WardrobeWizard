import os
from io import BytesIO

import cv2
import numpy as np
import tensorflow as tf
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from .forms import ImageUploadForm
from .models import UserImage


# Load the pre-trained MobileNet model
model_path = os.path.join(os.path.dirname(__file__), 'custom_fashion_model.h5')
model = load_model(model_path)


def home(request):
    return render(request, 'wardrobe/home.html')

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Account created for {username}!')
            return redirect('login')
    else:
        form = UserCreationForm()
    return render(request, 'wardrobe/register.html', {'form': form})

# Define the class labels (replace with your actual labels)
class_labels = [
    "T-Shirt", "Longsleeve", "Pants", "Shoes", "Shirt", "Dress", "Outwear",
    "Shorts", "Not sure", "Hat", "Skirt", "Polo", "Undershirt", "Blazer",
    "Hoodie", "Body", "Other", "Top", "Blouse", "Skip"
]

@login_required
def upload_image(request):
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            files = request.FILES.getlist('image')
            for file in files:
                # Save the image to the database
                user_image = UserImage.objects.create(user=request.user, image=file)

                # Load and preprocess the image
                img = image.load_img(user_image.image.path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0  # Normalize pixel values

                # Predict the label
                predictions = model.predict(img_array)
                predicted_class = np.argmax(predictions, axis=1)
                predicted_label = class_labels[predicted_class[0]]

                # Save the predicted label
                user_image.labels = predicted_label
                user_image.save()

                # Debug: Print the predicted label
                print(f"Predicted Label: {predicted_label}")

            return redirect('image_result', image_id=user_image.id)
    else:
        form = ImageUploadForm()
    return render(request, 'wardrobe/upload.html', {'form': form})

@login_required
def history(request):
    # pylint: disable=no-member
    images = UserImage.objects.filter(user=request.user)
    return render(request, 'wardrobe/history.html', {'images': images})

@login_required
def delete_image(request, image_id):
    # Get the image object or return a 404 error if it doesn't exist
    image = get_object_or_404(UserImage, id=image_id, user=request.user)
    
    # Delete the image
    image.delete()
    
    # Redirect to the history page
    return redirect('history')

@login_required
def menu(request):
    return render(request, 'wardrobe/menu.html')

@login_required
def profile(request):
    user_images = UserImage.objects.filter(user=request.user)
    return render(request, 'wardrobe/profile.html', {
        'user': request.user,
        'user_images': user_images,
    })

@login_required
def image_result(request, image_id):
    image = get_object_or_404(UserImage, id=image_id, user=request.user)
    user_images = UserImage.objects.filter(user=request.user)
    recommendations = advanced_recommend_outfits(user_images)
    return render(request, 'wardrobe/image_result.html', {
        'image': image,
        'recommendations': recommendations,
    })

def advanced_recommend_outfits(user_images):
    recommendations = []
    for image in user_images:
        # Extract labels
        labels = image.labels.split(', ') if image.labels else []
        print(f"Processing image {image.id}: Labels={labels}")  # Debug

        for other_image in user_images:
            if image.id != other_image.id:
                other_labels = other_image.labels.split(', ') if other_image.labels else []
                print(f"Comparing with image {other_image.id}: Labels={other_labels}")  # Debug

                # Style Matching Rules
                if ('T-Shirt' in labels and any(label in other_labels for label in ['Pants', 'Shorts', 'Skirt'])) or \
                   ('Shirt' in labels and any(label in other_labels for label in ['Pants', 'Shorts', 'Skirt'])) or \
                   ('Polo' in labels and any(label in other_labels for label in ['Pants', 'Shorts', 'Skirt'])) or \
                   ('Dress' in labels and any(label in other_labels for label in ['Outwear', 'Blazer', 'Hoodie'])) or \
                   ('Top' in labels and any(label in other_labels for label in ['Pants', 'Shorts', 'Skirt'])) or \
                   ('Blouse' in labels and any(label in other_labels for label in ['Pants', 'Shorts', 'Skirt'])) or \
                   ('Outwear' in labels and any(label in other_labels for label in ['Dress', 'T-Shirt', 'Shirt'])) or \
                   ('Blazer' in labels and any(label in other_labels for label in ['Dress', 'T-Shirt', 'Shirt'])) or \
                   ('Hoodie' in labels and any(label in other_labels for label in ['Pants', 'Shorts', 'Skirt'])) or \
                   ('Skirt' in labels and any(label in other_labels for label in ['T-Shirt', 'Shirt', 'Top', 'Blouse'])) or \
                   ('Shorts' in labels and any(label in other_labels for label in ['T-Shirt', 'Shirt', 'Top', 'Blouse'])) or \
                   ('Pants' in labels and any(label in other_labels for label in ['T-Shirt', 'Shirt', 'Top', 'Blouse'])):
                    print(f"Style match found: {image.id} and {other_image.id}")  # Debug
                    recommendations.append((image, other_image))

    print(f"Total recommendations: {len(recommendations)}")  # Debug
    return recommendations

def are_colors_complementary(color1, color2):
    """
    Check if two colors are complementary.
    This is a basic example; you can use a color theory library for better results.
    """
    # Convert RGB strings to tuples
    def rgb_to_tuple(rgb):
        return tuple(map(int, rgb.strip('rgb()').split(',')))

    color1_rgb = rgb_to_tuple(color1)
    color2_rgb = rgb_to_tuple(color2)

    # Basic complementary color check
    return abs(color1_rgb[0] - color2_rgb[0]) > 100 and \
           abs(color1_rgb[1] - color2_rgb[1]) > 100 and \
           abs(color1_rgb[2] - color2_rgb[2]) > 100