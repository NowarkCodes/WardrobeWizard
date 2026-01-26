import os
from datetime import date
from collections import Counter

import numpy as np
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.db.models import Count, Sum
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from .forms import ImageUploadForm, ClothingItemEditForm
from .models import ClothingItem, Outfit, WearLog


# Load the pre-trained MobileNet model
model_path = os.path.join(os.path.dirname(__file__), 'custom_fashion_model.h5')
model = load_model(model_path)


def home(request):
    """Landing page with value proposition for new users."""
    if request.user.is_authenticated:
        return redirect('dashboard')
    return render(request, 'wardrobe/home.html')


def register(request):
    """User registration view."""
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
def dashboard(request):
    """Dashboard view showing wardrobe statistics and quick actions."""
    items = ClothingItem.objects.filter(user=request.user, is_active=True)
    saved_outfits = Outfit.objects.filter(user=request.user, is_saved=True)
    recent_items = items.order_by('-uploaded_at')[:6]

    # Calculate statistics
    total_items = items.count()
    category_counts = items.values('category').annotate(count=Count('category')).order_by('-count')
    most_worn_items = items.order_by('-wear_count')[:5]
    least_worn_items = items.filter(wear_count=0).order_by('uploaded_at')[:5]

    # Get outfit recommendations preview
    recommendations = advanced_recommend_outfits(items)[:3]

    return render(request, 'wardrobe/dashboard.html', {
        'total_items': total_items,
        'saved_outfits_count': saved_outfits.count(),
        'category_counts': category_counts,
        'recent_items': recent_items,
        'most_worn_items': most_worn_items,
        'least_worn_items': least_worn_items,
        'recommendations': recommendations,
    })


@login_required
def upload_image(request):
    """Upload and classify clothing items."""
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            files = request.FILES.getlist('image')
            last_item = None
            for file in files:
                # Save the image to the database
                clothing_item = ClothingItem.objects.create(user=request.user, image=file)

                # Load and preprocess the image
                img = image.load_img(clothing_item.image.path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array / 255.0  # Normalize pixel values

                # Predict the label
                predictions = model.predict(img_array)
                predicted_class = np.argmax(predictions, axis=1)
                confidence = float(predictions[0][predicted_class[0]])
                predicted_label = class_labels[predicted_class[0]]

                # Map special labels to valid categories
                if predicted_label in ['Not sure', 'Skip']:
                    predicted_label = 'Other'

                # Save the predicted label and confidence
                clothing_item.category = predicted_label
                clothing_item.classification_confidence = confidence
                clothing_item.save()
                last_item = clothing_item

            if last_item:
                messages.success(request, f'Successfully uploaded {len(files)} item(s)!')
                return redirect('image_result', image_id=last_item.id)
    else:
        form = ImageUploadForm()
    return render(request, 'wardrobe/upload.html', {'form': form})


@login_required
def history(request):
    """View wardrobe history/gallery with filtering."""
    items = ClothingItem.objects.filter(user=request.user, is_active=True)

    # Filter by category if specified
    category = request.GET.get('category')
    if category:
        items = items.filter(category=category)

    # Get unique categories for filter dropdown
    categories = ClothingItem.objects.filter(
        user=request.user, is_active=True
    ).values_list('category', flat=True).distinct()

    return render(request, 'wardrobe/history.html', {
        'images': items,
        'categories': categories,
        'selected_category': category,
    })


@login_required
def delete_image(request, image_id):
    """Delete a clothing item (soft delete by setting is_active=False)."""
    item = get_object_or_404(ClothingItem, id=image_id, user=request.user)

    if request.method == 'POST':
        # Soft delete - set is_active to False
        item.is_active = False
        item.save()
        messages.success(request, 'Item removed from your wardrobe.')

    return redirect('history')


@login_required
def edit_item(request, image_id):
    """Edit a clothing item's details."""
    item = get_object_or_404(ClothingItem, id=image_id, user=request.user)

    if request.method == 'POST':
        form = ClothingItemEditForm(request.POST, instance=item)
        if form.is_valid():
            form.save()
            messages.success(request, 'Item updated successfully!')
            return redirect('image_result', image_id=item.id)
    else:
        form = ClothingItemEditForm(instance=item)

    return render(request, 'wardrobe/edit_item.html', {
        'form': form,
        'item': item,
    })


@login_required
def mark_worn(request, image_id):
    """Mark a clothing item as worn today."""
    item = get_object_or_404(ClothingItem, id=image_id, user=request.user)

    if request.method == 'POST':
        # Update wear count and last worn date
        item.wear_count += 1
        item.last_worn_date = date.today()
        item.save()

        # Create a wear log entry
        wear_log = WearLog.objects.create(
            user=request.user,
            date_worn=date.today()
        )
        wear_log.items.add(item)

        messages.success(request, f'Marked "{item.category}" as worn today!')

    return redirect('history')


@login_required
def menu(request):
    """Menu page with navigation options."""
    return render(request, 'wardrobe/menu.html')


@login_required
def profile(request):
    """User profile with wardrobe overview."""
    items = ClothingItem.objects.filter(user=request.user, is_active=True)
    saved_outfits = Outfit.objects.filter(user=request.user, is_saved=True)

    # Calculate wardrobe statistics
    total_items = items.count()
    total_wears = items.aggregate(total=Sum('wear_count'))['total'] or 0
    category_breakdown = items.values('category').annotate(count=Count('category')).order_by('-count')

    return render(request, 'wardrobe/profile.html', {
        'user': request.user,
        'user_images': items,
        'total_items': total_items,
        'total_wears': total_wears,
        'saved_outfits_count': saved_outfits.count(),
        'category_breakdown': category_breakdown,
    })


@login_required
def image_result(request, image_id):
    """View details of a specific clothing item."""
    item = get_object_or_404(ClothingItem, id=image_id, user=request.user)
    user_items = ClothingItem.objects.filter(user=request.user, is_active=True)
    recommendations = advanced_recommend_outfits(user_items, item)

    return render(request, 'wardrobe/image_result.html', {
        'image': item,
        'recommendations': recommendations,
    })


@login_required
def wardrobe_stats(request):
    """View detailed wardrobe statistics and analytics."""
    items = ClothingItem.objects.filter(user=request.user, is_active=True)
    wear_logs = WearLog.objects.filter(user=request.user)

    # Category breakdown
    category_counts = items.values('category').annotate(count=Count('category')).order_by('-count')

    # Most and least worn items
    most_worn = items.order_by('-wear_count')[:10]
    never_worn = items.filter(wear_count=0)

    # Wear history by month (last 6 months)
    # This is a simplified version - can be enhanced later

    return render(request, 'wardrobe/stats.html', {
        'total_items': items.count(),
        'category_counts': category_counts,
        'most_worn': most_worn,
        'never_worn': never_worn,
        'total_wears': items.aggregate(total=Sum('wear_count'))['total'] or 0,
    })


@login_required
def save_outfit(request):
    """Save a new outfit combination."""
    if request.method == 'POST':
        item_ids = request.POST.getlist('items')
        name = request.POST.get('name', 'My Outfit')
        occasion = request.POST.get('occasion', '')
        season = request.POST.get('season', '')

        if item_ids:
            outfit = Outfit.objects.create(
                user=request.user,
                name=name,
                occasion=occasion if occasion else None,
                season=season if season else None,
            )
            items = ClothingItem.objects.filter(id__in=item_ids, user=request.user)
            outfit.items.set(items)
            messages.success(request, f'Outfit "{name}" saved successfully!')
            return redirect('saved_outfits')

    # Show form with available items
    items = ClothingItem.objects.filter(user=request.user, is_active=True)
    return render(request, 'wardrobe/save_outfit.html', {
        'items': items,
        'occasion_choices': Outfit.OCCASION_CHOICES,
        'season_choices': Outfit.SEASON_CHOICES,
    })


@login_required
def saved_outfits(request):
    """View all saved outfits."""
    outfits = Outfit.objects.filter(user=request.user, is_saved=True)
    return render(request, 'wardrobe/saved_outfits.html', {
        'outfits': outfits,
    })


@login_required
def delete_outfit(request, outfit_id):
    """Delete a saved outfit."""
    outfit = get_object_or_404(Outfit, id=outfit_id, user=request.user)

    if request.method == 'POST':
        outfit.delete()
        messages.success(request, 'Outfit deleted successfully!')

    return redirect('saved_outfits')


def advanced_recommend_outfits(user_items, current_item=None):
    """
    Generate outfit recommendations based on clothing compatibility.
    Returns pairs of items that go well together.
    """
    recommendations = []
    seen_pairs = set()

    for item in user_items:
        category = item.category or ''

        for other_item in user_items:
            if item.id == other_item.id:
                continue

            other_category = other_item.category or ''

            # Create a sorted pair key to avoid duplicates
            pair_key = tuple(sorted([item.id, other_item.id]))
            if pair_key in seen_pairs:
                continue

            # Style Matching Rules based on clothing compatibility
            is_match = False

            # Tops pair with bottoms
            tops = ['T-Shirt', 'Shirt', 'Polo', 'Top', 'Blouse', 'Hoodie', 'Longsleeve', 'Undershirt']
            bottoms = ['Pants', 'Shorts', 'Skirt']

            if category in tops and other_category in bottoms:
                is_match = True
            elif category in bottoms and other_category in tops:
                is_match = True

            # Outerwear pairs with many things
            outerwear = ['Outwear', 'Blazer']
            if category in outerwear and other_category in (tops + ['Dress']):
                is_match = True
            elif other_category in outerwear and category in (tops + ['Dress']):
                is_match = True

            # Dress pairs with outerwear
            if category == 'Dress' and other_category in outerwear:
                is_match = True
            elif other_category == 'Dress' and category in outerwear:
                is_match = True

            if is_match:
                seen_pairs.add(pair_key)
                recommendations.append((item, other_item))

    # Prioritize recommendations involving the current item if provided
    if current_item:
        recommendations.sort(
            key=lambda x: (0 if current_item.id in [x[0].id, x[1].id] else 1)
        )

    return recommendations


def are_colors_complementary(color1, color2):
    """
    Check if two colors are complementary.
    This is a basic example; you can use a color theory library for better results.
    """
    def rgb_to_tuple(rgb):
        return tuple(map(int, rgb.strip('rgb()').split(',')))

    color1_rgb = rgb_to_tuple(color1)
    color2_rgb = rgb_to_tuple(color2)

    return (abs(color1_rgb[0] - color2_rgb[0]) > 100 and
            abs(color1_rgb[1] - color2_rgb[1]) > 100 and
            abs(color1_rgb[2] - color2_rgb[2]) > 100)