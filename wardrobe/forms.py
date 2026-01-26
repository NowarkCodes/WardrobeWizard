from django import forms
from .models import ClothingItem


class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True


class MultipleFileField(forms.FileField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput())
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        single_file_clean = super().clean
        if isinstance(data, (list, tuple)):
            result = [single_file_clean(d, initial) for d in data]
        else:
            result = single_file_clean(data, initial)
        return result


class ImageUploadForm(forms.ModelForm):
    class Meta:
        model = ClothingItem
        fields = ['image']

    # Use the custom MultipleFileField
    image = MultipleFileField()


class ClothingItemEditForm(forms.ModelForm):
    """Form for editing clothing item details."""
    class Meta:
        model = ClothingItem
        fields = ['category', 'notes']
        widgets = {
            'notes': forms.Textarea(attrs={'rows': 3, 'placeholder': 'Add notes about this item...'}),
        }