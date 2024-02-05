from django import forms
from .models import PlateLog

class PlateLogForm(forms.ModelForm):
    class Meta:
        model = PlateLog
        fields = '__all__'  # Include all fields from the model

    # You can customize form fields or widgets here if needed
