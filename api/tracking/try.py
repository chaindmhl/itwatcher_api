import os
import django

# Set the DJANGO_SETTINGS_MODULE environment variable
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tricycle.settings")

# Initialize the Django settings
django.setup()

# Now you can use cache-related code
from django.core.cache import cache

# Clear the default cache
cache.clear()
