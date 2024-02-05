from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from tracking.views import (
    MyView,
    PlateView,
    FrameView,
    FrameColorView,
    MapView,
)
#original
urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("tracking.urls")),
    path('my-url/', MyView.as_view(), name='my-view'),
    path('display_plates/', PlateView.as_view(), name='display_plates'),
    path('view_frame/<int:log_id>/', FrameView.view_frame, name='view_frame'),
    path('view_colorframe/<int:log_id>/', FrameColorView.view_colorframe, name='view_colorframe'),
    path('view_camera_map/<int:log_id>/', MapView.view_camera_map, name='view_camera_map'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

