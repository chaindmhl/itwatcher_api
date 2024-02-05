from django.conf import settings
from django.conf.urls.static import static
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import DownloadRequestListCreateView, DownloadRequestDetailView, success_page
from tracking.views import (
    VideoUploadViewSet,
    ProcessVideoViewSet,
    CatchAllViewSet,
    LPRViewSet,
    LPRAllViewSet,
    TrikeAllViewSet,
    LPRFrontSideViewSet,
    ColorViewSet,

    MyView,
    PlateView,
    FrameView,
    FrameColorView,
    MapView,
    ColorView,
    CountLogViewSet,
    CountLogListView,
    TrikeVehicleLogListView,
    VehicleLogListView,
    TricycleCountGraphView,
    VehicleCountGraphView,
    
)

router = DefaultRouter()
router.register("tracking/video", VideoUploadViewSet, basename="tracking-video")
router.register("tracking/tric", ProcessVideoViewSet, basename="tracking-tric")
router.register("tracking/color", ColorViewSet, basename="tracking-color")
router.register("tracking/lpr", LPRViewSet, basename="LPR-tric")
router.register("tracking/count-logs", CountLogViewSet, basename="tracking-count")
router.register("tracking/catchall", CatchAllViewSet, basename="tracking-catchall")
router.register("tracking/trikeall", TrikeAllViewSet, basename="tracking-trike_all")
router.register("tracking/lpr_all", LPRAllViewSet, basename="LPR-All_Vehicle")
router.register("tracking/lpr_frontside", LPRFrontSideViewSet, basename="LPR-Trike_FrontSide")

urlpatterns = [
    path('', include(router.urls)),
    path('my-url/', MyView.as_view(), name='my-view'),
    path('download-requests/', DownloadRequestListCreateView.as_view(), name='downloadrequest-list-create'),
    path('download-requests/<int:pk>/', DownloadRequestDetailView.as_view(), name='downloadrequest-detail'),
    path('success/', success_page, name='success-page'),
    path('display_plates/', PlateView.as_view(), name='display_plates'),
    path('display_color/', ColorView.as_view(), name='display_color'),
    path('view_frame/<int:log_id>/', FrameView.view_frame, name='view_frame'),
    path('view_colorframe/<int:log_id>/', FrameColorView.view_colorframe, name='view_colorframe'),
    path('view_camera_map/<int:log_id>/', MapView.view_camera_map, name='view_camera_map'),
    path('count_logs/', CountLogListView.as_view(), name='count_log_list'),
    path('vehicle_logs/', VehicleLogListView.as_view(), name='vehicle_log_list'),
    path('trikeall_logs/', TrikeVehicleLogListView.as_view(), name='trikeall_log_list'),
    path('tricycle_count_graph/<int:log_id>/', TricycleCountGraphView.as_view(), name='tricycle_count_graph'),
    #path('vehicle_count_graph/<int:log_id>/', VehicleCountGraphView.as_view(), name='vehicle_count_graph'),
    path('vehicle_count_graph/<str:log_date>/<int:log_id>/', VehicleCountGraphView.as_view(), name='vehicle_count_graph'),

] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
