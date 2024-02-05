from django.contrib.auth.models import User
from rest_framework.permissions import IsAuthenticated
from django.views import View
from django.shortcuts import render
from rest_framework import viewsets, status, generics
from rest_framework.response import Response
from tracking.serializers import (
    VideoSerializers,
    ProcessVideoSerializers,
    LPRSerializers,
    CountLogSerializer,
    VehicleLogSerializer,
    DownloadRequestSerializer,
)
from tracking.models import Video, PlateLog, CountLog, VehicleLog, DownloadRequest
from tracking.process import process_vid
from tracking.process_lpr_comb import process_lpd_comb
from tracking.process_tc_all import process_trackcount_all
from tracking.process_tc_comb import process_trackcount_comb
from tracking.process_lpr_all import process_alllpr
#from tracking.process_frontside import process_frontside

# Define your Django Rest Framework view
from rest_framework.response import Response
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE
import io
import base64
from django.db.models import Count, F
from datetime import timedelta, datetime
import shlex

# Create your views here.
class MyView(View):
    
    permission_classes = [IsAuthenticated]

    def get(self, request):
        users = User.objects.all()
        videos = Video.objects.filter(user=request.user)
        context = {'users': users, 'videos': videos}
        return render(request, 'index.html', context)

class CountLogViewSet(viewsets.ModelViewSet):
    queryset = CountLog.objects.all()
    serializer_class = CountLogSerializer

class VehicleLogViewSet(viewsets.ModelViewSet):
    queryset = VehicleLog.objects.all()
    serializer_class = VehicleLogSerializer
       
# Create your views here.
class VideoUploadViewSet(viewsets.ModelViewSet):
    """
    Uploads a File
    """

    queryset = Video.objects.all()
    serializer_class = VideoSerializers
    permission_classes = [IsAuthenticated]


class ProcessVideoViewSet(viewsets.ViewSet):
    """
    Perform tricycle Detection in Videos
    """

    serializer_class = ProcessVideoSerializers

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("camera_feed_url")

            if video_path:
                context = process_vid(video_path=video_path)
            elif livestream_url:
                stream_path = livestream_url
                #stream_file = cv2.VideoCapture(stream_path)
                context = process_vid(livestream_url=livestream_url, video_stream=stream_path)
            else:
                return Response({"error": "Either video or camera_feed_url must be provided"}, status=status.HTTP_400_BAD_REQUEST)

             # Return the path to the output video
            return Response({'output_video_path': context['output_video_path']}, status=status.HTTP_200_OK)

        # Return validation errors if any
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class CatchAllViewSet(viewsets.ViewSet):
    """
    Perform Vehicle Detection in Videos
    """

    serializer_class = ProcessVideoSerializers

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("camera_feed_url")

            if video_path:
                context = process_trackcount_all(video_path=video_path)
            elif livestream_url:
                stream_path = livestream_url
                #stream_file = cv2.VideoCapture(stream_path)
                context = process_trackcount_all(livestream_url=livestream_url, video_stream=stream_path)
            else:
                return Response({"error": "Either video or camera_feed_url must be provided"}, status=status.HTTP_400_BAD_REQUEST)

             # Return the path to the output video
            return Response({'output_video_path': context['output_video_path']}, status=status.HTTP_200_OK)

        # Return validation errors if any
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class TrikeAllViewSet(viewsets.ViewSet):
    """
    Perform Vehicle Detection (Trike and Vehicle) in Videos
    """

    serializer_class = ProcessVideoSerializers

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("camera_feed_url")

            if video_path:
                context = process_trackcount_comb(video_path=video_path)
            elif livestream_url:
                stream_path = livestream_url
                #stream_file = cv2.VideoCapture(stream_path)
                context = process_trackcount_comb(livestream_url=livestream_url, video_stream=stream_path)
            else:
                return Response({"error": "Either video or camera_feed_url must be provided"}, status=status.HTTP_400_BAD_REQUEST)

             # Return the path to the output video
            return Response({'output_video_path': context['output_video_path']}, status=status.HTTP_200_OK)

        # Return validation errors if any
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
class LPRViewSet(viewsets.ViewSet):
    """
    Perform LPR-trike in Videos
    """

    serializer_class = LPRSerializers

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("camera_feed_url")

            if video_path:
                context = process_lpd_comb(video_path=video_path)   
            elif livestream_url:
                stream_path = livestream_url
                context = process_lpd_comb(livestream_url=livestream_url, video_stream=stream_path)
            else:
                return Response({"error": "Either video or camera_feed_url must be provided"}, status=status.HTTP_400_BAD_REQUEST)

             # Return the path to the output video
            return Response({'output_video_path': context['output_video_path']}, status=status.HTTP_200_OK)

        # Return validation errors if any
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class LPRAllViewSet(viewsets.ViewSet):
    """
    Perform LPR-all in Videos
    """

    serializer_class = LPRSerializers

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("camera_feed_url")

            if video_path:
                context = process_alllpr(video_path=video_path)
            elif livestream_url:
                stream_path = livestream_url
                #stream_file = cv2.VideoCapture(stream_path)
                context = process_alllpr(livestream_url=livestream_url, video_stream=stream_path)
            else:
                return Response({"error": "Either video or camera_feed_url must be provided"}, status=status.HTTP_400_BAD_REQUEST)

             # Return the path to the output video
            return Response({'output_video_path': context['output_video_path']}, status=status.HTTP_200_OK)

        # Return validation errors if any
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)  

class LPRFrontSideViewSet(viewsets.ViewSet):
    """
    Perform LPR_comb in Videos
    """

    serializer_class = LPRSerializers

    def create(self, request):
        serializer = ProcessVideoSerializers(data=request.data)
        if serializer.is_valid():
            data = serializer.validated_data
            video_path = data.get("video")
            livestream_url = data.get("camera_feed_url")

            if video_path:
                context = process_lpd_comb(video_path=video_path)
            elif livestream_url:
                stream_path = livestream_url
                #stream_file = cv2.VideoCapture(stream_path)
                context = process_lpd_comb(livestream_url=livestream_url, video_stream=stream_path)
            else:
                return Response({"error": "Either video or camera_feed_url must be provided"}, status=status.HTTP_400_BAD_REQUEST)

             # Return the path to the output video
            return Response({'output_video_path': context['output_video_path']}, status=status.HTTP_200_OK)

        # Return validation errors if any
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST) 
        

class PlateView(View):
    def get(self, request):
        # Select distinct filenames with timestamps within a 5-minute range and having three or more characters
        distinct_filenames = PlateLog.objects.filter(
            timestamp__range=(F('timestamp') - timedelta(minutes=5), F('timestamp') + timedelta(minutes=5)),
            filename__in=PlateLog.objects.values('filename').annotate(count=Count('filename')).filter(count__gte=3).values('filename')
        ).values('filename').distinct()

        # Create a list of IDs to exclude
        exclude_ids = PlateLog.objects.exclude(
            filename__in=distinct_filenames
        ).values('id')

        # Exclude the unwanted entries
        PlateLog.objects.exclude(id__in=exclude_ids).delete()

        # Retrieve the remaining PlateLog records
        plate_logs = PlateLog.objects.all()

        context = {
            'plate_logs': plate_logs,
        }
        return render(request, '/home/icebox/itwatcher_api/tracking/display_plates.html', context)

    def post(self, request):
        # Handle POST requests if needed
        pass

class FrameView(View):

    def view_frame(request, log_id):
        # Retrieve the PlateLog instance based on the log_id
        plate_log = PlateLog.objects.get(id=log_id)
        context = {
            'plate_log': plate_log,
        }
        return render(request, '/home/icebox/itwatcher_api/tracking/view_frame.html', context)
    
class MapView(View):

    def view_camera_map(request, log_id):
        plate_log = PlateLog.objects.get(id=log_id)
        context = {
            'plate_log': plate_log,
        }
        return render(request, '/home/icebox/itwatcher_api/tracking/view_camera_map.html', context)

class CountLogListView(View):

    template_name = 'count_log_list.html'

    def get(self, request, *args, **kwargs):
        count_logs = CountLog.objects.all()
        context = {'count_logs': count_logs}
        return render(request, self.template_name, context)
            
class VehicleLogListView(View):

    template_name = 'vehicle_log_list.html'

    def get(self, request, *args, **kwargs):
        vehicle_logs = VehicleLog.objects.all()
        context = {'vehicle_logs': vehicle_logs}
        return render(request, self.template_name, context)
    
class TrikeVehicleLogListView(View):

    template_name = 'trikeall_log_list.html'

    def get(self, request, *args, **kwargs):
        trikeall_logs = VehicleLog.objects.all()
        context = {'trikeall_logs': trikeall_logs}
        return render(request, self.template_name, context)

class TricycleCountGraphView(View):
    template_name = 'tricycle_count_graph.html'  # Path to your template

    def get(self, request, log_id):
        # Retrieve the log entry based on log_id
        log = CountLog.objects.get(id=log_id)  # Adjust this based on your model

        # Extract class names and counts from the log
        class_names = list(log.class_counts.keys())
        class_counts = list(log.class_counts.values())

        # Generate the bar graph
        bar_figure = plt.figure(figsize=(6, 4))
        plt.bar(class_names, class_counts)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Bar Graph')
        bar_buffer = io.BytesIO()
        plt.savefig(bar_buffer, format='png')
        plt.close(bar_figure)

        # Generate the pie chart
        pie_figure = plt.figure(figsize=(6, 4))
        plt.pie(class_counts, labels=class_names, autopct='%1.1f%%')
        plt.title('Pie Chart')
        pie_buffer = io.BytesIO()
        plt.savefig(pie_buffer, format='png')
        plt.close(pie_figure)

        # Generate the line graph
        line_figure = plt.figure(figsize=(6, 4))
        plt.plot(class_names, class_counts, marker='o')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Line Graph')
        line_buffer = io.BytesIO()
        plt.savefig(line_buffer, format='png')
        plt.close(line_figure)

        # Convert the buffer data to base64 for embedding in the HTML
        bar_graph_data = base64.b64encode(bar_buffer.getvalue()).decode()
        pie_graph_data = base64.b64encode(pie_buffer.getvalue()).decode()
        line_graph_data = base64.b64encode(line_buffer.getvalue()).decode()

        context = {
            'bar_graph_data': bar_graph_data,
            'pie_graph_data': pie_graph_data,
            'line_graph_data': line_graph_data,
            'log_id': log_id,
        }

        return render(request, self.template_name, context)

class VehicleCountGraphView(View):
    template_name = 'vehicle_count_graph.html'  # Path to your template

    def get(self, request, log_date, log_id):
        # Retrieve the log entry based on log_id
        log = VehicleLog.objects.get(id=log_id)  # Adjust this based on your model
        # logs = VehicleLog.objects.filter(timestamp__date=log_date)

        # Extract class names and counts from the log
        class_names = list(log.class_counts.keys())
        class_counts = list(log.class_counts.values())
        # Extract vehicle types and counts from the logs
        # class_names = list(set(log.vehicle_type for log in logs))
        # class_counts = [logs.filter(vehicle_type=vehicle_type).count() for vehicle_type in class_names]
        print("Class Names:", class_names)
        print("Class Counts:", class_counts)

        # Generate the bar graph
        bar_figure = plt.figure(figsize=(8, 6))
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'yellow', 'black']  # Add more colors if needed
        #plt.bar(class_names, class_counts)
        # Calculate cumulative counts for each vehicle type
        cumulative_counts = [0] * len(class_names)
        for i, count in enumerate(class_counts):
            plt.bar(log_date, count, bottom=cumulative_counts, color=colors[i], label=class_names[i])
            cumulative_counts[i] += count

        plt.xlabel('Date')
        plt.ylabel('Count')
        plt.title('Bar Graph for {log_date}')
        bar_buffer = io.BytesIO()
        plt.savefig(bar_buffer, format='png')
        plt.close(bar_figure)

        # Generate the pie chart
        pie_figure = plt.figure(figsize=(6, 4))
        plt.pie(class_counts, labels=class_names, autopct='%1.1f%%')
        plt.title('Pie Chart')
        pie_buffer = io.BytesIO()
        plt.savefig(pie_buffer, format='png')
        plt.close(pie_figure)

        # Generate the line graph
        line_figure = plt.figure(figsize=(6, 4))
        plt.plot(class_names, class_counts, marker='o')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Line Graph')
        line_buffer = io.BytesIO()
        plt.savefig(line_buffer, format='png')
        plt.close(line_figure)

        # Convert the buffer data to base64 for embedding in the HTML
        bar_graph_data = base64.b64encode(bar_buffer.getvalue()).decode()
        pie_graph_data = base64.b64encode(pie_buffer.getvalue()).decode()
        line_graph_data = base64.b64encode(line_buffer.getvalue()).decode()

        context = {
            'bar_graph_data': bar_graph_data,
            'pie_graph_data': pie_graph_data,
            'line_graph_data': line_graph_data,
            'log_date': log_date,
            
        }

        return render(request, self.template_name, context)

class DownloadRequestListCreateView(generics.ListCreateAPIView):
    template_name = 'download_video.html'
    queryset = DownloadRequest.objects.all()
    serializer_class = DownloadRequestSerializer

    def get(self, request, *args, **kwargs):
        return render(request, self.template_name)
    def post(self, request, *args, **kwargs):
        # Handle form submission and run the media_download.py script
        # Assuming you have form handling logic here...
        start = '00:00:00'
        end = '23:59:59'
        # Run the media_download.py script
        script_path = '/home/icebox/itwatcher_api/tracking/hikvision/media_download.py'
        camera_ip = request.POST.get('camera_ip')  # Adjust to match your form field name
        start_date = request.POST.get('start_date')
        start_time = request.POST.get('start_time', start)
        end_date = request.POST.get('end_date')
        end_time = request.POST.get('end_time', end)
        content_type = request.POST.get('content_type')  # Assuming you have a content_type field
        channel = request.POST.get('channel')

        if start_date:
            start_datetime = datetime.strptime(f"{start_date} {start_time}", "%Y-%m-%d %H:%M:%S")
        else:
            # Set default value for start_datetime
            start_datetime = datetime.now()

        if end_date:
            end_datetime = datetime.strptime(f"{end_date} {end_time}", "%Y-%m-%d %H:%M:%S")
        else:
            # Set default value for end_datetime (24 hours after start_datetime)
            start_datetime = datetime.now()
            end_datetime = start_datetime + timedelta(hours=24)

        # Check if start_datetime is greater than or equal to end_datetime
        if start_datetime >= end_datetime:
            error_message = "Start Datetime must be before End Datetime."
            return render(request, self.template_name, {'error_message': error_message})
    
        # Form the command with arguments
        command = f"python3 {script_path} {camera_ip} {start_date} {start_time} {end_date} {end_time} {channel}"

        if content_type:
            command += f" -p"

        process = Popen(shlex.split(command), stdout=PIPE, stderr=PIPE)
        output, error = process.communicate()

        # Check the output and error if needed
        print('Output:', output.decode())
        print('Error:', error.decode())

        # Render the form again (or handle success differently if needed)
        return render(request, self.template_name)

def success_page(request):
    return render(request, 'download_success.html')
           
class DownloadRequestDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = DownloadRequest.objects.all()
    serializer_class = DownloadRequestSerializer