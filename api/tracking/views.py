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
from tracking.models import Video, PlateLog, CountLog, ColorLog, VehicleLog, DownloadRequest
from tracking.process import process_vid
from tracking.process_lpr_comb import process_lpd_comb
from tracking.process_tc_all import process_trackcount_all
from tracking.process_tc_comb import process_trackcount_comb
from tracking.process_lpr_all import process_alllpr
from tracking.process_color import process_color
#from tracking.process_frontside import process_frontside

# Define your Django Rest Framework view
from rest_framework.response import Response
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE
import io, os
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
        return render(request, 'html_files/index.html', context)

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
    
class ColorViewSet(viewsets.ViewSet):
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
                context = process_color(video_path=video_path)
            elif livestream_url:
                stream_path = livestream_url
                #stream_file = cv2.VideoCapture(stream_path)
                context = process_color(livestream_url=livestream_url, video_stream=stream_path)
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
        
class ColorView(View):

    def get(self, request):
        color_logs = ColorLog.objects.all()
        context = {
            'color_logs': color_logs,
        }
        return render(request, '/home/icebox/itwatcher_api/tracking/html_files/display_color.html', context)
    def post(self, request):
        # Handle POST requests if needed
        pass


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
        return render(request, '/home/icebox/itwatcher_api/tracking/html_files/display_plates.html', context)

    def post(self, request):
        # Handle POST requests if needed
        pass

class FrameColorView(View):

    def view_colorframe(request, log_id):
        # Retrieve the PlateLog instance based on the log_id
        color_log = ColorLog.objects.get(id=log_id)
        context = {
            'color_log': color_log,
        }
        return render(request, '/home/icebox/itwatcher_api/tracking/html_files/view_colorframe.html', context)
    
class FrameView(View):

    def view_frame(request, log_id):
        # Retrieve the PlateLog instance based on the log_id
        plate_log = PlateLog.objects.get(id=log_id)
        context = {
            'plate_log': plate_log,
        }
        return render(request, '/home/icebox/itwatcher_api/tracking/html_files/view_frame.html', context)
    
class MapView(View):

    def view_camera_map(request, log_id):
        plate_log = PlateLog.objects.get(id=log_id)
        context = {
            'plate_log': plate_log,
        }
        return render(request, '/home/icebox/itwatcher_api/tracking/html_files/view_camera_map.html', context)

class CountLogListView(View):

    template_name = 'html_files/count_log_list.html'

    def get(self, request, *args, **kwargs):
        count_logs = CountLog.objects.all()
        context = {'count_logs': count_logs}
        return render(request, self.template_name, context)
            
class VehicleLogListView(View):

    template_name = 'html_files/vehicle_log_list.html'

    def get(self, request, *args, **kwargs):
        vehicle_logs = VehicleLog.objects.all()
        context = {'vehicle_logs': vehicle_logs}
        return render(request, self.template_name, context)
    
class TrikeVehicleLogListView(View):

    template_name = 'html_files/trikeall_log_list.html'

    def get(self, request, *args, **kwargs):
        trikeall_logs = VehicleLog.objects.all()
        context = {'trikeall_logs': trikeall_logs}
        return render(request, self.template_name, context)

class TricycleCountGraphView(View):
    template_name = 'html_files/tricycle_count_graph.html'  # Path to your template

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
    template_name = 'html_files/vehicle_count_graph.html'  # Path to your template

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
    template_name = 'html_files/download_video.html'
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

        directory_name = f"{start_date}_{start_time}_{end_date}_{end_time}_{channel}"
        directory_path = os.path.join('/home/icebox/itwatcher_api/nvr_videos/10.101.60.148', start_date, channel, start_date)
        directory_exists = os.path.exists(directory_path)

        print(f"Directory path: {directory_path}")
        print(f"Directory exists: {directory_exists}")


        # Check if the directory already exists
        if directory_exists:
            # Directory already exists, prompt user for action
            user_response = request.POST.get('user_response', '').lower().strip()
            if user_response != 'y':
                # User chose not to append, return a response or redirect
                error_message = f"Video Files for {start_date} already exists. Download aborted."
                print(f"Error message: {error_message}")
                return render(request, 'html_files/download_success.html', {'error_message': error_message, 'directory_exists': directory_exists})
                    
            
        # Form the command with arguments
        command = f"python3 {script_path} {camera_ip} {start_date} {start_time} {end_date} {end_time} {channel}"

        if content_type:
            command += " -p"

        # Run the script only if the directory doesn't exist
        if not directory_exists:
            # Run the script and wait for its completion
            process = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
            process.communicate()  # Wait for the process to complete

            # Check the return code for success or failure
            if process.returncode == 0:
                print('Download executed successfully.')
            else:
                print(f'Download execution failed. Return code: {process.returncode}')

        # Render the form again (or handle success differently if needed)
        return render(request, 'html_files/download_success.html', {'directory_exists': directory_exists})

def success_page(request):
    return render(request, 'html_files/download_success.html')
           
class DownloadRequestDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = DownloadRequest.objects.all()
    serializer_class = DownloadRequestSerializer