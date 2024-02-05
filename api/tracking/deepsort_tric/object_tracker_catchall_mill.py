import os
# comment out below line to enable tensorflow logging outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
# from absl import app, flags, logging
# from absl.flags import FLAGS
import tracking.deepsort_tric.core.utils as utils
from tracking.deepsort_tric.core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from tracking.deepsort_tric.core.config_all import cfg
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tracking.models import VehicleLog
# deep sort imports
from tracking.deepsort_tric.deep_sort import preprocessing, nn_matching
from tracking.deepsort_tric.deep_sort.detection import Detection
from tracking.deepsort_tric.deep_sort.tracker import Tracker
from tracking.deepsort_tric.tools import generate_detections as gdet
import datetime
from collections import Counter, deque
import math
import pytesseract

import re

""""
!python object_tracker.py --video /content/road1.mp4 --output ./outputs/custom3.avi --model yolov4 --dont_show --info
"""



class VehiclesCounting():
    def __init__(self, file_counter_log_name, framework='tf', weights='./checkpoints/yolov4-416',
                size=416, tiny=False, model='yolov4', video='./data/videos/cam0.mp4',
                output=None, output_format='XVID', iou=0.45, score=0.5,
                dont_show=False, info=False,
                detection_line=(0.5,0)):
    
        self._file_counter_log_name = file_counter_log_name
        self._framework = framework
        self._weights = weights
        self._size = size
        self._tiny = tiny
        self._model = model
        self._video = video
        self._output = output
        self._output_format = output_format
        self._iou = iou
        self._score = score
        self._dont_show = dont_show
        self._info = info
        self._detect_line_position = detection_line[0]
        self._detect_line_angle = detection_line[1]
        
        self.total_counter = 0
        self.up_count = 0
        self.down_count = 0
        self.class_counts = 0

        
        

    def _intersect(self, A, B, C, D):
        return self._ccw(A,C,D) != self._ccw(B, C, D) and self._ccw(A,B,C) != self._ccw(A,B,D)


    def _ccw(self, A,B,C):
        return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

    def _vector_angle(self, midpoint, previous_midpoint):
        x = midpoint[0] - previous_midpoint[0]
        y = midpoint[1] - previous_midpoint[1]
        return math.degrees(math.atan2(y, x))
    
    def get_total_counter(self):
        return self.total_counter
    
    def get_up_count(self):
        return self.up_count

    def get_down_count(self):
        return self.down_count
    
    def get_class_counts(self):
        return self.class_counts

    def run(self):
        # Definition of the parameters
        max_cosine_distance = 0.4
        nn_budget = None
        nms_max_overlap = 1.0
        show_detections = False

        # initialize deep sort
        model_filename = '/home/icebox/itwatcher_api/tracking/deepsort_tric/model_data/mars-small128.pb'
        
        encoder = gdet.create_box_encoder(model_filename, batch_size=1)
        # calculate cosine distance metric
        metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        # initialize tracker
        tracker = Tracker(metric)

        #initialize color map
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i)[:3] for i in np.linspace(0, 1, 20)]

        # load configuration for object detector
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        #STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)
        input_size = self._size
        video_path = self._video

        # load tflite model if flag is set
        if self._framework == 'tflite':
            interpreter = tf.lite.Interpreter(model_path=self._weights)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            print(input_details)
            print(output_details)
        # otherwise load standard tensorflow saved model
        else:
            saved_model_loaded = tf.saved_model.load(self._weights, tags=[tag_constants.SERVING])
            infer = saved_model_loaded.signatures['serving_default']

        # begin video capture
        try:
            if isinstance(self._video, str):
                vid = cv2.VideoCapture(int(video_path))
            else:
                vid = self._video
        except:
            vid = cv2.VideoCapture(video_path)

        out = None

        # get video ready to save locally if flag is set
        if self._output:
            # by default VideoCapture returns float instead of int
            width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(vid.get(cv2.CAP_PROP_FPS))
            codec = cv2.VideoWriter_fourcc(*self._output_format)
            out = cv2.VideoWriter(self._output, codec, fps, (width, height))

        frame_num = 0
        current_date = datetime.datetime.now().date()
        count_dict = {}  # initiate dict for storing counts

        total_counter = 0
        up_count = 0
        down_count = 0

        class_counter = Counter()  # store counts of each detected class
        already_counted = deque(maxlen=50)  # temporary memory for storing counted IDs
        intersect_info = []  # initialise intersection list

        memory = {}

        cv2.namedWindow(self._file_counter_log_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._file_counter_log_name, 640, 480)

        prev_time = cv2.getTickCount()
        delay_sum = 0
        delay_count = 1

        skip_frames = 3

        class_counts = class_counter

        while True:
            return_value, frame = vid.read()    
            #if return_value:
            if not return_value:
                print('Video has ended or failed, try a different video format!')
                break
            if vid.get(cv2.CAP_PROP_POS_FRAMES) % skip_frames != 0:
                #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
            frame_num +=1
            print('Frame #: ', frame_num)
            frame_size = frame.shape[:2]
            
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.

            image_data = image_data[np.newaxis, ...].astype(np.float32)
            start_time = time.time()

            # run detections on tflite if flag is set
            if self._framework == 'tflite':
                interpreter.set_tensor(input_details[0]['index'], image_data)
                interpreter.invoke()
                pred = [interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]
                # run detections using yolov3 if flag is set
                if self._model == 'yolov3' and self._tiny == True:
                    boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
                else:
                    boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25,
                                                    input_shape=tf.constant([input_size, input_size]))
            else:
                batch_data = tf.constant(image_data)
                pred_bbox = infer(batch_data)
                for _, value in pred_bbox.items():
                    boxes = value[:, :, 0:4]
                    pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=self._iou,
                score_threshold=self._score
            )

            # convert data to numpy arrays and slice out unused elements
            num_objects = valid_detections.numpy()[0]
            bboxes = boxes.numpy()[0]
            bboxes = bboxes[0:int(num_objects)]
            scores = scores.numpy()[0]
            scores = scores[0:int(num_objects)]
            classes = classes.numpy()[0]
            classes = classes[0:int(num_objects)]

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, width, height
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(bboxes, original_h, original_w)

            # store all predictions in one parameter for simplicity when calling functions
            pred_bbox = [bboxes, scores, classes, num_objects]

            # read in all class names from config
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)

            # by default allow all classes in .names file
            allowed_classes = list(class_names.values())

            # custom allowed classes (uncomment line below to customize tracker for only people)
            #allowed_classes = ['person']

            # loop through objects and use class index to get class name, allow only classes in allowed_classes list
            names = []
            deleted_indx = []
            for i in range(num_objects):
                class_indx = int(classes[i])
                class_name = class_names[class_indx]
                if class_name not in allowed_classes:
                    deleted_indx.append(i)
                else:
                    names.append(class_name)
                    ymin, xmin, ymax, xmax = bboxes[i]
                   
            names = np.array(names)
            count = len(names)
            if count:
                cv2.putText(frame, "Objects being tracked: {}".format(count), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
                print("Objects being tracked: {}".format(count))
            
            # delete detections that are not in allowed_classes
            bboxes = np.delete(bboxes, deleted_indx, axis=0)
            scores = np.delete(scores, deleted_indx, axis=0)

            # encode yolo detections and feed to tracker
            features = encoder(frame, bboxes)
            detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in zip(bboxes, scores, names, features)]

            # run non-maxima supression
            boxs = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            classes = np.array([d.class_name for d in detections])
            indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # Call the tracker
            tracker.predict()
            tracker.update(detections)

            # calculate line position and angle
            # (0, pos*y+y'), (x, pos*y-y')
            # y' = tan(angle) * x / 2

            #yp = math.tan(self._detect_line_angle*math.pi/180) * frame.shape[1] / 2
            x1 = int(frame.shape[1] / 2)
            y1 = 0
            x2 = int(frame.shape[1] / 2)
            y2 = frame.shape[0]
            xa = 0
            ya = int(frame.shape[0] / 2) + 100
            xb = frame.shape[1]
            yb = int(frame.shape[0] / 2) - 100
                

            line = [(x1, y1), (x2, y2)]
            line1 = [(xa,ya), (xb,yb)]
            # draw yellow line
            cv2.line(frame, line[0], line[1], (0, 255, 255), 2)
            cv2.line(frame, line1[0], line1[1], (0, 255, 255), 2)


            # update tracks
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue

                bbox = track.to_tlbr()
                class_name = track.get_class()

                midpoint = track.tlbr_midpoint(bbox)
                origin_midpoint = (midpoint[0], frame.shape[0] - midpoint[1])

                if track.track_id not in memory:
                        memory[track.track_id] = deque(maxlen=2)

                memory[track.track_id].append(midpoint)
                previous_midpoint = memory[track.track_id][0]

                origin_previous_midpoint = (previous_midpoint[0], frame.shape[0] - previous_midpoint[1])
                cv2.line(frame, midpoint, previous_midpoint, (0, 255, 0), 2)

                if self._intersect(midpoint, previous_midpoint, line[0], line[1]) and track.track_id not in already_counted:
                        class_counter[class_name] += 1
                        total_counter += 1

                        # draw red line
                        cv2.line(frame, line[0], line[1], (0, 0, 255), 2)

                        # Set already counted for ID to true.
                        already_counted.append(track.track_id)  

                        intersection_time = datetime.datetime.now() - datetime.timedelta(microseconds=datetime.datetime.now().microsecond)
                        angle = self._vector_angle(origin_midpoint, origin_previous_midpoint)
                        intersect_info.append([class_name, origin_midpoint, angle, intersection_time])

                        if angle > 0:
                            up_count += 1
                        if angle < 0:
                            down_count += 1

                if self._intersect(midpoint, previous_midpoint, line1[0], line1[1]) and track.track_id not in already_counted:
                        class_counter[class_name] += 1
                        total_counter += 1

                        # draw red line
                        cv2.line(frame, line[0], line[1], (0, 0, 255), 2)

                        # Set already counted for ID to true.
                        already_counted.append(track.track_id)  

                        intersection_time = datetime.datetime.now() - datetime.timedelta(microseconds=datetime.datetime.now().microsecond)
                        angle = self._vector_angle(origin_midpoint, origin_previous_midpoint)
                        intersect_info.append([class_name, origin_midpoint, angle, intersection_time])

                        if angle > 0:
                            up_count += 1
                        if angle < 0:
                            down_count += 1
                if class_name != class_names[0]:
                    cv2.rectangle(frame,  (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,0,0), 2)
                    cv2.putText(frame, "" ,(int(bbox[0]), int(bbox[1])), 0,
                                1.1e-3 * frame.shape[0], (255, 0, 0), 2)
                if class_name == class_names[0]:
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255,255,255), 2)  # WHITE BOX
                    cv2.putText(frame, "" + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0,
                                1.1e-3 * frame.shape[0], (0, 255, 0), 2)


                if show_detections:
                    adc = "%.2f" % (track.adc * 100) + "%"  # Average detection confidence
                    cv2.putText(frame, str(class_name), (int(bbox[0]), int(bbox[3])), 0,
                                    1e-3 * frame.shape[0], (0, 255, 0), 2)
                    cv2.putText(frame, 'ADC: ' + adc, (int(bbox[0]), int(bbox[3] + 2e-2 * frame.shape[1])), 0,
                                    1e-3 * frame.shape[0], (0, 255, 0), 2)


            # if enable info flag then print details about each track
                if self._info:
                    print("Tracker ID: {}, Class: {},  BBox Coords (xmin, ymin, xmax, ymax): {}".format(str(track.track_id), class_name, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))

            # Delete memory of old tracks.
            # This needs to be larger than the number of tracked objects in the frame.
            if len(memory) > 50:
                del memory[list(memory)[0]]

            # Draw total count.
            cv2.putText(frame, "Total: {} ({} left, {} right)".format(str(total_counter), str(up_count),
                        str(down_count)), (int(0.05 * frame.shape[1]), int(0.1 * frame.shape[0])), 0,
                        1.5e-3 * frame.shape[0], (0, 255, 255), 2)

            if show_detections:
                track_dict = {}

                for det in detections:
                    bbox = det.to_tlbr()
                    class_name = det.get_class_name()
                    score = "%.2f" % (det.confidence * 100) + "%"
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)  # BLUE BOX
                    if len(classes) > 0:
                        det_cls = det.cls
                        cv2.putText(frame, str(det_cls) + " " + score, (int(bbox[0]), int(bbox[3])), 0,
                                    1.5e-3 * frame.shape[0], (0, 255, 0), 2)


            # display counts for each class as they appear
            y = 0.2 * frame.shape[0]
            for cls in class_counter:
                class_count = class_counter[cls]
                cv2.putText(frame, str(cls) + " " + str(class_count), (int(0.05 * frame.shape[1]), int(y)), 0,
                            1.5e-3 * frame.shape[0], (0, 255, 255), 2)
                y += 0.05 * frame.shape[0]

            # Break the loop if the video has ended
            if not return_value:
                print('Video has ended or failed, try a different video format!')
                break

            # Display the frame if required
            if not self._dont_show:
                cv2.imshow(self._file_counter_log_name, frame)

            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # calculate current minute
        now = datetime.datetime.now()
        #rounded_now = now - datetime.timedelta(microseconds=now.microsecond)  # round to nearest second
        current_minute = now.time().minute  # reset counts every hour

        if current_minute == 0 and len(count_dict) > 1:
            count_dict = {}  # reset counts every hour

        # Save the count log to the database
        vehicle_log = VehicleLog.objects.create(
            filename=self._file_counter_log_name,
            total_count=total_counter,
            up_count=up_count,
            down_count=down_count,
            class_counts=class_counts
        )
        vehicle_log.save()

                    

            # calculate frames per second of running detections
        fps = 1.0 / (time.time() - start_time)
            #print("FPS: %.2f" % fps)
        result = np.asarray(frame)
        result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Calculate delay between consecutive frames
            #current_time = cv2.getTickCount()
            #delay = (current_time - prev_time) / cv2.getTickFrequency() * 1000
            #prev_time = current_time
    
            # Accumulate delay statistics
            #delay_sum += delay
            #delay_count += 1
        if self._output:
            out.write(result)

        # Calculate the average delay per frame
        avg_delay = delay_sum / delay_count
        print(f"Average delay per frame: {avg_delay:.2f} milliseconds")
        print("total_counter:", total_counter)
        print("up_count:", up_count)
        print("down_count:", down_count)
        print("class_counts:", class_counts)

        cv2.destroyAllWindows() 