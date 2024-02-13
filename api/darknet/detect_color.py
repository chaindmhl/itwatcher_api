import cv2
import numpy as np
import os

class YOLOv4Inference:
    def __init__(self, weights_path="/home/icebox/itwatcher_api/darknet/color/color.weights",
                 config_path="/home/icebox/itwatcher_api/darknet/color/yolov4-custom.cfg",
                 class_names_path="/home/icebox/itwatcher_api/darknet/color/color.names",
                 confidence_threshold=0.5, nms_threshold=0.4):
        self.weights_path = weights_path
        self.config_path = config_path
        self.class_names_path = class_names_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        with open(self.class_names_path, 'r') as f:
            self.class_names = f.read().strip().split('\n')

    def infer_image(self, img):
        height, width, _ = img.shape
        blob = cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        outputs = self.net.forward(self.output_layers)
        
        class_ids = []
        confidences = []
        boxes = []
        
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.confidence_threshold:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
            
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
            
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        detected_classes = []
        for i in indexes:
            detected_classes.append((self.class_names[class_ids[i]], boxes[i]))

        return img, detected_classes

    def infer_image_only(self, img):
        _, detected_classes = self.infer_image(img)
        # Check if any classes were detected
        if detected_classes:
            return {"detected_class": detected_classes[0]}  # Return the detected class only
        else:
            return {"detected_class": "Unrecognized Color"}  # Return a default value or handle the absence of detections

    def infer_and_save(self, plate_image, track_id):
        # Perform inference and get the detected class
        _, detected_classes = self.infer_image(plate_image)
        
        
        if detected_classes:
            # Assuming there's only one detected class
            detected_class, box = detected_classes[0]
            x, y, w, h = box

            # Generate the output filename based on the detected class and track ID
            output_filename = f"{detected_class}_{track_id}.jpg"
            
            output_path = output_filename
            
            # Save the plate image
            cv2.imwrite(output_path, plate_image)

            return {"detected_class": detected_class}
        
        else:
            # Handle the case where no objects were detected
            print("No objects detected")
