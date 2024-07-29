import cv2 as cv
import numpy as np
import os
import time


def save_results(accuracy, process_time):
    with open(output_path, "w") as file:
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"Processing Time: {process_time}\n")


def load_labels(label_path):
    labels = []
    with open(label_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            labels.append((class_id, x_center, y_center, width, height))
    return labels

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

# Load class names and YOLOv3-tiny model
classes = open('qrcode.names').read().strip().split('\n')
net = cv.dnn.readNetFromDarknet('qrcode-yolov3-tiny.cfg', 'qrcode-yolov3-tiny.weights')
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)

# Directory containing images and labels
image_dir = '../data/qr code detection.v2i.yolov7pytorch/valid/images/'
label_dir = '../data/qr code detection.v2i.yolov7pytorch/valid/labels/'
output_path='../results/YOLO_results.txt'

threshold = 0.6
iou_list = []

# Lists to store timing information
load_image_times = []
load_label_times = []
convert_blob_times = []
get_layers_times = []
forward_pass_times = []
postprocess_times = []

for image_name in os.listdir(image_dir):
    if not image_name.endswith(('.jpg', '.png')):
        continue

    # Measure time to load image
    start_time = time.monotonic()
    frame = cv.imread(os.path.join(image_dir, image_name))
    imgHeight, imgWidth = frame.shape[:2]
    elapsed_ms = (time.monotonic() - start_time) * 1000
    load_image_times.append(elapsed_ms)
    print(f'Load image {image_name} in {elapsed_ms:.1f} ms')

    # Measure time to load labels
    start_time = time.monotonic()
    label_path = os.path.join(label_dir, image_name.replace('.jpg', '.txt').replace('.png', '.txt'))
    labels = load_labels(label_path)
    elapsed_ms = (time.monotonic() - start_time) * 1000
    load_label_times.append(elapsed_ms)
    print(f'Load labels for {image_name} in {elapsed_ms:.1f} ms')

    # Measure time to convert frame to blob
    start_time = time.monotonic()
    blob = cv.dnn.blobFromImage(frame, 1/255, (416, 416), swapRB=True, crop=False)
    elapsed_ms = (time.monotonic() - start_time) * 1000
    convert_blob_times.append(elapsed_ms)
    print(f'Convert {image_name} to blob in {elapsed_ms:.1f} ms')

    # Measure time to determine the output layer
    start_time = time.monotonic()
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    elapsed_ms = (time.monotonic() - start_time) * 1000
    get_layers_times.append(elapsed_ms)
    print(f'Get output layers for {image_name} in {elapsed_ms:.1f} ms')

    # Measure time for forward pass
    net.setInput(blob)
    start_time = time.monotonic()
    outs = net.forward(ln)
    elapsed_ms = (time.monotonic() - start_time) * 1000
    forward_pass_times.append(elapsed_ms)
    print(f'Forward pass for {image_name} in {elapsed_ms:.1f} ms')

    # Measure time for postprocessing
    start_time = time.monotonic()
    frameHeight, frameWidth = frame.shape[:2]
    classIds = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > threshold:
                x, y, width, height = detection[:4] * np.array([frameWidth, frameHeight, frameWidth, frameHeight])
                left = int(x - width / 2)
                top = int(y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, int(width), int(height)])

    indices = cv.dnn.NMSBoxes(boxes, confidences, threshold, threshold - 0.1)
    detected_boxes = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        detected_boxes.append((left, top, left + width, top + height))
        elapsed_ms = (time.monotonic() - start_time) * 1000
        postprocess_times.append(elapsed_ms)
        
        # Draw bounding box for objects
        cv.rectangle(frame, (left, top), (left + width, top + height), (0, 0, 255), 2)

        # Draw class name and confidence
        label = '%s:%.2f' % (classes[classIds[i]], confidences[i])
        cv.putText(frame, label, (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

    # Calculate IoU for each detected box with the ground truth boxes
    for label in labels:
        gt_x_center = label[1] * imgWidth
        gt_y_center = label[2] * imgHeight
        gt_width = label[3] * imgWidth
        gt_height = label[4] * imgHeight
        gt_left = int(gt_x_center - gt_width / 2)
        gt_top = int(gt_y_center - gt_height / 2)
        gt_right = int(gt_x_center + gt_width / 2)
        gt_bottom = int(gt_y_center + gt_height / 2)
        gt_box = (gt_left, gt_top, gt_right, gt_bottom)

        for detected_box in detected_boxes:
            iou = calculate_iou(gt_box, detected_box)
            iou_list.append(iou)

    
    print(f'Postprocess {image_name} in {elapsed_ms:.1f} ms')

    # Display the image with bounding boxes
    #cv.imshow('QR Detection', frame)
    
    cv.imshow('Detected QR Codes', frame)
    cv.imwrite(f'Detected QR Code of {image_name}.jpg', frame)
    cv.waitKey(0)
    cv.waitKey(0)  # Press any key to move to the next image

cv.destroyAllWindows()

# Calculate mean IoU
mean_iou = np.mean(iou_list) if iou_list else 0
print('Mean IoU: %.2f' % mean_iou)

# Calculate and print average times
avg_load_image_time = np.mean(load_image_times) if load_image_times else 0
avg_load_label_time = np.mean(load_label_times) if load_label_times else 0
avg_convert_blob_time = np.mean(convert_blob_times) if convert_blob_times else 0
avg_get_layers_time = np.mean(get_layers_times) if get_layers_times else 0
avg_forward_pass_time = np.mean(forward_pass_times) if forward_pass_times else 0
avg_postprocess_time = np.mean(postprocess_times) if postprocess_times else 0
sum_avg=avg_forward_pass_time+avg_postprocess_time
save_results(mean_iou,sum_avg)

#print(f'Average load image time: {avg_load_image_time:.1f} ms')
#print(f'Average load label time: {avg_load_label_time:.1f} ms')
#print(f'Average convert blob time: {avg_convert_blob_time:.1f} ms')
#print(f'Average get output layers time: {avg_get_layers_time:.1f} ms')
#print(f'Average forward pass time: {avg_forward_pass_time:.1f} ms')
#print(f'Average postprocess time: {avg_postprocess_time:.1f} ms')
#print(f'Avarage sum of forward pass and postprocess time: {sum_avg:.lf} ms')