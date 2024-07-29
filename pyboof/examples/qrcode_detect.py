import numpy as np
import pyboof as pb
import cv2
import time
import os

def read_labels(label_path, img_shape):
    with open(label_path, 'r') as file:
        lines = file.readlines()
        ground_truths = []
        for line in lines:
            _, cx, cy, w, h = map(float, line.strip().split())
            img_height, img_width = img_shape[:2]
            x1 = int((cx - w / 2) * img_width)
            y1 = int((cy - h / 2) * img_height)
            x2 = int((cx + w / 2) * img_width)
            y2 = int((cy + h / 2) * img_height)
            ground_truths.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        return ground_truths

def iou(box1, box2):
    x1, y1 = box1[0]
    x2, y2 = box1[2]
    x1_2, y1_2 = box2[0]
    x2_2, y2_2 = box2[2]

    xi1 = max(x1, x1_2)
    yi1 = max(y1, y1_2)
    xi2 = min(x2, x2_2)
    yi2 = min(y2, y2_2)

    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area != 0 else 0

def process_image(image_path, label_path, load_image_times, read_label_times, process_times):
    # Measure time to load image
    start_time = time.monotonic()
    image = pb.load_single_band(image_path, np.uint8)
    opencv_image = pb.boof_to_ndarray(image)
    elapsed_ms = (time.monotonic() - start_time) * 1000
    load_image_times.append(elapsed_ms)
    print(f'Loaded image {image_path} in {elapsed_ms:.1f} ms')

    # Measure time to read labels
    start_time = time.monotonic()
    try:
        with open(label_path, 'r') as file:
            lines = file.readlines()
            ground_truths = []
            for line in lines:
                _, cx, cy, w, h = map(float, line.strip().split())
                img_height, img_width = opencv_image.shape[:2]
                x1 = int((cx - w / 2) * img_width)
                y1 = int((cy - h / 2) * img_height)
                x2 = int((cx + w / 2) * img_width)
                y2 = int((cy + h / 2) * img_height)
                ground_truths.append([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        elapsed_ms = (time.monotonic() - start_time) * 1000
        read_label_times.append(elapsed_ms)
        print(f'Read labels from {label_path} in {elapsed_ms:.1f} ms')
    except Exception as e:
        elapsed_ms = (time.monotonic() - start_time) * 1000
        read_label_times.append(elapsed_ms)
        print(f"Label file error for {label_path}: {e}")
        return None, None

    # Measure time to detect object
    start_time = time.monotonic()
    detector = pb.FactoryFiducial(np.uint8).qrcode()
    detector.detect(image)
    elapsed_ms = (time.monotonic() - start_time) * 1000
    process_times.append(elapsed_ms)


    min_x = min_y = float('inf')
    max_x = max_y = -float('inf')

    detected_boxes = []
    for qr in detector.detections:
        print("Message: " + qr.message)
        print("     at: " + str(qr.bounds))

        points = np.array([[p.x, p.y] for p in qr.bounds.vertexes], dtype=np.int32)

        # Update extreme points
        for p in points:
            x, y = p
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y

        cv2.putText(opencv_image, qr.message, (points[0][0], points[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Add detected box to list
        detected_boxes.append([(min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)])

    # Draw the encompassing bounding box using extreme points
    if min_x != float('inf') and max_x != -float('inf') and min_y != float('inf') and max_y != -float('inf'):
        top_left = (min_x, min_y)
        bottom_right = (max_x, max_y)
        cv2.rectangle(opencv_image, top_left, bottom_right, (0, 0, 0), 2)

    
    print(f'Processed image {image_path} in {elapsed_ms:.1f} ms')

    # Compute accuracy
    iou_threshold = 0.5
    correct_detections = 0
    for detected_box in detected_boxes:
        for ground_truth in ground_truths:
            if iou(detected_box, ground_truth) > iou_threshold:
                correct_detections += 1
                break

    accuracy = correct_detections / len(ground_truths) if ground_truths else 0
    return opencv_image, accuracy

def main(images_folder, labels_folder):
    example_result_count=5
    total_accuracy = 0
    valid_count = 0
    load_image_times = []
    read_label_times = []
    process_times = []

    for filename in os.listdir(images_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(images_folder, filename)
            label_path = os.path.join(labels_folder, filename.replace('.jpg', '.txt').replace('.png', '.txt'))

            if os.path.isfile(label_path):
                processed_image, accuracy = process_image(image_path, label_path, load_image_times, read_label_times, process_times)
                if processed_image is not None:
                    total_accuracy += accuracy
                    valid_count += 1
                    print(f'Accuracy for {filename}: {accuracy * 100:.2f}%')

                    #Save and display the result
                    #if example_result_count!=0:
                    cv2.imshow('Detected QR Codes', processed_image)
                    cv2.imwrite(f'Detected QR Code of {filename}.jpg', processed_image)
                    #    example_result_count-=1
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

    if valid_count > 0:
        average_accuracy = total_accuracy / valid_count
        print(f'Average Accuracy: {average_accuracy * 100:.2f}%')
    else:
        print("No valid images or labels found.")

    # Calculate and print average times
    avg_load_image_time = np.mean(load_image_times) if load_image_times else 0
    avg_read_label_time = np.mean(read_label_times) if read_label_times else 0
    avg_process_time = np.mean(process_times) if process_times else 0
    save_results(average_accuracy,avg_process_time)

    print(f'Average load image time: {avg_load_image_time:.1f} ms')
    print(f'Average read label time: {avg_read_label_time:.1f} ms')
    print(f'Average process time: {avg_process_time:.1f} ms')

def save_results(accuracy, process_time):
    with open(output_path, "w") as file:
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"Processing Time: {process_time}\n")

if __name__ == "__main__":
    output_path='../../results/pyboof_results.txt'
    images_folder = '../../data/qr code detection.v2i.yolov7pytorch/valid/images'
    labels_folder = '../../data/qr code detection.v2i.yolov7pytorch/valid/labels'
    main(images_folder, labels_folder)
