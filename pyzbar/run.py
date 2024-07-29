import cv2
import numpy as np
from pyzbar.pyzbar import decode
import time
import os


#function for saving results
def save_results(accuracy, process_time):
    with open(output_path, "w") as file:
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"Processing Time: {process_time}\n")

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
    box1 = cv2.boundingRect(np.array(box1, dtype=np.int32))
    box2 = cv2.boundingRect(np.array(box2, dtype=np.int32))
    
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area

def process_image(img_path, label_path, load_image_times, load_label_times, process_times):
    # Measure time to load image
    start_time = time.monotonic()
    img = cv2.imread(img_path)
    elapsed_ms = (time.monotonic() - start_time) * 1000
    load_image_times.append(elapsed_ms)
    print(f'Load image {img_path} in {elapsed_ms:.1f} ms')

    # Measure time to read labels
    start_time = time.monotonic()
    ground_truths = read_labels(label_path, img.shape)
    elapsed_ms = (time.monotonic() - start_time) * 1000
    load_label_times.append(elapsed_ms)
    print(f'Read labels for {img_path} in {elapsed_ms:.1f} ms')

    # Measure time to detect object
    start_time = time.monotonic()
    min_x = min_y = float('inf')
    max_x = max_y = -float('inf')

    detected_boxes = []
    for barcode in decode(img):
        myData = barcode.data.decode("utf-8")
        
        color = (0, 0, 255)
        pts = np.array([barcode.polygon], np.int32)
        pts = pts.reshape((-1, 1, 2))
        elapsed_ms = (time.monotonic() - start_time) * 1000
        process_times.append(elapsed_ms)
        detected_boxes.append(pts)
        
        for p in pts:
            x, y = p[0]
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y
        
        pts2 = barcode.rect
        cv2.putText(img, myData, (pts2[0], pts2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
    if min_x != float('inf') and max_x != -float('inf') and min_y != float('inf') and max_y != -float('inf'):
        top_left = (min_x, min_y)
        bottom_right = (max_x, max_y)
        cv2.rectangle(img, top_left, bottom_right, (255, 0, 0), 2)

    
    print(f'Processed {img_path} in {elapsed_ms:.1f} ms')

    iou_threshold = 0.5
    correct_detections = 0
    for detected_box in detected_boxes:
        for ground_truth in ground_truths:
            if iou(detected_box, ground_truth) > iou_threshold:
                correct_detections += 1
                break

    accuracy = correct_detections / len(ground_truths) if ground_truths else 0
    print(f'Accuracy for {img_path}: {accuracy * 100:.2f}%')

    return accuracy, img

def main(image_dir, label_dir):
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    total_accuracy = 0
    num_images = len(image_files)
    
    if num_images == 0:
        print("No images found in the directory.")
        return

    load_image_times = []
    load_label_times = []
    process_times = []

    for image_file in image_files:
        img_path = os.path.join(image_dir, image_file)
        label_file = image_file.replace('.jpg', '.txt')
        label_path = os.path.join(label_dir, label_file)

        if not os.path.exists(label_path):
            print(f"Label file {label_path} does not exist. Skipping image {img_path}.")
            continue

        accuracy, img = process_image(img_path, label_path, load_image_times, load_label_times, process_times)
        total_accuracy += accuracy
        #extract the base name
        base_name = os.path.splitext(image_file)[0]
        # Save or display the resulting image
        
        cv2.imshow('Result', img)
        cv2.imwrite(f'Result {base_name}.jpg', img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    average_accuracy = total_accuracy / num_images if num_images > 0 else 0
    print(f'Overall Accuracy: {average_accuracy * 100:.2f}%')

    # Calculate and print average times
    avg_load_image_time = np.mean(load_image_times) if load_image_times else 0
    avg_load_label_time = np.mean(load_label_times) if load_label_times else 0
    avg_process_time = np.mean(process_times) if process_times else 0
    save_results(average_accuracy,avg_process_time)


    print(f'Average load image time: {avg_load_image_time:.1f} ms')
    print(f'Average load label time: {avg_load_label_time:.1f} ms')
    print(f'Average process time: {avg_process_time:.1f} ms')

if __name__ == "__main__":
    output_path='../results/pyzbar_results.txt'
    image_directory = '../data/qr code detection.v2i.yolov7pytorch/valid/images'
    label_directory = '../data/qr code detection.v2i.yolov7pytorch/valid/labels'
    main(image_directory, label_directory)
