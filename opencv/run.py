import cv2
import numpy as np
import os
import time

def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area != 0 else 0
    return iou

def normalize_to_absolute(box, img_width, img_height):
    x_center, y_center, width, height = box
    x_center *= img_width
    y_center *= img_height
    width *= img_width
    height *= img_height

    x1 = int(x_center - (width / 2))
    y1 = int(y_center - (height / 2))
    x2 = int(x_center + (width / 2))
    y2 = int(y_center + (height / 2))

    return (x1, y1, x2, y2)

def process_image(image_path, label_path, load_image_times, read_label_times, process_times):
    # Measure time to load image
    start_time = time.monotonic()
    img = cv2.imread(image_path)
    elapsed_ms = (time.monotonic() - start_time) * 1000
    load_image_times.append(elapsed_ms)
    print(f'Loaded image {image_path} in {elapsed_ms:.1f} ms')

    # Measure time to read labels
    start_time = time.monotonic()
    try:
        with open(label_path, 'r') as f:
            label_data = f.readline().strip().split()
        read_label_times.append((time.monotonic() - start_time) * 1000)
        label_box = list(map(float, label_data[1:]))
        img_width, img_height = img.shape[1], img.shape[0]
        ground_truth_box = normalize_to_absolute(label_box, img_width, img_height)
    except Exception as e:
        read_label_times.append((time.monotonic() - start_time) * 1000)
        print(f"Label file error for {label_path}: {e}")
        return None, None

    # Measure time to process image
    start_time = time.monotonic()
    qcd = cv2.QRCodeDetector()
    retval, decoded_info, points, _ = qcd.detectAndDecodeMulti(img)
    process_times.append((time.monotonic() - start_time) * 1000)
    
    if retval:
        points = np.array(points).astype(int)

        min_x = min_y = float('inf')
        max_x = max_y = -float('inf')

        for point in points:
            for p in point:
                x, y = p
                if x < min_x:
                    min_x = x
                if y < min_y:
                    min_y = y
                if x > max_x:
                    max_x = x
                if y > max_y:
                    max_y = y

        top_left = (min_x, min_y)
        bottom_right = (max_x, max_y)
        img = cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 3)

        for s, p in zip(decoded_info, points):
            p_tuple = tuple(p[0])
            img = cv2.putText(img, s, p_tuple, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        detected_box = (min_x, min_y, max_x, max_y)
        iou = calculate_iou(ground_truth_box, detected_box)
        print(f'IoU for {image_path}: {iou}')
        
        return iou, img
    else:
        print(f"QR code not found in {image_path}.")
        return None, img

def main(images_folder, labels_folder):
    total_iou = 0
    valid_count = 0
    load_image_times = []
    read_label_times = []
    process_times = []

    for filename in os.listdir(images_folder):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(images_folder, filename)
            label_path = os.path.join(labels_folder, filename.replace('.jpg', '.txt').replace('.png', '.txt'))

            if os.path.isfile(label_path):
                iou, processed_image = process_image(image_path, label_path, load_image_times, read_label_times, process_times)
                valid_count += 1
                if iou is not None:
                    total_iou += iou

                # Save or display the resulting image
                if processed_image is not None:
                    base_name = os.path.splitext(filename)[0]
                    
                    cv2.imshow('Result', processed_image)
                    cv2.imwrite(f'result examples{base_name}_result.jpg', processed_image)
                    print('saved')
                    cv2.waitKey(0)

    cv2.destroyAllWindows()

    if valid_count > 0:
        average_iou = total_iou / valid_count
        print(f'Average IoU: {average_iou:.4f}')
    else:
        print("No valid images or labels found.")

    # Calculate and print average times
    avg_load_image_time = np.mean(load_image_times) if load_image_times else 0
    avg_read_label_time = np.mean(read_label_times) if read_label_times else 0
    avg_process_time = np.mean(process_times) if process_times else 0
    save_results(average_iou, avg_process_time)

    print(f'Average load image time: {avg_load_image_time:.1f} ms')
    print(f'Average read label time: {avg_read_label_time:.1f} ms')
    print(f'Average process time: {avg_process_time:.1f} ms')

def save_results(accuracy, process_time):
    with open(output_path, "w") as file:
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"Processing Time: {process_time}\n")

if __name__ == "__main__":
    output_path='../results/OpenCV_results.txt'
    images_folder = '../data/qr code detection.v2i.yolov7pytorch/valid/images'
    labels_folder = '../data/qr code detection.v2i.yolov7pytorch/valid/labels'
    main(images_folder, labels_folder)
