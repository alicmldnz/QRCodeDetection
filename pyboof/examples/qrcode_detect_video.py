import numpy as np
import pyboof as pb
import cv2
import time

# QR Kodları tespit eden ve mesajlarını yazdıran kod

# Webcam'i başlat
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

detector = pb.FactoryFiducial(np.uint8).qrcode()

while True:
    start_time = time.monotonic()
    
    # Kameradan bir kare yakala
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame (stream end?). Exiting ...")
        break
    
    # OpenCV formatındaki kareyi BoofCV formatına dönüştür
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boof_image = pb.ndarray_to_boof(gray)
    
    # QR kodlarını tespit et
    detector.detect(boof_image)
    opencv_image = pb.boof_to_ndarray(boof_image)
    print("Detected a total of {} QR Codes".format(len(detector.detections)))
    
    elapsed_ms = (time.monotonic() - start_time) * 1000    
    print('process in %.1fms' % (elapsed_ms))

    for qr in detector.detections:
        print("Message: " + qr.message)
        print("     at: " + str(qr.bounds))
        
        # QR kodu sınırlarını çizmek için köşe noktalarını alın
        points = np.array([[p.x, p.y] for p in qr.bounds.vertexes], dtype=np.int32)
        cv2.polylines(opencv_image, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # QR kodu mesajını görüntü üzerinde ekle
        cv2.putText(opencv_image, qr.message, (points[0][0], points[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Sonucu göster
    cv2.imshow('Detected QR Codes', opencv_image)
    
    # 'q' tuşuna basarak çıkış yap
    if cv2.waitKey(1) == ord('q'):
        break

# Kamerayı serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()

