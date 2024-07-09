import torch
import cv2
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

external_cam_index = 1
camera = cv2.VideoCapture(external_cam_index)

if not camera.isOpened():
    exit()

def get_color_label(avg_color):
    r, g, b = avg_color

    if r > 0 and b > 0 and g > 0:
        return 'person'
    else:
        return 'undefined'

frame_count = 0
while camera.isOpened():
    ret, img = camera.read()  
    if not ret:
        break

    frame_count += 1
    if frame_count % 2 != 0: 
        continue

    scale_percent = 100  
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    result = model(img)

    detect = result.xyxy[0].cpu().numpy()

    
    for i in range(len(detect)):
        class_id = int(detect[i, 5])
        #
        if class_id == 0:
            x1 = int(detect[i, 0])
            y1 = int(detect[i, 1])
            x2 = int(detect[i, 2])
            y2 = int(detect[i, 3])

            center_x1 = int(x1 + (x2 - x1) * 0.25)
            center_y1 = int(y1 + (y2 - y1) * 0.25)
            center_x2 = int(x1 + (x2 - x1) * 0.75)
            center_y2 = int(y1 + (y2 - y1) * 0.75)

            person_roi = img[center_y1:center_y2, center_x1:center_x2]
            avg_color_per_row = np.average(person_roi, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)
            avg_color = avg_color.astype(int)

            label = get_color_label(avg_color)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    cv2.imshow('Camera Feed', img)

    if cv2.waitKey(1) & 0xFF == ord('1'):
        break

camera.release()
cv2.destroyAllWindows()
