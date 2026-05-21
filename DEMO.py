import cv2
import numpy as np
from tensorflow.keras.models import load_model

def preProcessingImage(img, IMG_SIZE=(128, 128)):
    # Resize ảnh về kích thước chuẩn
    img_resized = cv2.resize(img, IMG_SIZE)
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_resized = img_resized / 255.0  # Chuẩn hóa pixel [0, 1]

    return img_resized

# Load mô hình đã lưu
model_face_detection = load_model("face_detection_model_2_5.keras")
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h_img, w_img, _ = frame.shape  # Lấy kích thước ảnh gốc
    print(w_img, h_img)

    pre_processing_image = preProcessingImage(frame)

    # Chuyển thành tensor có shape phù hợp (1, 128, 128, 1)
    input = np.expand_dims(pre_processing_image, axis=-1)  # Thêm channel
    input = np.expand_dims(input, axis=0)  # Thêm batch size

    # Dự đoán tọa độ (x, y, w, h)
    prediction = model_face_detection.predict(input)[0]  # Lấy kết quả đầu ra

    # Giải chuẩn hóa về kích thước gốc
    x, y, w, h, conf = prediction

    x = int(x * w_img)
    y = int(y * h_img)
    w = int(w * w_img)
    h = int(h * h_img)

    x = x - int(w/2)
    y = y - int(y/2)
    w = int(w*1.5)
    h = int(h*1.5)

    if conf > 0.6:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"Conf: {conf:.2f}"

        cv2.putText(
        frame,
        text,
        (x, y - 10),   # vị trí text phía trên box
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

    cv2.imshow("Kết quả nhận diện", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()

