import os
import cv2

# Đường dẫn dataset
dataset_path = "WIDER_train/images"
label_file = "wider_face_train_bbx_gt.txt"

def getImageByFileName(filename):
    # Đọc file nhãn
    with open(label_file, "r") as f:
        lines = f.readlines()

    faces = []

    # Lấy ngẫu nhiên 1 ảnh
    idx = 0
    while not lines[idx].strip().endswith(filename):  # Tìm dòng chứa tên file ảnh
        idx += 1

    file_name = lines[idx].strip()
    num_faces = int(lines[idx + 1].strip())
    faces = lines[idx + 2 : idx + 2 + num_faces]

    return file_name, num_faces, faces, lines, idx

def preProcessingImage(img, IMG_SIZE=(128, 128)):
    # Resize ảnh về kích thước chuẩn
    img_resized = cv2.resize(img, IMG_SIZE)
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_resized = img_resized / 255.0  # Chuẩn hóa pixel [0, 1]

    return img_resized

def calculate_confidence(blur, expression, illumination, invalid, occlusion, pose):
    confidence = 1.0

    # Blur
    if blur == 1:
        confidence -= 0.1
    elif blur == 2:
        confidence -= 0.3

    # Expression
    if expression == 1:
        confidence -= 0.1

    # Illumination
    if illumination == 1:
        confidence -= 0.1

    # Invalid box
    if invalid == 1:
        return 0.0

    # Occlusion
    if occlusion == 1:
        confidence -= 0.2
    elif occlusion == 2:
        confidence -= 0.4

    # Pose
    if pose == 1:
        confidence -= 0.2

    # Clamp về khoảng [0, 1]
    return max(0.0, min(1.0, confidence))

IMG_SIZE = (128, 128) 

#Lấy 1 ảnh ngẫu nhiên
file_name, num_faces, faces, _, _ = getImageByFileName("8--Election_Campain/8_Election_Campain_Election_Campaign_8_71.jpg")
print(file_name)

# Đọc ảnh
img_path = os.path.join(dataset_path, file_name)
image = cv2.imread(img_path)

ih, iw, _ = image.shape
print(ih, iw)

# image = preProcessingImage(image)

scale_x = IMG_SIZE[0] / iw
scale_y = IMG_SIZE[1] / ih


for face in faces: 
    face_data = list(map(int, face.split()))
    x, y, w, h, blur, expression, illumination, invalid, occlusion, pose = face_data[:10]
    
    confidence = calculate_confidence(blur=blur, expression=expression, illumination=illumination, invalid=invalid, occlusion=occlusion, pose=pose)
    
    x = int(x * scale_x)
    y = int(y * scale_y)
    w = int(w * scale_x)
    h = int(h * scale_y)
    print(x, y, w, h, confidence)
    
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("hinh", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(image.shape)
print("This is image being RGB")
print("Because the value in 1 layout is increment")