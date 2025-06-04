import os
import cv2
from ultralytics import YOLO

# Đường dẫn tới model và thư mục ảnh
model_path = 'v7/best.pt'
input_folder = 'input'      # Thư mục chứa ảnh cần detect
output_folder = 'output'    # Thư mục lưu ảnh sau detect

# Tạo thư mục output nếu chưa có
os.makedirs(output_folder, exist_ok=True)

# Load model YOLO
model = YOLO(model_path)

# Duyệt qua tất cả ảnh trong thư mục
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        image_path = os.path.join(input_folder, filename)
        image = cv2.imread(image_path)

        # Chạy YOLOv8 detect
        results = model(image)

        # Vẽ bounding box
        for result in results:
            annotated_image = result.plot()

        # Lưu ảnh đã annotate
        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, annotated_image)
        print(f"✅ Đã xử lý: {filename}")
