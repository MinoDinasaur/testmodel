import os, sys, time
import cv2
import torch
import psutil
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Thêm đường dẫn repo yolov7 (phải đúng!)
sys.path.append("testonv7/yolov7")

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression
from utils.torch_utils import select_device

def run_yolov7_ram(model_path, image_dir):
    device = select_device("cpu")  # hoặc 'cuda:0' nếu muốn đo GPU
    model = attempt_load(model_path, map_location=device)
    model.eval()

    image_paths = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.png', '.bmp'))
    ])

    proc = psutil.Process(os.getpid())
    ram_usage = []
    timestamps = []
    processing_times = []
    image_names = []

    start = time.time()
    for i, img_path in enumerate(image_paths):
        t0 = time.time()

        img0 = cv2.imread(img_path)
        img = letterbox(img0, new_shape=640)[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = torch.from_numpy(img.copy()).float() / 255.0
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img)[0]
            pred = non_max_suppression(pred, 0.25, 0.45)[0]

        t1 = time.time()
        duration = t1 - t0
        current_time = t1 - start
        current_ram = proc.memory_info().rss / (1024 * 1024)

        image_names.append(Path(img_path).name)
        processing_times.append(duration)
        timestamps.append(current_time)
        ram_usage.append(current_ram)

        print(f"[{i+1}] {Path(img_path).name} – RAM: {current_ram:.2f} MB – Time: {duration:.3f}s")

    # Thống kê
    total_time = time.time() - start
    fps = len(image_paths) / total_time
    peak_ram = max(ram_usage)

    print(f"\n[✓] Done! FPS: {fps:.2f}, Peak RAM: {peak_ram:.2f} MB, Time: {total_time:.2f} s")

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 5))
    plt.scatter(timestamps, ram_usage, s=25, color="blue", label=os.path.basename(model_path))
    plt.xlabel("Thời gian (s)")
    plt.ylabel("RAM sử dụng (MB)")
    plt.title(f"RAM Usage – {os.path.basename(model_path)}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_name = f"ram_plot_{Path(model_path).stem}.png"
    plt.savefig(plot_name)
    plt.show()

    # Lưu file CSV
    df = pd.DataFrame({
        "image_name": image_names,
        "processing_time_s": processing_times,
        "timestamp_s": timestamps,
        "ram_MB": ram_usage
    })

    csv_name = f"ram_data_{Path(model_path).stem}.csv"
    df.to_csv(csv_name, index=False)
    print(f"[✓] Dữ liệu chi tiết đã lưu tại: {csv_name}")

if __name__ == "__main__":
    run_yolov7_ram("v7/best.pt", "input")
