import torch
import os
import time
import cv2
import psutil
import matplotlib.pyplot as plt
import pandas as pd

def run_yolov5_and_monitor_ram(model_path, image_dir):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, trust_repo=True)

    images = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".bmp", ".png"))
    ])

    proc = psutil.Process(os.getpid())
    ram_usage = []
    timestamps = []
    processing_times = []
    image_names = []

    print(f"[✓] Running model: {model_path} on {len(images)} images")

    start = time.time()

    for i, img_path in enumerate(images):
        t0 = time.time()
        img = cv2.imread(img_path)
        model(img, size=640)
        t1 = time.time()

        duration = t1 - t0
        current_time = t1 - start
        current_ram = proc.memory_info().rss / (1024 * 1024)

        image_names.append(os.path.basename(img_path))
        processing_times.append(duration)
        timestamps.append(current_time)
        ram_usage.append(current_ram)

        print(f"[{i+1}] RAM: {current_ram:.2f} MB - {os.path.basename(img_path)} - Time: {duration:.3f}s")

    end = time.time()
    total_time = end - start
    fps = len(images) / total_time
    peak_ram = max(ram_usage)

    print(f"\n[✓] Done. FPS: {fps:.2f}, Peak RAM: {peak_ram:.2f} MB, Time: {total_time:.2f}s")

    # Vẽ biểu đồ scatter
    plt.figure(figsize=(10, 5))
    plt.scatter(timestamps, ram_usage, s=25, color='blue', label='YOLOv5')
    plt.xlabel("Thời gian (s)")
    plt.ylabel("RAM sử dụng (MB)")
    plt.title(f"RAM Usage – {os.path.basename(model_path)}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_name = f"ram_plot_{os.path.basename(model_path).replace('.pt','')}.png"
    plt.savefig(plot_name)
    plt.show()

    # Lưu file CSV
    df = pd.DataFrame({
        "image_name": image_names,
        "processing_time_s": processing_times,
        "timestamp_s": timestamps,
        "ram_MB": ram_usage
    })

    csv_name = f"ram_data_{os.path.basename(model_path).replace('.pt','')}.csv"
    df.to_csv(csv_name, index=False)
    print(f"[✓] Dữ liệu chi tiết đã lưu tại: {csv_name}")

if __name__ == "__main__":
    run_yolov5_and_monitor_ram("v5/best.pt", "input")
