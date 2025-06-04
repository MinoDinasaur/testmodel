import os
import time
import psutil
import matplotlib.pyplot as plt
import pandas as pd
from ultralytics import YOLO

def run_model_and_monitor_ram(model_path, image_dir, interval=0.2):
    model = YOLO(model_path)
    VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    images = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.lower().endswith(VALID_EXTS)
    ])

    print(f"[✓] Running model: {model_path} on {len(images)} images")

    proc = psutil.Process(os.getpid())
    ram_usage = []
    timestamps = []
    processing_times = []
    image_names = []

    start_time = time.time()

    for i, img in enumerate(images):
        t0 = time.time()
        model.predict(source=img, save=False, stream=False, verbose=False)
        t1 = time.time()

        duration = t1 - t0
        current_ram = proc.memory_info().rss / (1024 * 1024)
        current_timestamp = t1 - start_time

        processing_times.append(duration)
        ram_usage.append(current_ram)
        timestamps.append(current_timestamp)
        image_names.append(os.path.basename(img))

        print(f"Image {i+1}/{len(images)} - Time: {duration:.3f}s - RAM: {current_ram:.2f} MB")

    total_time = time.time() - start_time
    fps = len(images) / total_time
    peak_ram = max(ram_usage)

    print(f"\n[✓] Done.")
    print(f"Peak RAM: {peak_ram:.2f} MB")
    print(f"FPS: {fps:.2f}")
    print(f"Total Time: {total_time:.2f} seconds")

    # Vẽ biểu đồ RAM
    plt.figure(figsize=(10, 5))
    plt.scatter(timestamps, ram_usage, label=model_path, s=25, color='blue')
    plt.xlabel("Thời gian (s)")
    plt.ylabel("RAM sử dụng (MB)")
    plt.title(f"RAM Usage theo thời gian – {os.path.basename(model_path)}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plot_name = f"ram_plot_{os.path.basename(model_path).replace('.pt','')}.png"
    plt.savefig(plot_name)
    plt.show()

    # ✅ Lưu bảng dữ liệu
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
    model_path = "v8/bestv8.pt"     
    image_dir = "input"             
    run_model_and_monitor_ram(model_path, image_dir)
