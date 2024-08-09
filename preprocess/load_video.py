import cv2
import os
import numpy as np

def extract_frames(video_path, output_folder, fps=None, interval=None):
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("无法打开视频文件")
        return
    
    # 获取视频的FPS（帧率）
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / video_fps

    print(f"视频时长: {duration} 秒")
    print(f"视频总帧数: {frame_count} 帧")
    print(f"视频帧率: {video_fps} FPS")
    
    if fps is not None:
        interval = 1 / fps
    elif interval is not None:
        fps = 1 / interval
    else:
        print("请提供fps或者interval参数")
        return
    
    print(f"每秒提取帧数: {fps} FPS")
    print(f"每 {interval} 秒提取一帧")
    
    frame_interval = int(video_fps * interval)
    print(f"帧间隔: {frame_interval}")
    
    frame_id = 0
    extracted_frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_id % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"{extracted_frame_id:04d}.png")
            cv2.imwrite(frame_filename, frame)
            extracted_frame_id += 1
        
        frame_id += 1
    
    cap.release()
    print("帧提取完成")


def calculate_average_brightness(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return None

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_brightness = 0
    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_brightness = np.mean(gray_frame)
        total_brightness += frame_brightness
        frame_id += 1

    cap.release()
    average_brightness = total_brightness / frame_count
    return average_brightness


def count_frames_by_brightness(video_path, threshold_high, threshold_low):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return None, None

    high_count = 0
    low_count = 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_brightness = np.mean(gray_frame)

        if frame_brightness > threshold_high or frame_brightness < threshold_low:
            high_count += 1
        else:
            low_count += 1

    cap.release()
    return high_count, low_count, frame_count

def adaptive_extract_frames(video_path, output_folder, high_fps=None, low_fps=None, high_interval=None, low_interval=None, threshold_ratio=0.3):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"video fps is {video_fps}")
    average_brightness = calculate_average_brightness(video_path)
    print(f"average brightness is {average_brightness}")
    threshold_high = average_brightness * (1 + threshold_ratio)
    threshold_low = average_brightness * (1 - threshold_ratio)

    if high_fps is not None:
        high_interval = 1 / high_fps
    if low_fps is not None:
        low_interval = 1 / low_fps

    if high_interval is None or low_interval is None:
        print("请提供 high_fps 和 low_fps 或者 high_interval 和 low_interval 参数")
        return

    high_frame_interval = int(video_fps * high_interval)
    low_frame_interval = int(video_fps * low_interval)
    print(f"high freq sample --- {video_fps / high_frame_interval} frame per second")
    print(f"low freq sample --- {video_fps / low_frame_interval} frame per second")

    frame_id = 0
    extracted_frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_brightness = np.mean(gray_frame)

        if frame_brightness > threshold_high or frame_brightness < threshold_low:
            if frame_id % high_frame_interval == 0:
                frame_filename = os.path.join(output_folder, f"{extracted_frame_id:04d}.png")
                cv2.imwrite(frame_filename, frame)
                print(f"write to {frame_filename}")
                extracted_frame_id += 1
        else:
            if frame_id % low_frame_interval == 0:
                frame_filename = os.path.join(output_folder, f"{extracted_frame_id:04d}.png")
                cv2.imwrite(frame_filename, frame)
                print(f"write to {frame_filename}")
                extracted_frame_id += 1

        frame_id += 1

    cap.release()
    print("帧提取完成")


def extract_fixed_number_frames(video_path, output_folder, target_frame_count, threshold_ratio=0.3):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("无法打开视频文件")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"video fps is {video_fps}")
    average_brightness = calculate_average_brightness(video_path)
    print(f"average brightness is {average_brightness}")
    threshold_high = average_brightness * (1 + threshold_ratio)
    threshold_low = average_brightness * (1 - threshold_ratio)

    high_count, low_count, frame_count = count_frames_by_brightness(video_path, threshold_high, threshold_low)
    
    total_duration = frame_count / video_fps

    # Calculate high_fps and low_fps based on the target frame count and the proportion of high and low brightness frames
    if high_count + low_count == 0:
        print("无法计算高低亮度帧的数量")
        return

    high_fps = (target_frame_count * (high_count / frame_count)) / total_duration
    low_fps = (target_frame_count * (low_count / frame_count)) / total_duration

    high_interval = 1 / high_fps if high_fps > 0 else float('inf')
    low_interval = 1 / low_fps if low_fps > 0 else float('inf')

    high_frame_interval = int(video_fps * high_interval)
    low_frame_interval = int(video_fps * low_interval)

    print(f"高亮度帧数量: {high_count}, 低亮度帧数量: {low_count}, 视频总帧数: {frame_count}")
    print(f"高亮度帧FPS: {high_fps}, 低亮度帧FPS: {low_fps}")
    print(f"高亮度帧间隔: {high_frame_interval}, 低亮度帧间隔: {low_frame_interval}")

    frame_id = 0
    extracted_frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_brightness = np.mean(gray_frame)

        if frame_brightness > threshold_high or frame_brightness < threshold_low:
            if frame_id % high_frame_interval == 0:
                frame_filename = os.path.join(output_folder, f"{extracted_frame_id:04d}.png")
                cv2.imwrite(frame_filename, frame)
                print(f"write to {frame_filename}")
                extracted_frame_id += 1
        else:
            if frame_id % low_frame_interval == 0:
                frame_filename = os.path.join(output_folder, f"{extracted_frame_id:04d}.png")
                cv2.imwrite(frame_filename, frame)
                print(f"write to {frame_filename}")
                extracted_frame_id += 1

        frame_id += 1

    cap.release()
    print("帧提取完成")


def extract_fix_num_frames(video_path, num_frames, save_dir):
    # 检查并创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    # 获取视频的总帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 计算每个要提取帧的位置
    step = total_frames // num_frames
    frame_indices = [i * step for i in range(num_frames)]
    
    frame_count = 0
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检查当前帧是否在我们需要的位置上
        if i in frame_indices:
            # 保存提取的帧为图像文件，文件名从0000开始顺序命名
            frame_filename = os.path.join(save_dir, f'{frame_count:04d}.png')
            cv2.imwrite(frame_filename, frame)
            frame_count += 1
    
    # 释放视频捕获对象
    cap.release()

if __name__ == "__main__":
    for scene in ["sofa", "table", "dressing_table"]:
        os.system(f"rm -rf /cfs/wangboyuan/dataset/furniture/sparse_{scene}/database.db")
        os.system(f"rm -rf /cfs/wangboyuan/dataset/furniture/sparse_{scene}/dense")
        os.system(f"rm -rf /cfs/wangboyuan/dataset/furniture/sparse_{scene}/sparse")
        os.system(f"rm -rf /cfs/wangboyuan/dataset/furniture/sparse_{scene}/images")
        os.system(f"rm -rf /cfs/wangboyuan/dataset/furniture/sparse_{scene}/mask")
        input  = f'/cfs/wangboyuan/dataset/furniture/sparse_{scene}/video.mp4'
        save_dir = os.path.join(os.path.split(input)[0], "images")
        extract_fix_num_frames(input, 50, save_dir)