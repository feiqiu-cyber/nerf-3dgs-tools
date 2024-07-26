import cv2
import os

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
            frame_filename = os.path.join(output_folder, f"frame_{extracted_frame_id:04d}.png")
            cv2.imwrite(frame_filename, frame)
            extracted_frame_id += 1
        
        frame_id += 1
    
    cap.release()
    print("帧提取完成")