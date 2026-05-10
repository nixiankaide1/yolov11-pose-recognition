import cv2

video_path = "path/to/your/video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("无法打开视频文件")
else:
    print("视频文件打开成功")
    ret, frame = cap.read()
    if ret:
        print("成功读取第一帧")
    else:
        print("无法读取视频帧")

cap.release()