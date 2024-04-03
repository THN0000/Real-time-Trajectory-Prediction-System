import cv2

# 打开视频文件
cap = cv2.VideoCapture('video/2.mp4')

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Could not open video file")

# 循环读取视频帧
while cap.isOpened():
    # 读取一帧
    ret, frame = cap.read()

    # 检查是否成功读取帧
    if not ret:
        break
    frame = cv2.resize(frame, (640, 480))
    # 显示帧
    cv2.imshow('Video', frame)

    # 检查是否按下 'q' 键退出循环
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# 关闭窗口和视频流
cap.release()
cv2.destroyAllWindows()
