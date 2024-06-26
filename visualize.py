import numpy as np
import cv2


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [i for i in line if not i == '']
            # line = [float(i) for i in line]
            # 将每个数据项按制表符分割，然后转换为浮点数
            line = [float(item) for subline in line for item in subline.split('\t')]
            # 将数据行添加到数据列表中
            data.append(line)
            # data.append(line)
    return np.asarray(data)


if __name__ == '__main__':
    data = read_file('./datasets/eth/test/obsmat.txt')


    H = np.array([
        [2.8128700e-02, 2.0091900e-03, -4.6693600e+00],
        [8.0625700e-04, 2.5195500e-02, -5.0608800e+00],
        [3.4555400e-04, 9.2512200e-05, 4.6255300e-01]
    ])
    H_inv = np.linalg.inv(H)
    frames = np.unique(data[:, 0]).tolist()  # 总帧数
    frame_data = []
    pixel_poses = []
    for frame in frames:
        frame_data.append(data[frame == data[:, 0], :])  # 重组gt, frame_data列表的每一个元素代表当前帧下所有Person的出现情况
    for frame_data_ in frame_data:
        print(frame_data_.shape)
        world_pos = np.vstack((frame_data_[:, 2], frame_data_[:, 4]))
        world_pos = np.vstack((world_pos, np.ones((world_pos.shape[1]))))
        pixel_pos = np.dot(H_inv, world_pos)
        pixel_pos_ = pixel_pos[:2, :] / pixel_pos[2:, :]
        pixel_poses.append(pixel_pos_)

    video = cv2.VideoCapture('./ewap_dataset/seq_eth/seq_eth.avi')
    k = 0
    index = 0
    while True:
        _, frame = video.read()
        if frame is None:
            break
        img = frame.copy()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if k == frames[index]:
            positions = pixel_poses[index]
            positions = np.transpose(positions)
            for p in positions:
                cy, cx = np.int32(p)
                cv2.rectangle(img, (cx - 10, cy - 20), (cx + 10, cy + 20), (255, 255, 255), thickness=2)
            index = index + 1
            cv2.imwrite('./imgs/{}.jpg'.format(k), img)
        cv2.imshow('video', img)
        cv2.waitKey(10)
        k = k + 1