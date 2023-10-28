import numpy as n
from target_tracking_utils import *

n.random.seed(0)


# 得分函数
def score(fea1, fea2):
    sco = 1 / n.linalg.norm(fea1 - fea2)
    return sco


# 目标追踪
def target_detect(frames, target_info):

    # 粒子数
    sample_nums = 100
    # RGB各通道特征向量维度
    his_range = 4
    
    # 目标RGB特征
    features = rgb_histogram(target_info['image'], range_=his_range, is_show=False).reshape(-1)

    range_y = target_info['range_y']
    range_x = target_info['range_x']

    mean_y = target_info['center_y']
    mean_x = target_info['center_x']
    std_y = target_info['range_y'] / 2
    std_x = target_info['range_x'] / 2

    result_ims = []
    record_yx = [{'y': mean_y, 'x': mean_x}]
    for i, frame in enumerate(frames):
        print('processing frame: {} / {}'.format(i+1, len(frames)))

        y_up_bound = frame.shape[0]
        x_up_bound = frame.shape[1]

        # 线性预测
        if len(record_yx) >= 2:
            offset_y = record_yx[-1]['y'] - record_yx[-2]['y']
            offset_x = record_yx[-1]['x'] - record_yx[-2]['x']
        else:
            offset_y = 0
            offset_x = 0
        
        # 采样
        sample_y = n.random.randn(sample_nums) * std_y + (mean_y + offset_y)
        sample_x = n.random.randn(sample_nums) * std_x + (mean_x + offset_x)
        samples_yx = n.concatenate((sample_y.reshape(-1, 1), sample_x.reshape(-1, 1)), axis=1)

        samples_scores = []
        for yx in samples_yx:
            
            y_low = yx[0] - 0.5*range_y if yx[0] - 0.5*range_y > 1 else 0
            y_up = yx[0] + 0.5*range_y if yx[0] + 0.5*range_y < y_up_bound - 1 else y_up_bound - 1

            x_low = yx[1] - 0.5*range_x if yx[1] - 0.5*range_x > 1 else 0
            x_up = yx[1] + 0.5*range_x if yx[1] + 0.5*range_x < x_up_bound - 1 else x_up_bound - 1

            im = frame[int(y_low):int(y_up), int(x_low):int(x_up)]

            # 粒子RGB特征
            samples_fea = rgb_histogram(im, range_=his_range).reshape(-1)

            # 度量得分
            sco = score(features, samples_fea)
            samples_scores.append(sco)

        # 根据得分计算融合权重
        samples_scores = n.array(samples_scores)**2
        weighs = samples_scores / samples_scores.sum()
        
        # 融合得到新的位置估计
        new_yx = n.sum(weighs.reshape(-1, 1) * samples_yx, axis=0)

        mean_y = new_yx[0]
        mean_x = new_yx[1]

        record_yx.append({'y': mean_y, 'x': mean_x})

        # 将结果绘制到该帧
        im = draw_on_frame(frame, (mean_y, mean_x), range_yx=(range_y, range_x))
        result_ims.append(im)
    return result_ims


# 从第一帧截取目标图像
def find_first_target(first):
    start_x, start_y = 320, 280
    range_x, range_y = 90, 120
    target_image = first[start_y:start_y+range_y, start_x:start_x+range_x]
    draw_image = draw_rec(first, (start_y, start_x), (start_y+range_y, start_x+range_x))
    # 目标信息字典
    target_info = {
        'image': target_image,
        'center_y': start_y + range_y / 2,
        'center_x': start_x + range_x / 2,
        'range_y': range_y,
        'range_x': range_x,
    }
    
    return target_info, draw_image


def run():
    video = Video('video.mp4')
    frames = video.read_video_list()

    # 目标图像
    target_info, draw_image = find_first_target(frames[0])
    # show_image(draw_image)
    
    # 检测
    ims = target_detect(frames[1:], target_info)

    # 制作gif
    make_gif(ims, 'target_tracking.gif', video.fps/2)


if __name__ == '__main__':
    run()






















