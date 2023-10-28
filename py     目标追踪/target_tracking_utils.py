import numpy as n
import cv2 as cv


# 视频类
class Video:
    def __init__(self, path):
        self.cap = cv.VideoCapture(path)

        self.width = self.cap.get(cv.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        self.fps = self.cap.get(cv.CAP_PROP_FPS)
        self.max_index = self.cap.get(cv.CAP_PROP_FRAME_COUNT)
        self.max_time = self.index_to_time(int(self.max_index))

    @staticmethod
    def change_bgr_to_rgb(bgr):
        return cv.cvtColor(bgr, cv.COLOR_BGR2RGB)

    def is_open(self):
        return self.cap.isOpened()

    def index_to_time(self, index):
        return index / self.fps

    def read_video_list(self):
        self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        ret = []
        while True:
            flag, frame = self.cap.read()
            if not flag:
                break
            ret.append(self.change_bgr_to_rgb(frame))
        return ret

    def release(self):
        self.cap.release()


# 制作gif
def make_gif(ims, save_path, fps):
    import imageio
    duration = 1 / fps
    imageio.mimsave(save_path, ims, 'GIF', duration=duration)


# 显示图像
def show_image(im):
    # im.shape = h,w,3  .dtype = uint8 or float
    if len(im.shape) not in (2, 3):
        raise Exception('图像形状错误 im.shape={}'.format(im.shape))
    if len(im.shape) == 3 and im.shape[-1] == 1:
        im = im[:, :, 0]
    import matplotlib.pyplot as p
    p.figure()
    p.imshow(im)
    p.axis('off')
    p.show()


# 统计rgb直方图
def rgb_histogram(im, range_, is_show=False):
    # 直方图
    # im.shape = h,w,3
    import matplotlib.pyplot as p
    r = cv.calcHist([im], [0], None, [range_], [0, 255])
    g = cv.calcHist([im], [1], None, [range_], [0, 255])
    b = cv.calcHist([im], [2], None, [range_], [0, 255])
    
    r = r.reshape(1, -1)
    g = g.reshape(1, -1)
    b = b.reshape(1, -1)

    rgb = n.concatenate((r, g, b), axis=0)[:, 1:]

    if is_show:
        colors = ['r', 'g', 'b']
        p.grid()
        for i in range(3):
            p.plot(rgb[i], colors[i])
        p.show()

    return rgb


# 绘制矩形
def draw_rec(frame, start_yx, end_yx):
    start_y, start_x = start_yx
    range_y, range_x = end_yx
    draw_image = cv.rectangle(frame, (start_x, start_y), (range_x, range_y), color=(255, 255, 255), thickness=2)
    return draw_image


# 帧中绘制矩形
def draw_on_frame(frame, yx, range_yx):
    y_up_bound = frame.shape[0]
    x_up_bound = frame.shape[1]
    range_y = range_yx[0]
    range_x = range_yx[1]
    start_y = yx[0] - 0.5*range_y if yx[0] - 0.5*range_y > 1 else 0
    end_y = yx[0] + 0.5*range_y if yx[0] + 0.5*range_y < y_up_bound - 1 else y_up_bound - 1

    start_x = yx[1] - 0.5*range_x if yx[1] - 0.5*range_x > 1 else 0
    end_x = yx[1] + 0.5*range_x if yx[1] + 0.5*range_x < x_up_bound - 1 else x_up_bound - 1

    im = draw_rec(frame, start_yx=(int(start_y), int(start_x)), end_yx=(int(end_y), int(end_x)))
    return im









