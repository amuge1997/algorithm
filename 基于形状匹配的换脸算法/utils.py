
import numpy as n
import cv2 as cv
from PIL import Image
import os


def read_video_one_frame_by_index(path, index):
    cap = cv.VideoCapture(path)
    cap.set(cv.CAP_PROP_POS_FRAMES, index)
    _, im = cap.read()
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    return im


def read_video_one_frame_by_index_rate(path, index_rate):
    cap = cv.VideoCapture(path)
    frame_nums = cap.get(cv.CAP_PROP_FRAME_COUNT)
    index = int(index_rate * frame_nums)
    im = read_video_one_frame_by_index(path, index)
    return im


def show_image(im):
    # im.shape = h,w,3  .dtype = uint8 or float
    import matplotlib.pyplot as p
    p.figure()
    p.imshow(im)
    p.axis('off')
    p.show()


def show_image_by_cv(im):
    cv.imshow('im', im)
    cv.waitKey()


def show_image4(x):
    # x.dtype = float or uint8
    # x.shape = n, s, s, 1 or 3
    import matplotlib.pyplot as p
    if x.dtype in [n.float64, n.float32, n.float16]:
        if x.min() < 1e-7:          # tanh
            x = (x + 1) / 2
    nums = 4
    for i in range(nums):
        xi = x[i]   # s,s,1 or 3
        if xi.shape[2] == 1:
            xi = n.squeeze(xi, 2)   # s,s
        p.subplot(2, 2, i+1)
        p.axis('off')
        p.imshow(xi)
    p.show()


def show_image9(x):
    # x.dtype = float or uint8
    # x.shape = n, s, s, 1 or 3
    import matplotlib.pyplot as p
    if x.dtype in [n.float64, n.float32, n.float16]:
        if x.min() < 1e-7:          # tanh
            x = (x + 1) / 2
    nums = 9
    for i in range(nums):
        xi = x[i]   # s,s,1 or 3
        if xi.shape[2] == 1:
            xi = n.squeeze(xi, 2)   # s,s
        p.subplot(3, 3, i+1)
        p.axis('off')
        p.imshow(xi)
    p.show()


def one_hot(x, n_class):
    # x.shape = n,
    x = n.eye(n_class, dtype='int')[x]
    return x


def draw_line(im, points):
    # im.shape = h,w,3  im.dtype = uint8
    for i in range(len(points) - 1):
        sx, sy = points[i]
        ex, ey = points[i+1]
        sx = int(sx)
        sy = int(sy)
        ex = int(ex)
        ey = int(ey)
        # print(sx)
        im = cv.line(im, (sx, sy), (ex, ey), (0, 0, 0), thickness=2)
    return im


def draw_circle(im, point, radius):
    x, y = point
    im = cv.circle(im, (x, y), radius, (0, 255, 0))
    return im


# def read_image_by_cv(path):
#     im = cv.imread(path)
#     # if len(im.shape) == 2:
#     #     im = n.repeat(im, axis=2, repeats=3)
#     im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
#     # im.shape = h,w,3  .dtype = uint8
#     return im


def read_image_by_cv(path):
    im = cv.imdecode(n.fromfile(path, dtype=n.uint8), -1)
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    return im


def read_image(path):
    im = Image.open(path)
    im = n.array(im)
    # print(im.dtype)
    # exit()
    # im.shape = h,w,3  .dtype = uint8
    return im


def save_image_by_cv(im, path):
    im = cv.cvtColor(im, cv.COLOR_RGB2BGR)
    cv.imwrite(path, im)


def save_image(im, path):
    im = im.copy()
    if im.dtype in [n.float, n.float32]:
        im *= 255
        im = n.uint8(im)
    im = Image.fromarray(im)
    im.save(path, quality=100)


def padding_image_to_same_height_width(im):
    # im.shape = h,w,3  im.dtype = uint8
    h, w, _ = im.shape
    if h > w:
        pad = h - w
        pad_half = int((h - w) / 2)

        pad_left = pad_half
        pad_righ = pad - pad_half

        im = n.pad(im, ((0, 0), (pad_left, pad_righ), (0, 0)))
        return im, (0, 0), (pad_left, pad_righ)
    elif w > h:
        pad = w - h
        pad_half = int((w - h) / 2)

        pad_top = pad_half
        pad_bottom = pad - pad_half

        im = n.pad(im, ((pad_top, pad_bottom), (0, 0), (0, 0)))
        return im, (pad_top, pad_bottom), (0, 0)
    else:
        return im, (0, 0), (0, 0)


def padding_image(im, left_pad, righ_pad, top_pad, bottom_pad):
    # im.shape = h,w,3
    # w_pad_tuple = (int, int)
    # h_pad_tuple = (int, int)
    if len(im.shape) == 3:
        im = n.pad(im, ((top_pad, bottom_pad), (left_pad, righ_pad), (0, 0)))
    elif len(im.shape) == 2:
        im = n.pad(im, ((top_pad, bottom_pad), (left_pad, righ_pad)))
    return im


def image_255_to_11(im):
    # im.dtype = uint8  .range = 0,255
    im = (im / 255) * 2 - 1
    im = im.astype('float')
    # im.dtype = float  .range = -1,1
    return im


def image_11_to_255(im):
    # im.dtype = float  .range = -1,1
    im = (im + 1) / 2 * 255
    im = im.astype('uint8')
    # im.dtype = uint8  .range = 0,255
    return im


def image_255_to_01(im):
    # im.dtype = uint8  .range = 0,255
    im = im / 255
    im = im.astype('float')
    # im.dtype = float  .range = 0,1
    return im


def image_01_to_255(im):
    # im.dtype = float  .range = 0,1
    im = im * 255
    im = im.astype('uint8')
    # im.dtype = uint8  .range = 0,255
    return im


def mse_loss(inp, tar):
    if inp.shape != tar.shape:
        raise Exception('形状不匹配')
    return n.mean((inp - tar) ** 2)


def mse_loss_torch(inp, tar):
    import torch
    loss = torch.mean(torch.pow(inp - tar, 2))
    print()
    print(n.round(inp.cpu().detach().numpy(), 3))
    print(n.round(tar.cpu().detach().numpy(), 3))
    print(n.round(loss.cpu().detach().numpy(), 3))
    return loss


def find_children(modules, indexes):
    new_modules = list(modules.children())[indexes[0]]
    if len(indexes[1:]) == 0:
        return new_modules
    else:
        return find_children(new_modules, indexes[1:])


def freeze_batch_norm_layers(model):
    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


def get_files_by_type_from_dir(dir_path, types):
    files_ = os.listdir(dir_path)
    files = []
    for f in files_:
        for t in types:
            if t in f:
                # print(t in f)
                files.append(f)
    return files


def read_images_from_dir(dir_path):
    types = ['jpg', 'jpeg', 'png']
    files = get_files_by_type_from_dir(dir_path, types)
    images = []
    for f in files:
        path = dir_path + '/' + f
        im = read_image(path)
        images.append(im)
    return images


def make_stack_images(images, row_and_col_size=5, picture_cell_size=256):
    ps = row_and_col_size
    cs = picture_cell_size

    nums = ps ** 2
    h = ps*cs
    w = h

    if len(images) > nums:
        raise Exception('图像数量过多')

    picture = n.zeros((h, w, 3), dtype='uint8')
    index = 0
    for im in images[:nums]:

        if len(im.shape) != 3 or im.shape[2] == 1:
            im = im[..., n.newaxis]
            im = n.repeat(im, axis=2, repeats=3)

        im, _, _ = padding_image(im)
        # print(im.shape)
        # exit()
        im = cv.resize(im, (cs, cs))
        y = index // ps
        x = index % ps
        # print(y, x)
        picture[y*cs: (y+1)*cs, x*cs: (x+1)*cs] = im

        index += 1

    # print(picture.shape)
    # cv.imshow('im', picture)
    # cv.waitKey()
    # cv.imwrite(output_path, picture)
    return picture


def read_labelme_data_json(path, shape_types):
    import json
    with open(path) as fp:
        js = json.load(fp)
        shapes = js['shapes']
        # print(shapes)
        ret = []
        for dc in shapes:
            st = dc['shape_type']
            if st in shape_types:
                ret.append({
                    'label': dc['label'],
                    'points': dc['points']
                })
        return ret


def labelme_json_to_dataset(json_file, out_dir):
    # 需要安装labelme
    # 该函数根据 Lib\site-packages\labelme\cli 下的 json_to_dataset.py 改写
    import argparse
    import base64
    import json
    import os
    import os.path as osp

    import imgviz
    import PIL.Image

    from labelme.logger import logger
    from labelme import utils
    logger.warning(
        "This script is aimed to demonstrate how to convert the "
        "JSON file to a single image dataset."
    )
    logger.warning(
        "It won't handle multiple JSON files to generate a "
        "real-use dataset."
    )

    # parser = argparse.ArgumentParser()
    # parser.add_argument("json_file")
    # parser.add_argument("-o", "--out", default=None)
    # args = parser.parse_args()

    if out_dir is None:
        out_dir = osp.basename(json_file).replace(".", "_")
        out_dir = osp.join(osp.dirname(json_file), out_dir)
    else:
        out_dir = out_dir
    if not osp.exists(out_dir):
        os.mkdir(out_dir)

    data = json.load(open(json_file))
    imageData = data.get("imageData")

    if not imageData:
        imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"])
        with open(imagePath, "rb") as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode("utf-8")
    img = utils.img_b64_to_arr(imageData)

    label_name_to_value = {"_background_": 0}
    for shape in sorted(data["shapes"], key=lambda x: x["label"]):
        label_name = shape["label"]
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
    lbl, _ = utils.shapes_to_label(
        img.shape, data["shapes"], label_name_to_value
    )

    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name

    lbl_viz = imgviz.label2rgb(
        label=lbl, img=imgviz.asgray(img), label_names=label_names, loc="rb"
    )

    PIL.Image.fromarray(img).save(osp.join(out_dir, "1.png"))
    utils.lblsave(osp.join(out_dir, "label.png"), lbl)
    PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, "label_viz.png"))

    with open(osp.join(out_dir, "label_names.txt"), "w") as f:
        for lbl_name in label_names:
            f.write(lbl_name + "\n")

    logger.info("Saved to: {}".format(out_dir))


def max_min_value_filter(image, ksize=3, mode=1):
    # 最大最小值滤波
    """
    :param image: 原始图像
    :param ksize: 卷积核大小
    :param mode:  最大值：1 或最小值：2
    :return:
    """
    import cv2
    img = image.copy()

    if len(img.shape) == 2:
        rows, cols = img.shape
    elif len(img.shape) == 3:
        rows, cols, channels = img.shape
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        raise Exception('维度错误!')
    padding = (ksize-1) // 2
    new_img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=255)
    for i in range(rows):
        for j in range(cols):
            roi_img = new_img[i:i+ksize, j:j+ksize].copy()
            min_val, max_val, min_index, max_index = cv2.minMaxLoc(roi_img)
            if mode == 1:
                img[i, j] = max_val
            elif mode == 2:
                img[i, j] = min_val
            else:
                raise Exception("please Select a Mode: max(1) or min(2)")
    return new_img


def dilate(im, size):
    # 膨胀
    kernel = n.ones((size, size), n.uint8)
    return cv.dilate(im, kernel)


def erode(im, size):
    # 腐蚀
    kernel = n.ones((size, size), n.uint8)
    return cv.erode(im, kernel)


def rgb_histogram(im, range_=255, is_show=False):
    # 直方图
    # im.shape = h,w,3
    import matplotlib.pyplot as p
    r = cv.calcHist([im], [0], None, [range_], [0, 255])
    g = cv.calcHist([im], [1], None, [range_], [0, 255])
    b = cv.calcHist([im], [2], None, [range_], [0, 255])

    r = r.reshape(1, -1)
    g = g.reshape(1, -1)
    b = b.reshape(1, -1)

    rgb = n.concatenate((r, g, b), axis=0)

    if is_show:
        colors = ['r', 'g', 'b']
        p.grid()
        for i in range(3):
            p.plot(rgb[i], colors[i])
        p.show()

    return rgb


# 将模型的参数转换为1维向量
def torch_state_dict_to_array(state_dict):
    # state_dict.type = dict    模型的state_dict 即net.state_dict()

    info = {
        'names': [],
        'shapes': [],
        'arrays': []
    }
    for name, value in state_dict.items():
        value = value.detach().numpy()
        info['names'].append(name)
        info['shapes'].append(value.shape)
        info['arrays'].append(value)

    vectors = []
    for v in info['arrays']:
        v = v.reshape(-1)
        vectors.append(v)
    vectors = n.concatenate(vectors, axis=0)

    # vectors.shape = m
    # info.type     = dict
    return vectors, info


# 将1维向量转换为torch模型的参数
def array_to_torch_state_dict(vectors, info):
    # vectors.shape = m
    # info.type     = dict

    import torch
    from functools import reduce
    names = info['names']
    shapes = info['shapes']
    nums = 0
    state_dict = {}
    for shape, name in zip(shapes, names):
        sha_ = reduce(lambda x, y: x * y, shape)
        state_dict[name] = torch.tensor(vectors[nums:nums + sha_].reshape(shape))

    # state_dict.type = dict
    return state_dict


def rgb_to_gray(im):
    # im.shape = h,w,3
    # im.dtype = uint8
    return n.mean(im.astype('float32'), axis=2).astype('uint8')


def resize_image(im, size):
    # im.shape = h,w,3
    # im.dtype = uint8
    im = cv.resize(im, size)
    return im


def detect_only_one_face(image, dlib_shape_path):
    # face_recognition  的特征点与 dlib 的特征点不相同
    # face_recognition  和 dlib 的 nose 分别是 5点 和 9点
    # face_recognition  的 mouth 分为 上唇和下唇
    # dlib              的 mouth 分为 外唇和内唇

    # image.shape = h,w,3
    import dlib
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dlib_shape_path)
    faces_pos_ = detector(image, 0)

    image_draw = image.copy()

    faces_pos = []
    faces_shapes = []
    if len(faces_pos_) > 1:
        raise Exception('人脸过多')
    elif len(faces_pos_) == 0:
        raise Exception('未检测到人脸')
    for face in faces_pos_:
        left = face.left()
        right = face.right()
        top = face.top()
        bottom = face.bottom()
        image_draw = cv.rectangle(image_draw, (left, top), (right, bottom), (0, 255, 0), 2)
        faces_pos.append(face)
        if dlib_shape_path is not None:
            pos = predictor(image, face).parts()
            faces_shapes.append(pos)
            for i, p in enumerate(pos):
                image_draw = cv.circle(image_draw, (p.x, p.y), 1, color=(0, 255, 0))

    faces_shapes = n.array([[p.y, p.x] for p in faces_shapes[0]])

    # faces_pos = [dlib.face, dlib.face, ...]
    # faces_shapes = [[y,x], [y,x]]
    # image_draw = array
    return faces_pos, faces_shapes, image_draw


def face_recognition_shapes_indexes():
    JAW_POINTS = list(range(0, 17))             # 下颚
    RIGHT_BROW_POINTS = list(range(17, 22))     # 右眉毛
    LEFT_BROW_POINTS = list(range(22, 27))      # 左眉毛
    NOSE_POINTS = list(range(27, 32))           # 鼻
    RIGHT_EYE_POINTS = list(range(32, 38))      # 右眼
    LEFT_EYE_POINTS = list(range(38, 44))       # 左眼
    MOUTH_TOP_POINTS1 = list(range(44, 56))          # 嘴
    MOUTH_BOTTOM_POINTS2 = list(range(56, 68))          # 嘴2
    return \
        JAW_POINTS, \
        RIGHT_BROW_POINTS, \
        LEFT_BROW_POINTS, \
        NOSE_POINTS, \
        RIGHT_EYE_POINTS, \
        LEFT_EYE_POINTS, \
        MOUTH_TOP_POINTS1, \
        MOUTH_BOTTOM_POINTS2


def detect_only_one_face_by_face_recognition(image):
    # face_recognition  的特征点与 dlib 的特征点不相同
    # face_recognition  和 dlib 的 nose 分别是 5点 和 9点
    # face_recognition  的 mouth 分为 上唇和下唇
    # dlib              的 mouth 分为 外唇和内唇

    # image.shape = h,w,3

    import face_recognition
    faces_pos_ = face_recognition.face_locations(image)

    if len(faces_pos_) > 1:
        raise Exception('人脸过多')
    elif len(faces_pos_) == 0:
        raise Exception('未检测到人脸')

    image_draw = image.copy()

    def face_pos_dict_to_array(pos_):
        JAW_POINTS = pos_['chin']  # 下颚
        RIGHT_BROW_POINTS = pos_['right_eyebrow']  # 右眉毛
        LEFT_BROW_POINTS = pos_['left_eyebrow']  # 左眉毛
        NOSE_POINTS = pos_['nose_tip']  # 鼻
        RIGHT_EYE_POINTS = pos_['right_eye']  # 右眼
        LEFT_EYE_POINTS = pos_['left_eye']  # 左眼
        MOUTH_TOP_POINTS1 = pos_['top_lip']  # 嘴
        MOUTH_BOTTOM_POINTS2 = pos_['bottom_lip']  # 嘴
        # show_matrix(
        #     n.array(JAW_POINTS),
        #     n.array(RIGHT_BROW_POINTS),
        #     n.array(LEFT_BROW_POINTS),
        #     n.array(NOSE_POINTS),
        #     n.array(RIGHT_EYE_POINTS),
        #     n.array(LEFT_EYE_POINTS),
        #     n.array(MOUTH_TOP_POINTS1),
        #     n.array(MOUTH_BOTTOM_POINTS2),
        #     names=[
        #         'JAW_POINTS',
        #         'RIGHT_BROW_POINTS',
        #         'LEFT_BROW_POINTS',
        #         'NOSE_POINTS',
        #         'RIGHT_EYE_POINTS',
        #         'LEFT_EYE_POINTS',
        #         'MOUTH_TOP_POINTS1',
        #         'MOUTH_BOTTOM_POINTS2'
        #     ]
        # )
        pos_ = n.concatenate(
            (
                JAW_POINTS,
                RIGHT_BROW_POINTS,
                LEFT_BROW_POINTS,
                NOSE_POINTS,
                RIGHT_EYE_POINTS,
                LEFT_EYE_POINTS,
                MOUTH_TOP_POINTS1,
                MOUTH_BOTTOM_POINTS2
            ),
            axis=0
        )
        return pos_

    faces_pos = []
    faces_shapes = []
    for face in faces_pos_:
        top = face[0]
        right = face[1]
        bottom = face[2]
        left = face[3]
        image_draw = cv.rectangle(image_draw, (left, top), (right, bottom), (0, 255, 0), 2)
        faces_pos.append(face)

        # pos = predictor(image, face).parts()
        pos = face_recognition.face_landmarks(image)[0]
        pos = face_pos_dict_to_array(pos)

        faces_shapes.append(pos)
        for i, p in enumerate(pos):
            image_draw = cv.circle(image_draw, (p[0], p[1]), 1, color=(0, 255, 0))

    faces_shapes = n.array([[p[1], p[0]] for p in faces_shapes[0]])

    # faces_pos = [dlib.face, dlib.face, ...]
    # faces_shapes = [[y,x], [y,x]]
    # image_draw = array
    return faces_pos, faces_shapes, image_draw


def find_mask_bound(mask):
    # mask.shape = h,w
    # 获取mask边界, mask为uint8类型
    h, w = mask.shape[:2]

    bound = n.zeros(mask.shape, dtype='uint8')
    for j in range(h):
        for i in range(w):
            # print(mask.shape)
            top = j-1 if j-1 >= 0 else 0
            left = i-1 if i-1 >= 0 else 0
            bottom = j+2 if j+2 <= h else h
            right = i+2 if i+2 <= w else w
            patch = mask[top:bottom, left:right]
            s = (bottom - top) * (right - left)
            patch = n.sum(patch)
            if patch != s and patch != 0 and mask[j, i] != 0:
                bound[j, i] = 1
    return bound


def find_mask_mul_bound(mask, reps):
    # mask.shape = h,w
    # reps.dtype = int
    # 获取mask多层边界, mask为uint8类型, reps为层数
    bounds = []
    mask = mask.copy()
    bs = n.zeros_like(mask, dtype='float')
    for i in range(reps):
        print('find_mask_mul_bound {}/{}'.format(i+1, reps))
        bound = find_mask_bound(mask)
        bounds.append(bound)
        bs += bound*1.
        mask[bound != 0] = 0
    bs /= 255
    # show_image(bs)
    return bounds


def show_matrix_v1(x, name='array', new_line=True):
    if new_line:
        print()
    print(name, x.shape, x.dtype, x.min(), x.max())


def show_matrix(*xs, names=None, new_line=True, show_detail=False):
    if names is None:
        names = ['array'] * len(xs)
    elif type(names) == str:
        names = [names] * len(xs)
    for x, name in zip(xs, names):
        if new_line:
            print()
        print('\033[36mname: \033[33m{}'.format(name))
        print('\033[36m             shape\033[0m: {}'.format(x.shape))
        print('\033[36m             dtype\033[0m: \033[31m{}'.format(x.dtype))
        print('\033[36m             min  \033[0m: \033[34m{}'.format(x.min()))
        print('\033[36m             max  \033[0m: \033[34m{}\033[0m'.format(x.max()))
        if show_detail:
            print('\033[36m             mea  \033[0m: \033[34m{}'.format(x.mean()))
            print('\033[36m             std  \033[0m: \033[34m{}\033[0m'.format(x.std()))


class RGBNumericalCompute:
    # 图像 RGB统计 的数值计算
    def __init__(self):
        pass

    @staticmethod
    def d_sigmoid(x):
        import torch
        return torch.sigmoid(x) * (1 - torch.sigmoid(x))

    @staticmethod
    def d_sigmoid_offset_rate(x, offset, rate):
        return RGBNumericalCompute.d_sigmoid((x - offset) * rate)

    @staticmethod
    def check_inputs(im, mask, allow_flow):
        import torch
        if \
                im.dtype not in [torch.float32, torch.float64] or \
                mask.dtype not in [torch.int32, torch.int64, torch.uint8]:
            raise Exception('数据类型错误')
        if len(im.shape) != 3 or len(mask.shape) != 2:
            # print(im.shape, mask.shape)
            raise Exception('数据形状错误')
        if (torch.sum(mask == 0) + torch.sum(mask == 1)) != mask.shape[0] * mask.shape[1]:
            raise Exception('数值错误')
        if ((im < 0.).any() or (im > 1.).any()) and not allow_flow:
            raise Exception('数值错误')

    @staticmethod
    def compute_rate(r, mask):
        import torch
        rate = 5

        mask_zeros = mask > 0
        r_pix_nums = torch.sum(mask_zeros)
        offset = torch.arange(0, 256, 1) / 255

        ret = torch.zeros(offset.shape[0], dtype=torch.float32, device=r.device)
        r = r[mask_zeros]
        for i, o in enumerate(offset):
            r_offset = RGBNumericalCompute.d_sigmoid_offset_rate(r, o, rate)
            temp = torch.sum(r_offset)
            ret[i] = temp / r_pix_nums

        return ret


def rgb_numerical_compute_by_torch(im, mask, allow_flow):
    import torch
    # im.shape = h, w, 3
    # im.dtype = float
    # im.range = 0,1

    # mask.shape = h, w
    # mask.dtype = int
    # mask.range = 0,1

    if mask is None:
        mask = torch.ones(im.shape[:2], dtype=torch.int32, device=im.device)

    RGBNumericalCompute.check_inputs(im, mask, allow_flow)

    r = RGBNumericalCompute.compute_rate(im[:, :, 0], mask).reshape(1, -1)
    g = RGBNumericalCompute.compute_rate(im[:, :, 1], mask).reshape(1, -1)
    b = RGBNumericalCompute.compute_rate(im[:, :, 2], mask).reshape(1, -1)

    # import matplotlib.pyplot as p
    # p.grid()
    # p.plot(r.cpu().detach().numpy()[0], c='r')
    # p.plot(g.cpu().detach().numpy()[0], c='g')
    # p.plot(b.cpu().detach().numpy()[0], c='b')
    # p.show()

    rgb = torch.cat((r, g, b), dim=0)

    # rgb.shape = 3, 256
    # rgb.dtype = float32
    return rgb


def draw_mask_by_points(im_shape, points):
    # mask 的 类型统一为 uint8 或 int32 或 int64

    # im_shape.shape = 2,  ->  h,w
    # points.shape = n,2  ->  [[y,x],[y,x],...]
    # points.dtype = int

    import cv2, numpy
    points = n.concatenate((points[:, 1:2], points[:, 0:1]), axis=1)

    def draw_convex_hull(im_, points_, color_):
        points_ = cv2.convexHull(points_)
        cv2.fillConvexPoly(im_, points_, color=color_)

    mask = numpy.zeros(im_shape, dtype=numpy.float64)
    draw_convex_hull(mask, points, color_=1)
    mask = numpy.where(mask > 0, 1, 0)
    mask = mask.astype('int32')

    # mask.dtype = int32
    return mask


def draw_mask_by_points_on_image(im, points, mask_color=0):
    # mask 的 类型统一为 uint8 或 int32 或 int64

    # im.shape = h,w,3
    # im.dtype = uint8

    # points.shape = n,2  ->  [[h,w],[h,w],...]
    # points.dtype = int

    import numpy
    mask = draw_mask_by_points(im.shape[:2], points)
    mask = numpy.array([mask, mask, mask], dtype='int32').transpose((1, 2, 0))
    im[mask != 0] = mask_color
    return im


def draw_mask_on_image(im, masks, mask_color=0):
    # mask 的 类型统一为 uint8 或 int32 或 int64

    # im.shape = h,w,3
    # im.dtype = uint8

    # masks = [mask1, mask2, ...]

    import numpy
    im = im.copy()
    for mask in masks:
        mask = numpy.array([mask, mask, mask], dtype='int32').transpose((1, 2, 0))
        im[mask != 0] = mask_color
    return im


# def dlib_shapes_indexes():
#     # FACE_POINTS = list(range(17, 68))
#     MOUTH_POINTS = list(range(48, 61))          # 嘴
#     RIGHT_BROW_POINTS = list(range(17, 22))     # 右眉毛
#     LEFT_BROW_POINTS = list(range(22, 27))      # 左眉毛
#     RIGHT_EYE_POINTS = list(range(36, 42))      # 右眼
#     LEFT_EYE_POINTS = list(range(42, 48))       # 左眼
#     NOSE_POINTS = list(range(27, 35))           # 鼻
#     JAW_POINTS = list(range(0, 17))             # 下颚
#     return MOUTH_POINTS, \
#            RIGHT_BROW_POINTS, \
#            LEFT_BROW_POINTS, \
#            RIGHT_EYE_POINTS, \
#            LEFT_EYE_POINTS, \
#            NOSE_POINTS, \
#            JAW_POINTS

def dlib_shapes_indexes():
    JAW_POINTS = list(range(0, 17))             # 下颚
    RIGHT_BROW_POINTS = list(range(17, 22))     # 右眉毛
    LEFT_BROW_POINTS = list(range(22, 27))      # 左眉毛
    NOSE_POINTS = list(range(27, 36))           # 鼻
    RIGHT_EYE_POINTS = list(range(36, 42))      # 右眼
    LEFT_EYE_POINTS = list(range(42, 48))       # 左眼
    MOUTH_POINTS1 = list(range(48, 60))          # 嘴
    MOUTH_POINTS2 = list(range(60, 68))          # 嘴2
    return MOUTH_POINTS1, MOUTH_POINTS2, \
           RIGHT_BROW_POINTS, \
           LEFT_BROW_POINTS, \
           RIGHT_EYE_POINTS, \
           LEFT_EYE_POINTS, \
           NOSE_POINTS, \
           JAW_POINTS


def plot_line(*points, colors=None, title=None, labels=None, is_show=False):
    # points = [pt, pt, ...]
    # pt.shape = n, or n,1 or n,2
    # pt = [y, y, ...] or [[y], [y], ...] or [[y,x], [y,x], ...]
    import matplotlib.pyplot as p

    p.figure()

    if colors is None:
        colors = ['b'] * len(points)
    if isinstance(colors, str):
        colors = [colors] * len(points)
    if isinstance(title, str):
        p.title(title)
    if labels is None:
        labels_ = [None] * len(points)
    else:
        labels_ = labels
    p.grid()
    for pt, co, la in zip(points, colors, labels_):
        nums = pt.shape[0]
        if pt.shape == (nums,):
            p.plot(pt, c=co, label=la)
        if pt.shape == (nums, 1):
            p.plot(pt[:, 0], c=co, label=la)
        if pt.shape == (nums, 2):
            p.plot(pt[:, 1], pt[:, 0], c=co, label=la)
    if labels is not None:
        p.legend()
    if is_show:
        p.show()


def plot_show():
    import matplotlib.pyplot as p
    p.show()


def mask_1c_to_3c(mask):
    # mask.shape = h,w
    return n.repeat(mask[..., n.newaxis], axis=2, repeats=3)


def mask_1c_to_3c_torch(mask):
    # mask.shape = h,w
    import torch
    return torch.repeat_interleave(mask.unsqueeze(2), repeats=3, dim=2)


# 色彩迁移
def histogram_match(src_im, src_mask, dst_im, dst_mask):
    # src_im.shape = h,w,3
    # src_im.dtype = uint8

    # src_mask.shape = h,w
    # src_mask.dtype = int

    # dst_im.shape = h,w,3
    # dst_im.dtype = uint8

    # dst_mask.shape = h,w
    # dst_mask.dtype = int

    def match(src_im_, src_mask_pix_nums_, dst_im_, dst_mask_pix_nums_):
        # print(src_mask_pix_nums_, dst_mask_pix_nums_)
        # src_im.shape = h,w,3
        img = src_im_
        imgRef = dst_im_

        _, _, channel = img.shape
        imgOut = n.zeros_like(img)
        for i in range(channel):
            histImg, _ = n.histogram(img[:, :, i], 256)  # 计算原始图像直方图
            # print(histImg)
            histImg = histImg[1:]
            histImg = histImg / src_mask_pix_nums_
            histRef, _ = n.histogram(imgRef[:, :, i], 256)  # 计算匹配模板直方图
            histRef = histRef[1:]
            histRef = histRef / dst_mask_pix_nums_
            # exit()
            # plot_line(histImg[1:], histRef[1:], colors=['r', 'b'])
            # plot_show()
            cdfImg = n.cumsum(histImg)  # 计算原始图像累积分布函数 CDF
            # print(cdfImg)
            cdfRef = n.cumsum(histRef)  # 计算匹配模板累积分布函数 CDF
            for j in range(256-1):
                tmp = abs(cdfImg[j] - cdfRef)
                tmp = tmp.tolist()
                index = tmp.index(min(tmp))  # find the smallest number in tmp, get the index of this number
                # print(index)
                imgOut[:, :, i][img[:, :, i] == j+1] = index
        return imgOut

    def get_im_with_mask(im_, mask_):
        ret_ = im_.copy()
        mask3 = mask_1c_to_3c(mask_)
        ret_[mask3 == 0] = 0
        return ret_

    src_im_mask = get_im_with_mask(src_im, src_mask)
    dst_im_mask = get_im_with_mask(dst_im, dst_mask)

    src_mask_pix_nums = n.sum(src_mask != 0)
    dst_mask_pix_nums = n.sum(dst_mask != 0)
    ret = match(src_im_mask, src_mask_pix_nums, dst_im_mask, dst_mask_pix_nums)

    return ret


# LAB色彩迁移
def lab_match(src_im, src_mask, dst_im, dst_mask):
    # src_im.shape = h,w,3
    # src_im.dtype = uint8

    # src_mask.shape = h,w
    # src_mask.dtype = int

    # dst_im.shape = h,w,3
    # dst_im.dtype = uint8

    # dst_mask.shape = h,w
    # dst_mask.dtype = int

    dst_lab = cv.cvtColor(dst_im, cv.COLOR_RGB2Lab)
    src_lab = cv.cvtColor(src_im, cv.COLOR_RGB2Lab)

    dst_lab = dst_lab.astype('float')
    src_lab = src_lab.astype('float')

    dst_lab_mean = n.mean(dst_lab[dst_mask != 0], axis=0)
    dst_lab_std = n.std(dst_lab[dst_mask != 0], axis=0)

    src_lab_mean = n.mean(src_lab[src_mask != 0], axis=0)
    src_lab_std = n.std(src_lab[src_mask != 0], axis=0)

    combine_lab = src_lab.copy()

    combine_lab[src_mask != 0] = \
        (combine_lab[src_mask != 0] - src_lab_mean) / src_lab_std * dst_lab_std + dst_lab_mean

    combine_lab[combine_lab < 0.] = 0.
    combine_lab[combine_lab > 255.] = 255.

    combine_lab = combine_lab.astype('uint8')

    ret_im = cv.cvtColor(combine_lab, cv.COLOR_Lab2RGB)
    return ret_im


def tensor_to_numpy(tensor):
    return tensor.cpu().detach().numpy()


def image_tensor_rgb_to_gray(tensor):
    # tensor.shape = h,w,3
    import torch
    return torch.mean(tensor, dim=2, keepdim=True)


def image_tensor_shape_to_conv_tensor_shape(im_tensor):
    # im_tensor.shape = h,w,c
    import torch
    im_tensor = torch.unsqueeze(im_tensor, 0)
    return im_tensor.permute(0, 3, 1, 2)


def conv_tensor_shape_to_image_tensor_shape(conv_tensor):
    # conv_tensor.shape = n, c, h, w
    return conv_tensor.permute(0, 2, 3, 1)


def write_xpath(*lis):
    # demo: print(write_xpath('//', 'div', 'id', 'yes', 1, 'div', None, None, 1))

    ret = ''
    index = 0
    while index < len(lis):
        if lis[index] == '//':
            ret += '//'
            index += 1
        else:
            condition0 = lis[index]
            condition1 = lis[index + 1]
            condition2 = lis[index + 2]
            condition3 = lis[index + 3]
            condition = '{}'.format(condition0)
            if condition1 is not None:
                condition += '[@{}="{}"]'.format(condition1, condition2)
            if condition3 is not None:
                condition += '[position()={}]'.format(condition3)
            if index + 4 < len(lis) and lis[index + 4] != '//':
                condition += '/'

            ret += condition
            index += 4
    return ret


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
        # show_matrix(bgr)
        return cv.cvtColor(bgr, cv.COLOR_BGR2RGB)

    def get_frame_by_index(self, index):
        if index < self.max_index:
            self.cap.set(cv.CAP_PROP_POS_FRAMES, index)
            _, frame = self.cap.read()
            frame = self.change_bgr_to_rgb(frame)
            # show_image(frame)
            return frame
        else:
            raise Exception('索引超出视频长度')

    def is_open(self):
        return self.cap.isOpened()

    # def set_start_time(self, second):
    #     index = int(second * self.fps)
    #     self.set_start_index(index)
    #
    # def set_start_index(self, index):
    #     self.cap.set(cv.CAP_PROP_POS_FRAMES, index)

    def time_to_index(self, second):
        return int(second * self.fps)

    def index_to_time(self, index):
        return index / self.fps

    def read_video_list_by_range_time(self, start_time, end_time=None):
        start_index = self.time_to_index(start_time)
        if end_time is None:
            end_time = self.max_time
        end_index = self.time_to_index(end_time)
        range_index = end_index - start_index
        if start_time < 0 or end_index > self.max_index or range_index < 0:
            raise Exception('时间范围错误')
        self.cap.set(cv.CAP_PROP_POS_FRAMES, start_index)
        ret = []
        for i in range(range_index):
            flag, frame = self.cap.read()
            if not flag:
                break
            ret.append(self.change_bgr_to_rgb(frame))
        return ret

    def read_video_list_by_range_time_by_generate(self, start_time, end_time=None):
        start_index = self.time_to_index(start_time)
        if end_time is None:
            end_time = self.max_time
        end_index = self.time_to_index(end_time)
        range_index = end_index - start_index
        if start_time < 0 or end_index > self.max_index or range_index < 0:
            raise Exception('时间范围错误')
        self.cap.set(cv.CAP_PROP_POS_FRAMES, start_index)
        # ret = []
        for i in range(range_index):
            flag, frame = self.cap.read()
            if not flag:
                break
            yield self.change_bgr_to_rgb(frame)
            # ret.append()
        # return ret

    def read_video_list(self):
        self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        ret = []
        while True:
            flag, frame = self.cap.read()
            if not flag:
                break
            ret.append(self.change_bgr_to_rgb(frame))
        return ret

    def read_video(self):
        self.cap.set(cv.CAP_PROP_POS_FRAMES, 0)
        while True:
            flag, frame = self.cap.read()
            if not flag:
                break
            yield self.change_bgr_to_rgb(frame)


def rotate_image_by_cut(image, angle, center=None, scale=1.0):  # 1
    # 可能会截断图像
    # angle 角度制
    (h, w) = image.shape[:2]  # 2
    if center is None:  # 3
        center = (w // 2, h // 2)  # 4

    M = cv.getRotationMatrix2D(center, angle, scale)  # 5

    rotated = cv.warpAffine(image, M, (w, h))  # 6
    return rotated


def rotate_image_0_90_180_270_by_all(im):
    # 非截断图像
    # im.shape = h,w,3  .dtype=uint8
    h, w, _ = im.shape

    im0 = im.copy()

    im180 = im[::-1, ::-1, :].copy()

    im90 = n.transpose(im, (1, 0, 2)).copy()

    im90 = im90[::-1, :, :].copy()

    im270 = im90[::-1, ::-1, :].copy()

    return im0, im90, im180, im270


def make_gif(ims, save_path, fps):
    import imageio
    duration = 1 / fps
    imageio.mimsave(save_path, ims, 'GIF', duration=duration)


def make_gif_from_dir(im_dir, save_path, fps, rotate=0):
    import os
    from PIL import Image
    print()
    print('图像以整数数字作为序号!')
    files = os.listdir(im_dir)
    # files_number = [int(f.split('.')[0]) for f in files]
    files = sorted(files, key=lambda x: int(x.split('.')[0]))
    ims = []
    for f in files:
        path = im_dir + '/' + f
        im = Image.open(path)
        if rotate != 0:
            im = im.rotate(rotate)
        ims.append(im)
    make_gif(ims, save_path, fps)


def make_mp4(ims, save_path, fps):
    h, w = ims[0].shape[:2]
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    video = cv.VideoWriter(save_path, fourcc, fps, (w, h))
    for im in ims:
        im = cv.cvtColor(im, cv.COLOR_RGB2BGR)
        video.write(im)


def make_mp4_from_dir(im_dir, save_path, fps, rotate=0, select_index_callback=None):
    import os
    print()
    print('图像以整数数字作为序号!')
    files = os.listdir(im_dir)
    # files_number = [int(f.split('.')[0]) for f in files]
    files = sorted(files, key=lambda x: int(x.split('.')[0]))
    ims = []
    for f in files:
        if select_index_callback is not None:
            index = int(f.split('.')[0])
            flag = select_index_callback(index)
            if not flag:
                continue
        path = im_dir + '/' + f
        im = read_image(path)
        if rotate != 0:
            im0, im90, im180, im270 = rotate_image_0_90_180_270_by_all(im)
            if rotate == 90:
                im = im90
            elif rotate == 180:
                im = im180
            elif rotate == 270:
                im = im270
            else:
                raise Exception('角度必须为 0 90 180 270 之一')
        ims.append(im)
    make_mp4(ims, save_path, fps)


def encode_face(im):
    import face_recognition
    encodes = face_recognition.face_encodings(im)
    return encodes


def encode_only_one_face(im):
    import face_recognition
    encodes = face_recognition.face_encodings(im)
    if len(encodes) == 0:
        raise Exception('未检测到人脸')
    elif len(encodes) > 1:
        raise Exception('检测到多个人脸')
    return encodes[0]


def compute_distance_by_two_face_image(face_im1, face_im2):
    # 距离低于 0.5~0.6 可认为相似
    import face_recognition
    e1 = encode_only_one_face(face_im1)
    e2 = encode_only_one_face(face_im2)
    dis = face_recognition.face_distance([e1], e2)
    return dis[0]


def compute_distance_by_two_face_encode(encode1, encode2):
    # 距离低于 0.5~0.6 可认为相似
    import face_recognition
    dis = face_recognition.face_distance([encode1], encode2)[0]
    return dis


def compute_is_same_person_by_two_face_image(face_im1, face_im2):
    import face_recognition
    e1 = encode_only_one_face(face_im1)
    e2 = encode_only_one_face(face_im2)
    boolean = face_recognition.compare_faces([e1], e2)[0]
    return boolean


def test():
    pass


if __name__ == '__main__':
    test()




