
import numpy as n
import cv2 as cv
from PIL import Image
import os


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

        pos = face_recognition.face_landmarks(image)[0]
        pos = face_pos_dict_to_array(pos)

        faces_shapes.append(pos)
        for i, p in enumerate(pos):
            image_draw = cv.circle(image_draw, (p[0], p[1]), 1, color=(0, 255, 0))

    faces_shapes = n.array([[p[1], p[0]] for p in faces_shapes[0]])

    return faces_pos, faces_shapes, image_draw


def show_image(im):
    # im.shape = h,w,3  .dtype = uint8 or float
    import matplotlib.pyplot as p
    p.figure()
    p.imshow(im)
    p.axis('off')
    p.show()


def read_image(path):
    im = Image.open(path)
    im = n.array(im)
    return im

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


def erode(im, size):
    # 腐蚀
    kernel = n.ones((size, size), n.uint8)
    return cv.erode(im, kernel)

def dilate(im, size):
    # 膨胀
    kernel = n.ones((size, size), n.uint8)
    return cv.dilate(im, kernel)

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


def mask_1c_to_3c(mask):
    # mask.shape = h,w
    return n.repeat(mask[..., n.newaxis], axis=2, repeats=3)


def save_image(im, path):
    im = im.copy()
    if im.dtype in [n.float, n.float32]:
        im *= 255
        im = n.uint8(im)
    im = Image.fromarray(im)
    im.save(path, quality=100)


def save_image(im, path):
    im = im.copy()
    if im.dtype in [n.float, n.float32]:
        im *= 255
        im = n.uint8(im)
    im = Image.fromarray(im)
    im.save(path, quality=100)


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













