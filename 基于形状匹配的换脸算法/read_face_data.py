import utils
from utils import show_image, read_image, detect_only_one_face, save_image, show_matrix
from utils import erode, dilate, detect_only_one_face_by_face_recognition
from sklearn.metrics import mean_absolute_error
import dlib
import numpy as n
import os


def get_face_mask(im, landmarks):
    import numpy

    JAW_POINTS, \
    RIGHT_BROW_POINTS, \
    LEFT_BROW_POINTS, \
    NOSE_POINTS, \
    RIGHT_EYE_POINTS, \
    LEFT_EYE_POINTS, \
    MOUTH_TOP_POINTS1, \
    MOUTH_BOTTOM_POINTS2 = utils.face_recognition_shapes_indexes()

    OVERLAY_POINTS2 = [
        JAW_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS
    ]

    def draw_convex_hull(im_, points_, color_):
        import cv2
        points_ = cv2.convexHull(points_)
        cv2.fillConvexPoly(im_, points_, color=color_)

    im = numpy.zeros(im.shape[:2], dtype=numpy.float64)
    for group in OVERLAY_POINTS2:
        draw_convex_hull(im,
                         landmarks[group],
                         color_=1)
    im = numpy.array([im, im, im]).transpose((1, 2, 0))
    im = im[:, :, 0]
    im = numpy.where(im > 0, 1, 0)
    return im


def get_face_anchors(im):
    faces_pos, faces_shapes, image_draw = detect_only_one_face_by_face_recognition(im)
    # show_image(image_draw)
    # anchors = [(p.y, p.x) for p in faces_shapes[0]]
    # anchors = n.array(anchors)
    anchors = faces_shapes
    return anchors


def read_data(im, dlib_shapes_predictor_path):
    # JAW_POINTS = list(range(0, 17))
    # RIGHT_BROW_POINTS = list(range(17, 22))
    # LEFT_BROW_POINTS = list(range(22, 27))
    # NOSE_POINTS = list(range(27, 36))
    # RIGHT_EYE_POINTS = list(range(36, 42))
    # LEFT_EYE_POINTS = list(range(42, 48))
    # MOUTH_POINTS1 = list(range(48, 60))
    # MOUTH_POINTS2 = list(range(60, 68))
    JAW_POINTS, \
    RIGHT_BROW_POINTS, \
    LEFT_BROW_POINTS, \
    NOSE_POINTS, \
    RIGHT_EYE_POINTS, \
    LEFT_EYE_POINTS, \
    MOUTH_TOP_POINTS1, \
    MOUTH_BOTTOM_POINTS2 = utils.face_recognition_shapes_indexes()

    # select_anchors = \
    #     RIGHT_BROW_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS + LEFT_BROW_POINTS + RIGHT_BROW_POINTS + LEFT_BROW_POINTS + \
    #     RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS + \
    #     NOSE_POINTS + \
    #     MOUTH_TOP_POINTS1 + MOUTH_TOP_POINTS1 + MOUTH_BOTTOM_POINTS2 + MOUTH_BOTTOM_POINTS2 + \
    #     JAW_POINTS + JAW_POINTS + JAW_POINTS + JAW_POINTS + JAW_POINTS

    # 5*3 + 5*3 = 0: 30
    # 6*3 + 6*3 = 30: 66
    # 5         = 66: 71
    # 12*2 + 12*2   = 71: 119
    # 17*5      = 119: 204

    anchors = get_face_anchors(im)
    seg = get_face_mask(im, n.concatenate((anchors[:, 1:2], anchors[:, 0:1]), axis=1))
    # anchors = anchors[select_anchors, :]

    return im, anchors, seg

















