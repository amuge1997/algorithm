import utils
from utils import detect_only_one_face_by_face_recognition
import numpy as n


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
    anchors = faces_shapes
    return anchors


def read_data(im, dlib_shapes_predictor_path):
    JAW_POINTS, \
    RIGHT_BROW_POINTS, \
    LEFT_BROW_POINTS, \
    NOSE_POINTS, \
    RIGHT_EYE_POINTS, \
    LEFT_EYE_POINTS, \
    MOUTH_TOP_POINTS1, \
    MOUTH_BOTTOM_POINTS2 = utils.face_recognition_shapes_indexes()

    anchors = get_face_anchors(im)
    seg = get_face_mask(im, n.concatenate((anchors[:, 1:2], anchors[:, 0:1]), axis=1))

    return im, anchors, seg

















