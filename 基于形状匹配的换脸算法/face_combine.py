import utils as u
from utils import read_image, show_image, show_matrix, erode, lab_match, mask_1c_to_3c
import cv2 as cv
import numpy as n


def blur_mask(mask):
    y, x = n.where(mask > 0)
    top, bottom, left, right = n.min(y), n.max(y), n.min(x), n.max(x)
    y_range = bottom - top
    x_range = right - left
    size = int(n.min((y_range, x_range)) * 0.12)
    size = size if size % 2 != 0 else size + 1
    blur_size = int(size*1)
    blur_size = blur_size if blur_size % 2 != 0 else blur_size + 1

    mask = mask.copy()
    mask = erode(mask, size)
    mask = cv.GaussianBlur(mask, (blur_size, blur_size), size * 2)
    return mask


def combine(src_im, face_mask, dst_im):
    src_im = lab_match(
        (src_im*255).astype('uint8'),
        (face_mask*255).astype('int'),
        (dst_im*255).astype('uint8'),
        (face_mask*255).astype('int')
    ) / 255
    face_mask = blur_mask(face_mask)
    face_mask3 = mask_1c_to_3c(face_mask)
    im_combine = src_im * face_mask3 + dst_im * (1 - face_mask3)
    return im_combine


def run():
    combine_im = read_image('results/out_im.png') / 255
    combine_mask = read_image('results/out_seg.png') / 255

    dst_im = read_image('results/base_im.png') / 255
    im = combine(combine_im, combine_mask, dst_im)
    u.save_image(im, 'results/combine.png')
    return im


if __name__ == '__main__':
    run()

















