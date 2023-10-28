from utils import read_image
import read_face_data
import face_swap
import face_combine


def run(face_path, base_path, dlib_shapes_predictor_path):
    # face_path = 'data/face/lyl.jpg'
    # base_path = 'data/image/1.png'

    im1, anchors1, seg1 = \
        read_face_data.read_data(
            read_image(face_path),
            dlib_shapes_predictor_path=dlib_shapes_predictor_path)
    im2, anchors2, seg2 = \
        read_face_data.read_data(
            read_image(base_path),
            dlib_shapes_predictor_path=dlib_shapes_predictor_path)

    face_swap.run(
        im1, anchors1, seg1,
        im2, anchors2, seg2,
        epochs=10000,
        saved_model=None
    )

    im = face_combine.run()

    return im


if __name__ == '__main__':
    run(
        face_path='data/face/lyl.jpg',
        base_path='data/image/1.png',
        dlib_shapes_predictor_path='data/shape_predictor_68_face_landmarks.dat'
    )

















