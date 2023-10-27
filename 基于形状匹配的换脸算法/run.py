from face_swap_and_combine_by_image import run
from utils import show_image
import utils as u


if __name__ == '__main__':
    face_path = r'./杨幂.jpg'

    base_path = r'./伊万卡.jpg'

    save_path = r'./融合结果.png'

    im = run(
        face_path=face_path,
        base_path=base_path,
        dlib_shapes_predictor_path='data/shape_predictor_68_face_landmarks.dat'
    )

    u.save_image(im, save_path)












