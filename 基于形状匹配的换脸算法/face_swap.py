
import utils
from utils import show_image, read_image, detect_only_one_face, save_image, show_matrix
from utils import erode, dilate
import numpy as n
import os


FLOAT_TYPE = 'float32'


def torch_nn_model(x, y, max_epochs, target_loss, lr, saved_model=None):
    # x.shape = n,2
    # y.shape = n,2
    import torch.utils.data as Data
    import torch
    import torch.nn as nn

    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float()

    torch_dataset = Data.TensorDataset(x, y)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=x.shape[0],
        shuffle=True,
        drop_last=True
    )

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            hidden = 1024
            proba = 0.4
            self.backbone = nn.Sequential(
                nn.Linear(2, 2),        # 第一层为 线性旋转层, 用于将坐标向量对齐
                nn.Linear(2, hidden),   # 第二层为 非线性映射层
                nn.ReLU(),
                nn.Dropout(proba),      # dropout可以使得输出平滑
                nn.Linear(hidden, 2)
            )

        def forward(self, inp):
            return self.backbone(inp)

    lf = nn.MSELoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    g = Net()
    if saved_model is not None and os.path.exists(saved_model):
        g.load_state_dict(torch.load(saved_model))
    gopt = torch.optim.Adam(g.parameters(), lr=lr)
    g.to(device)

    best_loss = n.inf
    best_state_dict = g.state_dict()
    for ep in range(max_epochs):
        g.train()
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            gopt.zero_grad()
            predict = g(x)
            g_loss = lf(predict, y)
            g_loss.backward()
            gopt.step()

        if ep % 200 == 0:
            eval_losses = []
            for i, (x, y) in enumerate(loader):
                x = x.to(device)
                y = y.to(device)
                g.eval()
                predict = g(x)
                eval_loss = lf(predict, y)
                eval_losses.append(eval_loss.cpu().detach().numpy())
            eval_losses = n.mean(eval_losses)
            if eval_losses < best_loss:
                best_loss = eval_losses
                best_state_dict = g.state_dict()
            if eval_losses < target_loss:
                break

            print(
                'epoch: {:>5}/{:<5}    target loss:{:<5.2f}    epochs loss:{:<5.2f}'.format(
                    ep + 1, max_epochs, target_loss, eval_losses)
            )

    if saved_model is not None:
        torch.save(g.state_dict(), saved_model)

    class REG:
        def __init__(self, model, model_best_state_dict):
            self.model = model
            self.model.load_state_dict(model_best_state_dict)

        def predict(self, inps):
            device_ = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.eval()
            self.model.to(device_)
            inps = torch.tensor(inps, dtype=torch.float32).to(device_)
            outs = self.model(inps)
            outs = outs.cpu().detach().numpy()
            return outs
    return REG(g, best_state_dict)


def swap(im, seg, map_function, out_im, out_seg):
    in_h, in_w = im.shape[:2]
    in_xy_vec_int = []
    for j in range(in_h):
        for i in range(in_w):
            if seg[j, i] > 0:
                in_xy_vec_int.append([j, i])
    in_xy_vec_float = n.array(in_xy_vec_int, dtype=FLOAT_TYPE)

    out_xy_vec_float = map_function.predict(in_xy_vec_float)

    out_xy_vec_int = out_xy_vec_float.astype('int')

    out_im = out_im.copy()
    out_im_ = n.zeros_like(out_im)
    for k in range(len(in_xy_vec_int)):
        in_y, in_x = in_xy_vec_int[k]
        out_y, out_x = out_xy_vec_int[k]

        try:
            out_im[out_y, out_x] = im[in_y, in_x]
            out_im_[out_y, out_x] = im[in_y, in_x]
        except Exception as e:
            print('ERRPR: {}'.format(e))

    # out_im = blank_pad(out_im, out_im_, out_seg)
    zeros = n.array((0, 0, 0))
    seg3 = n.zeros_like(out_seg)
    for j in range(out_seg.shape[0]):
        for i in range(out_seg.shape[1]):
            if (out_im_[j, i] != zeros).all() or (out_seg[j, i] != zeros).all():
                seg3[j, i] = 1
    seg3 = erode(erode(dilate(dilate(seg3.astype('uint8'), 5), 5), 5), 5)
    seg3 = seg3.astype('int')

    out_im = blank_pad_v2(out_im, out_im_, seg3)
    # out_im_ = blank_pad_v2(out_im_, out_im_, seg3)
    # show_image(out_im)
    # show_image(seg3)
    # show_image(out_im_)

    print()
    print('Change Complete!')

    return out_im, seg3


def blank_pad_v1(real_ground_image, blank_ground_image, seg):
    h, w = real_ground_image.shape[:2]
    size = 2
    ret = real_ground_image.copy()
    zeros = n.array((0, 0, 0))
    for j in range(h):
        for i in range(w):
            seg_ji = seg[j, i]
            out_ji = blank_ground_image[j, i]
            if seg_ji > 0 and (out_ji == 0).all():
                patch = blank_ground_image[j - size: j + size + 1, i - size: i + size + 1, :]
                patch = patch.reshape(-1, 3)
                ps = []
                for p in patch:
                    if (p != zeros).any():
                        ps.append(p)
                if len(ps) == 0:
                    continue
                p = ps[n.random.randint(0, len(ps))]
                # p = n.mean(ps)
                ret[j, i] = p
    return ret


def f(patch):
    # patch = 3,3,3
    # patch = patch.astype(n.float)

    zeros = n.zeros((3,), dtype='uint8')

    patch_ = patch.reshape(-1, 3)
    ps = []
    for p in patch_:
        if (p != zeros).any():
            ps.append(p)
    if len(ps) == 0:
        return False, None

    ret = ps[n.random.randint(0, len(ps))].copy()

    [
        [p1, p2, p3],
        [p4, p5, p6],
        [p7, p8, p9]
    ] \
        = patch.tolist()

    # 垂直
    h = [p2, p8]
    w = [p4, p6]
    x1 = [p1, p9]
    x2 = [p3, p7]

    y = [h, w, x1, x2]

    for yi in y:
        o1, o3 = yi
        if n.not_equal(o1, zeros).any() and n.not_equal(o3, zeros).any():
            o1 = n.array(o1, dtype='float')
            o3 = n.array(o3, dtype='float')
            ret = (o1 + o3) / 2
            break
    ret = ret.astype('uint8')
    # print(p1, p2, p8)
    # exit()
    return True, ret


def blank_pad_v2(real_ground_image, blank_ground_image, seg):

    h, w = real_ground_image.shape[:2]
    ret = real_ground_image.copy()
    zeros = n.array((0, 0, 0))
    for j in range(2, h-2):
        for i in range(2, w-2):
            seg_ji = seg[j, i]
            out_ji = blank_ground_image[j, i]
            if seg_ji > 0 and (out_ji == 0).all():
                size = 1
                patch = blank_ground_image[j - size: j + size + 1, i - size: i + size + 1, :]
                flag, pix = f(patch)
                if flag:
                    ret[j, i] = pix
                else:
                    size = 2
                    patch = blank_ground_image[j - size: j + size + 1, i - size: i + size + 1, :]
                    patch = patch.reshape(-1, 3)
                    ps = []
                    for pix in patch:
                        if (pix != zeros).any():
                            ps.append(pix)
                    if len(ps) == 0:
                        continue
                    pix = ps[n.random.randint(0, len(ps))]
                    ret[j, i] = pix
    # show_image(ret)
    return ret


def regress(inputs, outputs, epochs, saved_model):
    # inputs.shape  = n, idims  .dtype = float
    # outputs.shape = n, odims  .dtype = float

    h_max, w_max = outputs.max(axis=0)
    h_min, w_min = outputs.min(axis=0)
    h_range = (h_max - h_min)
    w_range = (w_max - w_min)
    target_loss = n.min((h_range, w_range)) * 0.07
    print()
    print('Target Loss: {:.2f}'.format(target_loss))
    print()

    reg = torch_nn_model(
        inputs,
        outputs,
        max_epochs=epochs,
        target_loss=target_loss,
        lr=1e-3,
        saved_model=saved_model
    )
    return reg


def local_compute(im, anchors, seg):
    # anchors = [[y,x], [y,x], ...]

    # im_cut = im.copy()
    # seg_cut = seg.copy()

    # show_matrix(anchors)

    y, x = n.where(seg > 0)
    top, bottom, left, right = n.min(y), n.max(y), n.min(x), n.max(x)
    y_range = bottom - top
    x_range = right - left

    offset = int(0.2 * n.min((y_range, x_range)))
    # offset = 0

    top_ = top - offset
    bottom_ = bottom + offset
    left_ = left - offset
    right_ = right + offset

    anchors_cut = anchors.copy()
    anchors_cut[:, 0] -= top_
    anchors_cut[:, 1] -= left_

    im_cut = im[top_:bottom_, left_:right_].copy()
    seg_cut = seg[top_:bottom_, left_:right_].copy()

    # show_matrix(anchors_cut)

    # show_image(im_cut)
    return im_cut, anchors_cut, seg_cut, top_, left_


def local_compute_reverse(im_cut, seg_cut, im_src, seg_src, top, left):

    y_range, x_range = im_cut.shape[:2]

    im = im_src.copy()
    # show_matrix(im)
    # show_matrix(im[top:top+y_range, left:left+x_range], im_cut)
    im[top:top+y_range, left:left+x_range] = im_cut

    seg = seg_src.copy()
    seg[top:top + y_range, left:left + x_range] = seg_cut

    # show_image(im)
    return im, seg


def save_results(face_im, face_seg, base_im, base_seg, out_im, out_seg):
    save_image(face_im, 'results/face_im.png')
    save_image((face_seg * 255).astype('uint8'), 'results/face_seg.png')

    save_image(base_im, 'results/base_im.png')
    save_image((base_seg * 255).astype('uint8'), 'results/base_seg.png')

    save_image(out_im, 'results/out_im.png')
    save_image((out_seg * 255).astype('uint8'), 'results/out_seg.png')


def run(
        im1_src, anchors1_src, seg1_src,
        im2_src, anchors2_src, seg2_src,
        epochs,
        saved_model=None
        ):

    # im2_src, anchors2_src, seg2_src = read_data(base_im)

    im1_cut, anchors1_cut, seg1_cut, top1, left1 = local_compute(im1_src, anchors1_src, seg1_src)
    im2_cut, anchors2_cut, seg2_cut, top2, left2 = local_compute(im2_src, anchors2_src, seg2_src)
    # show_matrix(im1_cut, names='源剪切图像')
    # print()

    import time
    start = time.time()
    reg = regress(anchors1_cut, anchors2_cut, epochs, saved_model)
    end = time.time()
    print()
    print('TRAIN USE TIME: {:.2f}'.format(end - start))
    out_im, seg3 = swap(im1_cut, seg1_cut, reg, im2_cut, seg2_cut)
    out_im, seg3 = local_compute_reverse(out_im, seg3, im2_src, seg2_src, top2, left2)

    save_results(im1_src, seg1_src, im2_src, seg2_src, out_im, seg3)


def test():
    import read_face_data

    face_path = 'data/face/lyl.jpg'
    base_path = 'data/image/1.png'

    im1, anchors1, seg1 = \
        read_face_data.read_data(
            read_image(face_path),
            'data/shape_predictor_68_face_landmarks.dat')
    im2, anchors2, seg2 = \
        read_face_data.read_data(
            read_image(base_path),
            'data/shape_predictor_68_face_landmarks.dat')

    run(
        im1, anchors1, seg1,
        im2, anchors2, seg2,
        epochs=5000,
        saved_model=None
    )


if __name__ == '__main__':
    test()











































