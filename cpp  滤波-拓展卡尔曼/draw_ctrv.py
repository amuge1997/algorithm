import numpy as n


def read(name):
    with open(name) as fp:
        data = []
        lines = fp.readlines()
        for li in lines:
            di = [float(i)for i in li.split('\t')[:-1]]
            data.append(di)
        data = n.array(data).T
        return data


def run():
    now_x = read('now_x.txt')
    now_z = read('now_z.txt')
    now_real_x = read('now_real_x.txt')
    predict_x = read('predict_x.txt')
    
    linewidth = 1.8

    import matplotlib.pyplot as plt
    # 绘制位置
    plt.figure()
    plt.title('Position')
    plt.plot(now_real_x[0, :], now_real_x[1, :], linewidth=linewidth, label='real', c='black')
    plt.plot(now_z[0, :], now_z[1, :], linewidth=linewidth, label='meansure', c='orange')
    plt.plot(predict_x[0, :], predict_x[1, :], linewidth=linewidth, label='predict', c='green')
    plt.plot(now_x[0, :], now_x[1, :], linewidth=linewidth, label='fusion', c='blue')
    plt.legend()
    plt.grid()

    plt.figure()
    plt.title('X')
    plt.plot(now_real_x[0, :], linewidth=linewidth, label='real', c='black')
    plt.plot(now_z[0, :], linewidth=linewidth, label='meansure', c='orange')
    plt.plot(predict_x[0, :], linewidth=linewidth, label='predict', c='green')
    plt.plot(now_x[0, :], linewidth=linewidth, label='fusion', c='blue')
    plt.legend()
    plt.grid()

    plt.figure()
    plt.title('Y')
    plt.plot(now_real_x[1, :], linewidth=linewidth, label='real', c='black')
    plt.plot(now_z[1, :], linewidth=linewidth, label='meansure', c='orange')
    plt.plot(predict_x[1, :], linewidth=linewidth, label='predict', c='green')
    plt.plot(now_x[1, :], linewidth=linewidth, label='fusion', c='blue')
    plt.legend()
    plt.grid()

    plt.show()


run()









