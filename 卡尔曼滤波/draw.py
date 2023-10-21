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
    plt.plot(now_real_x[0, :], label='x real', c='black', linewidth=linewidth)
    plt.plot(now_z[0, :], label='x meansure', c='orange', linewidth=linewidth)
    plt.plot(predict_x[0, :], label='x predict', c='green', linewidth=linewidth)
    plt.plot(now_x[0, :], label='x fusion', c='blue', linewidth=linewidth)
    plt.legend()
    plt.grid()

    # 绘制速度
    plt.figure()
    plt.title('Velocity')
    plt.plot(now_real_x[1, :], label='v real', c='black', linewidth=linewidth)
    plt.plot(now_z[1, :], label='v meansure', c='orange', linewidth=linewidth)
    plt.plot(predict_x[1, :], label='v predict', c='green', linewidth=linewidth)
    plt.plot(now_x[1, :], label='v fusion', c='blue', linewidth=linewidth)
    plt.legend()
    plt.grid()

    plt.show()


run()









