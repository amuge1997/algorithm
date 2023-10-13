import numpy as np

# 固定随机种子
np.random.seed(0)


# 卡尔曼滤波
def KF(x, z, A, P, Q, R, H):
    # A: 状态转移矩阵
    # P: 状态误差矩阵
    # Q: 转移过程误差矩阵

    # R: 观测误差矩阵
    # H: 观测矩阵

    # x: 上一刻状态
    # z: 该时刻观测
    rank = P.shape[0]

    x_predict = A @ x
    P_predict = A @ P @ A.T + Q

    K = P_predict @ H.T @ np.linalg.inv((H @ P_predict @ H.T + R))

    x_new = x_predict + K @ (z - H @ x_predict)
    P_new = ((np.identity(rank)) - K @ H) @ P_predict

    return x_new, P_new, x_predict


# 仿真环境
class Simulate:
    def __init__(self, x_init, A, Q, H, R):
        self.x_real = [x_init]
        self.z_real = []
        self.z_noise = []
        self.A = A
        self.Q = Q
        self.R = R
        self.H = H

    # 初始状态
    def x_start(self):
        x_this = self.x_real[-1]
        return x_this

    # 状态步进
    def x_step(self):
        x_this = self.x_real[-1]
        x_next_real = np.random.multivariate_normal((self.A @ x_this).reshape(-1), self.Q).reshape(-1, 1)
        self.x_real.append(x_next_real)
        return x_next_real

    # 观测步进
    def z_step(self):
        x_next = self.x_real[-1]
        z_next = self.H @ x_next
        z_next_noise = np.random.multivariate_normal(z_next.reshape(-1), self.R).reshape(-1, 1)
        
        self.z_real.append(z_next)
        self.z_noise.append(z_next_noise)
        return z_next_noise


def run():
    '''
        状态向量:
            x = [s, v].T
            s:位置
            v:速度
        其中初始位置0,初始速度1
        初始状态向量:
            x = [0, 1].T

        状态转移:
            A = [
                [1, t],
                [0, 1]
            ]
        其中 t=1
            A = [
                [1, 1],
                [0, 1]
            ]
    '''

    # 系统矩阵
    A = np.array([
        [1, 1],
        [0, 1]
    ])

    # 误差协方差
    P = np.array([
        [1.0**2, 0],
        [0, 1.0**2],
    ])

    # 过程噪声协方差
    Q = np.array([
        [0.1**2, 0],
        [0, 0.1**2]
    ])

    # 测量噪声协方差
    R = np.array([
        [0.5**2, 0],
        [0, 0.5**2],
    ])

    # 测量矩阵
    H = np.array([
        [1, 0],
        [0, 1],
    ])

    # 初始位置和速度
    x_real_init = np.array([
        [0.],
        [1.]
    ])
    
    # 仿真环境初始化
    sim = Simulate(x_real_init, A, Q, H, R)

    x_old = sim.x_start()
    A_old = A
    P_old = P

    z_new_noise_record = []
    x_new_record = []
    x_predict_record = []
    
    # 滤波
    for _ in range(20):
        sim.x_step()
        z_new_noise = sim.z_step()
        x_new, P_new, x_predict = KF(x_old, z_new_noise, A_old, P_old, Q, R, H)

        z_new_noise_record.append(z_new_noise)
        x_new_record.append(x_new)
        x_predict_record.append(x_predict)
        
        x_old = x_new
        P_old = P_new

    # 列表转向量
    z_new_noise_record = np.concatenate(z_new_noise_record, axis=1)
    x_new_record = np.concatenate(x_new_record, axis=1)
    x_predict_record = np.concatenate(x_predict_record, axis=1)

    import matplotlib.pyplot as plt
    # 绘制位置
    plt.figure()
    plt.plot(np.array(sim.x_real[1:])[:, 0, 0], label='x real', c='black')
    plt.plot(z_new_noise_record[0, :], label='x meansure', c='orange')
    plt.plot(x_predict_record[0, :], label='x predict', c='green')
    plt.plot(x_new_record[0, :], label='x fusion', c='blue')
    plt.legend()
    plt.grid()

    # 绘制速度
    plt.figure()
    plt.plot(np.array(sim.x_real[1:])[:, 1, 0], label='v real', c='black')
    plt.plot(z_new_noise_record[1, :], label='v meansure', c='orange')
    plt.plot(x_predict_record[1, :], label='v predict', c='green')
    plt.plot(x_new_record[1, :], label='v fusion', c='blue')
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == '__main__':
    run()

























