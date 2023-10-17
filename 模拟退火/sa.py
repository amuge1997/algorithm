import numpy as n
import random
import matplotlib.pyplot as plt

n.random.seed(0)


# 初始解（初始城市排列）
def init_solation(num_cities):
    return random.sample(range(num_cities), num_cities)


# 计算解的总距离
def sum_distance(solution, dis_matrix):
    total_dist = 0
    num_cities = len(solution)
    for i in range(num_cities - 1):
        total_dist += dis_matrix[solution[i]][solution[i+1]]
    total_dist += dis_matrix[solution[-1]][solution[0]]
    return total_dist


# 模拟退火算法
def sa(dis_matrix, init_temp, rate, epochs):
    num_cities = len(dis_matrix)
    current_solution = init_solation(num_cities)
    current_dist = sum_distance(current_solution, dis_matrix)
    best_solution = current_solution
    best_dist = current_dist

    for _ in range(epochs):
        # 生成新解
        new_solution = current_solution.copy()
        i, j = random.sample(range(num_cities), 2)
        new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
        new_dist = sum_distance(new_solution, dis_matrix)

        # 计算温度
        temp = init_temp * rate**epochs

        # 接受新解
        if new_dist < current_dist or random.random() < n.exp(- (new_dist - current_dist) / temp):
            current_solution = new_solution
            current_dist = new_dist

            # 更新最佳解
            if new_dist < best_dist:
                best_solution = new_solution
                best_dist = new_dist
    
    return best_solution, best_dist


# 绘图
def draw(position, solution):
    path_y = [position[i, 0] for i in solution]
    path_x = [position[i, 1] for i in solution]
    plt.plot(path_x, path_y, color='r', linestyle='-', linewidth=2, label='Path')

    # 绘制点
    plt.scatter(position[:, 1], position[:, 0], color='b', marker='o', label='Points')
    plt.grid()
    # plt.axis('off')
    plt.show()


# 计算距离矩阵
def cal_dis_matrix(position):
    nums = position.shape[0]
    dis_matrix = n.zeros((nums, nums))
    for i in range(nums):
        for j in range(i + 1, nums):
            # 计算点i和点j之间的欧氏距离
            distance = n.linalg.norm(position[i] - position[j])
            dis_matrix[i, j] = distance
            dis_matrix[j, i] = distance
    return dis_matrix


def run():
    position = n.random.normal(0, 1, (10, 2))
    dis_matrix = cal_dis_matrix(position)

    # 参数设置
    init_temp = 100.0
    rate = 0.99
    epochs = 1000

    # 运行模拟退火算法
    best_solution, best_dist = sa(dis_matrix, init_temp, rate, epochs)

    print("最佳路径:", best_solution)
    print("最短距离:", best_dist)
    draw(position, best_solution)
    


if __name__ == '__main__':
    run()















