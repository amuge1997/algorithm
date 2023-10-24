
import numpy as n
import matplotlib.pyplot as plt


def read_position_solution(position_file_name, solution_file_name):
    position = []
    solution = []
    with open(position_file_name) as fp:
        pos = fp.readlines()
        for p in pos:
            p = p.replace('\n', '')
            p = p.split('\t')
            y, x = p
            y, x = float(y), float(x)
            position.append([y, x])
        position = n.array(position)
    
    with open(solution_file_name) as fp:
        sol = fp.readlines()
        for s in sol:
            s = s.replace('\n', '')
            i = int(s)
            solution.append(i)
    return position, solution


def draw(position_file, solution_file, title):
    position, solution = read_position_solution(position_file, solution_file)

    path_y = [position[i, 0] for i in solution] #+ [position[solution[0], 0]]
    path_x = [position[i, 1] for i in solution] #+ [position[solution[0], 1]]
    plt.figure(figsize=(5, 5))
    plt.plot(path_x, path_y, color='r', linestyle='-', linewidth=2, label='Path')
    plt.scatter(position[:, 1], position[:, 0], color='b', marker='o', label='Points')
    plt.title(title)
    plt.grid()
    plt.show()


def draw3():
    draw(
        position_file='position.txt',
        solution_file='solution_start.txt',
        title='Start Solution'
    )
    draw(
        position_file='position.txt',
        solution_file='solution_middle.txt',
        title='Middle Solution'
    )
    draw(
        position_file='position.txt',
        solution_file='solution_best.txt',
        title='Best Solution'
    )


if __name__ == "__main__":
    draw3()

    











