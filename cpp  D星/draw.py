import tkinter as tk
import numpy as n
from PIL import ImageGrab


def read_data(fn):
    with open(fn) as fp:
        lines = fp.readlines()
        data = []
        for li in lines:
            li = li.replace('\n', '')
            li = li.split('\t')
            # pos
            pos = li[0:2]
            parent_pos = li[2:4]
            j_value = li[4]
            h_value = li[5]
            state = li[6]
            map_type = li[7]
            dc = {
                'pos': [int(pos[0]), int(pos[1])],
                'parent_pos': [int(parent_pos[0]), int(parent_pos[1])],
                'j_value': float(j_value),
                'h_value': float(h_value),
                'state': state,
                'map_type': int(map_type),
            }
            data.append(dc)
    return data


def get_data(data, y_pos, x_pos):
    for di in data:
        if di['pos'][0] == y_pos and di['pos'][1] == x_pos:
            return di


def draw_grid_with_arrows(file_name, root):

    INF = 100000.
    fn = file_name

    data = read_data(fn)
    y_max = max(data, key=lambda dc:dc['pos'][0])['pos'][0]
    x_max = max(data, key=lambda dc:dc['pos'][1])['pos'][1]

    y_length = y_max + 1
    x_length = x_max + 1

    im = n.zeros((y_length, x_length), dtype='int64')

    # 自定义颜色列表，每个元素都是一个RGB颜色元组
    colors1 = [
        (1., 1., 1.),     # 原障碍
        (0.99, 0., 0),  
        (255/255, 165/255, 0/255),  # open
        (0., 0., 0.99)
    ]
    
    colors2 = []
    j_value_ex_inf = [dc['j_value'] for dc in data if dc['j_value'] < INF]
    j_max = max(j_value_ex_inf)
    j_value_uni = [int(j / j_max * 100) for j in j_value_ex_inf]
    j_value_uni = filter(lambda d:d>=0, j_value_uni)
    j_value_uni = sorted(set(j_value_uni))
    
    for i in range(len(j_value_uni)):
        i += 1
        colors2.append((1, 1 - i / len(j_value_uni), 1 - i / len(j_value_uni)))
    
    colors = colors1 + colors2

    for ri in range(y_length):
        for ci in range(x_length):
            datai = get_data(data, ri, ci)
            if datai['map_type'] == 0:
                if datai['j_value'] == -1:
                    pass
                else:
                    if datai['j_value'] >= INF:
                        im[ri, ci] = 2
                    else:
                        colo = int(datai['j_value'] / j_max * 100)
                        ii = j_value_uni.index(colo)
                        coloi = len(colors1) + ii
                        im[ri, ci] = coloi
                    if datai['state'] == 'o':
                        im[ri, ci] = 2

    n_rows, n_cols = len(im), len(im[0])

    for i in range(n_rows):
        for j in range(n_cols):
            this = get_data(data, i, j)
            color = colors[im[i][j]]
            
            if this['map_type'] == 1:
                color = (0., 0., 0.)
            elif this['h_value'] >= INF and this['j_value'] != -1:
                color = (0.5, 0.5, 0.5)

            this_pos = this['pos']
            parent_pos = this['parent_pos']
            dy = parent_pos[0] - this_pos[0]
            dx = parent_pos[1] - this_pos[1]

            if parent_pos[1] == -1 or parent_pos[0] == -1:
                direction = 8
            elif dx == 0 and dy == -1:
                direction = 0
            elif dx == 0 and dy == 1:
                direction = 1
            elif dx == -1 and dy == 0:
                direction = 2
            elif dx == 1 and dy == 0:
                direction = 3
            elif dx == -1 and dy == -1:
                direction = 4
            elif dx == 1 and dy == -1:
                direction = 5
            elif dx == -1 and dy == 1:
                direction = 6
            elif dx == 1 and dy == 1:
                direction = 7
            else:
                direction = 8
            
            color_hex = "#{:02X}{:02X}{:02X}".format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))
            
            frame = tk.Frame(root, width=50, height=50, bg=color_hex)
            frame.grid(row=i, column=j, padx=1, pady=1)

            canvas = tk.Canvas(frame, width=50, height=50, bg=color_hex, highlightthickness=0)
            canvas.grid(row=0, column=0)

            
            if this['pos'][1] == start_pos[1] and this['pos'][0] == start_pos[0]:
                canvas.create_polygon(arrow_center[0] - arrow_size, arrow_center[1] - arrow_size, 
                                    arrow_center[0] + arrow_size, arrow_center[1] - arrow_size, 
                                    arrow_center[0] - arrow_size, arrow_center[1] + arrow_size, 
                                    arrow_center[0] + arrow_size, arrow_center[1] + arrow_size, 
                                    fill='black')
            else:

                arrow_size = 15
                arrow_center = (25, 25)


                if direction == 0:
                    canvas.create_polygon(arrow_center[0], arrow_center[1] - arrow_size, 
                                        arrow_center[0] - arrow_size//2, arrow_center[1] + arrow_size, 
                                        arrow_center[0] + arrow_size//2, arrow_center[1] + arrow_size, 
                                        fill='black')
                elif direction == 1:
                    canvas.create_polygon(arrow_center[0], arrow_center[1] + arrow_size, 
                                        arrow_center[0] - arrow_size//2, arrow_center[1] - arrow_size, 
                                        arrow_center[0] + arrow_size//2, arrow_center[1] - arrow_size, 
                                        fill='black')
                elif direction == 2:
                    canvas.create_polygon(arrow_center[0] + arrow_size, arrow_center[1] - arrow_size//2, 
                                        arrow_center[0] - arrow_size, arrow_center[1], 
                                        arrow_center[0] + arrow_size, arrow_center[1] + arrow_size//2, 
                                        fill='black')
                elif direction == 3:
                    canvas.create_polygon(arrow_center[0] - arrow_size, arrow_center[1] - arrow_size//2, 
                                        arrow_center[0] - arrow_size, arrow_center[1] + arrow_size//2, 
                                        arrow_center[0] + arrow_size, arrow_center[1], 
                                        fill='black')
                elif direction == 4:
                    canvas.create_polygon(arrow_center[0] + arrow_size+5, arrow_center[1], 
                                        arrow_center[0] - arrow_size, arrow_center[1] - arrow_size, 
                                        arrow_center[0] + arrow_size-5, arrow_center[1] + arrow_size, 
                                        fill='black')
                elif direction == 5:
                    canvas.create_polygon(arrow_center[0] - arrow_size-5, arrow_center[1], 
                                        arrow_center[0] + arrow_size, arrow_center[1] - arrow_size, 
                                        arrow_center[0] - arrow_size+5, arrow_center[1] + arrow_size, 
                                        fill='black')
                elif direction == 6:
                    canvas.create_polygon(arrow_center[0] + arrow_size+5, arrow_center[1], 
                                        arrow_center[0] - arrow_size, arrow_center[1] + arrow_size, 
                                        arrow_center[0] + arrow_size-5, arrow_center[1] - arrow_size, 
                                        fill='black')
                elif direction == 7:
                    canvas.create_polygon(arrow_center[0] - arrow_size-5, arrow_center[1], 
                                        arrow_center[0] + arrow_size, arrow_center[1] + arrow_size, 
                                        arrow_center[0] - arrow_size+5, arrow_center[1] - arrow_size, 
                                        fill='black')
                    

file_name_format = "./grid/grid_{}.txt"
def update_grid(canvas):
    global index
    index += 1
    if index > end_index:
        return
    file_name = file_name_format.format(index)
    for widget in canvas.winfo_children():
        widget.destroy()
    draw_grid_with_arrows(file_name, canvas)

    # 将Canvas内容保存为图像
    x=canvas.winfo_rootx()+canvas.winfo_x()+40
    y=canvas.winfo_rooty()+canvas.winfo_y()+35
    x1=x+canvas.winfo_width()+120
    y1=y+canvas.winfo_height()+120
    ImageGrab.grab().crop((x,y,x1,y1)).save(f"./png/{index}.png")
    
    rt.after(time_step, lambda: update_grid(canvas))


start_pos = (4, 4)
target_pos = (9, 9)

index = 0
end_index = 87

time_step = 1000
rt = tk.Tk()
rt.title("Grid with Arrows")
canvas = tk.Frame(rt)
canvas.pack()
draw_grid_with_arrows(file_name_format.format(index), canvas)
rt.after(time_step, lambda: update_grid(canvas))
rt.mainloop()








