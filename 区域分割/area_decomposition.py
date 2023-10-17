import numpy as n


VAC = n.int32(0)    # 空置
OBS = n.int32(1)    # 阻碍
BOD = n.int32(2)    # 边界
NUS = n.int32(3)    # 起始序号


# 绘图
def show(image, cmap=None):
    import matplotlib.pyplot as p

    if cmap is not None:
        p.imshow(image, cmap=cmap)
        p.axis('off')
        p.show()
    else:
        p.imshow(image)
        p.axis('off')
        p.show()



# 区域分割
def area_decomposition(area):
    rows, cols = area.shape

    decomposition_map = area.copy()
    
    # 逐列扫描
    previous_segment_nums = None
    is_detect_first_col = False
    for ci in range(cols):
        obstacle_start_row = []
        obstacle_end_row = []
        in_obs = False
        for ri in range(rows):
            state = area[ri, ci]
            if in_obs is False and state == OBS:
                obstacle_start_row.append(ri)
                in_obs = True
            if in_obs is True and state == VAC:
                obstacle_end_row.append(ri - 1)
                in_obs = False
        
        obstacle_start_row = obstacle_start_row[0:-1]
        obstacle_end_row = obstacle_end_row[0:]

        this_segment_nums = len(obstacle_start_row)
        if this_segment_nums > 0 and is_detect_first_col is False:
            is_detect_first_col = True
            previous_segment_nums = this_segment_nums
            continue
        
        if is_detect_first_col:
            # 当前段数大于前段数
            if this_segment_nums > previous_segment_nums:
                for yi in range(rows):
                    state = area[yi, ci]
                    if state == OBS:
                        # 判断是否发生间断
                        is_grap = True
                        for i in [-1, 0, 1]:
                            t = n.min([n.max([0, yi + i]), rows - 1])
                            if area[t, ci - 1] == OBS:
                                is_grap = False
                        if is_grap:
                            for j in range(-1, -yi, -1):
                                t = n.min([n.max([0, yi + j]), rows - 1])
                                if area[t, ci] == VAC:
                                    decomposition_map[t, ci] = BOD
                                else:
                                    break
                            
                            for j in range(1, rows-yi-1, 1):
                                t = n.min([n.max([0, yi + j]), rows - 1])
                                if area[t, ci] == VAC:
                                    decomposition_map[t, ci] = BOD
                                else:
                                    break
                                    
            # 当前段数小于前段数
            if this_segment_nums < previous_segment_nums:
                for yi in range(rows):
                    state = area[yi, ci - 1]
                    if state == OBS:
                        is_grap = True
                        for i in [-1, 0, 1]:
                            t = n.min([n.max([0, yi + i]), rows - 1])
                            if area[t, ci] == OBS:
                                is_grap = False
                        if is_grap:
                            for j in range(-1, -yi, -1):
                                t = n.min([n.max([0, yi + j]), rows - 1])
                                if area[t, ci - 1] == VAC:
                                    decomposition_map[t, ci - 1] = BOD
                                else:
                                    break
                            
                            for j in range(1, rows-yi-1, 1):
                                t = n.min([n.max([0, yi + j]), rows - 1])
                                if area[t, ci - 1] == VAC:
                                    decomposition_map[t, ci - 1] = BOD
                                else:
                                    break
            previous_segment_nums = this_segment_nums
    return decomposition_map


# 序号扩散
def spread(result, decomposition_map, start_yx, next_number):
    unfinished = []
    finished = []

    result[start_yx[0], start_yx[1]] = next_number
    unfinished.append(start_yx)
    while len(unfinished) > 0:
        this = unfinished[-1]

        for j in [-1, 0, 1]:
            for i in [-1, 0, 1]:
                if j == i:
                    continue
                ri, ci = this
                y = ri + j
                x = ci + i

                oth = (y, x)
                oth_state = decomposition_map[y, x]
                if oth_state == VAC and oth not in finished:
                    result[y, x] = next_number
                    unfinished.append(oth)
        
        unfinished.remove(this)
        finished.append(this)


# 区域添加序号
def add_area_number(decomposition_map):
    rows, cols = decomposition_map.shape

    result = n.zeros_like(decomposition_map)
    next_number = NUS
    for ri in range(rows):
        for ci in range(cols):
            state = decomposition_map[ri, ci]
            number = result[ri, ci]

            if state == VAC and number == 0:
                start_yx = (ri, ci)
                spread(result, decomposition_map, start_yx, next_number)
                next_number += 1

    for ri in range(rows):
        for ci in range(cols):
            state = decomposition_map[ri, ci]
            if state == BOD:
                up_state = decomposition_map[ri-1, ci]
                if up_state == BOD:
                    result[ri, ci] = result[ri-1, ci]
                else:
                    state_ = decomposition_map[ri, ci-1]
                    if state_ != OBS:
                        result[ri, ci] = result[ri, ci-1]
                    else:
                        result[ri, ci] = result[ri, ci+1]
    return result


def run():
    from PIL import Image
    im = Image.open("map2.png")
    im = n.array(im)
    im = im.astype('int32')
    im = n.sum(im, axis=2)
    area = n.where(im == 255*3, 0, 1)
    
    # 边界必须填充1
    area = n.pad(area, pad_width=1, mode='constant', constant_values=1)
    decomposition_map = area_decomposition(area)

    from matplotlib.colors import ListedColormap
    colors = [(0.9, 0.9, 0.9), 'black', (0.6, 0.6, 0.6)]
    cmap = ListedColormap(colors)
    show(decomposition_map, cmap)

    result = add_area_number(decomposition_map)
    from matplotlib.colors import ListedColormap
    cmap = 'inferno'
    show(result, cmap)


if __name__ == "__main__":
    run()



















