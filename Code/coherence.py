# -*-coding:utf-8-*-
# 联通法获取独立的文字
import queue
def cfs(img, del_point):
    """传入二值化后的图片进行连通域分割"""
    pixdata = img.load()
    w, h = img.size
    visited = set()
    q = queue.Queue()
    # offset = [(-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    offset = [(0, -1),(-1, 0), (1, 0),(0, 1)]
    cuts_x = []
    cuts_y = []
    for x in range(w):
        for y in range(h):
            x_axis = []
            y_axis = []
            if pixdata[x, y] == 0 and (x, y) not in visited:
                q.put((x, y))
                visited.add((x, y))
            while not q.empty():
                x_p, y_p = q.get()
                for x_offset, y_offset in offset:
                    x_c, y_c = x_p + x_offset, y_p + y_offset
                    if (x_c, y_c) in visited:
                        continue
                    visited.add((x_c, y_c))
                    try:
                        if pixdata[x_c, y_c] == 0:
                            q.put((x_c, y_c))
                            x_axis.append(x_c)
                            y_axis.append(y_c)
                    except:
                        pass
            if x_axis:
                min_x, max_x = min(x_axis) if min(x_axis) > 0 else 0, max(x_axis)
                if max_x - min_x > del_point:
                    # 宽度小于3的认为是噪点，根据需要修改
                    cuts_x.append((min_x, max_x + 1))
            if y_axis:
                min_y, max_y = min(y_axis) if min(y_axis) > 0 else 0, max(y_axis)
                if max_y - min_y > del_point:
                    # 宽度小于3的认为是噪点，根据需要修改
                    cuts_y.append((min_y, max_y + 1))
    return (cuts_x, cuts_y)


def saveSmall(pic, cuts, image):
    cuts_x, cuts_y = cuts
    return_list = []
    return_image_list = []
    try:
        for i, item in enumerate(cuts_x):
            box = (item[0] if item[0] > 0 else 0, cuts_y[i][0], item[1], cuts_y[i][1])
            add_pic = pic.crop(box)
            add_image = image.crop(box)
            return_image_list.append(add_image)
            return_list.append(add_pic)
            # import numpy as np
            # from PIL import Image
            # aa = np.array(add_pic) * 255
            # Image.fromarray(aa.astype('uint8')).convert('RGB').show()
    except:
        print("saveSmall ERROR:",cuts)
    return return_list, return_image_list


def TwoValue(pic,THRESHOLD):
    LUT = [0] * THRESHOLD + [1] * (256 - THRESHOLD)
    capt_gray = pic.convert("L")
    capt_bw = capt_gray.point(LUT, "1")
    return capt_bw


def check(cut_list):
    len0 = len(cut_list[0])
    len1 = len(cut_list[1])
    try:
        if len0 != len1:
            del_count = abs(len0-len1)
            del_number = 0
            if len0 > len1:
                for i in range(1, len0):
                    if cut_list[0][i - 1][1] > cut_list[0][i][0]:
                        del (cut_list[0][i-1])
                        del_number += 1
                        if del_count == del_number:
                            break
            else:
                for i in range(1, len1):
                    if cut_list[1][i - 1][1] > cut_list[1][i][0]:
                        del (cut_list[1][i-1])
                        del_number += 1
                        if del_count == del_number:
                            break
    except:
        return ()
    return cut_list


# 传入正常图片处理方法
def coherence_function(pic, THRESHOLD=140, del_point=1):
    handl_pic = TwoValue(pic, THRESHOLD)
    cut_list = cfs(handl_pic, del_point)
    cut_list = check(cut_list)
    if cut_list == ():
        return [],[],[]
    return_list, return_image_list = saveSmall(handl_pic, cut_list, pic)
    return return_list, cut_list, return_image_list


# 传入二值图处理方法
def coherence_function2(binary, image, del_point=3):
    cut_list = cfs(binary, del_point)
    cut_list = check(cut_list)
    return_list, return_image_list = saveSmall(binary, cut_list, image)
    return return_list, cut_list, return_image_list
