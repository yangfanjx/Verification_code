from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from Code import coherence, extractColor
import traceback
from skimage import morphology

import matplotlib.pyplot as plt

# 降噪
def check_over(chack_array, x, y, x_all, y_all):
    # 0:黑色点
    # 1:白色点
    if y == 0:
        left = 1
        right = bool(chack_array[x, y + 1])
    elif y == y_all - 1:
        right = 1
        left = bool(chack_array[x, y - 1])
    else:
        left = bool(chack_array[x, y - 1])
        right = bool(chack_array[x, y + 1])
    if x == 0:
        up = 1
        down = bool(chack_array[x + 1, y])
    elif x == x_all - 1:
        down = 1
        up = bool(chack_array[x - 1, y])
    else:
        up = bool(chack_array[x - 1, y])
        down = bool(chack_array[x + 1, y])

    if up == down == left == right == 1:
        return 1
    else:
        return 0


# 纵向切割图片空白区域的坐标list分组
def get_y_coordinate(x_list: list) -> list:
    all_list = []
    item_list = []
    for index in range(len(x_list)):
        if index == 0:
            continue
        if index < len(x_list) and x_list[index] - x_list[index - 1] == 1:
            item_list.append(x_list[index])
        else:
            all_list.append(item_list.copy())
            item_list = []
            item_list.append(x_list[index])

    all_list.append(item_list.copy())
    return all_list


# 纵向分割图片，去除图片空白部分
def y_cut_fun(all_list: list, pic: Image, x_shape) -> list:
    y_cut_list = []
    for item_list in all_list:
        try:
            x_start = x_end = 0
            x_start = item_list[0]
            x_end = item_list[-1]
            capt_per_char = pic.crop((x_start, 0, x_end + 1, x_shape))  # 分割
            y_cut_list.append(capt_per_char)
        except:
            print("item_list：",item_list)
    return y_cut_list


# 遍历已经完胜纵向分割的图片list，对每张图进行提取含有文字的行坐标信息
def get_all_y(y_cut_list: Image) -> list:
    return_list = []
    for y_cut in y_cut_list:
        item_list = []
        y_cut_array = np.array(y_cut)
        _x, _y, _ = y_cut_array.shape
        for m_x in range(_x):
            count = 0
            for m_y in range(_y):
                if list(y_cut_array[m_x, m_y]) == [255, 255, 255]:
                    count += 1
            if count != _y:
                item_list.append(m_x)
        return_list.append(item_list)
    return return_list


# 横向去除图片空白部分
def x_cut_fun(y_cut_list, y_cut):
    x_cut_list = []
    for index, item_list in enumerate(y_cut_list):
        try:
            y_start = item_list[0]
            y_end = item_list[len(item_list) - 1]
            _x, _y, _ = np.array(y_cut[index]).shape
            capt_per_char = y_cut[index].crop((0, y_start, _y, y_end + 1))  # 分割
            # img = capt_per_char.resize((20, 20), Image.ANTIALIAS)
            x_cut_list.append(capt_per_char)
        except Exception as ex:
            print("x_cut_fun:",ex)
            x_cut_list = []
    return x_cut_list

#返回没有背景的彩色图片
def ergodic_pic(char_array, image):
    # 获取原图的三维数组,用于截取原图中的文字
    img_array = np.array(image)

    # 获取二维数组的行、列数，用于循环遍历每个像素
    x_shape, y_shape = char_array.shape

    # 因为图片信息为三位数组，建立空的三位数组保存新图片
    new_pic = np.zeros((x_shape, y_shape, 3))

    # 记录该列不全部为空白的x坐标
    x_list = []
    for y in range(y_shape):
        empty_count = 0
        for x in range(x_shape):
            # 如果该像素不是空白，则判断是否为噪点
            # (白底黑字)    0:黑色点,1:白色点
            if char_array[x, y] == 0:
                _current_point = check_over(char_array, x, y, x_shape, y_shape)
            else:
                _current_point = char_array[x, y]

            # 使用降噪后的黑白图片作为模板，对原来的彩色图片进行剪切
            count = list(img_array[x, y].shape)[0] if list(img_array[x, y].shape) != [] else 1
            if int(_current_point) == 1:
                new_pic[x, y] = [255] * count
                empty_count += 1
            else:
                new_pic[x, y] = img_array[x, y]
        if empty_count != x_shape:
        # if x_shape - empty_count <= 2:
            x_list.append(y)
    # 二维数组转图片
    pic = Image.fromarray(new_pic.astype('uint8')).convert('RGB')

    return pic, x_list

#返回没有背景的彩色图片,删除边框
def ergodic_pic_ShangHai(char_array, image):
    # 获取原图的三维数组,用于截取原图中的文字
    img_array = np.array(image)

    # 获取二维数组的行、列数，用于循环遍历每个像素
    x_shape, y_shape = char_array.shape

    # 因为图片信息为三位数组，建立空的三位数组保存新图片
    new_pic = np.zeros((x_shape, y_shape, 3))
    # 记录该列不全部为空白的x坐标
    x_list = []
    for y in range(y_shape):
        empty_count = 0
        for x in range(x_shape):
            if y == 0 or y == y_shape - 1 or x == 0 or x == x_shape - 1:
                char_array[x, y] = 1
            # 如果该像素不是空白，则判断是否为噪点
            # (白底黑字)    0:黑色点,1:白色点
            if char_array[x, y] == 0:
                _current_point = check_over(char_array, x, y, x_shape, y_shape)
            else:
                _current_point = char_array[x, y]

            # 使用降噪后的黑白图片作为模板，对原来的彩色图片进行剪切
            count = list(img_array[x, y].shape)[0] if list(img_array[x, y].shape) != [] else 1
            if int(_current_point) == 1:
                new_pic[x, y] = [255] * count
                empty_count += 1
            else:
                new_pic[x, y] = img_array[x, y]
        if x_shape - empty_count <=1:
            # new_pic[:,y] = [255] * count
            for x in range(x_shape):
                new_pic[x, y] = [255] * count
    # 二维数组转图片
    pic = Image.fromarray(new_pic.astype('uint8')).convert('RGB')

    return pic, x_list

'''
# 将图片转换为没有噪点的黑白图片
def change_pic_coherence(char_array, image: Image, m_count) -> list:
    return_dict = {"type": 0}

    # 获取没有噪点的彩色原图，当有文字不能切割是，使用彩色图按照颜色拆分
    pic, x_list= ergodic_pic(char_array, image)
    return_dict["pic"] = pic

    # 使用联通法拆分文字
    pic_list, cut_list = coherence.coherence_function2(Image.fromarray(char_array))
    return_dict["cut_list"] = cut_list
    if len(cut_list) == m_count:
        return_dict["type"] = 1
    return_dict["pic_list"] = pic_list
    return return_dict
'''

# 将图片转换为没有噪点的黑白图片
def change_pic(char_array: list, image: Image, m_count,_pic=None, _x_list=None) -> list:
    return_dict = {"type": 0}
    #如果这两项不空，则说明传入图像是已经去除背景的彩色图片，不需要再转换
    if _pic and _x_list:
        pic = _pic
        x_list = _x_list
    else:
        # 获取没有噪点的彩色原图，当有文字不能切割是，使用彩色图按照颜色拆分
        pic, x_list = ergodic_pic(char_array, image)

    # 按照字母切割成多个图片
    m_all_list = get_y_coordinate(x_list)
    m_y_cut = y_cut_fun(m_all_list, pic, pic.size[1])
    m_get_all_y = get_all_y(m_y_cut)
    train_list = x_cut_fun(m_get_all_y, m_y_cut)

    if len(train_list) == m_count:
        return_dict["type"] = 1
    return_dict["pic"] = train_list
    return return_dict


# 处理图片
def handle_pic(image: Image, global_count: int, LUT_value=140, ergodic_tuple=()):
    # 降噪
    def noice(binary):  # 二值图
        char_array = np.array(binary)
        x_shape, y_shape = char_array.shape

        # 降噪方法
        def check_over(chack_array, x, y, x_all, y_all):
            # False:黑色点
            # True:白色点
            if y == 0:
                left = True
                right = bool(chack_array[x, y + 1])
            elif y == y_all - 1:
                right = True
                left = bool(chack_array[x, y - 1])
            else:
                left = bool(chack_array[x, y - 1])
                right = bool(chack_array[x, y + 1])
            if x == 0:
                up = True
                down = bool(chack_array[x + 1, y])
            elif x == x_all - 1:
                down = True
                up = bool(chack_array[x - 1, y])
            else:
                up = bool(chack_array[x - 1, y])
                down = bool(chack_array[x + 1, y])

            if up == down == left == right == True:
                return True
            else:
                return False

        for y in range(y_shape):
            for x in range(x_shape):
                # (白底黑字)False:黑色点,True:白色点
                if char_array[x, y] == False:
                    a = check_over(char_array, x, y, x_shape, y_shape)
                    if a == True:
                        char_array[x, y] = 255
                    else:
                        char_array[x, y] = 0
                else:
                    char_array[x, y] = 255
        return char_array

    LUT = [0] * LUT_value + [1] * (256 - LUT_value)
    # 转灰度图
    imgry = image.convert('L')

    # 转二值图
    binary = imgry.point(LUT, '1')

    if ergodic_tuple != ():
        get_dict = change_pic(None, None, global_count, ergodic_tuple[0], ergodic_tuple[1])
    else:
        # +0为了将原二维数组中bool类型元素全部转为0，1形式
        char_array = np.array(binary) + 0
        x_shape, _ = char_array.shape
        get_dict = change_pic(char_array, image, global_count)
    if get_dict["type"] == 1:
        _list = []
        for i in get_dict["pic"]:
            imgry = i.convert('L')
            binary = imgry.point(LUT, '1')
            _list.append(binary)
        return _list
    else:
        return_list = []
        try:
            #r如果图片没有分割成指定的个数，就降原图的二值化进行连通性分割
            pic_list, cut_list, return_image_list = coherence.coherence_function2(binary, image)
            if len(pic_list) == global_count:
                return pic_list

            # elif len(pic_list) == 1:
            #
            #如果连通性分割没有分出需要的个数，进行颜色聚类分割
            elif global_count - len(pic_list) == 1:
                widthest = 0
                widthest_pic = None
                widthest_index = 0
                for _index, _width in enumerate(cut_list[0]):
                    d_value = _width[1]-_width[0]
                    if d_value > widthest:
                        widthest_index = _index
                        widthest = d_value
                        widthest_pic = return_image_list[_index]
                pic_list = extractColor.handle_pic(widthest_pic, 3)
                _dict = {}
                if len(pic_list) == 3:
                    all=0
                    for _index, i in enumerate(pic_list):
                        imgry = i.convert('L')
                        binary = imgry.point(LUT, '1')
                        x_shape, y_shape = binary.size
                        binary = noice(binary)
                        # iii.show()
                        # print(np.array(binary))
                        dst = morphology.remove_small_objects(np.array(binary), min_size=5, connectivity=1)

                        new_array = np.empty(shape=[y_shape,x_shape])

                        for y in range(y_shape):
                            for x in range(x_shape):
                                # (白底黑字)False:黑色点,True:白色点
                                if dst[y, x] == False:
                                    new_array[y, x] = 0
                                else:
                                    new_array[y, x] = 255
                        imm = Image.fromarray(new_array.astype('uint8')).convert('RGB')
                        pic_list, cut_list, return_image_list = coherence.coherence_function(imm)
                        min_x = max_x = min_y = max_y = 0
                        for x_x in cut_list[0]:
                            if x_x[0] < min_x:
                                min_x = x_x[0]
                            if x_x[1] > max_x:
                                max_x = x_x[1]
                        for y_y in cut_list[1]:
                            if y_y[0] < min_y:
                                min_y = y_y[0]
                            if y_y[1] > max_y:
                                max_y = y_y[1]
                        if min_x == 0 and max_x == x_shape:
                            continue
                        elif min_x < x_shape // 4 and max_x < x_shape:
                            _dict["1"] = imm.crop((min_x, min_y, max_x, max_y))
                            all += 1
                        elif min_x >= x_shape // 4 and max_x <= x_shape:
                            _dict["2"] = imm.crop((min_x, min_y, max_x, max_y))
                            all += 1

                    for _i, _item in enumerate(get_dict["pic"]):
                        if _i == widthest_index:
                            if "1" in _dict.keys() and "2" in _dict.keys():
                                return_list.append(_dict["1"])
                                return_list.append(_dict["2"])
                            else:
                                return return_list
                        else:
                            imgry = _item.convert('L')
                            binary = imgry.point(LUT, '1')
                            return_list.append(binary)
        except:
            print("Noice:", traceback.print_exc())
        return return_list


def handle_pic_bak(image: Image, global_count: int, LUT_value=140, ergodic_tuple=()):
    LUT = [0] * LUT_value + [1] * (256 - LUT_value)
    # 转灰度图
    imgry = image.convert('L')

    # 转二值图
    binary = imgry.point(LUT, '1')

    if ergodic_tuple != ():
        get_dict = change_pic(None, None, global_count, ergodic_tuple[0], ergodic_tuple[1])
    else:
        # +0为了将原二维数组中bool类型元素全部转为0，1形式
        char_array = np.array(binary) + 0
        x_shape, _ = char_array.shape

        get_dict = change_pic(char_array, image, global_count)
    if get_dict["type"] == 1:
        _list = []
        for i in get_dict["pic"]:
            imgry = i.convert('L')
            binary = imgry.point(LUT, '1')
            _list.append(binary)
        return _list
    else:
        return_list = []
        try:
            width_list = []
            for i in get_dict["pic"]:
                width_list.append(i.size[0])
            # 当分割个数少于文字数，按照分割数量计算哪张子图片是包含多文字图片
            disparity_count = global_count - len(width_list)
            if disparity_count >= 1:
                # 当只有2个文字相连时，找到宽度最大的认为是相连的文字，在彩色图中进行切割
                _index = width_list.index(max(width_list))
                width_pic = get_dict["pic"][_index]
                #尝试使用联通性分割
                pic_list, cut_list, return_image_list = coherence.coherence_function(width_pic)
                if len(pic_list) == 2:
                    for _i, _item in enumerate(get_dict["pic"]):
                        if _i == _index:
                            for m_i in pic_list:
                                return_list.append(m_i)
                        else:
                            imgry = _item.convert('L')
                            binary = imgry.point(LUT, '1')
                            return_list.append(binary)
                    return return_list

                #如果连通性不能分割，使用颜色聚类方法分割
                pic_list = extractColor.handle_pic(width_pic, 3)

                _dict = {}
                if len(pic_list) == 3:
                    for i in pic_list:
                        show_array = np.array(i)
                        show_list = [show_array[0][0], show_array[0][-1], show_array[-1][0], show_array[-1][-1]]
                        if sum(show_list) <= 255 * 2:
                            i = ImageOps.invert(i).point(LUT, '1')

                        imgry = i.convert('L')
                        binary = imgry.point(LUT, '1')
                        _x, _y = binary.size
                        pic_list, cut_list, return_image_list = coherence.coherence_function2(binary, i)

                        if cut_list[0][0][0] == 0 and cut_list[0][0][1] == _x:
                            continue
                        elif cut_list[0][0][0] < _x // 4 and cut_list[0][0][1] < _x:
                            _dict["1"] = pic_list[0]
                        elif cut_list[0][0][0] >= _x // 4 and cut_list[0][0][1] <= _x:
                            _dict["2"] = pic_list[0]

                    for _i, _item in enumerate(get_dict["pic"]):
                        if _i == _index and len(_dict.keys())==2:
                            return_list.append(_dict["1"])
                            return_list.append(_dict["2"])
                        else:
                            imgry = _item.convert('L')
                            binary = imgry.point(LUT, '1')
                            return_list.append(binary)


        except:
            print("Noice:", traceback.print_exc())
        return return_list


# 颜色分割以后的黑白图使用裁掉空白区域方法
def colur_cut(image):  # 二值图
    LUT_value=50
    LUT = [0] * LUT_value + [1] * (256 - LUT_value)
    # 转灰度图
    imgry = image.convert('L')
    # 转二值图
    binary = imgry.point(LUT, '1')
    char_array = np.array(binary)
    x_shape, y_shape = char_array.shape

    # 降噪方法
    def check_over(chack_array, x, y, x_all, y_all):
        # False:黑色点
        # True:白色点
        if y == 0:
            left = True
            right = bool(chack_array[x, y + 1])
        elif y == y_all - 1:
            right = True
            left = bool(chack_array[x, y - 1])
        else:
            left = bool(chack_array[x, y - 1])
            right = bool(chack_array[x, y + 1])
        if x == 0:
            up = True
            down = bool(chack_array[x + 1, y])
        elif x == x_all - 1:
            down = True
            up = bool(chack_array[x - 1, y])
        else:
            up = bool(chack_array[x - 1, y])
            down = bool(chack_array[x + 1, y])

        if up == down == left == right == True:
            return True
        else:
            return False

    cut_x_list = []
    cut_y_list = []

    for y in range(y_shape):
        empty_point = 0
        for x in range(x_shape):
            # (白底黑字)False:黑色点,True:白色点
            if char_array[x, y] == False:
                a = check_over(char_array, x, y, x_shape, y_shape)
                if a == True:
                    char_array[x, y] = 255
                    empty_point+=1
                else:
                    char_array[x, y] = 0
            else:
                char_array[x, y] = 255
                empty_point+=1
        if empty_point != x_shape:
            cut_x_list.append(y)
    if cut_x_list!=[]:
        cut_x_start = cut_x_list[0]
        cut_x_end = cut_x_list[-1]
    else:
        return image,"0"

    for x in range(x_shape):
        empty_point = 0
        for y in range(y_shape):
            if char_array[x, y] == True:
                empty_point+=1
        if empty_point!=y_shape:
            cut_y_list.append(x)

    cut_y_start = cut_y_list[0]
    cut_y_end = cut_y_list[-1]

    #判断宽度，方式应为噪点过多影响图片切割
    if y_shape - (cut_x_end - cut_x_start) > 4 and cut_x_start <= 2:
        cut_x_end = cut_x_start + 20
    elif y_shape - (cut_x_end - cut_x_start) > 4 and y_shape - cut_x_end <= 2:
        cut_x_start = cut_x_end - 20

    return_pic = image.crop((cut_x_start, cut_y_start, cut_x_end, cut_y_end))
    if cut_x_start<=2:
        return_index = "1"
    else:
        return_index = "2"
    return return_pic,return_index

