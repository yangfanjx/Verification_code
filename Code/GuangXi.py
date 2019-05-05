from PIL import Image
import pytesseract, os, cv2, numpy as np
from Code import coherence

THRESHOLD = 158
LUT = [0] * THRESHOLD + [1] * (256 - THRESHOLD)

CAPT_PATH = "C:/Users/LENOVO/Desktop/pandas/capt/GuangXi/"
SAVE_PATH = "C:/Users/LENOVO/Desktop/pandas/capt/GuangXi_test/"


# 降噪
def check_over(chack_array, x, y, x_all, y_all):
    # True:黑色点
    # False:白色点
    if y == 0:
        left = False
        right = bool(chack_array[x, y + 1])
    elif y == y_all - 1:
        right = False
        left = bool(chack_array[x, y - 1])
    else:
        left = bool(chack_array[x, y - 1])
        right = bool(chack_array[x, y + 1])
    if x == 0:
        up = False
        down = bool(chack_array[x + 1, y])
    elif x == x_all - 1:
        down = False
        up = bool(chack_array[x - 1, y])
    else:
        up = bool(chack_array[x - 1, y])
        down = bool(chack_array[x + 1, y])

    if (up == down == False) or (left == right == False):
        return False
    else:
        return True


# 降噪主方法
def clear_noice(_array):
    x_shape, y_shape = _array.shape  # 42,20
    show_array = np.empty(shape=[x_shape, y_shape])
    have_point_line = []
    for i in range(x_shape):
        empty_count = 0  # y_shape
        for j in range(y_shape):
            if _array[i, j]:
                c_result = check_over(_array, i, j, x_shape, y_shape)
                _array[i, j] = c_result
                if c_result:
                    show_array[i, j] = 0
                else:
                    show_array[i, j] = 255
                    empty_count += 1
            else:
                show_array[i, j] = 255
                empty_count += 1
        if empty_count != y_shape:
            have_point_line.append(i)
    return show_array, have_point_line


# 切割空白区域
def cut_array(_array):
    x_shape, y_shape = _array.shape
    show_array, have_point_line = clear_noice(_array)
    # Image.fromarray(show_array.astype('uint8')).convert('RGB').show()
    # print(have_point_line,have_point_line[-1],have_point_line[0])
    new_width = have_point_line[-1] - have_point_line[0] + 1

    return_array = np.empty(shape=[new_width, y_shape])
    start_x = 0
    for i in range(x_shape):
        if i not in have_point_line:
            continue
        for j in range(y_shape):
            return_array[start_x, j] = 255 if show_array[i, j] else 0
        start_x += 1

    return Image.fromarray(return_array.astype('uint8')).convert('RGB')


# 旋转图片
def rotate_bound(image):
    M = cv2.getRotationMatrix2D((43, 13), 30, 1.0)
    M[0, 2] += 0.5
    M[1, 2] += 19.5
    return cv2.warpAffine(image, M, (87, 65), borderValue=(255, 255, 255))


# 多个小图片拼接切大图
def image_compose(pic_list):
    to_image = Image.new('RGB', (100, 25), "#FFFFFF")  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for x in range(len(pic_list)):
        from_image = pic_list[x]
        if x == 0:
            to_image.paste(from_image, (0, 0))
        to_image.paste(from_image, (x * 25, 0))

    return to_image


def cut_pic():
    def cut_handle(_cut_pic, _cut_num, ):
        m_pic_list = []
        _width = _cut_pic.size[0]
        for i in range(_cut_num):
            x_start = (_width // _cut_num) * i
            x_end = (_width // _cut_num) * (i + 1)
            m_cut_pic = _cut_pic.crop((x_start, 0, x_end, _cut_pic.size[1]))
            imgry = m_cut_pic.convert('L')
            imgry = np.array(imgry)
            bwimg = (imgry < 180)
            # dst = morphology.remove_small_objects(bwimg, min_size=30, connectivity=1)
            _img = cut_array(bwimg)
            m_pic_list.append(_img)

        return m_pic_list


    files = os.listdir(CAPT_PATH)
    total_count = error_count = 0
    for index, f in enumerate(files):
        total_count+=1
        # f = "4737_1555918726.49018.jpg"
        im = cv2.imread(CAPT_PATH + f,0)
        ret, mask = cv2.threshold(im, 150, 255, cv2.THRESH_BINARY)
        aa = rotate_bound(mask)

        img = Image.fromarray(cv2.cvtColor(aa, cv2.COLOR_BGR2RGB))

        # 连通性分割
        pic_list = []
        return_list, cut_list, return_image_list = coherence.coherence_function(img, 100)

        if len(return_list) == 4:
            pic_list = return_list
        # 分割以后为一张图片
        elif len(return_list) == 1:
            _pic = return_list[0]
            pic_list = cut_handle(_pic, 4)
        # 分割以后为两张图片
        elif len(return_list) == 2:
            width1 = cut_list[0][0][1] - cut_list[0][0][0]
            width2 = cut_list[0][1][1] - cut_list[0][1][0]
            # 通过宽度区分切割的是1还是2，或者1,2都需要切割
            if width1 < width2:
                if width2 // width1 > 1:
                    pic_list += [return_list[0]]
                    pic_list += cut_handle(return_list[1], 3)
                else:
                    pic_list += cut_handle(return_list[0], 2)
                    pic_list += cut_handle(return_list[1], 2)
            else:
                if width1 // width2 > 1:
                    pic_list += cut_handle(return_list[0], 3)
                    pic_list += [return_list[1]]
                else:
                    pic_list += cut_handle(return_list[0], 2)
                    pic_list += cut_handle(return_list[1], 2)
        # 分割以后为三张图片
        elif len(return_list) == 3:
            width1 = cut_list[0][0][1] - cut_list[0][0][0]
            width2 = cut_list[0][1][1] - cut_list[0][1][0]
            width3 = cut_list[0][2][1] - cut_list[0][2][0]
            width_list = [width1, width2, width3]
            max_index = width_list.index(max(width_list))
            for i in range(3):
                if i != max_index:
                    pic_list += [return_list[i]]
                else:
                    pic_list += cut_handle(return_list[i], 2)

        all_pic = image_compose(pic_list)
        # all_pic.save(SAVE_PATH + f)
        result_pre = pytesseract.image_to_string(all_pic, lang='gx', config="-psm 7")
        if result_pre != f.split("_")[0]:
            error_count +=1
    print("totla:{},error:{}".format(total_count, error_count))


if __name__ == "__main__":
    cut_pic()
