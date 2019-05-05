from PIL import Image
import os, cv2, numpy as np
from skimage import morphology
from Code import Noise, coherence

CAPT_PATH = "C:/Users/LENOVO/Desktop/pandas/capt/ShanDong/"
SAVE_PATH = "C:/Users/LENOVO/Desktop/pandas/capt/ShanDong_1/"
MASK_PATH = "C:/Users/LENOVO/Desktop/pandas/capt/ShanDong_mask/"


def check_over_sd(chack_array, x, y):
    # True:黑色点
    # False:白色点
    left = bool(chack_array[x, y - 1])
    right = bool(chack_array[x, y + 1])

    up = bool(chack_array[x - 1, y])
    down = bool(chack_array[x + 1, y])

    if up == down == left == right == False:
        return False
    # elif left == right == False:
    #     return False
    else:
        return True


# 黑白图片
def t1():
    files = os.listdir(CAPT_PATH)
    for index, f in enumerate(files):
        img = cv2.imread(CAPT_PATH + f, 0)
        # bwimg = (img <= 0.7)
        ret, th1 = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)

        char_array = np.array(th1)
        x_shape, y_shape = char_array.shape
        char_array = (char_array < 10)
        new_array = np.empty(shape=[x_shape, y_shape])
        for y in range(y_shape):
            for x in range(x_shape):
                if x < 6 or x > 26 or y < 7 or y > 73:
                    new_array[x, y] = 255
                else:
                    # (白底黑字)True:黑色点,False:白色点
                    if char_array[x, y]:
                        c_o_s = check_over_sd(char_array, x, y)
                        if c_o_s:
                            new_array[x, y] = 0
                            th1[x, y] = 0
                        else:
                            new_array[x, y] = 255
                            th1[x, y] = 255
                    else:
                        new_array[x, y] = 255
        Image.fromarray(new_array.astype('uint8')).convert('RGB').show()
        th1 = (th1 < 10)
        dst = morphology.remove_small_objects(th1, min_size=30, connectivity=1)

        for y in range(y_shape):
            for x in range(x_shape):
                if x < 6 or x > 26 or y < 7 or y > 73:
                    continue
                # (白底黑字)True:黑色点,False:白色点
                if dst[x, y]:
                    c_o_s = check_over_sd(dst, x, y)
                    if not c_o_s:
                        new_array[x, y] = 255
                    else:
                        new_array[x, y] = 0
                else:
                    new_array[x, y] = 255
        Image.fromarray(new_array.astype('uint8')).save(MASK_PATH + "_1" + f)
    # return Image.fromarray(new_array.astype('uint8')).convert('RGB')


# 彩色，去掉底色与部分干扰线
def t2():
    files = os.listdir(CAPT_PATH)
    for index, f in enumerate(files):
        if index == 100:
            break
        # f="1PYFC_1555918464.3162677.jpg"
        # t1(CAPT_PATH + f).show()#save(SAVE_PATH+f)
        # continue
        img = cv2.imread(CAPT_PATH + f)

        # img = Image.open(CAPT_PATH + f)
        x_size, y_size, _colors = img.shape  # 30,75

        for y in range(y_size):
            for x in range(x_size):
                if x < 5 or x >= x_size-2 or y < 7 or y >= 73:
                    # img.putpixel((x, y), (255, 255, 255))
                    img[x, y] = [255, 255, 255]
                else:
                    data = img[x, y]  # 打印该图片的所有点
                    if (data[0] >= 180 and data[1] >= 180 and data[2] >= 180):  # RGBA的r值大于100，并且g值小于90,并且b值小于90
                        img[x, y] = [255, 255, 255]
        # 保存去除底色的彩色图片
        # cv2.imwrite("C:/Users/LENOVO/Desktop/pandas/capt/sd/" + f.split("_")[0] + ".png", img)
        def del_noice_point(direction="right"):
            for x in range(5, x_size - 1):
                for y in range(7, 73):
                    if list(img[x, y]) != [255, 255, 255]:
                        data_current = [int(i) for i in img[x, y]]
                        data_left = [int(i) for i in img[x, y - 1]]
                        data_right = [int(i) for i in img[x, y + 1]]
                        data_up = [int(i) for i in img[x - 1, y]]
                        data_down = [int(i) for i in img[x + 1, y]]
                        if abs(data_up[0] - data_current[0]) < 80 and abs(data_up[1] - data_current[1]) < 80 and abs(
                                data_up[2] - data_current[2]) < 80:
                            continue
                        if abs(data_down[0] - data_current[0]) < 80 and abs(data_down[1] - data_current[1]) < 80 and abs(
                                data_down[2] - data_current[2]) < 80:
                            continue
                        if direction=="right":
                            if abs(data_right[0] - data_current[0]) < 80 and abs(data_right[1] - data_current[1]) < 80 and abs(
                                    data_right[2] - data_current[2]) < 80:
                                if abs(data_left[0] - data_current[0]) < 80 and abs(
                                        data_left[1] - data_current[1]) < 80 and abs(data_left[2] - data_current[2]) < 80:
                                    continue
                                else:
                                    img[x, y] = [255, 255, 255]
                        else:
                            if abs(data_left[0] - data_current[0]) < 80 and abs(
                                    data_left[1] - data_current[1]) < 80 and abs(data_left[2] - data_current[2]) < 80:
                                if abs(data_right[0] - data_current[0]) < 80 and abs(data_right[1] - data_current[1]) < 80 and abs(
                                        data_right[2] - data_current[2]) < 80:
                                    continue
                                else:
                                    img[x, y] = [255, 255, 255]

                        img[x, y] = [255, 255, 255]


        del_noice_point("right")
        del_noice_point("left")

        cv2.imwrite(MASK_PATH + f.split("_")[0] + ".jpg", img)
        # cv2.imshow("11", img)
        # cv2.waitKey()

# 彩色，对t2继续降噪
def t3():
    def check_over_t3(chack_array, x, y, from_left=True):
        # True:黑色点
        # False:白色点
        if from_left:
            right = bool(chack_array[x, y + 1])
            up = bool(chack_array[x - 1, y])
            down = bool(chack_array[x + 1, y])
            if up == down == right == False:
                return False
            else:
                return True
        else:
            left = bool(chack_array[x, y - 1])
            up = bool(chack_array[x - 1, y])
            down = bool(chack_array[x + 1, y])
            if up == down == left == False:
                return False
            else:
                return True


    files = os.listdir(MASK_PATH)
    for index, f in enumerate(files):
        img = Image.open(MASK_PATH+f)
        imgry = img.convert('L')
        # print(img.size)  #(75, 30)
        x_size, y_size = img.size
        new_np = np.array(imgry)
        new_np = (new_np//10*10)
        th1 = (new_np < 210)
        dst = morphology.remove_small_objects(th1, min_size=20, connectivity=2)


        for _y in range(y_size):
            for _x in range(x_size):
                if dst[_y, _x]:
                    if _x == 0 or _y == 0 or _x == x_size - 1 or _y == y_size - 1:
                        dst[_y, _x] = False
                    else:
                        dst[_y, _x] = check_over_t3(dst, _y, _x, True)

        dst = morphology.remove_small_objects(dst, min_size=20, connectivity=2)

        for _x in range(x_size - 2, 0, -1):
            point_count = []
            for _y in range(y_size - 2, 0, -1):
                if dst[_y, _x]:
                    get_result = check_over_t3(dst, _y, _x, True)
                    if get_result:
                        point_count.append(_y)
                    dst[_y, _x] = get_result
            if len(point_count) == 1:
                dst[point_count[0], _x] = False

        dst = morphology.remove_small_objects(dst, min_size=20, connectivity=2)

        # for _y in range(y_size):
        #     _y_list = []
        #     for _x in range(x_size):
        #         if dst[_y, _x]:
        #             _y_list.append("*")
        #         else:
        #             _y_list.append(" ")
        #     print(_y_list)
        # print(binary.getpixel((x, y)))
        # putpixel

        for _y in range(y_size):
            for _x in range(x_size):
                if not dst[_y, _x]:
                    img.putpixel((_x, _y),(255,255,255))
        img.save(MASK_PATH+f)




def function(img,_value,_min,connectivity = 2):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, th1 = cv2.threshold(gray, _value, 255, cv2.THRESH_BINARY)
        bwimg = (th1 < 10)
        dst = morphology.remove_small_objects(bwimg, min_size=_min, connectivity=connectivity)
        x_shape, y_shape = bwimg.shape
        for y in range(y_shape):
            for x in range(x_shape):
                if dst[x, y] == True:
                    th1[x, y] = 255
        for y in range(y_shape):
            for x in range(x_shape):
                if th1[x, y] == 0:
                    img[x, y] = (255, 255, 255)
        return img


def change_black():
    files = os.listdir(SAVE_PATH)
    for index, f in enumerate(files):
        if ".png" in f:
            continue

        img = Image.open(SAVE_PATH + f)
        imgry = img.convert('L')
        # imgry.show()
        THRESHOLD = 210
        SH_LUT = [0] * THRESHOLD + [1] * (256 - THRESHOLD)

        # 转二值图
        binary = imgry.point(SH_LUT, '1')
        binary.save(SAVE_PATH + f)


        # cv2.imshow("3", th1)
        # cv2.waitKey()


def test():
    img = cv2.imread(r"C:\Users\LENOVO\Desktop\pandas\capt\ShanDong_1\1YNV1_1.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, th1 = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
    cv2.imshow("0", th1)

    bwimg = (th1<10)
    cv2.imshow("1", th1)
    # cv2.waitKey()
    dst = morphology.remove_small_objects(bwimg, min_size=20, connectivity=2)
    x_shape, y_shape = bwimg.shape
    for y in range(y_shape):
        for x in range(x_shape):
            if dst[x, y] == True:
                th1[x, y] = 255
    cv2.imshow("2", th1)
    for y in range(y_shape):
        for x in range(x_shape):
            if th1[x, y] == 0:
                img[x, y] = (255,255,255)
    cv2.imshow("3", img)
    cv2.waitKey()


# 多个小图片拼接切大图
def image_compose(pic_list):

    to_image = Image.new('RGB', (100, 22), "#FFFFFF")  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    total_width = 0
    for x in range(len(pic_list)):
        from_image = pic_list[x]
        if x != 0:
            total_width = total_width + 5
        to_image.paste(from_image, (total_width, 0))
        total_width += from_image.size[0]
    return to_image


def tt():
    files = os.listdir(SAVE_PATH)
    for index, f in enumerate(files):
        if ".png" in f:
            continue
        p = Image.open(SAVE_PATH + f)
        # 灰度
        imgry = p.convert('L')

        THRESHOLD = 155
        SH_LUT = [0] * THRESHOLD + [1] * (256 - THRESHOLD)

        # 转二值图
        binary = imgry.point(SH_LUT, '1')

        # 转二维数组
        char_array = np.array(binary) + 0
        # 获取维度
        x_shape, _ = char_array.shape

        img, x_list = Noise.ergodic_pic(char_array, p)
        # img.show()
        return_list, cut_list, return_image_list = coherence.coherence_function(img, 155)

        if cut_list == []:
            continue
        ii = image_compose(return_list)
        ii.show()
        # for i in return_list:
        #     i.show()
        # img.show()

# tt()
# change_black()
# test()
t2()
t3()
# img = Image.open("C:/Users/LENOVO/Desktop/pandas/capt/ShanDong/1CBRW_1555918532.8117578.jpg")
# img = img.convert('RGBA')
# print(np.array(img))