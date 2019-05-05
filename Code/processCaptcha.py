#!/usr/bin/python
# -*-coding:utf-8-*-

from PIL import Image
import pytesseract
import os, cv2, numpy as np
from Code import Noise, coherence, extractColor

THRESHOLD_dict = {"湖南": {"LUT": 130, "data": "num"},
                  "广东": {"LUT": 150, "data": "gd"},
                  "福建": {"LUT": 130, "data": "num"},
                  "云南": {"LUT": 120, "data": "num"},
                  "山西": {"LUT": 130, "data": "num"},
                  "贵州": {"LUT": 130, "data": "gz"},
                  "青海": {"LUT": 200, "data": "eng"},
                  "黑龙江": {"LUT": 200, "data": "eng"},
                  "吉林": {"LUT": 50, "data": "jl"},
                  "北京": {"LUT": 140, "data": "bj"},
                  "浙江": {"LUT": 155, "data": "zj"},
                  "江西": {"LUT": 130, "data": "num"}}



# 多个小图片拼接切大图
def image_compose(pic_list):
    to_image = Image.new('RGB', (25 * len(pic_list), 22), "#FFFFFF")  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for x in range(len(pic_list)):
        from_image = pic_list[x]
        to_image.paste(from_image, (x * 22 + 3, 0))
    return to_image


# 去边框
def delFrame(img, threashold):
    pixdata = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            if y == 0 or y == h - 1 or x == 0 or x == w - 1:
                pixdata[x, y] = 255
            elif pixdata[x, y] < threashold:
                pixdata[x, y] = 0
            else:
                pixdata[x, y] = 255
    return img


# 降噪
def noice(binary):  # 二值图
    char_array = np.array(binary)
    x_shape, y_shape = char_array.shape
    new_array = np.empty(shape=[x_shape, y_shape])


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
                new_array[x, y] = a
                if a == True:
                    new_array[x, y] = 255
                else:
                    new_array[x, y] = 0
            else:
                new_array[x, y] = 255
    return Image.fromarray(new_array.astype('uint8')).convert('L')


# 基本得二值化
def handleHuNanCaptcha(province, imagePath):
    THRESHOLD = THRESHOLD_dict[province]["LUT"]
    LUT = [0] * THRESHOLD + [1] * (256 - THRESHOLD)
    image = Image.open(imagePath)
    image = image.convert('L')
    image = image.point(LUT, '1')
    result_pre = pytesseract.image_to_string(image, lang=THRESHOLD_dict[province]["data"], config="-psm 7")
    result = ''.join(list(filter(str.isalnum, result_pre)))
    return result


# 吉林
def JiLin(province, imagePath):
    # 吉林降噪方法
    def noice_jl(chack_array, x, y, x_all, y_all):
        # False 0:黑色点
        # True 1:白色点
        if y == 0:
            left = 1
            right = 1 if bool(chack_array[x, y + 1]) == True else 0
        elif y == y_all - 1:
            right = 1
            left = 1 if bool(chack_array[x, y - 1]) == True else 0
        else:
            left = 1 if bool(chack_array[x, y - 1]) == True else 0
            right = 1 if bool(chack_array[x, y + 1]) == True else 0
        if x == 0:
            up = 1
            down = 1 if bool(chack_array[x + 1, y]) == True else 0
        elif x == x_all - 1:
            down = 1
            up = 1 if bool(chack_array[x - 1, y]) == True else 0
        else:
            up = 1 if bool(chack_array[x - 1, y]) == True else 0
            down = 1 if bool(chack_array[x + 1, y]) == True else 0

        if up + down + left + right > 3:
            return True
        else:
            return False


    # 吉林填充方法
    def fill_jl(chack_array, x, y, x_all, y_all):
        # False 0:黑色点
        # True 1:白色点
        if y == 0:
            left = 1
            right = 1 if bool(chack_array[x, y + 1]) == True else 0
        elif y == y_all - 1:
            right = 1
            left = 1 if bool(chack_array[x, y - 1]) == True else 0
        else:
            left = 1 if bool(chack_array[x, y - 1]) == True else 0
            right = 1 if bool(chack_array[x, y + 1]) == True else 0
        if x == 0:
            up = 1
            down = 1 if bool(chack_array[x + 1, y]) == True else 0
        elif x == x_all - 1:
            down = 1
            up = 1 if bool(chack_array[x - 1, y]) == True else 0
        else:
            up = 1 if bool(chack_array[x - 1, y]) == True else 0
            down = 1 if bool(chack_array[x + 1, y]) == True else 0

        if up + down + left + right <= 1:
            return False
        else:
            return True


    def function_jl(binary):
        char_array = np.array(binary)
        x_shape, y_shape = char_array.shape  # x_shape行数，y_shape列数

        new_array = np.empty(shape=[x_shape, y_shape])
        for y in range(y_shape):
            for x in range(x_shape):
                if 3 <= y <= 13 or 19 <= y <= 29 or 35 <= y <= 45 or 50 <= y <= 60:
                    # (白底黑字)False:黑色点,True:白色点
                    if char_array[x, y] == True:
                        a = fill_jl(char_array, x, y, x_shape, y_shape)
                        char_array[x, y] = a
                        if a == True:
                            new_array[x, y] = 255
                        else:
                            new_array[x, y] = 0
                else:
                    char_array[x, y] = True
                    new_array[x, y] = 255
        for y in range(y_shape):
            for x in range(x_shape):
                # (白底黑字)False:黑色点,True:白色点
                if char_array[x, y] == False:
                    a = noice_jl(char_array, x, y, x_shape, y_shape)
                    char_array[x, y] = a
                    if a:
                        new_array[x, y] = 255
                    else:
                        new_array[x, y] = 0
        return Image.fromarray(new_array.astype('uint8')).convert('1')


    THRESHOLD = THRESHOLD_dict[province]["LUT"]
    LUT = [0] * THRESHOLD + [1] * (256 - THRESHOLD)
    image = Image.open(imagePath)
    image = image.convert('L')
    image = image.point(LUT, '1')
    noice_image = function_jl(image)
    result_pre = pytesseract.image_to_string(noice_image, lang=THRESHOLD_dict[province]["data"], config="-psm 7")
    result = ''.join(list(filter(str.isalnum, result_pre)))
    return result


# 陕西使用去除边框
def ShanXi(province, imagePath):
    image = Image.open(imagePath)
    image = image.convert('L')
    threshold = THRESHOLD_dict[province]["LUT"]
    image = delFrame(image, threshold)
    result_pre = pytesseract.image_to_string(image, lang='num', config="-psm 7")
    result = ''.join(list(filter(str.isalnum, result_pre)))
    return result


# 福建使用，去除指定范围内的噪点
def FuJian(province, imagePath):
    # 福建使用按照颜色提取
    def getcolor(img):
        width = img.size[0]  # 长度
        height = img.size[1]  # 宽度
        for i in range(0, width):  # 遍历所有长度的点
            for j in range(0, height):  # 遍历所有宽度的点
                data = (img.getpixel((i, j)))  # 打印该图片的所有点
                if (data[0] >= 100 and data[1] < 90 and data[2] < 90):  # RGBA的r值大于100，并且g值小于90,并且b值小于90
                    img.putpixel((i, j), (0, 0, 0, 255))  # 则这些像素点的颜色改成黑色
                else:
                    img.putpixel((i, j), (255, 255, 255, 255))  # 则这些像素点的颜色改成白色
        img = img.convert("RGB")  # 把图片强制转成RGB
        return img


    image = Image.open(imagePath)
    image = getcolor(image)
    image = image.convert('L')
    threshold = THRESHOLD_dict[province]["LUT"]
    image = delFrame(image, threshold)
    result_pre = pytesseract.image_to_string(image, lang='num', config="-psm 7")
    result = ''.join(list(filter(str.isalnum, result_pre)))
    return result


# 云南使用方法
def YunNan(province, imagePath):
    THRESHOLD = THRESHOLD_dict[province]["LUT"]


    def m_function():
        LUT = [0] * THRESHOLD + [1] * (256 - THRESHOLD)
        image = Image.open(imagePath)
        # 转灰度图
        imgry = image.convert('L')
        # 转二值图
        binary = imgry.point(LUT, '1')

        char_array = np.array(binary)
        x_shape, y_shape = char_array.shape

        new_array = np.empty(shape=[x_shape, y_shape])
        for y in range(y_shape):
            for x in range(x_shape):
                if y == 0 or y == y_shape - 1 or x == 0:
                    new_array[x, y] = 255
                    # 防止字母部分与黑色边框重叠部分别剪切，进行灰度值判断
                elif 12 <= y <= 16 or 27 <= y <= 30 or 41 <= y <= 45 or x <= 3:
                    new_array[x, y] = 255
                else:
                    if x == x_shape - 1 and imgry.getpixel((y, x)) < 50:
                        new_array[x, y] = 255
                    else:
                        new_array[x, y] = char_array[x, y]
                        # (白底黑字)False:黑色点,True:白色点
                        if char_array[x, y] == False:
                            new_array[x, y] = 0
                        else:
                            new_array[x, y] = 255
        return Image.fromarray(new_array.astype('uint8')).convert('1')


    image = m_function()
    result_pre = pytesseract.image_to_string(image, lang='eng', config="-psm 7")
    result = ''.join(list(filter(str.isalnum, result_pre)))
    return result


# 山西
def Shan1Xi(province, imagePath):
    THRESHOLD = THRESHOLD_dict[province]["LUT"]


    def sx_function():
        image = Image.open(imagePath)
        width = image.size[0]  # 长度
        height = image.size[1]  # 宽度
        for i in range(0, width):  # 遍历所有长度的点
            for j in range(0, height):  # 遍历所有宽度的点
                data = (image.getpixel((i, j)))  # 打印该图片的所有点
                if j == 0:
                    if (data[0] >= 70 or data[1] > 70):  # RGBA的r值大于100，并且g值小于90,并且b值小于90
                        continue
                    else:
                        image.putpixel((i, j), (255, 255, 255, 255))  # 则这些像素点的颜色改成白色
                elif i == 0 or i == width - 1 or j == height - 1:
                    image.putpixel((i, j), (255, 255, 255, 255))  # 则这些像素点的颜色改成白色
        # img = image.convert("RGB")  # 把图片强制转成RGB
        LUT = [0] * THRESHOLD + [1] * (256 - THRESHOLD)
        image = Image.open(imagePath)
        # 转灰度图
        imgry = image.convert('L')
        # 转二值图
        binary = imgry.point(LUT, '1')
        return binary


    image = sx_function()
    result_pre = pytesseract.image_to_string(image, lang='eng', config="-psm 7")
    result = ''.join(list(filter(str.isalnum, result_pre)))
    return result


# 浙江
def ZheJiang(province, imagePath):

    # 降噪
    def check_over_zj(chack_array, x, y):
        # False:黑色点
        # True:白色点
        left = bool(chack_array[x, y - 1])
        right = bool(chack_array[x, y + 1])

        up = bool(chack_array[x - 1, y])
        down = bool(chack_array[x + 1, y])

        if (up == down == True) or (left == right == True):
            return True
        else:
            return False

    img = Image.open(imagePath)

    x_size, y_size = img.size  # 52,26
    for y in range(y_size):
        for x in range(x_size):
            if y < 8 or y > 19:
                img.putpixel((x, y), (255, 255, 255))
            elif 8 <= y <= 19 and (4 <= x <= 10 or 13 <= x <= 19 or 22 <= x <= 28 or 31 <= x <= 37):
                continue
            else:
                img.putpixel((x, y), (255, 255, 255))

    # 灰度
    imgry = img.convert('L')
    THRESHOLD = THRESHOLD_dict[province]["LUT"]
    ZJ_LUT = [0] * THRESHOLD + [1] * (256 - THRESHOLD)

    # 转二值图
    binary = imgry.point(ZJ_LUT, '1')

    char_array = np.array(binary)
    x_shape, y_shape = char_array.shape

    new_array = np.empty(shape=[x_shape, y_shape])
    for y in range(y_shape):
        for x in range(x_shape):
            # (白底黑字)False:黑色点,True:白色点
            if char_array[x, y] == False:
                a = check_over_zj(char_array, x, y)
                new_array[x, y] = a
                if a == True:
                    new_array[x, y] = 255
                else:
                    new_array[x, y] = 0
            else:
                new_array[x, y] = 255
    for y in range(y_shape):
        for x in range(x_shape):
            # (白底黑字)False:黑色点,True:白色点
            if char_array[x, y] == False:
                a = check_over_zj(char_array, x, y)
                new_array[x, y] = a
                if a == True:
                    new_array[x, y] = 255
                else:
                    new_array[x, y] = 0
            else:
                new_array[x, y] = 255
    result_pre = pytesseract.image_to_string(Image.fromarray(new_array.astype('uint8')).convert('1'), lang='zj', config="-psm 7")
    result = ''.join(list(filter(str.isalnum, result_pre)))
    return result


# 北京使用的单独模型


model = cv2.ml.KNearest_create()
with np.load(r"C:\Users\LENOVO\Desktop\pandas\capt\write.npz") as data:
    train = data['train']
    train_labels = data['train_labels']
    id_label_map = data['id_label']
    model.train(train, cv2.ml.ROW_SAMPLE, train_labels)
    id_label_map = id_label_map.tolist()


def BeiJing(province, imagePath):
    THRESHOLD = THRESHOLD_dict[province]["LUT"]
    img = Image.open(imagePath)

    ii = Noise.handle_pic(img, 4, THRESHOLD)
    return_lable = ""
    for box in ii:
        box = box.resize((22, 22), Image.ANTIALIAS)
        box.save("./123.jpg")

        im = cv2.imread("./123.jpg", cv2.IMREAD_GRAYSCALE)
        sample = im.reshape((1, 484)).astype(np.float32)
        ret, results, neighbours, distances = model.findNearest(sample, k=3)
        label_id = int(results[0][0])
        label = id_label_map[label_id]
        os.remove("./123.jpg")
        return_lable += label
        # box.show()
    return return_lable


def ShangHai():
    CAPT_PATH = "C:/Users/LENOVO/Desktop/pandas/capt/ShangHai_test/"
    files = os.listdir(CAPT_PATH)
    total_count = error_count = 0

    for index, f in enumerate(files):
        total_count += 1
        p = Image.open(CAPT_PATH + f)
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
        # 返回没有背景的彩色图片,删除边框以及像素连接小于等于1个的认为是分离
        img, x_list = Noise.ergodic_pic_ShangHai(char_array, p)

        return_list, cut_list, return_image_list = coherence.coherence_function(img, 155)

        # print(cut_list[0])
        # 使用联通法分割成功，进行匹配
        if len(cut_list[0]) == 4:
            sh_image = image_compose(return_list)
            result_pre = pytesseract.image_to_string(sh_image, lang='sh', config="-psm 7")
            result = ''.join(list(filter(str.isalnum, result_pre)))
        # 否则使用颜色聚类方法进行分割没有分开的部分图片
        elif len(cut_list[0]) == 3:
            max_width = max_index = 0
            for _index, _i in enumerate(cut_list[0]):
                chrrent_width = _i[1] - _i[0]
                if chrrent_width > max_width:
                    max_index = _index
                    max_width = chrrent_width
            use_pic = return_image_list[max_index]
            get_pic_list = extractColor.handle_pic(use_pic, 3)

            add_pic = {}
            for i in get_pic_list:
                cut_pic, cut_index = Noise.colur_cut(i)
                if max_width - cut_pic.size[0] < 3:
                    continue
                add_pic[cut_index] = cut_pic
            return_pic = []
            for m_index, m_i in enumerate(return_list):
                if m_index == max_index:
                    if "1" in add_pic.keys():
                        return_pic.append(add_pic["1"])
                    if "2" in add_pic.keys():
                        return_pic.append(add_pic["2"])
                else:
                    return_pic.append(m_i)

            sh_image = image_compose(return_pic)
            result_pre = pytesseract.image_to_string(sh_image, lang='sh', config="-psm 7")
            result = ''.join(list(filter(str.isalnum, result_pre)))
        else:
            error_count += 1
        if result != f.split("_")[0]:
            error_count += 1
        print("result:{},file:{}".format(result, f.split("_")[0]))
    print("total_count:{} , error_count:{}".format(total_count, error_count))


def handleCaptcha(province, imagePath):
    if province == "陕西":
        ret = ShanXi(province, imagePath)
    elif province == "福建":
        ret = FuJian(province, imagePath)
    elif province == "云南":
        ret = YunNan(province, imagePath)
    elif province == "山西":
        ret = Shan1Xi(province, imagePath)
    elif province == "吉林":
        ret = JiLin(province, imagePath)
    elif province == "北京":
        ret = BeiJing(province, imagePath)
    elif province == "浙江":
        ret = ZheJiang(province, imagePath)
    else:  # ["湖南", "重庆", "甘肃", "湖北", "宁夏", "新疆", "贵州", "江西"]
        ret = handleHuNanCaptcha(province, imagePath)
    return ret


test_dict = {"湖南": "HuNan", "重庆": "chongqing", "甘肃": "Gansu", "湖北": "hubei", "宁夏": "ningxia", "新疆": "xinjiang",
             "陕西": "shanxi"}

test_dict1 = {"浙江": "ZheJiang_test"}


# 测试准确度
def run():
    for key, value in test_dict1.items():
        error_count = 0
        total_count = 0
        _path = "C:/Users/LENOVO/Desktop/pandas/capt/{}/".format(value)
        files = os.listdir(_path)

        for index, f in enumerate(files):
            show_name = f.split("_")[0]
            # if index % 10 != 0:
            #     continue
            total_count += 1
            ret = handleCaptcha(key, _path + f)

            if show_name != ret:
                error_count += 1
                print(show_name, ret)
        print("{} total:{},error count:{}".format(key, total_count, error_count))


def run_save():
    for key, value in test_dict1.items():
        total_count = 0
        _path = "./capt/{}/".format(value)
        files = os.listdir(_path)
        for index, f in enumerate(files):
            if len(f) > 10:
                total_count += 1
                print(total_count)
                ret = handleCaptcha(key, _path + f)
                try:
                    os.rename(_path + f, "{}\{}.jpg".format(_path, ret))
                except:
                    continue
        print("{} total:{}".format(key, total_count))


# test_dict1={"天津": "tianjin"}
if __name__ == "__main__":
    run()
    # path = "C:/Users/LENOVO/Desktop/pandas/capt/ShangHai_1/black/"
    # files = os.listdir(path)
    # total_count = error_count = 0
    #
    # for index, f in enumerate(files):
    #     total_count+=1
    #     image = Image.open(path+f)
    #     result_pre = pytesseract.image_to_string(image, lang='eng', config="-psm 7")
    #     result = ''.join(list(filter(str.isalnum, result_pre)))
    #     _name = f.split("_")[0]
    #     if _name != result:
    #         error_count+=1
    #         print(_name , result)
    # print(total_count , error_count)
