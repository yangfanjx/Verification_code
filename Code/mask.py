from PIL import Image
import numpy as np
from skimage import io, morphology
import os, cv2
from Code import Noise, coherence, extractColor


# print(binary.getpixel((x, y)))
# putpixel
# 将文件夹中的全部图片转为不带噪点的二值图片

THRESHOLD = 158
LUT = [0] * THRESHOLD + [1] * (256 - THRESHOLD)

CAPT_PATH = "C:/Users/LENOVO/Desktop/pandas/capt/beijing/"
SAVE_PATH = "C:/Users/LENOVO/Desktop/pandas/capt/beijing_test/"


def yield_one():
    f = "92B7_1555553298.4571493.jpg"
    img = Image.open(r"C:\Users\LENOVO\Desktop\pandas\capt\ShangHai\{}".format(f))
    yield f, img


def yield_folder():
    files = os.listdir(CAPT_PATH)
    for index, f in enumerate(files):
        if index % 100 != 0:
            continue
        # train_labels += list(f.split("_")[0])
        img = Image.open(CAPT_PATH + f)
        yield f, img


# 降噪
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


# 通用方法
def main():
    for name, p in yield_folder():
        # 转灰度图
        imgry = p.convert('L')

        # 转二值图
        binary = imgry.point(LUT, '1')

        # +0为了将原二维数组中bool类型元素全部转为0，1形式
        # char_array = np.array(binary) + 0
        char_array = np.array(binary)
        x_shape, y_shape = char_array.shape

        new_array = np.empty(shape=[x_shape, y_shape])
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

        Image.fromarray(new_array.astype('uint8')).convert('1').save(SAVE_PATH + name)


# 福建
def main_fj():
    # 按照颜色提取图像
    def fj_function(img):
        width = img.size[0]  # 长度
        height = img.size[1]  # 宽度
        for i in range(0, width):  # 遍历所有长度的点
            for j in range(0, height):  # 遍历所有宽度的点
                data = (img.getpixel((i, j)))  # 打印该图片的所有点
                if (data[0] >= 90 and data[1] < 90 and data[2] < 90):  # RGBA的r值大于100，并且g值小于90,并且b值小于90
                    img.putpixel((i, j), (0, 0, 0, 255))  # 则这些像素点的颜色改成黑色
                else:
                    img.putpixel((i, j), (255, 255, 255, 255))  # 则这些像素点的颜色改成白色
        img = img.convert("RGB")  # 把图片强制转成RGB
        return img


    for name, p in yield_folder():
        p = fj_function(p)
        # 转灰度图
        imgry = p.convert('L')
        # 转二值图
        binary = imgry.point(LUT, '1')
        # +0为了将原二维数组中bool类型元素全部转为0，1形式
        # char_array = np.array(binary) + 0
        char_array = np.array(binary)
        x_shape, y_shape = char_array.shape

        new_array = np.empty(shape=[x_shape, y_shape])
        for y in range(y_shape):
            for x in range(x_shape):
                if y < 10 or y > 62 or x < 8 or x > 17 or y in [18, 27, 36, 45, 54]:
                    new_array[x, y] = 255
                else:
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
        Image.fromarray(new_array.astype('uint8')).convert('1').save(SAVE_PATH + name)


# 云南
def main_yn():
    for name, p in yield_folder():
        # 转灰度图
        imgry = p.convert('L')
        # 转二值图
        binary = imgry.point(LUT, '1')

        # +0为了将原二维数组中bool类型元素全部转为0，1形式
        # char_array = np.array(binary) + 0
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
        Image.fromarray(new_array.astype('uint8')).convert('1').save(SAVE_PATH + name)


# 山西
def main_sx():
    # 按照颜色提取图像
    def sx_function(img):
        width = img.size[0]  # 长度
        height = img.size[1]  # 宽度
        for i in range(0, width):  # 遍历所有长度的点
            for j in range(0, height):  # 遍历所有宽度的点
                data = (img.getpixel((i, j)))  # 打印该图片的所有点
                if j == 0:
                    if (data[0] >= 70 or data[1] > 70):  # RGBA的r值大于100，并且g值小于90,并且b值小于90
                        continue
                    else:
                        img.putpixel((i, j), (255, 255, 255, 255))  # 则这些像素点的颜色改成白色
                elif i == 0 or i == width - 1 or j == height - 1:
                    img.putpixel((i, j), (255, 255, 255, 255))  # 则这些像素点的颜色改成白色
        img = img.convert("RGB")  # 把图片强制转成RGB
        return img


    for name, p in yield_folder():
        p = sx_function(p)

        # 转灰度图
        imgry = p.convert('L')

        # 转二值图
        binary = imgry.point(LUT, '1')
        # print(SAVE_PATH + name)
        binary.save(SAVE_PATH + name)


# 吉林
def main_jl():
    def check_over_test(chack_array, x, y, x_all, y_all):
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


    def check_over_test2(chack_array, x, y, x_all, y_all):
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


    for name, p in yield_folder():
        # 转灰度图
        imgry = p.convert('L')

        # 转二值图
        binary = imgry.point(LUT, '1')

        char_array = np.array(binary)
        x_shape, y_shape = char_array.shape  # x_shape行数，y_shape列数

        new_array = np.empty(shape=[x_shape, y_shape])
        for y in range(y_shape):
            for x in range(x_shape):
                if 3 <= y <= 13 or 19 <= y <= 29 or 35 <= y <= 45 or 50 <= y <= 60:
                    # (白底黑字)False:黑色点,True:白色点
                    if char_array[x, y] == True:
                        a = check_over_test2(char_array, x, y, x_shape, y_shape)
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
                    a = check_over_test(char_array, x, y, x_shape, y_shape)
                    char_array[x, y] = a
                    if a:
                        new_array[x, y] = 255
                    else:
                        new_array[x, y] = 0

        # for y in range(y_shape):
        #     for x in range(x_shape):
        #         if 3 <= y <= 13 or 19 <= y <= 29 or 35 <= y <= 45 or 50 <= y <= 60:
        #             # (白底黑字)False:黑色点,True:白色点
        #             if char_array[x, y] == False:
        #                 a = check_over_test(char_array, x, y, x_shape, y_shape)
        #                 char_array[x, y] = a
        #                 if a == True:
        #                     new_array[x, y] = 255
        #                 else:
        #                     new_array[x, y] = 0
        #             else:
        #                 char_array[x, y] = True
        #                 new_array[x, y] = 255
        #         else:
        #             new_array[x, y] = 255

        Image.fromarray(new_array.astype('uint8')).convert('1').save(SAVE_PATH + name)


# 去边框
def delFrame(img):
    pixdata = img.load()
    w, h = img.size
    for y in range(h):
        for x in range(w):
            if y == 0 or y == h - 1 or x == 0 or x == w - 1:
                pixdata[x, y] = 255
    return img



# 没有背景的彩色图片
def BJ_no_bakground():
    for name, p in yield_one():
        # 转灰度图
        imgry = p.convert('L')

        # 转二值图
        binary = imgry.point(LUT, '1')
        char_array = np.array(binary) + 0
        x_shape, _ = char_array.shape
        pic, x_list = Noise.ergodic_pic(char_array, p)
        return pic
        # pic.save(SAVE_PATH + name)


# 北京验证码图片重新拼接为可用图片
def BJ_split_pic():
    # 分割北京为单个字母图片
    CAPT_PATH = "C:/Users/LENOVO/Desktop/pandas/capt/beijing/"
    SAVE_PATH = "C:/Users/LENOVO/Desktop/pandas/capt/beijing_test/"


    def image_compose(pic_list):
        to_image = Image.new('RGB', (100, 20), "#FFFFFF")  # 创建一个新图
        # 循环遍历，把每张图片按顺序粘贴到对应位置上
        for x in range(4):
            from_image = pic_list[x]
            to_image.paste(from_image, (x * 25, 0))
        return to_image


    files = os.listdir(CAPT_PATH)
    for index, f in enumerate(files):
        # f="0005_1553484889070.jpg"
        # if index % 50 != 0:
        #     continue
        img = Image.open(CAPT_PATH + f)
        use_name = f.split("_")[0]
        ii = Noise.handle_pic_bak(img, 4)
        if len(ii) != 4:
            continue
        for _index, i in enumerate(ii):
            _i = i.resize((22, 22), Image.ANTIALIAS)
            _name = "{}{}_{}".format(SAVE_PATH, use_name[_index], f)
            _i.save(_name)
        # aaa = image_compose(ii)
        # _name = SAVE_PATH+f
        # aaa.save(_name)


# 北京使用的训练与测试方法，使用切割完的单独字母黑白图片
def BJ_study_function():
    use_path = "C:/Users/LENOVO/Desktop/pandas/capt/beijing_use/"
    samples = np.empty((0, 484))

    if os.path.exists(r"C:\Users\LENOVO\Desktop\pandas\capt\write.npz"):
        model = cv2.ml.KNearest_create()
        with np.load(r"C:\Users\LENOVO\Desktop\pandas\capt\write.npz") as data:
            train = data['train']
            train_labels = data['train_labels']
            id_label_map = data['id_label']
            model.train(train, cv2.ml.ROW_SAMPLE, train_labels)
            id_label_map = id_label_map.tolist()

    else:
        labels = []
        files = os.listdir(SAVE_PATH)
        for index, f in enumerate(files):
            filepath = SAVE_PATH + f
            label = f.split("_")[0]
            labels.append(label)
            im = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
            sample = im.reshape((1, 484)).astype(np.float32)
            samples = np.append(samples, sample, 0)
            samples = samples.astype(np.float32)

        unique_labels = list(set(labels))
        unique_ids = list(range(len(unique_labels)))
        label_id_map = dict(zip(unique_labels, unique_ids))
        id_label_map = dict(zip(unique_ids, unique_labels))
        label_ids = list(map(lambda x: label_id_map[x], labels))
        label_ids = np.array(label_ids).reshape((-1, 1)).astype(np.float32)
        model = cv2.ml.KNearest_create()
        model.train(samples, cv2.ml.ROW_SAMPLE, label_ids)
        # id_label_map = id_label_map.tolist()
        print("start ok")
        # 保存模型
        np.savez(r"C:\Users\LENOVO\Desktop\pandas\capt\write", train=samples, train_labels=label_ids,
                 id_label=id_label_map)
        # #调用模型
        print("write ok")

    files_use = os.listdir(use_path)
    error_count = 0
    total_count = 0
    for _indes, box in enumerate(files_use):
        if _indes > 2:
            break
        total_count += 1
        filepath_use = use_path + box
        _label = box.split("_")[0]
        img = Image.open(filepath_use)
        ii = Noise.handle_pic(img, 4, 140)
        names = ""
        for box in ii:
            box = box.resize((22, 22), Image.ANTIALIAS)
            box.save("./123.jpg")

            im = cv2.imread("./123.jpg", cv2.IMREAD_GRAYSCALE)

            sample = im.reshape((1, 484)).astype(np.float32)
            ret, results, neighbours, distances = model.findNearest(sample, k=3)
            label_id = int((results[0, 0]))
            # label_id = int(results[0][0])
            label = id_label_map[label_id]

            os.remove("./123.jpg")
            names += label
        print(_label, names)
        if _label != names:
            error_count += 1

        #
        # im = cv2.imread(filepath_use, cv2.IMREAD_GRAYSCALE)
        # sample = im.reshape((1, 484)).astype(np.float32)
        # ret, results, neighbours, distances = model.findNearest(sample, k=3)
        # label_id = int(results[0, 0])
        # label = id_label_map[label_id]
        # if label!=_label:
        #     print(label,_label)
        # else:
        #     os.remove(filepath_use)
    print("total_count:{}, error_count:{}".format(total_count, error_count))


# 多个小图片拼接切大图
def image_compose(pic_list):
    to_image = Image.new('RGB', (25 * len(pic_list), 22), "#FFFFFF")  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for x in range(len(pic_list)):
        from_image = pic_list[x]
        to_image.paste(from_image, (x * 22 + 3, 0))
    return to_image


def ShangHai2():
    CAPT_PATH = "C:/Users/LENOVO/Desktop/pandas/capt/ShangHai/"
    SAVE_PATH = "C:/Users/LENOVO/Desktop/pandas/capt/ShangHai_1/"
    files = os.listdir(CAPT_PATH)
    for index, f in enumerate(files):

        # if index == 1:
        #     break
        # f = "ERR8_1555553317.806918.jpg"
        p = Image.open(CAPT_PATH + f)
        # 灰度
        imgry = p.convert('L')

        THRESHOLD=155
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
            image_compose(return_list).save(SAVE_PATH + f)
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

            # pi=image_compose(return_pic)
            # pi.show()
            image_compose(return_pic).save(SAVE_PATH+f)


def ShangHai_test():
    from PIL import Image
    CAPT_PATH = "C:/Users/LENOVO/Desktop/pandas/capt/ShangHai/"
    SAVE_PATH = "C:/Users/LENOVO/Desktop/pandas/capt/ShangHai_1/"

    files = os.listdir(CAPT_PATH)
    for index, f in enumerate(files):
        p = Image.open(CAPT_PATH + f)
        # 灰度
        imgry = p.convert('L')
        # 转二值图
        binary = imgry.point(LUT, '1')
        # 转二维数组
        char_array = np.array(binary) + 0
        # 获取维度
        x_shape, _ = char_array.shape
        # 返回没有背景的彩色图片,删除边框以及像素连接小于等于1个的认为是分离
        img, x_list = Noise.ergodic_pic_ShangHai(char_array, p)
        img.save(SAVE_PATH + f)



def ZheJiang():
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


    #只有234568这些数字组成
    CAPT_PATH = "C:/Users/LENOVO/Desktop/pandas/capt/ZheJiang/"
    SAVE_PATH = "C:/Users/LENOVO/Desktop/pandas/capt/ZheJiang_1/"
    files = os.listdir(CAPT_PATH)
    for index, f in enumerate(files):
        img = Image.open(CAPT_PATH + f)
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

        THRESHOLD = 155
        SH_LUT = [0] * THRESHOLD + [1] * (256 - THRESHOLD)

        # 转二值图
        binary = imgry.point(SH_LUT, '1')

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
        Image.fromarray(new_array.astype('uint8')).convert('1').save(SAVE_PATH+f)
        # Image.fromarray(new_array.astype('uint8')).convert('1').show()

def HeNan():
    # print(binary.getpixel((x, y)))
    # putpixel
    CAPT_PATH = "C:/Users/LENOVO/Desktop/pandas/capt/HeBei1/"
    SAVE_PATH = "C:/Users/LENOVO/Desktop/pandas/capt/HeBei1/"
    files = os.listdir(CAPT_PATH)
    for index, f in enumerate(files):

        # if index % 100 != 0:
        #     continue

        p = Image.open(CAPT_PATH + f)

        im_size = p.size

        for i in range(im_size[0]):
            for j in range(im_size[1]):
                _data = p.getpixel((i, j))
                if i == 0 or j == 0:
                    p.putpixel((i, j), (255, 255, 255))
                elif _data[0] < 50 and _data[1] < 50 and _data[2] < 50:
                    p.putpixel((i, j), (0, 0, 0))
                else:
                    p.putpixel((i, j), (255, 255, 255))
        p.save(SAVE_PATH+f)

def HeBei():

    CAPT_PATH = "C:/Users/LENOVO/Desktop/pandas/capt/HeBei1/"
    files = os.listdir(CAPT_PATH)
    for index, f in enumerate(files):

        # if index % 100 != 0:
        #     continue

        p = Image.open(CAPT_PATH + f)

        im_size = p.size

        for i in range(im_size[0]):#80
            line_cunt = 0
            for j in range(im_size[1]):#30
                _data = p.getpixel((i, j))
                if list(_data) == [0, 0, 0]:
                    line_cunt+=1
                    continue
                else:
                    _end = j
                    if line_cunt<=2:
                        for n in range(_end-line_cunt,_end):
                            p.putpixel((i, n), (255, 255, 255))
        # p.save(SAVE_PATH+f)
        p.show()


def ShangDong():
    # 按照颜色提取
    def SD_getcolor(img):
        width, height, _colors = img.shape  # 长度
        for i in range(0, width):  # 遍历所有长度的点
            for j in range(0, height):  # 遍历所有宽度的点
                data = img[i, j]  # 打印该图片的所有点
                if (data[0] >=190 and data[1] >= 140 and data[2] < 60):  # RGBA的r值大于100，并且g值小于90,并且b值小于90
                    img[i, j] = (0, 0, 0)
        return img


    #删除只有一行像素的部分，认为是噪点
    def check_over_sd(chack_array, x, y):
        # True:黑色点
        # False:白色点
        # left = bool(chack_array[x, y - 1])
        # right = bool(chack_array[x, y + 1])

        up = chack_array[x - 1, y]
        down = chack_array[x + 1, y]

        if up == down == 255:
            return 255
        # elif left == right == True:
        #     return 255
        else:
            return 0


    # 二值化删除联通数量少于特定值的点
    def t1(pic_path="", del_point=20):
        # pic_path = r"C:\Users\LENOVO\Desktop\pandas\capt\ShanDong_1\2UMYC_1555918589.8860226.jpg"
        # img = io.imread(pic_path, as_gray=True)  # 变成灰度图
        img = io.imread(pic_path)
        # img = SD_getcolor(img)
        # pic = cv2.resize(img, (300, 120), interpolation=cv2.INTER_CUBIC)


        # 简单方式二值化
        # ret, th1 = cv2.threshold(img, 190, 255, cv2.THRESH_BINARY)
        # 留下小于阈值的部分，及黑的部分
        bwimg = (img <= 0.7)
        # cv2.imshow('123', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #第一次删除连接数小于特定值的像素
        # dst = morphology.remove_small_objects(bwimg, min_size=del_point, connectivity=1)

        # char_array = np.array(dst)
        char_array = np.array(bwimg)
        x_shape, y_shape = char_array.shape

        new_array = np.empty(shape=[x_shape, y_shape])
        for y in range(y_shape):
            for x in range(x_shape):
                if x < 6 or x > 26 or y < 7 or y > 73:
                    new_array[x, y] = 255
                else:
                    # (白底黑字)True:黑色点,False:白色点
                    if char_array[x, y]:
                        c_o_s = check_over_sd(char_array, x, y)
                        new_array[x, y] = c_o_s
                        if not c_o_s:
                            bwimg[x, y] = False
                            new_array[x, y] = 255
                        else:
                            bwimg[x, y] = True
                            new_array[x, y] = 0
                    else:
                        new_array[x, y] = 255
        Image.fromarray(new_array.astype('uint8')).convert('RGB').show()
        dst = morphology.remove_small_objects(bwimg, min_size=del_point, connectivity=1)
        for y in range(y_shape):
            for x in range(x_shape):
                # (白底黑字)True:黑色点,False:白色点
                if dst[x, y]:
                    c_o_s = check_over_sd(char_array, x, y)
                    new_array[x, y] = c_o_s
                    if not c_o_s:
                        new_array[x, y] = 255
                    else:
                        new_array[x, y] = 0
                else:
                    new_array[x, y] = 255
        return Image.fromarray(new_array.astype('uint8')).convert('RGB')
        # Image.fromarray(new_array.astype('uint8')).convert('RGB').show()


    CAPT_PATH = "C:/Users/LENOVO/Desktop/pandas/capt/ShanDong/"
    SAVE_PATH = "C:/Users/LENOVO/Desktop/pandas/capt/ShanDong_1/"
    files = os.listdir(CAPT_PATH)
    for index, f in enumerate(files):
        if index == 100:
            break
        # f="16JMB_1555918624.0964966.jpg"
        # t1(CAPT_PATH + f).show()#save(SAVE_PATH+f)
        # continue
        img = cv2.imread(CAPT_PATH + f)

        # img = Image.open(CAPT_PATH + f)
        x_size, y_size ,_colors= img.shape  # 30,75

        for y in range(y_size):
            for x in range(x_size):
                if x < 5  or y < 7 or y > 73:
                    # img.putpixel((x, y), (255, 255, 255))
                    img[x, y] = [255, 255, 255]
                else:
                    data = img[x, y]  # 打印该图片的所有点

                    if (data[0] >= 180 and data[1] >= 180 and data[2] >= 180):  # RGBA的r值大于100，并且g值小于90,并且b值小于90
                        img[x, y] = [255, 255, 255]
                    elif x < x_size - 1:
                        data_1 = img[x + 1, y]
                        if list(img[x - 1, y]) == [255, 255, 255] and (
                                data_1[0] >= 180 and data_1[1] >= 180 and data_1[2] >= 180):
                            img[x, y] = [255, 255, 255]
                    elif list(img[x - 1, y]) == [255, 255, 255]:
                        img[x, y] = [255, 255, 255]


        cv2.imwrite(SAVE_PATH+f.split("_")[0]+".png",img)
        continue
        cv2.imshow("OpenCV1", img)
        cv2.waitKey()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 简单
        ret, th1 = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)
        # print(th1)

        # 自适应
        # th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)


        #
        # # 灰度
        # imgry = img.convert('L')
        # # imgry.show()
        # THRESHOLD = 210
        # SH_LUT = [0] * THRESHOLD + [1] * (256 - THRESHOLD)
        #
        # # 转二值图
        # binary = imgry.point(SH_LUT, '1')
        # print(th1[0,10])
        # new_array = np.empty(shape=[x_shape, y_shape])
        for i in range(0, 3):
            for y in range(1,y_size-1):
                for x in range(1,x_size-1):
                    # (白底黑字)False:黑色点,True:白色点
                    if th1[x, y] == 0:
                        th1[x, y] = check_over_sd(th1, x, y)

        # imgry = Image.fromarray(th1.astype('uint8')).convert('1')
        # imgry.show()
        # img = cv2.cvtColor(np.asarray(imgry), cv2.COLOR_RGB2BGR)
        # cv2.imshow("OpenCV1", th1)

        a = (th1 < 0.1)

        dst = morphology.remove_small_objects(a, min_size=30, connectivity=1)

        for y in range(y_size):
            for x in range(x_size):
                # (白底黑字)False:黑色点,True:白色点
                if dst[x, y]:
                    th1[x, y] = 0
                else:
                    th1[x, y] = 255
        # cv2.imshow("OpenCV2", th1)
        # cv2.waitKey()
        # Image.fromarray(th1.astype('uint8')).convert('1').save(SAVE_PATH+f.split("_")[0]+".png")
        Image.fromarray(th1.astype('uint8')).convert('1').show()

#将二维数组转换为图片显示，用于调试
def test_array(_array):
    save_path = "D:/360MoveData/Users/LENOVO/Desktop/"
    x_shape, y_shape = _array.shape

    show_array = np.empty(shape=[x_shape, y_shape])
    for i in range(x_shape):
        for j in range(y_shape):
            if _array[i,j]:
                show_array[i,j]=0
            else:
                show_array[i, j] = 255
    # Image.fromarray(show_array.astype('uint8')).convert('1').save(save_path+"tt.jpg")
    Image.fromarray(show_array.astype('uint8')).convert('RGB').show()


def test1():
    CAPT_PATH = "C:/Users/LENOVO/Desktop/pandas/capt/ShanDong/"
    SAVE_PATH = "C:/Users/LENOVO/Desktop/pandas/capt/ShanDong_mask/"


    #按照颜色过滤掉背景以及部分噪点
    def SD_getcolor():
        for i in range(0, height):
            for j in range(0, width):
                data = np_img[i, j]
                # print(sum(data))
                filter_color1 = data > 180
                filter_count1 = np.sum(filter_color1 == True)

                filter_color2 = data > 200
                filter_count2 = np.sum(filter_color2 == True)

                if filter_count1 > 2:   #rgb三项都大于180，此点为置白色
                    np_img[i, j] = (255, 255, 255)
                elif filter_count2 == 2:    #rgb三项中，有两项大于200，此点为置白色
                    np_img[i, j] = (255, 255, 255)
                elif sum(data) > 600:   #rgb三项数值的和大于600，此点为置白色
                    np_img[i, j] = (255, 255, 255)



    #按照连通性获得的二值图删除彩图的噪点
    def Use_dst_to_color():
        # print(img.size) # (75, 30)
        # print(dst.shape) #(30, 75)
        for i in range(0, width):
            for j in range(0, height):
                if not dst[j, i]:
                    img.putpixel((i, j), (255, 255, 255))

    def Use_dst_del_color():
        # print(img.size) # (75, 30)
        # print(dst.shape) #(30, 75)
        for i in range(0, width):
            for j in range(0, height):
                if dst[j, i]:
                    img.putpixel((i, j), (255, 255, 255))

    #去除噪点
    def check_over_test(chack_array, x, y, check_type="left"):
        #(30, 75) 75 30
        # True:黑色点
        # False:白色点
        if check_type == "left":
            left = bool(chack_array[x, y-1])
            up = bool(chack_array[x-1, y])
            down = bool(chack_array[x+1, y])

            if left == up == down == False:
                return False
            else:
                return True
        else:
            right = bool(chack_array[x, y + 1])
            up = bool(chack_array[x - 1, y])
            down = bool(chack_array[x + 1, y])

            if right == up == down == False:
                return False
            else:
                return True

    def return_different():
        # print(bwimg.shape,dst.shape,width,height)
        for i in range(0, height):
            for j in range(0, width):
                if bwimg[i,j] and not dst[i,j]:
                    dst[i, j] = True
                else:
                    dst[i, j] = False

    files = os.listdir(SAVE_PATH)
    for index, f in enumerate(files):
        if index%50 != 0:
            continue
        # if index == 1:
        #     break
        f="D5RGK_1555918445.931913.jpg"
        # print(f)
        img = Image.open(SAVE_PATH+f)
        width, height = img.size  # print(img.size) #(75, 30)


        for i in range(50,150,10):
            imgry = img.convert('L')
            imgry = np.array(imgry)

            # for d_x in range(1, height):  # 30
            #     a=[]
            #     for d_y in range(1, width):  # 75
            #         _num = imgry[d_x,d_y]
            #         if _num<10:
            #             a.append("**"+ str(_num))
            #         elif _num<100:
            #             a.append("*" + str(_num))
            #         else:
            #             a.append(str(_num))
            #     print(a)
            bwimg = (imgry < i)


            for d_x in range(1, height - 1):  # 30
                for d_y in range(1, width - 1):  # 75
                    if bwimg[d_x, d_y]:
                        bwimg[d_x, d_y] = check_over_test(bwimg, d_x, d_y)
            # 在dst上进行降噪处理(从右往左)
            for d_x in range(height - 2, 1, -1):  # 30
                for d_y in range(width - 2, 1, -1):  # 75
                    if bwimg[d_x, d_y]:
                        bwimg[d_x, d_y] = check_over_test(bwimg, d_x, d_y, "right")
            # test_array(bwimg)
            dst = morphology.remove_small_objects(bwimg, min_size=6, connectivity=1)
            # test_array(dst)
            return_different()
            # test_array(dst)
            Use_dst_del_color()
        img.show()




        np_img = np.array(img)

        SD_getcolor()
        img = Image.fromarray(np_img.astype('uint8')).convert('RGB')
        imgry = img.convert('L')
        imgry = np.array(imgry)

        #设置阈值为200，为了将浅色的部分不丢失（数组越小，过滤标准颜色越深）

        bwimg = (imgry < 180)
        dst = morphology.remove_small_objects(bwimg, min_size=30, connectivity=1)

        # print(dst.shape,width, height ) (30, 75) 75 30
        # 在dst上进行降噪处理(从左往右)
        for d_x in range(1, height - 1):  # 30
            for d_y in range(1, width - 1):  # 75
                if dst[d_x, d_y]:
                    dst[d_x, d_y] = check_over_test(dst, d_x, d_y)
        # 在dst上进行降噪处理(从右往左)
        for d_x in range(height - 2, 1, -1):  # 30
            for d_y in range(width - 2, 1, -1):  # 75
                if dst[d_x, d_y]:
                    dst[d_x, d_y] = check_over_test(dst, d_x, d_y, "right")


        for i in [50,120,160,200]:
            imgry = img.convert('L')
            imgry = np.array(imgry)
            bwimg = (imgry < i)


            for d_x in range(1, height - 1):  # 30
                for d_y in range(1, width - 1):  # 75
                    if bwimg[d_x, d_y]:
                        bwimg[d_x, d_y] = check_over_test(bwimg, d_x, d_y)
            # 在dst上进行降噪处理(从右往左)
            for d_x in range(height - 2, 1, -1):  # 30
                for d_y in range(width - 2, 1, -1):  # 75
                    if bwimg[d_x, d_y]:
                        bwimg[d_x, d_y] = check_over_test(bwimg, d_x, d_y, "right")
            test_array(bwimg)
            dst = morphology.remove_small_objects(bwimg, min_size=8, connectivity=2)
            test_array(dst)
            return_different()
            test_array(dst)
            Use_dst_del_color()
            img.show()



        for d_y in range(width):  # 75
            empty_count = 0  # 最后一遍遍历图片时，将纵向只有一两个像素的图片位置置空
            # empty_start = 0
            # empty_end = 0
            for d_x in range(height):  # 30
                if dst[d_x, d_y]:
                    empty_count += 1
                    # if empty_start == 0:
                    #     empty_start = d_x
                    # empty_end = d_x
            if empty_count < 2 :#and (empty_end - empty_start)<10
                for d_x in range(height):
                    dst[d_x, d_y] = False

        Use_dst_to_color()

        # test_ocr(img,f)
        # img.show()
        img.save(SAVE_PATH+f)


def test1_bak():
    CAPT_PATH = "C:/Users/LENOVO/Desktop/pandas/capt/ShanDong/"
    SAVE_PATH = "C:/Users/LENOVO/Desktop/pandas/capt/ShanDong_mask/"


    #按照颜色过滤掉背景以及部分噪点
    def SD_getcolor():
        for i in range(0, height):
            for j in range(0, width):
                data = np_img[i, j]
                # print(sum(data))
                filter_color1 = data > 180
                filter_count1 = np.sum(filter_color1 == True)

                filter_color2 = data > 200
                filter_count2 = np.sum(filter_color2 == True)

                if filter_count1 > 2:   #rgb三项都大于180，此点为置白色
                    np_img[i, j] = (255, 255, 255)
                elif filter_count2 == 2:    #rgb三项中，有两项大于200，此点为置白色
                    np_img[i, j] = (255, 255, 255)
                elif sum(data) > 600:   #rgb三项数值的和大于600，此点为置白色
                    np_img[i, j] = (255, 255, 255)
                # else:
                #     new_color = [0, 0, 0]
                #     for _index, _c in enumerate(data):
                #         if _c>200:
                #             new_color[_index] = 255
                #         else:
                #             # new_color[_index]=0
                #             new_color[_index] = _c//100*100
                #     img[i, j] = np.array(new_color)
                # new_color = [0,0,0]
                # _data = list(data)
                # max_index = _data.index(max(_data))
                # new_color[max_index] = 255
                # img[i, j] = np.array(new_color)


    #按照连通性获得的二值图删除彩图的噪点
    def Use_dst_to_color():
        # print(img.size) # (75, 30)
        # print(dst.shape) #(30, 75)
        for i in range(0, width):
            for j in range(0, height):
                if not dst[j, i]:
                    img.putpixel((i, j), (255, 255, 255))

    def Use_dst_del_color():
        # print(img.size) # (75, 30)
        # print(dst.shape) #(30, 75)
        for i in range(0, width):
            for j in range(0, height):
                if dst[j, i]:
                    img.putpixel((i, j), (255, 255, 255))

    #去除噪点
    def check_over_test(chack_array, x, y, check_type="left"):
        #(30, 75) 75 30
        # True:黑色点
        # False:白色点
        if check_type == "left":
            left = bool(chack_array[x, y-1])
            up = bool(chack_array[x-1, y])
            down = bool(chack_array[x+1, y])

            if left == up == down == False:
                return False
            else:
                return True
        else:
            right = bool(chack_array[x, y + 1])
            up = bool(chack_array[x - 1, y])
            down = bool(chack_array[x + 1, y])

            if right == up == down == False:
                return False
            else:
                return True

    def return_different():
        # print(bwimg.shape,dst.shape,width,height)
        for i in range(0, height):
            for j in range(0, width):
                if bwimg[i,j] and not dst[i,j]:
                    dst[i, j] = True
                else:
                    dst[i, j] = False

    files = os.listdir(CAPT_PATH)
    for index, f in enumerate(files):
        if index%50 != 0:
            continue
        # if index == 1:
        #     break
        # f="1PFU8_1555918456.2896287.jpg"
        # print(f)
        img = Image.open(CAPT_PATH+f)
        np_img = np.array(img)
        width, height = img.size  # print(img.size) #(75, 30)
        SD_getcolor()
        img = Image.fromarray(np_img.astype('uint8')).convert('RGB')
        imgry = img.convert('L')
        imgry = np.array(imgry)

        #设置阈值为200，为了将浅色的部分不丢失（数组越小，过滤标准颜色越深）

        bwimg = (imgry < 180)
        dst = morphology.remove_small_objects(bwimg, min_size=30, connectivity=1)

        # print(dst.shape,width, height ) (30, 75) 75 30
        # 在dst上进行降噪处理(从左往右)
        for d_x in range(1, height - 1):  # 30
            for d_y in range(1, width - 1):  # 75
                if dst[d_x, d_y]:
                    dst[d_x, d_y] = check_over_test(dst, d_x, d_y)
        # 在dst上进行降噪处理(从右往左)
        for d_x in range(height - 2, 1, -1):  # 30
            for d_y in range(width - 2, 1, -1):  # 75
                if dst[d_x, d_y]:
                    dst[d_x, d_y] = check_over_test(dst, d_x, d_y, "right")


        # for i in [50,120,160,200]:
        #     imgry = img.convert('L')
        #     imgry = np.array(imgry)
        #     bwimg = (imgry < i)
        #
        #
        #     for d_x in range(1, height - 1):  # 30
        #         for d_y in range(1, width - 1):  # 75
        #             if bwimg[d_x, d_y]:
        #                 bwimg[d_x, d_y] = check_over_test(bwimg, d_x, d_y)
        #     # 在dst上进行降噪处理(从右往左)
        #     for d_x in range(height - 2, 1, -1):  # 30
        #         for d_y in range(width - 2, 1, -1):  # 75
        #             if bwimg[d_x, d_y]:
        #                 bwimg[d_x, d_y] = check_over_test(bwimg, d_x, d_y, "right")
        #     test_array(bwimg)
        #     dst = morphology.remove_small_objects(bwimg, min_size=8, connectivity=2)
        #     test_array(dst)
        #     return_different()
        #     test_array(dst)
        #     Use_dst_del_color()
        #     img.show()



        for d_y in range(width):  # 75
            empty_count = 0  # 最后一遍遍历图片时，将纵向只有一两个像素的图片位置置空
            # empty_start = 0
            # empty_end = 0
            for d_x in range(height):  # 30
                if dst[d_x, d_y]:
                    empty_count += 1
                    # if empty_start == 0:
                    #     empty_start = d_x
                    # empty_end = d_x
            if empty_count < 2 :#and (empty_end - empty_start)<10
                for d_x in range(height):
                    dst[d_x, d_y] = False

        Use_dst_to_color()

        # test_ocr(img,f)
        # img.show()
        img.save(SAVE_PATH+f)

def test():
    save_path = "D:/360MoveData/Users/LENOVO/Desktop/"
    img = Image.open(r"C:\Users\LENOVO\Desktop\pandas\capt\ShanDong\1PFU8_1555918456.2896287.jpg")
    imgry = img.convert('L')
    _array = np.array(imgry)

    def check_over_test(chack_array, x, y):
        # True:黑色点
        # False:白色点
        left = bool(chack_array[x, y - 1])
        # right = bool(chack_array[x, y + 1])

        up = bool(chack_array[x - 1, y])
        down = bool(chack_array[x + 1, y])

        if up == down == False:
            return False
        else:
            return True

    # THRESHOLD = 100
    # SH_LUT = [0] * THRESHOLD + [1] * (256 - THRESHOLD)
    # binary = imgry.point(SH_LUT, '1')
    # binary.show()

    x_shape, y_shape = _array.shape  # 30,75
    new_array = np.empty(shape=[x_shape, y_shape])
    for _x in range(x_shape):
        # print_list = []
        for _y in range(y_shape):
            _value = _array[_x, _y] // 20 * 20
            if _value>=200:
                _value = 255
            new_array[_x, _y] = _value

    # ii = Image.fromarray(new_array.astype('uint8')).convert('L')
    # ii.save(save_path+"1.jpg")
    # THRESHOLD = 20
    # SH_LUT = [0] * THRESHOLD + [1] * (256 - THRESHOLD)
    # binary = ii.point(SH_LUT, '1')
    # binary.show()

    for i in [40, 60, 80, 100, 120, 140, 160, 180, 200]:  # [60,80,100,120,140,160,180,200][180, 160, 140, 120, 100]
        # for t in range(2):
        aa = (new_array <= i)
        bool_array = np.array(aa)

        # ii = Image.fromarray(new_array.astype('uint8')).convert('L')
        # ii.save(save_path+"1.jpg")
        # THRESHOLD = i
        # SH_LUT = [0] * THRESHOLD + [1] * (256 - THRESHOLD)
        # binary = ii.point(SH_LUT, '1')
        # binary.show()

        for _x in range(x_shape):
            for _y in range(y_shape):
                # (白底黑字)True:黑色点,False:白色点
                if bool_array[_x, _y] == True:
                    if _x == 1 or _x == x_shape - 1 or _y == 1 or _y == y_shape - 1:
                        new_array[_x, _y] = 255
                    else:
                        get_result = check_over_test(bool_array, _x, _y)
                        bool_array[_x, _y] = get_result
                        if get_result:
                            new_array[_x, _y] = i
                        else:
                            new_array[_x, _y] = 255
        Image.fromarray(new_array.astype('uint8')).convert('L').show()
        # Image.fromarray(new_array.astype('uint8')).convert('L').save(save_path+str(i)+".jpg")



def test_ocr(img, _name):
    SAVE_PATH = "C:/Users/LENOVO/Desktop/pandas/capt/ShanDong_mask/"

    # 多个小图片拼接切大图
    def image_compose(pic_list):
        to_image = Image.new('RGB', (20 * len(pic_list), 22), "#FFFFFF")  # 创建一个新图
        # 循环遍历，把每张图片按顺序粘贴到对应位置上
        total_width = 0
        for x in range(len(pic_list)):
            from_image = pic_list[x]
            if total_width == 0:
                to_image.paste(from_image, (3, 0))
            else:
                to_image.paste(from_image, (total_width, 0))
            total_width += from_image.size[0] + 5
        return to_image


    # img = Image.open(r"C:\Users\LENOVO\Desktop\pandas\capt\ShanDong_mask\JFRKY_1555918569.9829307.jpg")
    return_list, cut_list, return_image_list = coherence.coherence_function(img, 200)
    ii = None
    if len(return_list) == 4:
        max_width = 0
        max_index = 0
        max_pic=None
        for _index, i in enumerate(cut_list[0]):
            if (i[1] - i[0]) > max_width:
                max_width = i[1] - i[0]
                max_index = _index
                max_pic = return_list[_index]
        use_width = max_width // 2
        use_list = []
        for m_index, m_i in enumerate(return_list):
            if m_index == max_index:

                p1 = max_pic.crop((0,0,use_width,max_pic.size[1]))
                p2 = max_pic.crop((use_width, 0, max_pic.size[0], max_pic.size[1]))

                use_list.append(p1)
                use_list.append(p2)
            else:
                use_list.append(m_i)

        ii = image_compose(use_list)
    elif len(return_list) == 3:
        use_list = []
        width_list = []
        for _index, i in enumerate(cut_list[0]):
            width_list.append(i[1] - i[0])


        for _index_w, i in enumerate(width_list):
            if i < 15:
                use_list.append(return_list[_index_w])
            elif i > 18 and i < 25:
                max_pic=return_list[_index_w]
                use_width = i // 2
                p1 = max_pic.crop((0,0,use_width,max_pic.size[1]))
                p2 = max_pic.crop((use_width, 0, max_pic.size[0], max_pic.size[1]))

                use_list.append(p1)
                use_list.append(p2)
            else:
                max_pic = return_list[_index_w]
                use_width = i // 3
                p1 = max_pic.crop((0, 0, use_width, max_pic.size[1]))
                p2 = max_pic.crop((use_width, 0, use_width * 2, max_pic.size[1]))
                p3 = max_pic.crop((use_width * 2, 0, max_pic.size[0], max_pic.size[1]))
                use_list.append(p1)
                use_list.append(p2)
                use_list.append(p3)

        ii = image_compose(use_list)


    else:
        ii = image_compose(return_list)

    # ii.show()
    if ii:
        # result_pre = pytesseract.image_to_string(ii, lang='eng', config="-psm 7")
        try:
            ii.save(SAVE_PATH+_name)
        except:
            print("error")
        # print(result_pre)

if __name__ == "__main__":
    test1()
    # test_ocr()

