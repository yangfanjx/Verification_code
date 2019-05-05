from PIL import Image
import numpy as np
# import traceback
import os

#将文件夹中的全部图片转为不带噪点的彩色图片

THRESHOLD = 145
LUT = [0] * THRESHOLD + [1] * (256 - THRESHOLD)
CAPT_PATH = "C:/Users/LENOVO/Desktop/pandas/capt/tianjin/"
SAVE_PATH = "C:/Users/LENOVO/Desktop/pandas/capt/tianjin_noice/"

def yield_folder():
    files = os.listdir(CAPT_PATH)
    for f in files:
        # train_labels += list(f.split("_")[0])
        img = Image.open(CAPT_PATH + f)
        yield f,img


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


def main():
    for name, p in yield_folder():
        p_array = np.array(p)
        # 转灰度图
        imgry = p.convert('L')
        # 转二值图
        binary = imgry.point(LUT, '1')

        # +0为了将原二维数组中bool类型元素全部转为0，1形式
        # char_array = np.array(binary) + 0
        char_array = np.array(binary)
        x_shape, y_shape = char_array.shape
        new_pic = np.zeros((x_shape, y_shape, 3))
        for y in range(y_shape):
            for x in range(x_shape):
                # (白底黑字)False:黑色点,True:白色点
                if char_array[x, y] == False:
                    a = check_over(char_array, x, y, x_shape, y_shape)
                    char_array[x, y] = a
                    if a == True:
                        new_pic[x, y] = [255, 255, 255]
                    else:
                        new_pic[x, y] = p_array[x, y]
                else:
                    new_pic[x, y] = [255, 255, 255]
        # Image.fromarray(new_pic.astype('uint8')).convert('RGB').save(SAVE_PATH + name)
        Image.fromarray(new_pic.astype('uint8')).convert('RGB').show()
main()
