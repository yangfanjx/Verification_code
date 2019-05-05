from lxml import html
import requests
import base64
import time
from io import BytesIO
from PIL import Image


# 从尚德验证码识别系统下载验证码图片
def dowm_from_sunland():
    url = "http://172.16.100.115:5000/auto?model_name=14_self_current&url=http://zxks.gxeea.cn:8082/gxzkweb/SecCodeController"
    for i in range(300):
        res = requests.get(url)
        tree = html.fromstring(res.text)
        try:
            list = tree.xpath('//div[@class="col-md-4 column"]/form/img/@src')[0]
            name = tree.xpath('//div[@class="col-md-4 column"]/form/h2[@style="color: green"]')[0]
            pic_utl = list.split(",")[1]
            img = base64.b64decode(pic_utl)
            tt = str(time.time())
            pic_save_path = r"C:\Users\LENOVO\Desktop\pandas\capt\GuangXi\{}_{}.jpg".format(name.text, tt)
            file = open(pic_save_path, 'wb')
            file.write(img)
            file.close()
            time.sleep(0.3)
        except:
            continue


# 从官网下载验证码图片
def dowm_from_http():
    import random
    url = "http://202.121.151.78/shmeea/q/verifyImg?t=0.3632441769185033"
    for i in range(23,1000):
        try:
            res = requests.get(url, timeout=5)
            print('success')
        except requests.exceptions.RequestException as e:
            continue

        response = res.content

        BytesIOObj = BytesIO()
        BytesIOObj.write(response)
        img = Image.open(BytesIOObj)
        img.save("C:/Users/LENOVO/Desktop/pandas/capt/ShangHai_test/{}.jpg".format(i))
        # img.show()
        time.sleep(random.random())


dowm_from_sunland()
