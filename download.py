#coding=utf-8
import requests
from concurrent.futures import ThreadPoolExecutor
import os
import re
import time
import json

s = requests.session()
str = "<a href='{}'>{}</a>"
fr = open("apkINI.txt", 'r')
apkList = fr.read().split(",")[:-1]

fw = open("D:/wamp64/www/test1.html", 'w')

def getApkUrl(url):
    url = "https://www.apkmonk.com/check_file/?key="+url.split("/")[-2]+"&check=1"
    retu = s.get(url)
    temp = retu.content
    print (retu.status_code)
    if retu.status_code != 200:
        return False
    else:
        time.sleep(5)
        return json.loads(temp)['url']


# def getApkHTML(address):
#     for i in range(address, address+50):
#         downUrl = getApkUrl(apkList[i])
#         if downUrl != False:
#             fw.write(str.format(downUrl, downUrl))
#             print (downUrl)
#             time.sleep(2)



# pool = ThreadPoolExecutor(5)
# pool.submit(getApkHTML, 50)
# pool.submit(getApkHTML, 100)
# pool.submit(getApkHTML, 150)
# pool.submit(getApkHTML, 200)
# pool.submit(getApkHTML, 250)


for i in range(740, 760):
    downUrl = getApkUrl(apkList[i])
    if downUrl != False:
        print (downUrl)
        fw.write(str.format(downUrl, downUrl))
        fw.flush()
