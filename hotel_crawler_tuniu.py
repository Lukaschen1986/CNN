# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import random
import numpy as np
import pandas as pd
import pickle
import re
import time
import urllib.request
import urllib.parse
import os
os.getcwd()
os.chdir("D:\\my_project\\Python_Project\\test\\NN\\bedroom")

user_agent = ['Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/54.0.2840.87 Safari/537.36',  
              'Mozilla/5.0 (X11; U; Linux x86_64; zh-CN; rv:1.9.2.10) Gecko/20100922 Ubuntu/10.10 (maverick) Firefox/3.6.10',  
              'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',  
              'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/30.0.1599.101 Safari/537.36',  
              'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.71 Safari/537.1 LBBROWSER']

headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8', 
           'Accept-Encoding': 'gzip, deflate, sdch', 
           'Accept-Language': 'zh-CN,zh;q=0.8', 
           'User-Agent': user_agent[random.randint(0,4)]}

# 酒店搜索结果页
city_id = 2008
hotel_url = "http://hotel.tuniu.com/list?city=%d&checkindate=2017-11-21&checkoutdate=2017-11-22" % (city_id)
hotel_request = requests.get(hotel_url, headers)
hotel_text = hotel_request.text
hotel_soup = BeautifulSoup(hotel_text, "html.parser")
# 爬取hotel_id
hotel_ids = []
for data in hotel_soup.find("div", {"id":"content"}).find_all("a"):
    p = re.compile('data-id="(\d+)"')
#    p = re.compile('href="/detail/(\d+).')
    hotel_id = re.findall(p, str(data))
    hotel_ids.extend(hotel_id)
hotel_ids = list(set(hotel_ids))
# 爬取每个id下房型图片
#hotel_ids = ['198434747','62061','198169691','62386','597540']
for hid in hotel_ids:
    # hid = '233035312'
    detail_url = "http://hotel.tuniu.com/detail/%s?checkindate=2017-11-21&checkoutdate=2017-11-22" % (hid)
    detail_request = requests.get(detail_url, headers)
    detail_text = detail_request.text
    detail_soup = BeautifulSoup(detail_text, "html.parser")
    # 分别爬取每个id下外观、内景、房型url
    url_appearance = []; url_indoorScene= []; url_room = []
    # detail_soup.find("div", {"class":"hrela_spic"}).find_all("img")
    # p = re.compile("http://m.tuniucdn.com/\w+/\w+/\w+/\w+/\w+/\w+/\w+.jpg")
    for x in detail_soup.find("div", {"class":"hrela_spic"}).find_all("ul"):
        if x.attrs["data"] == "appearance":
            for y in x.find_all("img"):
                p = re.compile('data-midd="http://m.tuniucdn.com/\w+/\w+/\w+/\w+/\w+/\w+/\w+.jpg"')
                url = re.findall(p, str(y))
                url_appearance.extend(url)
        elif x.attrs["data"] == "indoorScene":
            for y in x.find_all("img"):
                p = re.compile('data-midd="http://m.tuniucdn.com/\w+/\w+/\w+/\w+/\w+/\w+/\w+.jpg"')
                url = re.findall(p, str(y))
                url_indoorScene.extend(url)
        elif x.attrs["data"] == "room":
            for y in x.find_all("img"):
                p = re.compile('data-midd="http://m.tuniucdn.com/\w+/\w+/\w+/\w+/\w+/\w+/\w+.jpg"')
                url = re.findall(p, str(y))
                url_room.extend(url)
        else:
            continue
    # 去重
    url_appearance = list(set(url_appearance))
    url_indoorScene = list(set(url_indoorScene))
    url_room = list(set(url_room))
    # 保存完整url格式
    url_appearance_2 = []; url_indoorScene_2 = []; url_room_2 = []
    for url_x in url_appearance:
        val = url_x.replace('data-midd="','').replace('"','')
        url_appearance_2.append(val)
    for url_y in url_indoorScene:
        val = url_y.replace('data-midd="','').replace('"','')
        url_indoorScene_2.append(val)
    for url_z in url_room:
        val = url_z.replace('data-midd="','').replace('"','')
        url_room_2.append(val)
    # 下载图片
    for i in range(len(url_appearance_2)):
        os.chdir("D:\\my_project\\Python_Project\\test\\NN\\bedroom\\appearance")
        filename = hid + "_appearance_" + str(i) + ".jpg"
        urllib.request.urlretrieve(url_appearance_2[i], filename)
    for j in range(len(url_indoorScene_2)):
        os.chdir("D:\\my_project\\Python_Project\\test\\NN\\bedroom\\indoorScene")
        filename = hid + "_indoorScene_" + str(j) + ".jpg"
        urllib.request.urlretrieve(url_indoorScene_2[j], filename)
    for n in range(len(url_room_2)):
        os.chdir("D:\\my_project\\Python_Project\\test\\NN\\bedroom\\room")
        filename = hid + "_room_" + str(n) + ".jpg"
        urllib.request.urlretrieve(url_room_2[n], filename)
    print("crawler finish, hid: " + hid)
    # 随机休眠n秒
    headers = {'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8', 
               'Accept-Encoding': 'gzip, deflate, sdch', 
               'Accept-Language': 'zh-CN,zh;q=0.8', 
               'User-Agent': user_agent[np.random.randint(0,4)]}
    s = np.random.randint(low=1, high=5, size=1)
    time.sleep(s)
    
