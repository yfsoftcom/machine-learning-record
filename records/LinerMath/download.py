# -*- coding: utf-8 -*-
import os, time, re, requests, json, subprocess
from lxml import etree

def download(url, encoding = 'utf-8'):
    UA = "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.13 Safari/537.36"
    header = { "User-Agent" : UA }
    r = requests.get(url, headers = header)
    r.encoding = encoding
    return r.text

def run():
    URL = 'http://open.163.com/special/opencourse/daishu.html'
    html = download(URL, encoding='gbk')
    root = etree.ElementTree(etree.HTML(html).xpath('//table[@id="list2"]')[0])
    td_tags = root.xpath('//td[@class="u-ctitle"]')
    json_data = []
    for td in td_tags:
        data = {}
        td_text = td.text.strip()
        td_tree = etree.ElementTree(td)
        a = td_tree.xpath('//a')[0]
        data['title'] = td_text + a.text
        data['href'] = a.get('href')
        shell = "C:/ProgramData/chocolatey/lib/you-get//tools/you-get.exe -o D:/Videos -O {title} {href}".format(title=data['title'], href=data['href'])
        retcode = subprocess.call(shell, shell=True)
        print(shell, retcode)
        json_data.append(data)

    with open('videos.json', 'w') as f:
        f.write(json.dumps(json_data))


if __name__ == '__main__':
    run()