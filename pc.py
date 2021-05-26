import requests
from bs4 import BeautifulSoup
from pandas import DataFrame


data = []
wb_data = requests.get('http://www.kugou.com/yy/rank/home/1-8888.html')
soup = BeautifulSoup(wb_data.text, 'lxml')
ranks = soup.select('span.pc_temp_num')
titles = soup.select('div.pc_temp_songlist > ul > li > a')
times = soup.select('span.pc_temp_tips_r > span')
for rank, title, time in zip(ranks, titles, times):
    a = {
        'rank': rank.get_text().strip(),
        'singer': title.get_text().split('-')[0],
        'song': title.get_text().split('-')[1],
        'time': time.get_text().strip()
    }
    data.append(a)
print(data)

df = DataFrame(data)
print(df)