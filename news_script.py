# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 10:32:05 2020

@author: zzhu1
"""


import bs4 as bs 
import urllib.request
import pandas as pd
import time

##read urls
urls=pd.read_excel('urls3.xlsx')
urls_list=urls['url'].tolist()
news=[]
#times=[]
links=[]
##get url and read in python
for url in urls_list:
    try:
        raw_html = urllib.request.urlopen(url)  
        raw_html = raw_html.read()
        links.append(url)
    except:
        print(url+'Blocked')
        continue
###parse with lxml
    article_html = bs.BeautifulSoup(raw_html, 'lxml')
    #grab timestamp
    #try:
     #   time = article_html.find('time').text
      #  times.append(time)
    #except:
     #   print('No info')
      #  continue
###find all paragraphs
    article_paragraphs = article_html.find_all('p')
##combine all paragraphs into text
    article_text = ''

    for para in article_paragraphs:  
        article_text += para.text
    news.append(article_text)
    txt=pd.DataFrame(news,columns=['txt'])
    txt.to_excel('txt3.xlsx',index=False)
    link=pd.DataFrame(links,columns=['url'])
    link.to_excel('link3.xlsx',index=False)
    time.sleep(1) 

##dataframe
txt1=pd.read_excel('txt.xlsx',skip_blank_lines=False)
txt2=pd.read_excel('txt2.xlsx',skip_blank_lines=False)
txt3=pd.read_excel('txt3.xlsx',skip_blank_lines=False)
frames=[txt1,txt2,txt3]
txt=pd.concat(frames)

link1=pd.read_excel('link.xlsx')
link2=pd.read_excel('link2.xlsx')
link3=pd.read_excel('link3.xlsx')
frames=[link1,link2,link3]
link=pd.concat(frames)

txt=txt.reset_index()
txt.drop(columns=['index'],inplace=True)

link=link.reset_index()
link.drop(columns=['index'],inplace=True)

#ti=pd.DataFrame(times,columns=['time'])
txt_link=txt.merge(link,left_index=True, right_index=True)
urls=pd.read_excel('urls.xlsx')

df=txt_link.merge(urls,on='url',how='left')
df=df.dropna()
df.to_excel('news_sample.xlsx')
    