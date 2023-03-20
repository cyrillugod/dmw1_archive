#!/usr/bin/env python
# coding: utf-8

# In[12]:


import json
import re
import sqlite3
import os
import pandas as pd
import numpy as np
from xml.etree import ElementTree
import requests
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import getpass


# In[2]:


def get_purchase_order_xpath():
    """Returns XPath of 'PurchaseOrderNumber' for items bound for USA"""
    return ".//Address[@Type='Shipping']/Country[.='USA']/../.."


# In[3]:


def has_comments_xpath():
    """Returns XPath of 'PurchaseOrderNumber' with 'Item' comment."""
    return ".//Items/Item/Comment/../../.."


# In[5]:


def plot_counts():
    """Returns a Series with date and total viewer count for each date."""
    val = []
    file = []
    directory = './sensor'
    df_dict = {}
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        if os.path.isfile(f):
            df = pd.read_json(f).T
            val_dic = df.loc["data", "viewer"]
            val.append(sum(
                [v for i in val_dic.values() for k, v in i.items()]))
            file.append(re.sub(r"(.*?).json", r"\1", filename))
    return pd.Series(data=val, index=file).sort_index()


# In[6]:


def get_age_population(path, min_age, max_age, gender='M/F'):
    """Return the total population of an age based on the supplied data
    
    Parameters
    ----------
    path : filepath
        Path to a census file
    min_age : int from 0 to 80, inclusive
        Lower bound of the age range
    max_age : int from 0 to 80, inclusive
        Upper bound of the age range
    gender : 'M', 'F' or 'M/F', optional
        Gender to consider
        
    Returns
    -------
    pop_count : int or None
        Total population of age range or `None` if the age range is not 
        allowed or the file cannot be located
    """
    if os.path.isfile(path):
        df = pd.read_excel(path, sheet_name="T2", skiprows=6, usecols=[1,2,3])
        df.dropna(inplace=True)
        (df.rename(columns={df.columns[0]: "M/F", df.columns[1]: "M",
                df.columns[2]: "F"},inplace=True))
        if (min_age < max_age) and ((min_age and max_age) in df.index):
            return((df.loc[min_age: max_age, gender]).sum().astype(int))
    return None


# In[7]:


def get_prev_word(inp, search_term):
    """Returns list of preceding words of a match within sentences."""
    return re.findall(fr'\b(\w*),?"?\s"?{re.escape(search_term)}"?\b', inp)


# In[8]:


def repeating_words(text):
    """Returns phrases containing repeating words."""
    return ([rep[0] for rep in 
             re.findall(r"((\w+)([- \.,]+\2)*[- \.,]+\2)+\b", text, re.I)])


# In[9]:


def red_stuff():
    """Returns a Series with words following the word 'Red'"""
    url = "/mnt/data/public/instacart/instacart_2017_05_01/products.csv"
    df = (pd.read_csv(url, index_col="product_id", 
        usecols=["product_id", "product_name"]))
    patt_1 = r"\bRed\s+[A-zÀ-ÿ\.'\-]+\s+[A-zÀ-ÿ\.'\-]+\b.*\b$"
    patt_2 = r"Red\s+[A-zÀ-ÿ\.'\-]+\s+[^&:,\d%]\b.*\b$"
    patt_3 = r"Red\s+[A-zÀ-ÿ\.'\-]+\b$"
    df["product_name"] = (df["product_name"].str
        .extract(f"({patt_1}|{patt_2}|{patt_3})"))
    df.dropna(inplace=True)
    ser = df.squeeze()
    ser = (ser.apply
        (lambda x: re.sub(r"(Red\s[\w-]+\s\w+|Red\s[\w-]+).*$", r"\1", x)))
    return ser


# In[10]:


def has_keyword():
    """Returns an SQL and conn string to generate title list per keyword."""
    return ("""
    SELECT
    DISTINCT m.title
    FROM movies m
    INNER JOIN movies_keywords mk
    ON m.idmovies = mk.idmovies
    INNER JOIN keywords k
    USING (idkeywords)
    WHERE k.keyword = ?
    ORDER BY m.title ASC
            """,
            sqlite3.connect('/mnt/data/public/imdb.db'))


# In[11]:


def aka_phils():
    """Returns SQL string for movie year and counts with alternate titles."""
    return ("""
    SELECT
        year,
        COUNT(location) AS title
    FROM
        aka_titles
    WHERE location LIKE '%Philippines: English title%'
    GROUP BY year
    HAVING COUNT(location) >= 10
    ORDER BY COUNT(location) DESC
            """,
            sqlite3.connect('/mnt/data/public/imdb.db'))


# In[12]:


def convert_twitter():
    """Returns SQLite connection to an in-memory db with table 'twitter'."""
    df = pd.read_json("data_twitter_sample.json", lines=True)
    df = (df[["id", "text", "is_quote_status", "favorite_count",
              "created_at", "timestamp_ms"]])
    df["created_at"] = df["created_at"].dt.strftime('%a %b %d %H:%M:%S %z %Y')
    df["timestamp_ms"] = (df["timestamp_ms"].dt.strftime('%s%f')
                              .apply(lambda x: x[:-3]).astype(int))
    conn = sqlite3.connect(":memory:")
    df.to_sql('twitter', conn, if_exists='replace', index = False)
    return conn


# In[2]:


def get_continent_urls():
    """
    Returns a dictionary with continent name as key
    and absolute url OAV page as value.
    """
    cont_dict = {}
    url = ('https://jojie.accesslab.aim.edu:9095/dmw_scraping/ASIA.html')
    response = requests.get(url)
    soup = BeautifulSoup(response.text)
    key = [e.text.strip() for e in soup.select('h3 > a[href]')]
    val = [urljoin(url, e['href']) for e in soup.select('h3 > a[href]')]
    cont_dict = {k: v for k, v in list(zip(key, val))}
    return cont_dict


# In[3]:


def get_fsp_urls(continent):
    """
    Returns a dictionary with foreign service posts in the continent as key
    and list of urls as values.
    """
    fsp_dict = {}
    url = ('https://jojie.accesslab.aim.edu:9095/dmw_scraping/' 
           + continent + '.html')
    response = requests.get(url)
    soup = BeautifulSoup(response.text)
    keys_raw = ([re.findall(r'(.*)\s-', e.text.strip())
                 for e in soup.select('h4 ~ a[href]')])
    fsp = [i for j in keys_raw for i in j]
    json_list = [urljoin(url, e['href']) for e in soup.select('h4 ~ a[href]')]
    df = pd.DataFrame({'key': fsp, 'val': json_list})
    fsp_dict = df.groupby('key')['val'].unique().to_dict()
    for k, v in fsp_dict.items():
        fsp_dict[k] = v.tolist()
    return fsp_dict


# In[4]:


def senator_votes(continent, fsp):
    """
    Returns a dictionary with candidate name as key and total votes received
    for the continent and FSP as values.
    """
    votes_list = []
    url = 'https://jojie.accesslab.aim.edu:9095/dmw_scraping/ASIA.html'
    url = urljoin(url, f'{continent}.html')
    response = requests.get(url)
    soup = BeautifulSoup(response.content)
    list_url = ([urljoin(url, e['href']) 
                 for e in soup.select(f'h4 ~ a[href*="{fsp}"]')])
    for url_ in list_url:
        resp = requests.get(url_).json()
        per_precinct = ([(resp['results'][i]['bName'],
                          resp['results'][i]['votes']) 
                         for i in range(len(resp['results']))])
        votes_list.extend(per_precinct)
    df = pd.DataFrame(votes_list)
    df.rename(columns={0: 'senator', 1: 'votes'}, inplace=True)
    df['votes'] = df['votes'].astype(int)
    return df.groupby('senator')['votes'].sum().to_dict()


# In[5]:


def latest_stories():
    """
    Returns a DataFrame with the category, title, and published timestamp
    of the latest stories in Inquirer sorted by order of appearance.
    """
    url = 'https://jojie.accesslab.aim.edu:9095/inquirer/'
    resp = requests.get(url)
    soup = BeautifulSoup(resp.content)
    h6_list = ([e.text.upper().strip() 
                for e in soup.select('div#tr_boxs3 > h6')])
    categ_list = h6_list[::2]
    title_list = ([e.text.strip() 
                   for e in soup.select('div#tr_boxs3 h2 a[href]')])
    published_list = [p for p in h6_list[1::2]]
    df = (pd.DataFrame(
        {'category': categ_list, 'title': title_list,
         'published': published_list}))
    df['published'] = (df['published']
                       .apply(lambda x: x[:-2]
                              .title()) + df['published'].str[-2:])
    return df


# In[6]:


def category_posts(topic):
    """
    Returns a DataFrame of the title and link to posts under the topic box in
    Inquirer sorted by order of appearance.
    """
    
    topic_cont = []
    if len(topic.split()) > 1:
        for i in topic.split():
            if i.isalnum():
                topic_cont.append(i.lower())
        topic = ('_').join(topic_cont)
    else:
        topic = topic.lower()
    
    url = 'https://jojie.accesslab.aim.edu:9095/inquirer/'
    resp = requests.get(url)
    soup = BeautifulSoup(resp.content)
    title_list = ([e.text.strip() 
                   for e in soup.select(f'div[id="track_{topic}"] h3 a')])
    link_list = ([e['href'] 
                  for e in soup.select(f'div[id="track_{topic}"] h3 a')])
    df = pd.DataFrame({'title': title_list, 'link': link_list})
    return df


# In[7]:


def get_books():
    """
    Returns a list of tuples of book titles and paperback selling prices of
    books in 'data_wrangling.html' sorted in order of appearance.
    """
    resp_txt = ''
    title_list = []
    price_list = []
    html_file = 'data_wrangling.html'
    t = open(html_file, "r")
    for v in t.readlines():
        resp_txt += v
    soup = BeautifulSoup(resp_txt)
    for i in range(len(soup.select('div[data-index]'))):
        title_list.append([e.text.strip() 
                           for e in soup.select(
                               f'div[data-index="{i}"] h2 > a')][0])
        supsoup = ([e.parent.parent.parent.parent.parent.parent 
                    for e in soup.select(f'div[data-index="{i}"] h2 > a')][0])
        price_raw = ([e.text.strip() 
                      for e in supsoup.select(
                          'div.a-row a.a-size-base span')])
        if len(price_raw) == 0:
            price_list.append(None)
        elif 'to buy' in price_raw:
            price_list.append(price_raw[9])
        else:
            price_list.append(price_raw[1])
    return [(t, p) for t, p in list(zip(title_list, price_list))]


# In[8]:


def get_revisions_timeseries():
    """
    Returns a Series of the number of revisions for the English Wikipedia
    article of Data Science by month, excluding months without revisions. 
    """
    revs = []
    params={
         'action': 'query',
         'prop': 'revisions',
         'rvstart': '2022-09-30T23:59:59',
         'rvprop': 'timestamp',
         'titles': 'Data science',
         'format': 'json'
     }

    while True:
        resp = (requests
         .get(
             'https://en.wikipedia.org/w/api.php',
             params=params,
             )
         .json()
        )
        
        revs.extend(resp['query']['pages']['35458904']['revisions'])
        if 'continue' in resp:
            params.update(resp['continue'])
        else:
            break
    ts_list = [t['timestamp'] for t in revs]
    df = pd.DataFrame({'count': 1}, index=ts_list)
    df.index = pd.to_datetime(df.index)
    df = df.resample('M').sum()
    df.index = df.index.strftime('%Y-%m')
    df = df[~(df['count'] == 0)]
    return df.squeeze()


# In[9]:


def get_foobar_link_revs_asof():
    """
    Returns a list of revision IDs as of 2022-09-01 UTC of each linked page
    in revision id '1114361016' of the English Wikipedia.
    """
    resp = (requests
     .get(
         'https://en.wikipedia.org/w/api.php',
         params={
         'action': 'query',
         'prop': 'links',
        'pllimit': 'max',
         'revids': '1114361016',
         'format': 'json'
     }
         )
     .json()
    )
    
    title_list = ([e['title'] 
                   for e in resp['query']['pages']['11178']['links'] 
                   if not ':' in e['title']])

    ids = []    

    for tit in title_list:

        params={
             'action': 'query',
             'prop': 'revisions',
            'rvstart': '2022-09-01T23:59:59',
             'rvprop': 'ids|timestamp',
            'titles': tit,
             'format': 'json'
         }

        resp = (requests
         .get(
             'https://en.wikipedia.org/w/api.php',
             params=params,
             )
         .json()
        )

        page_id = list(resp['query']['pages'].keys())[0]
        ids.append(resp['query']['pages'][page_id]['revisions'][0]['revid'])
    
    return sorted(ids)


# In[ ]:


bearer_token = getpass.getpass()


# In[10]:


def followed_accounts(username, bearer_token):
    """
    Returns the id, username, name, location, and created_at of Twitter
    accounts followed by the given username, sorted by id.
    """
    resp = (requests
     .get(
        'https://api.twitter.com/2/users/by',
         headers={
            'Authorization': f'Bearer {bearer_token}'
        },
         params={
             'usernames': username
         })
     .json())
    user_id = resp['data'][0]['id']

    twt = []
    
    params={
            'user.fields': 'location,created_at'
             }
    
    while True:
        resp_2 = (requests
         .get(
            f'https://api.twitter.com/2/users/{user_id}/following',
             headers={
                'Authorization': f'Bearer {bearer_token}'
            },
             params=params)
         .json())

        twt.extend(resp_2['data'])
        if 'next_token' in resp_2['meta']:
            params.update(dict(
                {'pagination_token': resp_2['meta']['next_token']}
            ))
        else:
            break
        
    id_ = [int(dic['id']) for dic in twt]
    created_at = [dic['created_at'] for dic in twt]
    name = [dic['name'] for dic in twt]
    username = [dic['username'] for dic in twt]
    location = []
    for dic in twt:
        try:
            location.append(dic['location'])
        except:
            location.append(np.nan)
    
    df = (pd.DataFrame(
        {'id': id_, 'username': username, 'name': name, 'location': location,
         'created_at': created_at})
          .sort_values(by='id', ignore_index=True))
    return df


# In[ ]:


api_key = getpass.getpass()


# In[14]:


def user_videos(channel_id, api_key):
    """
    Returns a DataFrame of the video_id, title, and publish_time of videos
    published in 2020 of given YouTube channel_id, sorted by publish_time.
    """
    vids = []
    params={
         'part': 'id,snippet',
        'type': 'video',
         'channelId': channel_id,
         'publishedBefore': '2020-12-31T23:59:59Z',
        'publishedAfter': '2020-01-01T00:00:00Z',
        'order': 'date',
         'key': api_key,
        'maxResults': 50
     }

    while True:
        resp = (requests
         .get(
            'https://www.googleapis.com/youtube/v3/search',
             params=params
         )
         .json())
        
        vids.extend(resp['items'])
        if 'nextPageToken' in resp:
            params.update({'pageToken': resp['nextPageToken']})
        else:
            break
    
    video_id = [e['id'].get('videoId') for e in vids]
    title = [e['snippet'].get('title') for e in vids]
    publish_time = [e['snippet'].get('publishedAt') for e in vids]
    
    df = (pd.DataFrame(
    {'video_id': video_id, 'title': title, 'publish_time': publish_time})
      .sort_values(by='publish_time', ignore_index=True))
    
    return df


# In[ ]:




