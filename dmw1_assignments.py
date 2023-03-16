#!/usr/bin/env python
# coding: utf-8

# In[4]:


import json
import re
import sqlite3
import os
import pandas as pd
import numpy as np
from xml.etree import ElementTree


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

