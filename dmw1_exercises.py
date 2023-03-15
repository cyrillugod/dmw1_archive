#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
import numpy as np
import csv
import json
import re
import os


# In[2]:


def list_recipes(recipe_xml):
    """Returns a list of ingredients sorted by order of appearance."""
    root = ET.fromstring(recipe_xml)
    return ([child.text for child in root.findall("ingredient")])


# In[3]:


def catalog_sizes(catalog_xml):
    """Returns unique catalog sizes."""
    size = []
    root = ET.fromstring(catalog_xml)
    for child in root.findall(".//size"):
        if child.get("description") not in size:
            size.append(child.get("description"))
    return size


# In[4]:


def read_fantasy(books_xml):
    """Returns a dataframe consisting of books in the Fantasy genre."""
    df = pd.read_xml(books_xml, parser="etree")
    return df.loc[df["genre"] == "Fantasy"]


# In[5]:


def dct2str(dct):
    """Accepts a dictionary and returns it as a JSON string."""
    return json.dumps(dct)


def dct2file(dct):
    """Accepts a disctionary and saves it as a JSON file."""
    with open("dct.json", "w") as fp:
        json.dump(dct, fp)


# In[7]:


def count_journals():
    """Returns a list of journals and count sorted by decreasing count and
    ascending lexicographic order of the journal name."""
    df = pd.read_json(
        "/mnt/data/public/covid19-lake/alleninstitute/CORD19/json/metadata/part-00000-81803174-7752-4489-8eeb-081318af9653-c000.json",
        lines=True)
    df = df[['journal']]
    df = df.groupby('journal').size().reset_index()
    df.rename(columns={0: 'count'}, inplace=True)
    df.sort_values(by=['count', 'journal'], ascending=[0, 1], inplace=True)
    return list(zip(df['journal'], df['count']))


# In[8]:


def business_labels():
    """Returns a series with labels for businesses listed on Yelp."""
    df = pd.read_json(
        "/mnt/data/public/yelp/challenge12/yelp_dataset/yelp_academic_dataset_photo.json",
        lines=True)
    return df.groupby("business_id")["label"].unique().apply(set)


# In[9]:


def get_businesses():
    """Returns a dataframe of Yelp entries."""
    bus_list = []
    with open(
            "/mnt/data/public/yelp/challenge12/yelp_dataset/yelp_academic_dataset_business.json",
            "r") as f:
        for line in f:
            bus_list.append(json.loads(line))
    bus_list = bus_list[:10_000]
    df = pd.json_normalize(bus_list)
    df.set_index("business_id", inplace=True)
    return df


# In[10]:


def pop_ncr():
    """Returns a dataframe of the total population per province, city,
    municipality, and barangay."""
    df = pd.read_excel("/mnt/data/public/census/2020/NCR.xlsx",
                       sheet_name="NCR by barangay",
                       usecols=[2, 3],
                       skiprows=4)
    df.dropna(inplace=True)
    df.rename(columns={
        "and Barangay": "Province, City, Municipality, and Barangay",
        "Population": "Total Population"
    },
        inplace=True)
    return df


# In[11]:


def dump_airbnb_beds():
    """Creates an excel file containing the host location and price for
    each bed type listed in Airbnb."""
    df = pd.read_csv(
        "/mnt/data/public/insideairbnb/data.insideairbnb.com/united-kingdom/england/london/2015-04-06/data/listings.csv.gz",
        compression="gzip",
        usecols=["bed_type", "host_location", "price"])
    types = df["bed_type"].unique().tolist()
    with pd.ExcelWriter("airbnb.xlsx") as f:
        for t in sorted(types):
            df.loc[df["bed_type"] == t][["host_location",
                                         "price"]].to_excel(f,
                                                            sheet_name=t,
                                                            index=False)


# In[13]:


def age_counts():
    """Returns a dataframe of the total and household population for male,
    female, and both sexes in the Philippines."""
    df = pd.read_excel(
        "/mnt/data/public/census/2015/_PHILIPPINES_Statistical Tables.xls",
        sheet_name=["T2", "T3"],
        skiprows=2,
        usecols=[0, 1, 2, 3, 4])
    df["T2"].drop(columns="Unnamed: 2", inplace=True)
    df["T3"].drop(columns="Unnamed: 4", inplace=True)
    df2 = pd.merge(df["T2"], df["T3"], on="Single-Year Age")
    df2.dropna(inplace=True)
    df2.rename(columns={
        "Both Sexes_x": "Both Sexes (Total)",
        "Male_x": "Male (Total)",
        "Female_x": "Female (Total)",
        "Both Sexes_y": "Both Sexes (Household)",
        "Male_y": "Male (Household)",
        "Female_y": "Female (Household)"
    },
        inplace=True)
    df2.set_index("Single-Year Age", inplace=True)
    return df2.iloc[1:]


# In[15]:


def is_date(test_str):
    """Returns True if date format is valid and returns False otherwise"""
    return bool(
        re.match(r"\d{4}([-/])\d{2}\1\d{2}\b|\d{2}([-/])\d{2}\2\d{4}\b",
                 test_str))


# In[16]:


def find_data(text):
    """Returns occurrences of 'data set' or 'dataset' in passed text in a
    chronological list."""
    pattern = re.compile(r"data\s?set\b")
    matches = pattern.findall(text)
    return ([m for m in matches])


# In[17]:


def find_lamb(text):
    """Returns all phrases that begins with 'little' and ends with 'lamb' in
    chronological order."""
    matches = re.findall(r"little[^\n,]*\blamb", text)
    return matches


# In[18]:


def repeat_alternate(text):
    """Retuns a string with every other word duplicated"""
    return re.sub("(\w+'?\w?\s*)(\w+'?\w?)", r"\1\1\2", text)


# In[19]:


def whats_on_the_bus(text):
    """Returns unique items that are on the bus from the text"""
    matches = re.findall(r"\w+(?=\son\sthe\sbus)", text)
    return list(set(matches))


# In[20]:


def to_list(text):
    """Returns list of items in text which are delimited by ',', '+',
    and 'and'"""
    return re.split("and|\+|,", text)


# In[21]:


def march_product(text):
    """Returns product of each m by n pair in text"""
    mult = re.findall(r"(\d+)\sby\s(\d+)", text)
    return [int(i) * int(j) for (i, j) in mult]


# In[22]:


def get_big(items):
    """Returns list of items beginning with 'Big' with an SKU that is not all
    numbers"""
    item = re.finditer("([A-Z]\d+|\d?[A-Z]+)\s(Big\s.*)", items)
    return [i.group(2) for i in item]


# In[23]:


def find_chris(names):
    """Returns a list of first names with a case-insensitive 'Chris' but a 
    last name that does not start with 'B' or 'M'"""
    name_list = re.findall("(?<=\s)\w*?chris\w*(?=\s[^BM])", names, re.I)
    return name_list


# In[24]:


def get_client(server_log):
    """Returns a list of client IP, date/time of server access, and status 
    code tuples from %logoff"""
    log = re.findall(
        r'(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}).*(\d{2}/\w*/\d{4}:\d{2}:\d{2}:\d{2}\s\+\d{4}).*HTTP/1.1"\s(\d{3})\s',
        server_log)
    return log


# In[25]:


def create_tables():
    """Creates an in-memory SQLite database with the following columns: 
    'full_name' (text, not nullable), 'age' (int, nullable),
    'rating' (float, not nullable), 'remarks' (text, nullable)"""
    conn = sqlite3.connect(":memory:")
    emp = conn.execute("""
    CREATE TABLE IF NOT EXISTS employee (
        full_name TEXT NOT NULL,
        age INT,
        rating REAL NOT NULL,
        remarks TEXT
    )
    """)
    return emp


# In[26]:


def insert_values(conn, rows):
    """Saves the rows from 'rows' into the table 'players'"""
    conn.executemany(
        "INSERT INTO players VALUES (" + "?, " * (len(rows[0]) - 1) + "?)",
        rows)


# In[27]:


def append_values(conn, df):
    """Appends content of dataframe 'df' to table 'reactions'"""
    conn.executemany(
        "INSERT INTO reactions VALUES (" + "?, " *
        (len(df.values.tolist()[0]) - 1) + "?)", df.values.tolist())


# In[28]:


def read_table(db):
    """Reads sqlite database file 'db' and returns contents of table
    'transactions' as a dataframe"""
    engine = create_engine("sqlite:///" + db)
    tran = pd.read_sql("SELECT * FROM transactions", engine)
    return tran


# In[29]:


def stocks_more_than_5():
    """Returns an SQL statement which returns rows of table 'transactions'
    with columns 'StockCode', 'Description', and 'Quantity' for entries in 
    'Quantity' greater than 5."""
    return """SELECT
        StockCode,
        Description,
        Quantity
    FROM transactions
    WHERE Quantity > 5
    """


# In[30]:


def get_invoices():
    """Returns an SQL statement that returns the rows of table 'transactions'
    grouped by 'Invoice No' with the following columns: 'InvoiceNo',
    'ItemCount', 'TotalQuantity'"""
    return """SELECT
            InvoiceNo,
            COUNT(*) AS ItemCount,
            SUM(Quantity) AS TotalQuantity
        FROM transactions
        GROUP BY InvoiceNo
        ORDER BY ItemCount DESC
    """


# In[31]:


def white_department(conn):
    """Returns a Series with 'department_id' as index and number of rows in
    'products' as values. Only returns rows where 'product_name' starts with
    case-insensitive word 'White' for that 'department_id' Rows are sorted in
    decreasing order."""
    return pd.read_sql("""
        SELECT
            department_id,
            COUNT(*) as count
        FROM products
        WHERE product_name LIKE "White %"
        GROUP BY department_id
        ORDER BY count DESC
    """,
                       conn,
                       index_col="department_id").squeeze()


# In[32]:


def count_aisle_products(conn):
    """Returns a dataframe with the columns 'aisle_id', 'aisle', and
    'product_count' where 'product_count is less than 100. Sorted by
    increasing 'product_count."""
    return pd.read_sql(
        """
    SELECT
        p.aisle_id AS aisle_id,
        a.aisle AS aisle,
        COUNT(p.product_id) AS product_count
    FROM products p
    LEFT JOIN aisles a
    ON p.aisle_id = a.aisle_id
    GROUP BY p.aisle_id
    HAVING product_count < 100
    ORDER BY product_count ASC
    """, conn)


# In[33]:


def concat_cols(conn):
    """Sets the value of 'col3' for rows with even values of 'col2' to the
    concatenation of 'col1' and 'col2'"""
    conn.execute("""
    UPDATE cols
    SET col3 = col1 || col2
    WHERE MOD(col2, 2) = 0
    """)()


# In[34]:


def del_row(conn):
    """Deletes rows where the 'col1' value is 'd'"""
    conn.execute("""
        DELETE FROM cols
        WHERE col1 = "d"
    """)


# In[1]:


def summarize_items():
    """Returns a DataFrame corresponding to the OASIS Item Summary table."""
    url = 'https://jojie.accesslab.aim.edu:9095/oasis/oi_item_summary.html'
    list_df = pd.read_html(url, header=0)
    return list_df[0]


# In[2]:


def item_values():
    """Returns a DataFrame corresponding to the OASIS Item Values table."""
    url = 'https://jojie.accesslab.aim.edu:9095/oasis/oi_d0700.html'
    list_df = pd.read_html(url, header=1)
    return list_df[2]


# In[3]:


def all_item_values():
    """Returns a DataFrame of all items values."""
    url = ('https://jojie.accesslab.aim.edu:9095'
                    '/oasis/oi_item_summary.html')
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.text)
    url_list= ([(i.text.strip(), urljoin(url, i.get('href'))) 
                for i in soup.select('a[href]')])
    df_list = []
    for text, url_ in url_list:
        df_item_val = pd.read_html(url_, match='Item Value', header=1)[0]
        df_item_val.insert(loc=0, column='Item', value=text)
        df_list.append(df_item_val)
    return pd.concat(df_list, ignore_index=True)


# In[4]:


def all_item_edits():
    """Returns ID, Type, Severity, and Text of all OASIS edits."""
    url = 'https://jojie.accesslab.aim.edu:9095/oasis/'
    soup = bs4.BeautifulSoup((requests.get(url)).content)
    url = urljoin(url, soup.select_one('frame').get('src'))
    soup = bs4.BeautifulSoup((requests.get(url)).content)
    url = urljoin(url, soup.select_one('a[href^=oe]').get('href'))
    soup = bs4.BeautifulSoup((requests.get(url)).content)
    url = urljoin(url, soup.select_one('a[href^=oe_edit]').get('href'))
    df = pd.read_html(url, header=0)[0]
    soup = bs4.BeautifulSoup((requests.get(url)).content)
    edit_list = [e.get('href') for e in soup.select('a[href]')]
    list_text = []
    for edit in edit_list:
        oe_html = urljoin(url, edit)
        df_oe = pd.read_html(oe_html, header=0)[1]
        edit_text = (df_oe[df_oe['Property'] == 'Edit Text']['Specification']
                         .values[0])
        list_text.append(edit_text)
    df['Edit Text'] = list_text
    df.rename(columns={'Type': 'Edit Type'}, inplace=True)
    return df.sort_values(by='Edit ID', ignore_index=True, ascending=False)


# In[5]:


def article_info(url):
    """Returns the title, author, and published timestamp from Rappler.com."""
    dic_art = {}
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.text)
    title = soup.select_one("div.post-single__header h1").text
    dic_art['title'] = title
    author = soup.select_one("div.post-single__authors").text
    dic_art['author'] = author
    published = soup.select_one("span.posted-on > time").text
    dic_art['published'] = published    
    return dic_art


# In[6]:


def latest_news():
    """Returns title, category, and timestamp of latest news from Rappler."""
    latest_dict = {}
    url = 'https://jojie.accesslab.aim.edu:9095/rappler/'
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.text)
    title = soup.select_one('div.latest-stories h3 a[href]').text.strip()
    latest_dict['title'] = title
    art_url = (urljoin(
        url, soup.select_one('div.latest-stories h3 a[href]').get('href')))
    resp_latest = requests.get(art_url)
    soup_latest = bs4.BeautifulSoup(resp_latest.text)
    category = (soup_latest.select_one('div.post-single__header a[href]')
                .text.strip())
    latest_dict['category'] = category
    date_string = soup_latest.select_one('time').get('datetime')
    timestamp = (datetime.datetime.strptime(
        date_string, '%Y-%m-%dT%H:%M:%S%z')
                 .astimezone(datetime.timezone.utc))
    latest_dict['timestamp'] = timestamp
    return latest_dict


# In[7]:


def get_category_posts(category):
    """Returns titles of posts from Rappler by category."""
    url = 'https://jojie.accesslab.aim.edu:9095/rappler/'
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.text)
    categ_dict = {}
    list_cat = [i.text.strip() for i in soup.select('a.post-card__category')]
    list_tit = [i.text.strip() for i in soup.select('h3.post-card__title')]
    df = pd.DataFrame({"category": list_cat, "titles": list_tit})
    df['category'] = df['category'].str.lower()
    df = df.groupby('category')['titles'].unique()
    return [e for e in df[category.lower()].tolist() if 'LIVE' not in e]


# In[8]:


def subsection_posts(url):
    """Returns a DataFrame of title and timestamp of posts in a subsection."""
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.content)
    title_list = ([e.text.strip() for e in soup.select(
        'div.archive-article__content h2 a[href]')])
    timestamp_list = ([d.text.strip() for d in soup.select(
        'div.archive-article__content time')])
    df = pd.DataFrame({'title': title_list, 'timestamp': timestamp_list})
    return df


# In[9]:


def subsection_authors(url):
    """Returns a DataFrame of title and author of posts under a subsection."""
    proxies = {
    'http': 'http://206.189.157.23'
    }
    response = requests.get(url, proxies=proxies)
    soup = bs4.BeautifulSoup(response.content)
    title_list = ([e.text.strip() for e in soup.select(
        'div.archive-article__content h2 a[href]')])
    art_list = ([e.get('href') for e in soup.select(
        'div.archive-article__content h2 a[href]')])
    auth_list = []
    for art in art_list:
        soup_art = (bs4.BeautifulSoup(
            requests.get(art, proxies=proxies).content))
        auth_list.append(soup_art.select_one('.post-single__authors').text)
    timestamp_list = ([d.text.strip() for d in soup.select(
        'div.archive-article__content time')])
    df = pd.DataFrame({'title': title_list, 'author': auth_list})
    return df.sort_values(by='title', ignore_index=True)


# In[11]:


def download_images(url):
    """Downloads post images to 'images' directory from Rappler subsection."""
    response = requests.get(url)
    soup = bs4.BeautifulSoup(response.content)
    img_list = ([e.get('src') 
                 for e in soup.select('figure.archive-article-image img')])
    if not os.path.isdir('./images'):
        os.mkdir('./images')
    for img in img_list:
        img_fname = basename(urlparse(img).path)
        with open(os.path.join('images', img_fname), 'wb') as f:
            f.write((requests.get(img)).content)

