from bs4 import BeautifulSoup
import pandas as pd
import requests
from pathlib import Path
import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.probability import FreqDist
#%% Parse Websites

# Recursively go through website
# Save name by taking the id="firstHeading" or by removing the - MIT Technology Roadmapping from title
# Save {Name:name, URL:url, Keywords:[keywords]} to a list []
# Convert dictionary to pd.DataFrame with from_dict
# Export pd.DataFrame to csv with to_csv
def get_page_content(url):
    response = requests.get(url)
    return response.text

def get_page_name(soup):
    name_tag = soup.find('h1',id='firstHeading')
    return name_tag.text

def parse_html(content):
    soup = BeautifulSoup(content, 'html.parser')
    return soup

def refine_html(soup):
    return soup.get_text()

def preprocess_text(text):
    tokens = word_tokenize(text)
    additional_stop_words = ['MIT', 'Technology', 'Roadmaps', 'Roadmaping', 'Systems', 'Work']
    stop_words = set(stopwords.words('english'))
    stop_words.update(additional_stop_words)
    filtered_tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
    
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    
    return stemmed_tokens

def get_main_keywords(tokens, n=10):
    freq_dist = FreqDist(tokens)
    main_keywords = freq_dist.most_common(n)
    print(main_keywords)
    return main_keywords

def get_keywords(soup):
    # refined_content = refine_html(soup)
    # preprocessed_text = preprocess_text(refined_content)
    # keywords = get_main_keywords(preprocessed_text)
    keywords = [];
    return keywords

def build_dictionary(url, depth):
    page_list = []
    name_list = []
    page_content = get_page_content(url)
    parsed_content = parse_html(page_content)
    
    keywords = get_keywords(parsed_content)
    name = get_page_name(parsed_content)
    
    page_list.append({'name':name, 'url': url, 'keywords': keywords })
    name_list.append(name)
    if depth > 0:
        links = parsed_content.find_all('a', href=True)
        for link in links:
            sub_url = link['href']
            if not sub_url.startswith('https://roadmaps'):
                continue
            sub_page_content = get_page_content(sub_url)
            sub_parsed_content = parse_html(sub_page_content)
            
            sub_keywords = get_keywords(sub_parsed_content)
            sub_name = get_page_name(sub_parsed_content)
            
            if sub_name not in name_list:
                name_list.append(sub_name)
                page_list.append({'name':sub_name, 'url': sub_url, 'keywords': sub_keywords, 'display':True })

    return page_list

def build_dataframe(url,depth):
    dict_list = build_dictionary(url, depth)
    return pd.DataFrame.from_dict(dict_list)

def export_dataframe(df, path):
    df.to_csv(path, index=False)
#%% Build Network
# Create an empty digraph
# Import pd.DataFrame of webpages
# Iterate through dataframe and create nodes with attributes url and keywords
# Add nodes using the names and the add_nodes_from([a, b])
#%% Save CSV
# Save data with headings: Node1 Node2 ListofKeywords
# Can use the networkx write_edgelist function to export a csv
# Make sure to include the path, delimeter, data(list), and encoding
#%% Build Streamlit App

# use streamlit-tags (pip install streamlit-tags) which returns a list
# can provide an initial value using the value=list arg

#%% DEBUGRUN
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('stopwords')
except LookupError:
    nltk.download('stopwords')

base_url = "https://roadmaps.mit.edu/index.php/Ground_Based_Radar_For_Space_Situational_Awareness"
depth = 0
page_path = Path('../data/page_path.csv')

df = build_dataframe(base_url,depth)
export_dataframe(df,page_path)

#print(df)