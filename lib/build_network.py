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
    # tokens = word_tokenize(text)
    # additional_stop_words = ['MIT', 'Technology', 'Roadmaps', 'Roadmaping', 'Systems', 'Work']
    # stop_words = set(stopwords.words('english'))
    # stop_words.update(additional_stop_words)
    # filtered_tokens = [token.lower() for token in tokens if token.lower() not in stop_words]
    
    # stemmer = PorterStemmer()
    # stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
    
    # return stemmed_tokens
    pass
def get_main_keywords(tokens, n=10):
    # freq_dist = FreqDist(tokens)
    # main_keywords = freq_dist.most_common(n)
    # print(main_keywords)
    # return main_keywords
    pass

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
    
    page_list.append({'name':name, 'url': url, 'keywords': keywords, 'main_page':True })
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
                page_list.append({'name':sub_name, 'url': sub_url, 'keywords': sub_keywords, 'main_page':False})

    return page_list

@st.cache_resource
def build_dataframe(url,depth):
    dict_list = build_dictionary(url, depth)
    return pd.DataFrame.from_dict(dict_list)

@st.cache_resource
def export_dataframe(df, path):
    df.to_hdf(path,'df')
#%% Build Network
# Create an empty digraph
# Import pd.DataFrame of webpages
# Iterate through dataframe and create nodes with attributes url and keywords
# Add nodes using the names and the add_nodes_from([a, b])

def import_hdf_as_df(hdf_path):
    return pd.read_hdf(hdf_path)

def create_edges_from_df(G,df):
    # Iterate through webpage dataframe
    page_edges_list = [];
    
    for page in df.itertuples():
        page_edges = [];
        
        for sub_page in df.itertuples():
            
            if sub_page.name != page.name:
                if page.main_page:
                    G.add_edge(page.name,sub_page.name)
                    page_edges.append(sub_page.name)
                else:
                    kw1 = set(page.keywords)
                    kw2 = set(sub_page.keywords)
                    common = kw1.intersection(kw2)
                    if common and kw1 and kw2:
                        G.add_edge(page.name,sub_page.name)
                        page_edges.append(sub_page.name)
                    
        page_edges_list.append({'roadmap':page.name, 'directed_to':page_edges, 'display':True})        
    
    return page_edges_list
@st.cache_resource
def build_network(webpage_list_hdf_path, graph_hdf_data_path, table_hdf_data_path):
    G = nx.DiGraph()
    df = import_hdf_as_df(webpage_list_hdf_path)
    list_of_connections = create_edges_from_df(G,df)
    print(G)
    export_graph_as_hdf(G, graph_hdf_data_path)
    export_tablelist_as_hdf(list_of_connections, table_hdf_data_path)

#%% Save HDF
# Save data with headings: Node1 Node2 ListofKeywords
# Can use the networkx write_edgelist function to export a HDF
# Make sure to include the path, delimeter, data(list), and encoding
def export_graph_as_hdf(G, graph_hdf_data_path):
    pandas_edgelist_df = nx.to_pandas_edgelist(G)
    export_dataframe(pandas_edgelist_df, graph_hdf_data_path)

def export_tablelist_as_hdf(alist, table_hdf_data_path):
    alist_df = pd.DataFrame.from_dict(alist)
    export_dataframe(alist_df, table_hdf_data_path)

#%% Build Streamlit App


# Import digraph df
# Import edgelist df

# Populate a table 'Webpage Name'(string) 'Connects To' (multiselect) 'Show' (bool) using edglist df

#%% RUN

base_url = "https://roadmaps.mit.edu/index.php/Technology_Roadmaps"
depth = 1
page_path = Path('../data/page_path.h5')
graph_hdf_data_path = Path('../data/digraph_pandas_dataframe.h5')
table_hdf_data_path = Path('../data/table_pandas_dataframe.h5')
df = build_dataframe(base_url,depth)
export_dataframe(df,page_path)
build_network(page_path, graph_hdf_data_path, table_hdf_data_path)
