from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests
from pathlib import Path
import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
import os
import sys

# Future libraries
# import json
# import nltk
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# from nltk.probability import FreqDist
#%% Parse Websites

# Recursively go through website
# Save name by taking the id="firstHeading" or by removing the - MIT Technology Roadmapping from title
# Save {Name:name, URL:url, Keywords:[keywords]} to a list []
# Convert dictionary to pd.DataFrame with from_dict
# Export pd.DataFrame to json with to_json
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
    df.to_json(path, orient='columns')
#%% Build Network
# Create an empty digraph
# Import pd.DataFrame of webpages
# Iterate through dataframe and create nodes with attributes url and keywords
# Add nodes using the names and the add_nodes_from([a, b])

def import_json_as_df(json_path,listcol):
    return pd.read_json(json_path)

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
                    
        page_edges_list.append({'roadmap':page.name, 'directed_to':page_edges, 'display':True, 'url':page.url})        
    
    return page_edges_list

@st.cache_resource
def build_network(webpage_list_json_path, graph_json_data_path, table_json_data_path):
    G = nx.DiGraph()
    df = import_json_as_df(webpage_list_json_path,'keywords')
    list_of_connections = create_edges_from_df(G,df)
    
    export_graph_as_json(G, graph_json_data_path)
    export_tablelist_as_json(list_of_connections, table_json_data_path)


#%% Save json
# Save data with headings: Node1 Node2 ListofKeywords
# Can use the networkx write_edgelist function to export a json
# Make sure to include the path, delimeter, data(list), and encoding
def export_graph_as_json(G, graph_json_data_path):
    pandas_edgelist_df = nx.to_pandas_edgelist(G)
    export_dataframe(pandas_edgelist_df, graph_json_data_path)

def export_tablelist_as_json(alist, table_json_data_path):
    alist_df = pd.DataFrame.from_dict(alist)
    export_dataframe(alist_df, table_json_data_path)

#%% Build Streamlit App

# Update edge_graph
def update_digraph(df):
    new_df_interact = pd.DataFrame({'source':[],'target':[]})
    for node in df.itertuples():
        if node.display:
            if type(node.directed_to) != list:
                targets = list(node.directed_to.split(',\n'))
            else:
                targets = node.directed_to
            for target in targets:
                new_df_interact.loc[len(new_df_interact.index)] = [node.roadmap, target]

    return new_df_interact

# Create pyvis network and display

def generate_pyvis_network(G, height_px):
    # Initiate PyVis network object
    node_net = Network(height=f'{height_px}px', 
                       bgcolor='white',
                       font_color='white', 
                       directed=True)
    # Take Networkx graph and translate it to a PyVis graph format
    node_net.from_nx(G)
    node_net.set_edge_smooth('dynamic')
    for anode in node_net.nodes:
        anode['shape'] = 'box'
        anode['color'] = 'blue'
        url_path = st.session_state.df_table['url'].loc[st.session_state.df_table['roadmap']==anode['label']].tolist()
        url_path = ''.join(url_path)
        anode['title'] = f"<a href='{url_path}' target='_blank; rel='noopener noreferrer'>{anode['label']}</a>"
    
    # Genereate network with specific layout settings
    node_net.repulsion(node_distance=420, central_gravity=0.33,
                       spring_length=110, spring_strength=0.10,
                       damping=0.95)
    
    # When running app locally, save pyvis html file to read later,
    # When running on Streamlit Cloud, use a tmp folder
    # Attempt both using try-except
    
    # Save and read graph as HTML file (on Streamlit Sharing)
    try:
        path = '/tmp'
        node_net.save_graph(f'{path}/pyvis_graph.html')
        HtmlFile = open(f'{path}/pyvis_graph.html','r',encoding='utf-8')
    
    # Save and read graph as HTML file (locally)
    except:
        path = './html'
        node_net.save_graph(f'{path}/pyvis_graph.html')
        HtmlFile = open(f'{path}/pyvis_graph.html','r',encoding='utf-8')
     
    # Load HTML file in HTML component for display on Streamlit page
    components.html(HtmlFile.read(), height=height_px)
    with open(f'{path}/pyvis_graph.html') as f:
        htmltext = f.read()
    return htmltext

# Populate a table 'Webpage Name'(string) 'Connects To' (multiselect) 'Show' (bool) using edglist df

def generate_table(df):
    # df['directed_to'] = df['directed_to'].apply(lambda x: ',\n'.join(x))
    df = df.sort_values(by=['roadmap'])
    st.dataframe(df,hide_index=True,
                column_config={'roadmap':'Roadmap',
                               'directed_to':st.column_config.ListColumn(label='Directed To', width='large'),
                               'display':None,
                               'url':None},
                use_container_width=True)

def update_table(df, from_node, to_nodes):
    url = df['url'].loc[df['roadmap']== from_node].item()
    df = df.loc[df['roadmap']!= from_node]
    new_row_array = np.asarray([from_node, to_nodes, True, url],dtype='object')
    df.loc[len(df.index)] = new_row_array
    df.sort_values(by=['roadmap'])
    return df

def convert_df_to_json(df):
    # df['directed_to'] = df['directed_to'].apply(lambda x: list(x.split(',\n')))
    return df.to_json()






# Initiate streamlit
def run_streamlit(graph_json_data_path,table_json_data_path):
    
    # Read datasets
    # if 'df_interact' not in st.session_state:
        # st.session_state.df_interact = pd.read_json(graph_json_data_path)
    if 'df_table' not in st.session_state:
        st.session_state.df_table = pd.read_json(table_json_data_path)
        st.session_state.df_interact = update_digraph(st.session_state.df_table)
    
    # Set Header Title
    st.title('Network Graph Visualization of Technology Roadmaps Interactions')
      
    # Create Upload json button
    with st.sidebar:
        df_uploaded_json = st.file_uploader('Upload Node json',type='json')
    if df_uploaded_json:
        st.session_state.df_table = import_json_as_df(df_uploaded_json,'directed_to')
        st.session_state.df_interact = update_digraph(st.session_state.df_table)
    
    
    # Allow users to select nodes the wish to visualize using multi-select
    # Define selection options and sort alphabetically
    node_list = st.session_state.df_table['roadmap'].tolist()
    node_list.sort()
    
    with st.sidebar:
        # Implement multiselect dropdown menu for option selection (returns list)
        selected_nodes = st.multiselect('Select Roadmaps(s) to visualize', node_list)
        select_all = st.checkbox('Visualize All')
        
    new_directed_to_nodes = []
    with st.sidebar:
        
        select_edit_node = st.selectbox('Select Roadmap to Edit',node_list, index=None)
        if select_edit_node:
            current_connections = st.session_state.df_table['directed_to'].loc[st.session_state.df_table['roadmap']== select_edit_node].tolist()
            current_connections = current_connections[0][:]
            current_connections.sort()
            st.text(set(node_list) - set(current_connections))
            new_directed_to_nodes = st.multiselect('Select Which Roadmaps to Direct To',options=node_list, default=current_connections)
            update_table_requested = st.button('Update Table')
            
            if update_table_requested:
                st.session_state.df_table = update_table(st.session_state.df_table, select_edit_node, new_directed_to_nodes)
                st.session_state.df_interact = update_digraph(st.session_state.df_table)
    
    with st.sidebar:
        # Add a button to download new table
        new_table_data = convert_df_to_json(st.session_state.df_table)
        st.download_button('Download Table As JSON',data=new_table_data,file_name='roadmap_connection_table.json')
        
    
    # Flow Control
    # Set info message on initial site load
    if select_all:
        # Code for filtering dataframe and generating network
        st.session_state.df_select = st.session_state.df_interact
        st.session_state.df_select = st.session_state.df_select.reset_index(drop=True)
        # Create networkx graph object from pandas dataframe
        G = nx.from_pandas_edgelist(st.session_state.df_select, 'source', 'target',create_using=nx.DiGraph())
        # Create pyvis network
        HtmlFile = generate_pyvis_network(G,400)
        # Add button to download network as html
        with st.sidebar:
            st.download_button('Download Visualization As HTML',data=HtmlFile,file_name='roadmap_visualization.html')
    elif len(selected_nodes) == 0:
        st.text('Please choose at least 1 Roadmap to get started')
        
    # Create network graph when user selects >= 1 item
    else:
        # Code for filtering dataframe and generating network
        st.session_state.df_select = st.session_state.df_interact.loc[st.session_state.df_interact['source'].isin(selected_nodes) | st.session_state.df_interact['target'].isin(selected_nodes)]
        # st.session_state.df_select = st.session_state.df_select.reset_index(drop=True)
        # Create networkx graph object from pandas dataframe
        G = nx.from_pandas_edgelist(st.session_state.df_select, 'source', 'target')
        G = nx.DiGraph()
        st.dataframe(st.session_state.df_select)
        # Create pyvis network
        HtmlFile = generate_pyvis_network(G,400)
        # Add button to download network as html
        with st.sidebar:
            st.download_button('Download Visualization As HTML',data=HtmlFile,file_name='roadmap_visualization.html')
        
    # Create table
    generate_table(st.session_state.df_table)
    # If table edited, generate new graph
    # if not st.session_state.df_table.equals(df_edited):
    #     st.session_state.df_interact = update_digraph(df_edited)


#%% RUN

# Make the webpage wide
st.set_page_config(layout="wide")

# base_url = "https://roadmaps.mit.edu/index.php/Technology_Roadmaps"
# depth = 1

# get the path of your current script (main.py in here)
script_path = os.path.dirname(os.path.abspath(__file__))

# add 'streamlit_app' folder to sys.path
app_folder = os.path.join(script_path, 'data')
sys.path.append(app_folder)
app_folder = os.path.join(script_path, 'html')
sys.path.append(app_folder)

page_path = Path('./data/page_path.json')
graph_json_data_path = Path('./data/digraph_pandas_dataframe.json')
table_json_data_path = Path('./data/table_pandas_dataframe.json')

# df = build_dataframe(base_url,depth)
# export_dataframe(df,page_path)
# build_network(page_path, graph_json_data_path, table_json_data_path)
  
run_streamlit(graph_json_data_path,table_json_data_path)
