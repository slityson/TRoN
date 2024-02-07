from bs4 import BeautifulSoup
import pandas as pd
import requests
from pathlib import Path
import networkx as nx
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
import json

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
    df.to_csv(path, index=False)
#%% Build Network
# Create an empty digraph
# Import pd.DataFrame of webpages
# Iterate through dataframe and create nodes with attributes url and keywords
# Add nodes using the names and the add_nodes_from([a, b])

def import_csv_as_df(csv_path,listcol):
    df_imported = pd.read_csv(csv_path)
    df_imported[listcol] = json.load(df_imported[listcol])
    return pd.read_csv(csv_path)

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
def build_network(webpage_list_csv_path, graph_csv_data_path, table_csv_data_path):
    G = nx.DiGraph()
    df = import_csv_as_df(webpage_list_csv_path,'keywords')
    list_of_connections = create_edges_from_df(G,df)
    
    export_graph_as_csv(G, graph_csv_data_path)
    export_tablelist_as_csv(list_of_connections, table_csv_data_path)


#%% Save CSV
# Save data with headings: Node1 Node2 ListofKeywords
# Can use the networkx write_edgelist function to export a csv
# Make sure to include the path, delimeter, data(list), and encoding
def export_graph_as_csv(G, graph_csv_data_path):
    pandas_edgelist_df = nx.to_pandas_edgelist(G)
    export_dataframe(pandas_edgelist_df, graph_csv_data_path)

def export_tablelist_as_csv(alist, table_csv_data_path):
    alist_df = pd.DataFrame.from_dict(alist)
    export_dataframe(alist_df, table_csv_data_path)

#%% Build Streamlit App


# Import digraph df

# Update edge_graph
def update_digraph(df):
    new_df_interact = pd.DataFrame({'source':[],'target':[]})
    for node in df.itertuples():
        if node.display:
            targets = json.loads(node.directed_to)
            st.text(targets)
            for target in targets:
                new_df_interact.loc[len(new_df_interact.index)] = [node.roadmap, target]

    return new_df_interact

# Create pyvis network and display

def generate_pyvis_network(G):
    # Initiate PyVis network object
    node_net = Network(height='800px', bgcolor='#222222',font_color='white')
    
    # Take Networkx graph and translate it to a PyVis graph format
    node_net.from_nx(G)
    
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
        path = '/html'
        node_net.save_graph('pyvis_graph.html')
        print(f'{path}/pyvis_graph.html')
        HtmlFile = open('pyvis_graph.html','r',encoding='utf-8')
     
    # Load HTML file in HTML component for display on Streamlit page
    components.html(HtmlFile.read(), height=435)

# Populate a table 'Webpage Name'(string) 'Connects To' (multiselect) 'Show' (bool) using edglist df
def debugmark():
    st.text('kiss')

def generate_table(df):
    edited_df = pd.DataFrame()
    edited_df = st.data_editor(df,num_rows='dynamic',hide_index=True ,column_config={'roadmap':'Roadmap',
                                                                                     'directed_to':st.column_config.TextColumn(label='Directed To'),
                                                                                     'display':'Display'})
    return edited_df
@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

# Initiate streamlit
def initiate_streamlit(graph_csv_data_path,table_csv_data_path):
    # Read datasets
    df_interact = pd.read_csv(graph_csv_data_path)
    df_table = pd.read_csv(table_csv_data_path)
    
    # Set Header Title
    st.title('Network Graph Visualization of Technology Roadmaps Interactions')
    # Generate column containers
    leftcol, rightcol = st.columns(2)
    
    # Create Upload CSV button
    with leftcol:
        df_uploaded_csv = st.file_uploader('Up Load Node CSV',type='csv')
    if df_uploaded_csv:
        df_table = import_csv_as_df(df_uploaded_csv,'directed_to')
        df_interact = update_digraph(df_uploaded_csv)
    
    
    # Allow users to select nodes the wish to visualize using multi-select
    # Define selection options and sort alphabetically
    node_list = df_table['roadmap'].tolist()
    node_list.sort()
    
    with leftcol:
        # Implement multiselect dropdown menu for option selection (returns list)
        selected_nodes = st.multiselect('Select Roadmaps(s) to visualize', node_list)
        # Create table
        df_edited = generate_table(df_table)
        # If table edited, generate new graph
        if not df_table.equals(df_edited):
            df_interact = update_digraph(df_edited)
        # Add a button to download new table
        csv = convert_df(df_table)
        st.download_button('Download Table As CSV',data=csv,file_name='roadmap_connection_table.csv')
    # Flow Control
    # Set info message on initial site load
    if len(selected_nodes) == 0:
        with rightcol:
            st.text('Please choose at least 1 Roadmap to get started')
        
    # Create network graph when user selects >= 1 item
    else:
        with rightcol:
            # Code for filtering dataframe and generating network
            #df_select = df_interact.loc[df_interact['source'].isin(selected_nodes) | df_interact['target'].isin(selected_nodes)]
            #df_select = df_select.reset_index(drop=True)
            st.text(df_interact)
            # Create networkx graph object from pandas dataframe
            G = nx.from_pandas_edgelist(df_interact, 'source', 'target')
            
            # Create pyvis network
            generate_pyvis_network(G)
        
        

#%% RUN

# Make the webpage wide
st.set_page_config(layout="wide")

base_url = "https://roadmaps.mit.edu/index.php/Technology_Roadmaps"
depth = 1

page_path = Path('../data/page_path.csv')
graph_csv_data_path = Path('../data/digraph_pandas_dataframe.csv')
table_csv_data_path = Path('../data/table_pandas_dataframe.csv')

df = build_dataframe(base_url,depth)
export_dataframe(df,page_path)
build_network(page_path, graph_csv_data_path, table_csv_data_path)
  
initiate_streamlit(graph_csv_data_path,table_csv_data_path)