import numpy as np
import pandas as pd
import networkx as nx
import scipy.io as sio
from tqdm import tqdm
from utils import *

# Load the SIoT network and create a graph
SIoT = sio.loadmat('SIoT.mat')['SIoT']
G = nx.from_numpy_array(SIoT)
n_nodes = len(SIoT)

# Load the final positions of the objects
last_nodes_positions = sio.loadmat('last_nodes_positions.mat')['last_nodes_positions']

# Load the types
tipologie = sio.loadmat('tipologie.mat')['tipologie']

# Generate node times
tempi_type_objects = [150, 60, 150, 60, 60, 150, 300, 150, 150, 60, 60, 300, 300, 60, 60, 60]
tempi_objects = objects_time_creation(tipologie, tempi_type_objects) 

# Number of queries
n_query = 10

# Query data
query_data = pd.DataFrame(columns=['source_node', 'destinations', 'app', 'servizio', 'space_0', 'space_1', 'time_query'])

# loading bar
pbar = tqdm(range(n_query), desc='Progress', unit='query')

# Loop over the number of queries
i = 0
while i < n_query:

    # Choose type
    type_node, type_node_matrix_n = tipologia()  

    # Choose node
    source_node = nodo(tipologie, type_node)  

    # Choose application 
    app = applicazione(type_node)  

    # Choose requirements
    range_mediana = [500, 1500]
    space, time_query = requisiti(source_node, last_nodes_positions, app, range_mediana) 

    # Space and time
    range = 200
    time_range = 50
    destination_nodes, servizi = query(app, last_nodes_positions, tipologie, space, range, tempi_objects, time_query, time_range)

    for idx, nodes in enumerate(destination_nodes):

        row = pd.DataFrame({
            'source_node': [source_node],
            'destinations': [str(nodes)],
            'app': [app],
            'servizio': [servizi[idx]],
            'space_0': [round(space[0],4)],
            'space_1': [round(space[0],1)],
            'time_query': [time_query]
        })
        query_data = pd.concat([query_data, row], ignore_index=True)

        # Update the progress bar
        pbar.update(1)
        i += 1
        if i >= n_query:
            break


# Define the file path
file_path = "output_query.csv"

# Write into CSV file
query_data.to_csv(file_path, index=False)
