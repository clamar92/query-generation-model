import numpy as np
import random


def objects_time_creation(tipologie, tempi_type_objects):
    """
    Create a matrix of information times; for each type of node there will be
    different sampling times.
    """

    # Initialize the list for storing the time objects
    tempi_objects = [None]*16216

    # Create times
    for i in range(16216):
        # Ensure tipologie[i] is an integer
        type_index = int(tipologie[i]) - 1
        # Check if type_index is within the valid range
        if type_index < len(tempi_type_objects):
            start = np.random.randint(0, 6)  # Random integer between 0 and 5
            step = tempi_type_objects[type_index]
            end = 12*60*60  # 12 hours in seconds
            tempi_objects[i] = np.arange(start, end, step)
        else:
            print(f"Invalid type_index: {type_index}. Skipping this iteration.")

    return tempi_objects



def tipologia():
    # choice of type

    # types: 1: SmartPhone, 2: Car, 3: Tablet, 4: Wearable Medical Devices,    
    #        5: SmartWatch, 6: Pc,  7: Printer, 8: Home sensors, 
    #        9: Point of Interest, 10. Environments and Weather,
    #       11: Transportation, 12: Indicator, 13: Waste, 14: Light, 
    #       15: Parking, 16: Alarms

    # objects: 1: SmartPhone, 2: Car, 3: Tablet, 4: Wearable Medical Devices,    
    #          5: SmartWatch, 6: Pc,  7: Printer, 8: Home sensors, 9: Parking,           
    #         10: Vehicle Speed, 11: Vehicle Counter, 12: Irrigation, 
    #         13: Agriculture, 14: Environmental, 15: Panel, 16: Sound,               
    #         17: Light, 18: Air, 19: Temperature, 20: Point of Interest
    #         21: bus, 22: taxi, 23: collection vehicles  

    # in the first column there are the types, in the second the objects belonging
    # to them and in the third the number of objects
    tipologie = {1: [1, 3640], 2: [2, 2200], 3: [3, 1600], 4: [4, 880], 5: [5, 200], 6: [6, 3360], 
                 7: [7, 2120], 8: [8, 600], 9: [20, 95], 10: [[16, 18, 19, 12, 13, 14], 140],
                 11: [[21, 22], 143], 12: [15, 10], 13: [23, 7], 14: [17, 506], 15: [9, 677],
                 16: [[10, 11], 38]}

    # I remove 7 (printer), 10 (Environments and Weather), 15 (parking) and
    # 8 (home sensors)
    # I consider all objects except these
    n_objects = 16216 - 2120 - 140 - 677 - 600

    list = np.zeros(16)
    somma = 0
    for i in range(1, 17):
        if i not in [7, 10, 15, 8]:
            list[i-1] = somma + (tipologie[i][1]/n_objects)
            somma = somma + (tipologie[i][1]/n_objects)
    list = list[[0, 1, 2, 3, 4, 5, 8, 10, 11, 12, 13, 15]]

    prob = np.random.rand()
    type_node = np.where(prob > list)

    if len(type_node[0]) == 0:    # for the first case of probability
        type_node = 1
    else:
        type_node = type_node[0][-1] + 2 

    # restore type values
    if type_node > 11:
        type_node = type_node + 4
    elif type_node > 7:
        type_node = type_node + 3
    elif type_node > 6:
        type_node = type_node + 2

    # I take up the values of the types in the matrix n
    type_node_matrix_n = tipologie[type_node][0]

    return type_node, type_node_matrix_n



def nodo(n, type_node):
    """
    Choose the node.

    """

    # Ensure type_node is a scalar or a one-dimensional array
    if np.ndim(type_node) > 1:
        type_node = type_node.flatten()[0]  # or select a specific element

    # Find the indices of the nodes with this type
    nodes = np.where(n[:, 0] == type_node)[0]

    # Random sample from the interval
    source_node = np.random.choice(nodes, 1)[0]

    return source_node



def applicazione(type_node):
    """
    Application choice

    """

    # number of services and applications
    # n_services = 16
    # n_app = 26

    # the numbering of the apps and services comes from the doc file related to
    # the paper

    # matrix of requested application types
    # for each position (type) a vector of apps
    type_app = {1: [4, 5, 6, 7, 8, 9, 10, 12, 16, 17, 19, 20, 21, 23, 24, 25, 26],  # 1 smartphone
                2: [1, 4, 6, 9, 11, 19, 23],  # 2 car
                3: [4, 5, 6, 7, 8, 9, 10, 12, 16, 17, 19, 20, 21, 23, 24, 25, 26],  # 3 tablet
                4: [26], 5: [4, 6, 7, 8, 16, 21, 26],  # 4 medical and 5 smartwatch
                6: [7, 8, 10, 16, 17, 25, 26],  # 6 pc
                9: [3, 6, 12, 18, 22, 26],  # 9 Point of Interest
                11: [1, 4, 6, 9],  # 11 Transportation
                12: [2, 3, 4, 5, 6, 8, 9, 12, 14, 15, 20, 23, 26],  # 12 Indicator
                13: [1],  # 13 Waste
                14: [2],  # 14 Light 
                16: [4]}  # 16 Alarms

    # matrix of fundamental services of applications
    # for each position (application) a vector of fundamental services
    app_services = {1: [5, 8, 10], 2: [4, 8, 9], 3: [7], 4: [5, 8, 10, 16],
                    5: [10, 11], 6: [5], 7: [15], 8: [3],
                    9: [13], 10: [4, 5], 11: [5, 6, 8, 9, 10], 12: [4, 5, 6],
                    13: [5, 6], 14: [3, 8, 13], 15: [5], 16: [14],
                    17: [15], 18: [4], 19: [6, 15],
                    20: [3, 4], 21: [4, 15], 22: [4, 5], 23: [12, 16],
                    24: [4], 25: [4], 26: [5, 6]}

    lista_app = type_app[type_node]
    len_list = len(lista_app)
    
    # if the list has only one element, I take that 
    if len_list == 1:
        app = lista_app[0]
    else:
        app = random.choice(lista_app)

    #numero_servizi = len(app_services[app])

    return app




def requisiti(source_node, last_nodes_positions, app, range_mediana):
    """
    Space requirement:
    The average value will correspond to a random position between
    the position of the node (x, y) + the first value of range median
    and the distance of "range_median" (the second value).
    """

    ax = last_nodes_positions[source_node, 0]
    bx = (range_mediana[1]/4000) - (range_mediana[0]/4000)    
    x_mu_space = ax + (bx-ax)*np.random.rand() + (range_mediana[0]/4000)
    
    ay = last_nodes_positions[source_node, 1]
    by = (range_mediana[1]/4000) - (range_mediana[0]/4000)    
    y_mu_space = ay + (by-ay)*np.random.rand() + (range_mediana[0]/4000)
    
    mu = [x_mu_space, y_mu_space]

    # Different sigma values for application
    sigma_app = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.1, 0.1, 0.05, 0.1, 0.05, 0.05, 0.1, 0.1,
                 0.1, 0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.05, 0.1, 0.1, 0.05, 0.05]

    sigma = sigma_app[app-1]

    space = np.random.normal(mu, sigma)
    # Truncate the values of x and y in space, in case they are negative or greater than 1
    for i in range(2):
        if space[i] > 1:
            space[i] = 1
        elif space[i] < 0:
            space[i] = 0
         
    # Time
    # It is taken in a uniform interval between 0 and 12 hours
    time_query = np.random.randint(0, 12*60*60)

    return space, time_query




def query(app, last_nodes_positions, tipologie, space, range_val, tempi_objects, time_query, time_range):
    # Query
    
    # Matrix of fundamental services of the applications
    # For each position (application), a vector of fundamental services
    app_services = [[5, 8, 10], [4, 8, 9], [7], [5, 8, 10, 16],
                    [10, 11], [5], [15], [3],
                    [13], [4, 5], [5, 6, 8, 9, 10], [4, 5, 6],
                    [5, 6], [3, 8, 13], [5], [14],
                    [15], [4], [6, 15],
                    [3, 4], [4, 15], [4, 5], [12, 16],
                    [4], [4], [5, 6]]

    # Matrix of offered service types
    # For each position (type), a vector of services
    type_service = [[4],  # 1 smartphone
                    [4, 5, 8, 10],  # 2 car
                    [4],  # 3 tablet
                    [14], [4],  # 4 medical and 5 smartwatch
                    [4], [], [15],  # 6 pc, 7 printer, and 8 home sensors
                    [3, 4, 7], [5, 6, 7], [8, 10, 11],  # 9 Point of Interest, 10 Environments and Weather, 12 Transportation
                    [4, 7], [12, 16], [7, 9],  # 12 Indicator, 13 Waste, 14 Light
                    [7, 13], [7, 8]]  # 15 Parking, 16 Alarms

    # Total queries from the chosen app
    query_services = app_services[app - 1]
    destinazioni = [None] * len(query_services)
    servizi = [None] * len(query_services)

    # SPACE AND TIME

    for j, query_service in enumerate(query_services):
        # Single query

        # List of nodes that can provide those services
        destination_type = [i for i in range(1, 17) if query_service in type_service[i - 1]]

        # Remove null values from the vector of node types that can provide such service
        destination_nodes = [i for i in range(len(tipologie)) if tipologie[i] in destination_type]

        # Rank nodes based on distance from the space in the query
        euclidean_distance = np.sqrt((last_nodes_positions[:,0] - space[0]) ** 2 + (last_nodes_positions[:,1] - space[1]) ** 2)

        euclidean_distance = np.column_stack((np.arange(1, len(euclidean_distance) + 1), euclidean_distance))

        # Select only the rows corresponding to destination_nodes
        euclidean_distance = euclidean_distance[destination_nodes]

        # Consider a range of "range" meters
        nodi_destinazione_spazio = euclidean_distance[euclidean_distance[:, 1] < (range_val / 4000)]

        # List nodes based on time distance in the query
        nodi_destinazione_spazio_tempo = []
        for i in range(len(nodi_destinazione_spazio)):
            element = int(nodi_destinazione_spazio[i, 0])
            if np.any((tempi_objects[element - 1] > (time_query - time_range)) & (tempi_objects[element - 1] < (time_query + time_range))):
                nodi_destinazione_spazio_tempo.append(element)
                #nodi_destinazione_spazio_tempo.append([element, nodi_destinazione_spazio[i, 1]])

        # Take the nearest one
        # if nodi_destinazione_spazio_tempo:
        #    # Take the nearest in terms of space
        #    min_distance = min(nodi_destinazione_spazio_tempo, key=lambda x: x[1])[1]
        #    idx_destinazioni = [i for i, v in enumerate(nodi_destinazione_spazio_tempo) if v[1] == min_distance]
        #    destinazioni[j] = nodi_destinazione_spazio_tempo[idx_destinazioni[0]][0]  # take the first
        #    servizi[j] = query_service
        # else:
        #    destinazioni[j] = 0
        #    servizi[j] = 0

        destinazioni[j] = nodi_destinazione_spazio_tempo
        servizi[j] = query_service


    return destinazioni, servizi

