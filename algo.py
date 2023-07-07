import pandas as pd
import numpy as np
import osmnx as ox
import glob, os
from datetime import datetime
import googlemaps
import matplotlib.pyplot as plt
import networkx as nx
import sklearn as sk
import heapq
import networkx as nx
import math

gmaps = googlemaps.Client(key="") ## use your own key lmao

# Google Maps

## origin = input('Enter your origin: ')
## destination = input('Enter your destination: ')

origin = "Buckingham Palace"
destination = "Trafalgar Square"

org_dict = gmaps.geocode(origin)[0]
org_lat, org_lng = float(org_dict['geometry']['location']['lat']), float(org_dict['geometry']['location']['lng'])
org = (org_lat,org_lng)

des_dict = gmaps.geocode(destination)[0]
des_lat, des_lng = float(des_dict['geometry']['location']['lat']), float(des_dict['geometry']['location']['lng'])
des = (des_lat,des_lng)

# OSMNX variables

graph = ox.graph.graph_from_bbox(org_lat,des_lat,des_lng,org_lng, 
                        network_type="walk",
                        simplify=True,) ##NSEW

org_node = ox.nearest_nodes(graph, org_lng, org_lat)
des_node = ox.nearest_nodes(graph, des_lng, des_lat)

# Databases

nodes, edges = ox.graph_to_gdfs(graph)
nodes, edges = nodes.reset_index(), edges.reset_index()

# Crime Spots dataframe

raw_df = pd.read_csv('crime_spots.csv')
df = raw_df.loc[(raw_df['Latitude'] >= nodes['y'].min()) & (raw_df['Latitude'] <= nodes['y'].max())].loc[(raw_df['Longitude'] >= nodes['x'].min()) & (raw_df['Longitude'] <= nodes['x'].max())].sort_values(['Latitude', 'Longitude'], ascending=[True,True]).reset_index()

# Algorithm (Final route)

l3=[]
for i in range(len(nodes)):
    l1=[]
    for j in range(len(df)):
        dist = ox.distance.euclidean_dist_vec(nodes['x'][i], 
                                              nodes['y'][i], 
                                              df['Longitude'][j], 
                                              df['Latitude'][j]) * 100000
        l1.append(dist)
    iminimum = l1.index(min(l1))
    l3.append(iminimum)## list of indexes of each minimum value
l4 = []
l5 = []
for i in range(len(l3)):
    score = df['Crime score'][l3[i]]
    count = df['Count'][l3[i]]
    l4.append(score)
    l5.append(count)
nodes['score'] = l4
nodes['count'] = l5

class Node(object):
    
    ## Special functions
    
    def __init__(self, lat, lng):
        self.id = ox.nearest_nodes(graph, lng, lat)
        self.lat = lat
        self.lng = lng
        self.score = nodes.loc[nodes['osmid']==self.id].reset_index()['score'][0]
        self.children = []
        self.cost = 0
        self.parent = None
    
    def __eq__(self, other): 
        return self.id == other.id
    
    def __lt__(self, other):
        return self.score < other.score
    
    def __str__(self):
        return f'(Node {self.id} at {self.lat},{self.lng} with score {self.score})'
    
    def __repr__(self):
        return f'Node ({self.lat},{self.lng})'
    
    ## Class functions
    
    def find_children(self):
        j = edges.loc[edges['u']==self.id].reset_index()['v']
        for i in range(len(j)):
            child = j[i]
            clat = nodes.loc[nodes['osmid']==child].reset_index()['y'][0]
            clng = nodes.loc[nodes['osmid']==child].reset_index()['x'][0]
            childNode = Node(clat,clng)
            self.children.append(childNode)
            childNode.parent = self
            
    def distance_to(self, other):
        lat1 = math.radians(self.lat)
        lon1 = math.radians(self.lng)
        lat2 = math.radians(other.lat)
        lon2 = math.radians(other.lng)

        # Compute the Haversine distance.
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = (math.sin(dlat/2)**2) + math.cos(lat1) * math.cos(lat2) * (math.sin(dlon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = 6371e3 * c  # 6371e3 is the radius of the Earth in meters.

        return distance

# BEST FIRST SEARCH
from queue import PriorityQueue

def best_first_search(start, goal):
    frontier = PriorityQueue()
    visited = []
    frontier.put(start)
    while not frontier.empty():
        current = frontier.get()
        if current == goal:
            return path(current)
        current.find_children()
        for child in current.children:
            if child not in visited:
                visited.append(child)
                child.parent = current
                frontier.put(child)
    return []

    
def path(node):
    if node.parent is None:
        return [node]
    else:
        return path(node.parent) + [node]

def node_to_osmid(listofnodes):
    osmid = []
    for i in range(len(listofnodes)):
        osmid.append(ox.nearest_nodes(graph, listofnodes[i].lng, listofnodes[i].lat))
    return osmid
    
org = Node(org_lat, org_lng)
des = Node(des_lat, des_lng)
path = best_first_search(org, des)

# MULTIHEURISTIC A* ALGORITHM
def calculate_cost(current_node, destination_node, score_weight=2):
    
    distance_cost = current_node.distance_to(destination_node)
    score_cost = current_node.score * score_weight
    return distance_cost + score_cost

def mha(start_node, end_node, score_weight=2):
    open_list = [(0, start_node)]
    closed_list = []
    while open_list:
        open_list.sort(key=lambda x: x[0])
        current_cost, current_node = open_list.pop(0)
        closed_list.append(current_node)
        if current_node == end_node:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = current_node.parent
            return path[::-1]
        current_node.find_children()
        for neighbor in current_node.children:
            if neighbor in closed_list:
                continue
            cost = calculate_cost(current_node, neighbor, score_weight=score_weight)
            in_open_list = False
            for i, n in enumerate(open_list):
                if n[1] == neighbor:
                    in_open_list = True
                    if n[0] > cost:
                        open_list[i] = (cost, neighbor)
                    break
            if not in_open_list:
                open_list.append((cost, neighbor))
        
path_astar = mha(org,des)