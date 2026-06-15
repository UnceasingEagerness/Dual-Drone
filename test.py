import numpy as np
import json
import folium
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import time
import random
from scipy.spatial import distance_matrix
import os
import sys
from typing import List, Dict, Tuple, Optional, Any

def load_geotags(json_file: str) -> List[Dict[str, Any]]:
    """
    Load geotags from a specified JSON file with error handling.
    
    Args:
        json_file (str): Path to the JSON file containing geotags.
        
    Returns:
        List[Dict[str, Any]]: A list of dictionaries representing the loaded geotags.
    """
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        print(f'✅ Successfully loaded {len(data)} geotags from {json_file}')
        return data
    except FileNotFoundError:
        print(f"❌ Error: File '{json_file}' not found!")
        print('Please ensure your geotags.json file is in the same directory as this script.')
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"❌ Error: Invalid JSON format in '{json_file}'!")
        sys.exit(1)

def generate_kmeans_ghost_points(geotags: List[Dict[str, Any]], num_points: int) -> np.ndarray:
    """
    Generates representative 'ghost points' using K-Means clustering.
    
    Args:
        geotags (List[Dict]): The list of all geotags.
        num_points (int): Number of clusters/ghost points to generate.
        
    Returns:
        np.ndarray: Array of generated ghost point coordinates.
    """
    coords = np.array([[tag['center_latitude'], tag['center_longitude']] for tag in geotags])
    kmeans = KMeans(n_clusters=num_points, random_state=42).fit(coords)
    return kmeans.cluster_centers_

def partition_geotags_knn(geotags: List[Dict[str, Any]], ghost_points: np.ndarray) -> List[List[Dict[str, Any]]]:
    """
    Partitions the geotags into sectors using K-Nearest Neighbors based on the ghost points.
    
    Args:
        geotags (List[Dict]): The list of all geotags.
        ghost_points (np.ndarray): The ghost points to act as sector centers.
        
    Returns:
        List[List[Dict]]: A list of clusters, where each cluster is a list of geotags.
    """
    coords = np.array([[tag['center_latitude'], tag['center_longitude']] for tag in geotags])
    neighbors = NearestNeighbors(n_neighbors=1).fit(ghost_points)
    _, indices = neighbors.kneighbors(coords)
    n_clusters = ghost_points.shape[0]
    clusters = [[] for _ in range(n_clusters)]
    for tag, index in zip(geotags, indices.flatten()):
        clusters[index].append(tag)
    return clusters

class AntColony:

    def __init__(self, dist_matrix, severities, n_ants=20, n_iterations=300, alpha=1, beta=2, gamma=2, rho=0.5, Q=100):
        self.dist_matrix = dist_matrix
        self.severities = severities
        self.pheromone = np.ones(self.dist_matrix.shape) * 0.1
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.rho = rho
        self.Q = Q
        self.num_nodes = dist_matrix.shape[0]
        self.best_path = None
        self.best_cost = float('inf')

    def run(self):
        for _ in range(self.n_iterations):
            all_paths = []
            all_costs = []
            for _ in range(self.n_ants):
                path = self.generate_path()
                cost = self.path_cost(path)
                all_paths.append(path)
                all_costs.append(cost)
                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_path = path
            self.update_pheromones(all_paths, all_costs)
        return (self.best_path, self.best_cost, None)

    def generate_path(self):
        path = []
        unvisited = list(range(self.num_nodes))
        current = np.argmax(self.severities)
        path.append(current)
        unvisited.remove(current)
        while unvisited:
            probabilities = self.calculate_probabilities(current, unvisited)
            next_node = random.choices(unvisited, weights=probabilities)[0]
            path.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        return path

    def calculate_probabilities(self, current, unvisited):
        pheromone = np.take(self.pheromone[current], unvisited)
        distances = np.take(self.dist_matrix[current], unvisited)
        distances = np.where(distances <= 1e-10, 1e-06, distances)
        heuristic = 1.0 / distances
        raw_severity = np.take(self.severities, unvisited)
        severity_range = np.max(self.severities) - np.min(self.severities)
        if severity_range < 1e-06:
            norm_severity = np.ones_like(raw_severity)
        else:
            norm_severity = 1 + (raw_severity - np.min(self.severities)) / (severity_range + 1e-10)
        desirability = pheromone ** self.alpha * heuristic ** self.beta * norm_severity ** self.gamma
        desirability_sum = np.sum(desirability)
        if desirability_sum <= 1e-10 or not np.isfinite(desirability_sum):
            return np.ones_like(desirability) / len(desirability)
        return desirability / desirability_sum

    def path_cost(self, path):
        total_cost = 0
        for i in range(len(path)):
            next_idx = (i + 1) % len(path)
            total_cost += self.dist_matrix[path[i]][path[next_idx]]
        return total_cost

    def update_pheromones(self, paths, costs):
        self.pheromone *= 1 - self.rho
        for path, cost in zip(paths, costs):
            for i in range(len(path)):
                next_idx = (i + 1) % len(path)
                self.pheromone[path[i]][path[next_idx]] += self.Q / cost
                self.pheromone[path[next_idx]][path[i]] += self.Q / cost

def two_opt(path, dist_matrix):
    improved = True
    best_path = path[:]
    best_cost = sum((dist_matrix[best_path[i]][best_path[(i + 1) % len(best_path)]] for i in range(len(best_path))))
    while improved:
        improved = False
        for i in range(len(path)):
            for j in range(i + 2, len(path)):
                if i == 0 and j == len(path) - 1:
                    continue
                new_path = best_path[:]
                new_path[i:j + 1] = best_path[j:i - 1:-1] if i > 0 else best_path[j::-1]
                new_cost = sum((dist_matrix[new_path[k]][new_path[(k + 1) % len(new_path)]] for k in range(len(new_path))))
                if new_cost < best_cost:
                    best_path = new_path
                    best_cost = new_cost
                    improved = True
                    break
            if improved:
                break
    return (best_path, best_cost)

def plan_ghost_point_tour(ghost_points, drone_location):
    drone_coords = np.array([drone_location[0], drone_location[1]])
    ghost_coords = np.array(ghost_points)
    all_points = np.vstack([drone_coords.reshape(1, -1), ghost_coords])
    dist_matrix = distance_matrix(all_points, all_points)
    if len(ghost_points) > 1:
        ghost_dist_matrix = dist_matrix[1:, 1:]
        colony = AntColony(ghost_dist_matrix, [1] * len(ghost_points), n_ants=30, n_iterations=200)
        best_path, _, _ = colony.run()
        drone_to_ghost_distances = dist_matrix[0, 1:]
        start_ghost_idx = np.argmin(drone_to_ghost_distances)
        start_pos = best_path.index(start_ghost_idx)
        reordered_path = best_path[start_pos:] + best_path[:start_pos]
        return reordered_path
    else:
        return [0]

def run_aco_optimized_sector_order(clusters, ghost_points, drone_location):
    sector_order = plan_ghost_point_tour(ghost_points, drone_location)
    global_aco_path = []
    for idx in sector_order:
        cluster = clusters[idx]
        if not cluster:
            continue
        coords = [(tag['center_latitude'], tag['center_longitude']) for tag in cluster]
        severities = np.array([tag['severity'] for tag in cluster])
        if len(coords) > 1:
            dist_matrix = distance_matrix(coords, coords)
            colony = AntColony(dist_matrix, severities, n_ants=30, n_iterations=300)
            best_path, _, _ = colony.run()
            best_path, _ = two_opt(best_path, dist_matrix)
            global_aco_path.extend([cluster[i] for i in best_path])
        elif len(coords) == 1:
            global_aco_path.append(cluster[0])
    return global_aco_path

def save_aco_path_json(global_aco_path, output_file='global_aco_path.json'):
    path_data = [{'sector_id': tag.get('sector_id', None), 'center_latitude': tag['center_latitude'], 'center_longitude': tag['center_longitude'], 'severity': tag['severity']} for tag in global_aco_path]
    with open(output_file, 'w') as f:
        json.dump(path_data, f, indent=4)
    print(f'✅ Global ACO path saved to {output_file}')

def plot_partitioned_geotags_with_path(geotags, clusters, ghost_points, global_aco_path, drone_location):
    center_lat = np.mean([tag['center_latitude'] for tag in geotags])
    center_lon = np.mean([tag['center_longitude'] for tag in geotags])
    m = folium.Map(location=[center_lat, center_lon], zoom_start=17)
    colors = ['blue', 'red', 'purple', 'orange', 'green', 'brown', 'pink', 'gray']
    for cluster_id, cluster in enumerate(clusters):
        if not cluster:
            continue
        for tag in cluster:
            folium.CircleMarker(location=(tag['center_latitude'], tag['center_longitude']), radius=3, color=colors[cluster_id % len(colors)], fill=True, fill_opacity=0.7).add_to(m)
        folium.Marker(location=(ghost_points[cluster_id][0], ghost_points[cluster_id][1]), popup=f'Ghost Point {cluster_id + 1}\nGeotags: {len(cluster)}', icon=folium.Icon(color='black', icon='info-sign')).add_to(m)
    folium.Marker(location=(drone_location[0], drone_location[1]), popup='Drone Home Location', icon=folium.Icon(color='darkgreen', icon='home')).add_to(m)
    if global_aco_path:
        global_coords = [(tag['center_latitude'], tag['center_longitude']) for tag in global_aco_path]
        folium.PolyLine(global_coords, color='black', weight=3, opacity=0.9, tooltip='Global ACO Path').add_to(m)
    return m

def main():
    """Main execution function"""
    print('🚁 Drone Path Optimization System')
    print('=' * 40)
    geotags_file = 'geotags.json'
    if not os.path.exists(geotags_file):
        print(f"❌ Error: '{geotags_file}' not found in current directory!")
        print('Please place your geotags.json file in the same directory as this script.')
        return
    geotags = load_geotags(geotags_file)
    if not geotags:
        print('❌ Error: No geotags found in the file!')
        return
    num_ghost_points = 8
    num_ghost_points = min(num_ghost_points, len(geotags))
    print(f'📊 Processing {len(geotags)} geotags with {num_ghost_points} sectors...')
    print('🎯 Generating ghost points using K-Means...')
    ghost_points = generate_kmeans_ghost_points(geotags, num_points=num_ghost_points)
    print('🗂️ Partitioning geotags into clusters...')
    clusters = partition_geotags_knn(geotags, ghost_points)
    PX4_HOME_LAT = 22.52
    PX4_HOME_LON = 75.92
    drone_location = (PX4_HOME_LAT, PX4_HOME_LON)
    print(f'🏠 Drone home location: {drone_location}')
    print('🐜 Running Ant Colony Optimization...')
    global_aco_path = run_aco_optimized_sector_order(clusters, ghost_points, drone_location)
    print('🗺️ Generating visualization...')
    combined_map = plot_partitioned_geotags_with_path(geotags, clusters, ghost_points, global_aco_path, drone_location)
    map_filename = 'combined_aco_partitioned_map.html'
    combined_map.save(map_filename)
    print(f'✅ Map saved to {map_filename}')
    print('💾 Saving optimized path...')
    save_aco_path_json(global_aco_path)
    print('\n📈 OPTIMIZATION SUMMARY')
    print('=' * 40)
    print(f'📊 Total geotags processed: {len(geotags)}')
    print(f'📊 Number of sectors: {len([c for c in clusters if c])}')
    print(f'📊 Global path length: {len(global_aco_path)}')
    print(f'📊 Ghost points generated: {len(ghost_points)}')
    print(f'✅ Optimization completed successfully!')
if __name__ == '__main__':
    main()