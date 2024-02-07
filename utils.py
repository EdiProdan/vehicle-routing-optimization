import csv
from typing import List, Tuple

import numpy as np

from vehicle_routing_optimization.Customer import Customer
from vehicle_routing_optimization.Vehicle import Vehicle

import matplotlib.pyplot as plt


def read_from_file(filename: str) -> Tuple[Customer, List[Customer], List[Vehicle]]:
    with open("data/" + filename, "r") as f:
        data = f.readlines()

    vehicle_number, max_capacity = data[2].split()
    vehicle_number, max_capacity = int(vehicle_number), int(max_capacity)

    customer_list = [[int(c) for c in customer.split()] for customer in data[7:]]
    customer_list = [Customer(*customer) for customer in customer_list]

    depot = customer_list[0]
    customer_list = customer_list[1:]

    vehicle_list = [Vehicle(i, max_capacity) for i in range(vehicle_number)]

    return depot, customer_list, vehicle_list


def plot_instance(depot: Customer, customer_list: List[Customer]):
    x_depot, y_depot = depot.x_coord, depot.y_coord
    x_customers = [customer.x_coord for customer in customer_list]
    y_customers = [customer.y_coord for customer in customer_list]

    plt.scatter(x_depot, y_depot, c="r", marker="s")
    plt.scatter(x_customers, y_customers, c="b", marker="o")

    plt.savefig("plt/instance_1.png")


def plot_vehicle_routes(depot: Customer, customer_time_list: list, customer_dict: dict):
    for route in customer_time_list:
        x_coords = []
        y_coords = []
        for customer_time in route:
            if customer_time[0] == 0:
                customer = depot
            else:
                customer = customer_dict[customer_time[0]]
            x_coords.append(customer.x_coord)
            y_coords.append(customer.y_coord)
        plt.plot(x_coords, y_coords, marker='o', linestyle='-')

    depot_x = depot.x_coord
    depot_y = depot.y_coord

    plt.scatter(depot_x, depot_y, marker='s', color='black', label='Depot')

    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Vehicle Routes')
    plt.legend()
    plt.savefig("plt/vehicle_routes.png")


def calculate_distance(depot: Customer, customer_list: List[Customer], instance):

    coordinates = np.array([(customer.x_coord, customer.y_coord) for customer in customer_list])
    depot_coord = np.array((depot.x_coord, depot.y_coord))

    # Calculate distances from the depot to each customer
    depot_to_customer_distances = np.linalg.norm(coordinates - depot_coord, axis=1)
    distances = [(distance, 0, customer.customer_id) for distance, customer in
                 zip(depot_to_customer_distances, customer_list)]

    # Calculate distances between each pair of customers
    customer_to_customer_distances = np.linalg.norm(coordinates[:, np.newaxis, :] - coordinates, axis=2)

    # Avoid calculating distances for the same customer or duplicates
    np.fill_diagonal(customer_to_customer_distances, np.inf)

    customer_pairs = np.triu_indices(len(customer_list), k=1)
    distances.extend(
        [(distance, customer_list[i].customer_id, customer_list[j].customer_id) for (i, j), distance in
         zip(zip(*customer_pairs), customer_to_customer_distances[customer_pairs])]
    )

    with open("heuristics/distances-"+instance+".csv", 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["distance", "node1", "node2"])
        for distance in distances:
            csv_writer.writerow([distance[0], distance[1], distance[2]])


def read_from_csv(filename: str) -> List[Tuple[float, int, int]]:
    distance_data = []
    with open("heuristics/" + filename, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        for row in csv_reader:
            distance_data.append((float(row[0]), int(row[1]), int(row[2])))
    return distance_data


def get_heuristic_matrix_and_avg_distance(n: int, instance: str) -> Tuple[np.ndarray, float]:
    distance_data = read_from_csv("distances-"+instance+".csv")

    distance_matrix = np.zeros((n + 1, n + 1))
    for distance in distance_data:
        distance_matrix[distance[1]][distance[2]] = 1/distance[0]
        distance_matrix[distance[2]][distance[1]] = 1/distance[0]

    avg_distance = np.mean([distance[0] for distance in distance_data])

    return distance_matrix, avg_distance


