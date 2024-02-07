import math
import sys
from typing import Dict, List, Set

import numpy as np
from time import time

from vehicle_routing_optimization.Customer import Customer
from vehicle_routing_optimization.Vehicle import Vehicle


def initialize_visited_customers_id_set(depot_id: int) -> set:
    visited_customers_id_set = set()
    visited_customers_id_set.add(depot_id)
    return visited_customers_id_set


def initialize_pheromone_matrix(avg_distance: float, n: int) -> np.ndarray:
    pheromone_matrix = np.zeros((n, n))
    pheromone_matrix.fill(1 / avg_distance)
    return pheromone_matrix


def apply_evaporation_strategy(pheromone_matrix: np.ndarray, rho: float) -> np.ndarray:
    return (1 - rho) * pheromone_matrix


def apply_reinforcement_strategy(pheromone_matrix: np.ndarray, delta: float, customer_time_list) -> np.ndarray:
    updated_pheromones = pheromone_matrix.copy()

    for route in customer_time_list:
        for i in range(len(route) - 1):
            updated_pheromones[route[i][0], route[i + 1][0]] += delta

    return updated_pheromones


def calculate_solution(customer_time_list, heuristic_matrix):
    solution = [0, 0]
    solution[0] = len(customer_time_list)
    for j, route in enumerate(customer_time_list):
        for i in range(len(route) - 1):
            solution[1] += 1 / heuristic_matrix[route[i][0], route[i + 1][0]]
    return solution


def print_output_to_file(customer_time_list: tuple, heuristic_matrix, instance: str, time: str, customer_dict):
    num_vehicles = customer_time_list[0]
    total_distance = customer_time_list[1]
    routes = customer_time_list[2]

    with open("out/res-{}-i{}.txt".format(time, instance), 'w') as f:

        f.write(f"{num_vehicles}\n")
        for j, route in enumerate(routes):
            route_string = ""
            for i in range(len(route)):
                customer_time = route[i][1]
                if i == len(route) - 1:
                    customer_time = route[i - 1][1]
                    customer_time += math.ceil(1 / heuristic_matrix[route[i - 1][0]][route[i][0]])
                    customer = customer_dict[route[i - 1][0]]
                    customer_time += customer.service_time

                route_string += f"{route[i][0]}({customer_time})->"
            route_string = route_string[:-2]
            f.write(f"{j + 1}: {route_string}\n")
        f.write(f"{round(total_distance, 2)}")


class AntColonyAlgorithm:
    def __init__(self, depot: Customer, customer_list: List[Customer], vehicle_list: List[Vehicle],
                 avg_distance: float, heuristic_matrix: np.ndarray, customer_dict: Dict[int, Customer], num_ants: int):
        self.depot = depot
        self.customer_list = customer_list
        self.n = len(self.customer_list) + 1
        self.vehicle_list = vehicle_list
        self.avg_distance = avg_distance
        self.heuristic_matrix = heuristic_matrix
        self.customer_dict = customer_dict
        self.num_ants = num_ants

    def reset_customer_time(self):
        for customer in self.customer_list:
            customer.time = 0

    def initialize_to_depot(self, ant: Vehicle):
        ant.reset_route()
        ant.add_customer(self.depot)

    def check_available_customers(self, ant: Vehicle, current_customer: Customer,
                                  unvisited_customers_set: Set[int]) -> list:
        if len(unvisited_customers_set) == 0:
            return []

        customers_to_remove = set()

        for customer in unvisited_customers_set:
            customer = self.customer_dict[customer]
            if customer.customer_id == 0:
                continue

            euclidean_distance_ceil = math.ceil(
                1 / self.heuristic_matrix[current_customer.customer_id][customer.customer_id])
            next_customer_time = current_customer.service_time + current_customer.time + euclidean_distance_ceil

            if next_customer_time > customer.due_date:
                customers_to_remove.add(customer)

            if next_customer_time < customer.ready_time:
                next_customer_time = customer.ready_time

            # Constraint that checks if the customer can reach the depot before the due date
            time_to_depot = math.ceil(1 / self.heuristic_matrix[customer.customer_id][0])
            if next_customer_time + time_to_depot + customer.service_time > self.depot.due_date:
                if customer not in customers_to_remove:
                    customers_to_remove.add(customer)

        if len(unvisited_customers_set) == 0:
            return []

        for customer in customers_to_remove:
            customer = self.customer_dict[customer.customer_id]
            unvisited_customers_set.remove(customer.customer_id)

        if len(unvisited_customers_set) == 0:
            return []

        customers_to_remove = set()
        for customer in unvisited_customers_set:
            customer = self.customer_dict[customer]
            if ant.current_capacity + customer.demand > ant.max_capacity:
                customers_to_remove.add(customer.customer_id)

        for customer in customers_to_remove:
            unvisited_customers_set.remove(customer)

        if len(unvisited_customers_set) == 0:
            return []

        return list(unvisited_customers_set)

    def prob_transition(self, ant: Vehicle, current_customer: Customer, unvisited_customers_set: Set[int],
                        pheromone_matrix: np.ndarray, alpha: float, beta: float) -> int:

        unvisited_customers_list = self.check_available_customers(ant, current_customer, unvisited_customers_set)

        if not unvisited_customers_list:
            return -1

        pheromone_values = pheromone_matrix[current_customer.customer_id, unvisited_customers_list]

        heuristic_values = self.heuristic_matrix[current_customer.customer_id, unvisited_customers_list]

        numerator = (pheromone_values ** alpha) * (heuristic_values ** beta)

        denominator = np.sum(numerator)

        probabilities = numerator / denominator

        probabilities = np.nan_to_num(probabilities)

        if np.sum(probabilities) != 1:
            probabilities /= np.sum(probabilities)
        next_customer_id = np.random.choice(unvisited_customers_list, p=probabilities)

        return next_customer_id

    def construct_solution(self, pheromone_matrix, alpha, beta):

        optimal_ant_solution: tuple[int, float, list] = (sys.maxsize, math.inf, [])

        vehicle_num = len(self.vehicle_list) + 1
        max_capacity = self.vehicle_list[0].max_capacity

        ant_list = [Vehicle(vehicle_num + i + 1, max_capacity) for i in range(self.num_ants)]

        for ant in ant_list:

            self.initialize_to_depot(ant)

            unvisited_customers_set = set(self.customer_list)

            routes = []

            self.reset_customer_time()

            while unvisited_customers_set:
                current_customer = ant.get_latest_customer()
                unvisited_customers_id_set = {customer.customer_id for customer in unvisited_customers_set}
                next_customer_id = self.prob_transition(ant, current_customer, unvisited_customers_id_set,
                                                        pheromone_matrix, alpha, beta)

                if next_customer_id == -1:
                    ant.add_customer(self.depot)
                    routes.append([(r.customer_id, r.time) for r in ant.route])

                    self.initialize_to_depot(ant)
                    self.reset_customer_time()

                    if len(unvisited_customers_set) == 0:
                        break
                    else:
                        continue

                next_customer = self.customer_dict[next_customer_id]
                euclidean_distance_ceil = math.ceil(
                    1 / self.heuristic_matrix[current_customer.customer_id][next_customer.customer_id])
                new_time = current_customer.service_time + current_customer.time + euclidean_distance_ceil

                if new_time < next_customer.ready_time:
                    new_time = next_customer.ready_time

                next_customer.time = new_time

                ant.add_customer(next_customer)
                ant.update_capacity(next_customer)
                unvisited_customers_set.remove(next_customer)

                if len(unvisited_customers_set) == 0:
                    ant.add_customer(self.depot)
                    routes.append([(r.customer_id, r.time) for r in ant.route])
                    self.reset_customer_time()

            num_vehicles, total_distance = calculate_solution(routes, self.heuristic_matrix)
            if num_vehicles < optimal_ant_solution[0] or (num_vehicles == optimal_ant_solution[0] and
                                                          total_distance < optimal_ant_solution[1]):
                optimal_ant_solution = (num_vehicles, total_distance, routes)

        return optimal_ant_solution

    def optimize(self, max_iterations: int, alpha: float, beta: float, rho: float, instance: str):
        pheromone_matrix = initialize_pheromone_matrix(self.avg_distance, self.n)

        optimal_solution: tuple[int, float, list] = (sys.maxsize, math.inf, [])

        one_min = True
        five_min = True

        start_time = time()

        for _ in range(max_iterations):

            print("Iteration number: ", _)
            # print("Time: ", time() - start_time)

            num_vehicles, total_distance, routes = self.construct_solution(pheromone_matrix, alpha, beta)

            if num_vehicles < optimal_solution[0] or (num_vehicles == optimal_solution[0]
                                                      and total_distance < optimal_solution[1]):
                optimal_solution = (num_vehicles, total_distance, routes)

            pheromone_matrix = apply_evaporation_strategy(pheromone_matrix, rho)
            pheromone_matrix = apply_reinforcement_strategy(pheromone_matrix, 1 / optimal_solution[1], optimal_solution[2])

            print("Best solution: ", optimal_solution[0], optimal_solution[1])
            print("Current solution: ", num_vehicles, total_distance)

            if time() - start_time > 60 and one_min:
                print_output_to_file(optimal_solution, self.heuristic_matrix, instance, '1m', self.customer_dict)
                one_min = False

            if time() - start_time > 300 and five_min:
                print_output_to_file(optimal_solution, self.heuristic_matrix, instance, '5m', self.customer_dict)
                five_min = False

            if not five_min and num_vehicles < optimal_solution[0]:
                print_output_to_file(optimal_solution, self.heuristic_matrix, instance, 'un', self.customer_dict)

        return optimal_solution
