from typing import List

from vehicle_routing_optimization.Customer import Customer


class Vehicle:
    def __init__(self, vehicle_id: int, max_capacity: int):
        self.vehicle_id = vehicle_id
        self.max_capacity = max_capacity
        self.current_capacity = 0
        self.route: List[Customer] = []
        self.service_schedule = []

    def __str__(self):
        return f'Vehicle {self.vehicle_id}, route: {self.route}'

    def __repr__(self):
        return f'Vehicle {self.vehicle_id}, route: {self.route}'

    def add_customer(self, customer: Customer):
        self.route.append(customer)

    def get_latest_customer(self):
        return self.route[-1]

    def update_capacity(self, customer: Customer):
        self.current_capacity += customer.demand

    def reset_route(self):
        self.route = []
        self.current_capacity = 0
