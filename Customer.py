class Customer:
    def __init__(self, customer_id: int, x_coord: int, y_coord: int, demand: int,
                 ready_time: int, due_date: int, service_time: int, neighbors=None, time: int = 0):
        if neighbors is None:
            neighbors = []
        self.customer_id = customer_id
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.demand = demand
        self.ready_time = ready_time
        self.due_date = due_date
        self.service_time = service_time
        self.neighbors = neighbors
        self.time = time

    def __str__(self):
        return f'{"Depot" if self.customer_id == 0 else "Customer"}: {self.customer_id}'

    def __repr__(self):
        return f'{"Depot" if self.customer_id == 0 else "Customer"}: {self.customer_id}'
