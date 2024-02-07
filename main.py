from vehicle_routing_optimization.AntColonyAlgorithm import AntColonyAlgorithm, print_output_to_file
from utils import read_from_file, get_heuristic_matrix_and_avg_distance, plot_vehicle_routes, calculate_distance

if __name__ == "__main__":

    instance = '7'

    depot, customer_list, vehicle_list = read_from_file(f"inst{instance}.txt")
    n = len(customer_list)

    calculate_distance(depot, customer_list, instance)

    heuristic_matrix, avg_distance = get_heuristic_matrix_and_avg_distance(n, instance)

    customer_dict = {customer.customer_id: customer for customer in customer_list}

    ant_colony = AntColonyAlgorithm(depot, customer_list, vehicle_list, avg_distance,
                                    heuristic_matrix, customer_dict, 5)
    ant_colony_solution = ant_colony.optimize(200, 5, 2, 0.45, instance)

    # plot_vehicle_routes(depot, ant_colony_solution, customer_dict)

    print_output_to_file(ant_colony_solution, heuristic_matrix, instance, 'un', customer_dict)
