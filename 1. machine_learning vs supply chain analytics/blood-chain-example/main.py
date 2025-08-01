import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import datetime
import csv

import matplotlib

matplotlib.use('TkAgg')
num_epochs = 5000
nb_samples = 100000
batch_size = int(nb_samples * 0.5)
risk_aversion = 0.7  # risk aversion coefficient, used to balance the risk and cost in the objective function
# ==============Global Variables================
g_node_list = []
g_internal_node_seq_dict = {}
g_external_node_id_dict = {}

g_supply_list = []
g_demand_list = []
g_demand_data_list = []
g_demand_info_dict = {}

g_link_list = []
g_internal_link_seq_dict = {}
g_external_link_id_dict = {}
g_link_pair_id_dict = {}
g_link_parameter_dict = {}

g_path_list = []
g_path_variable_list = []


# =================Classes================
class Node:
    def __init__(self, node_seq, node_name, node_id, node_type, x_coord, y_coord, geometry, ub_demand, lb_demand,
                 p_shortage, p_surplus):
        self.node_id = int(node_id)
        self.node_seq = node_seq
        self.node_name = str(node_name)
        self.node_type = str(node_type)
        self.x_coord = float(x_coord)
        self.y_coord = float(y_coord)
        self.geometry = str(geometry)
        # if upper_bound is '', then set it to -1
        self.ub_demand = float(ub_demand) if ub_demand else -1.0
        self.lb_demand = float(lb_demand) if lb_demand else -1.0
        self.p_shortage = float(p_shortage) if p_shortage else 0.0
        self.p_surplus = float(p_surplus) if p_surplus else 0.0
        self.outgoing_link_list = []
        self.incoming_link_list = []
        self.path_list = []  # List to hold paths associated with this node


class Link:
    def __init__(self, link_name, link_id, from_node_id, to_node_id, from_node_seq, to_node_seq,
                 link_seq, dir_flag, cost_coeff1, cost_coeff2,
                 risk_coeff1, risk_coeff2, discard_coeff1, discard_coeff2, decay_rate):
        self.link_name = str(link_name)
        self.link_id = int(link_id)
        self.from_node_id = int(from_node_id)
        self.to_node_id = int(to_node_id)
        self.from_node_seq = int(from_node_seq)
        self.to_node_seq = int(to_node_seq)
        self.link_seq = int(link_seq)
        self.dir_flag = int(dir_flag)
        self.cost_coeff1 = float(cost_coeff1)
        self.cost_coeff2 = float(cost_coeff2)
        self.risk_coeff1 = float(risk_coeff1)
        self.risk_coeff2 = float(risk_coeff2)
        self.discard_coeff1 = float(discard_coeff1)
        self.discard_coeff2 = float(discard_coeff2)
        self.decay_rate = float(decay_rate)
        self.flow = 0.0  # Initialize flow on the link
        g_node_list[self.from_node_seq].outgoing_link_list.append(self.link_id)
        g_node_list[self.to_node_seq].incoming_link_list.append(self.link_id)
        g_link_parameter_dict[self.link_id] = {
            'cost_coeff1': self.cost_coeff1,
            'cost_coeff2': self.cost_coeff2,
            'risk_coeff1': self.risk_coeff1,
            'risk_coeff2': self.risk_coeff2,
            'discard_coeff1': self.discard_coeff1,
            'discard_coeff2': self.discard_coeff2,
            'decay_rate': self.decay_rate
        }

    def calculate_cost(self):
        """
        Calculate the cost of the link based on the flow.
        :param flow: The flow on the link.
        :return: The cost of the link.
        """
        return self.cost_coeff2 * self.flow ** 2 + self.cost_coeff1 * self.flow

    def calculate_risk(self):
        """
        Calculate the risk of the link based on the flow.
        :param flow: The flow on the link.
        :return: The risk of the link.
        """
        return self.risk_coeff2 * self.flow ** 2 + self.risk_coeff1 * self.flow

    def calculate_discard_cost(self):
        """
        Calculate the discard cost of the link based on the flow.
        :param flow: The flow on the link.
        :return: The discard cost of the link.
        """
        return self.discard_coeff2 * self.flow ** 2 + self.discard_coeff1 * self.flow


class Network:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_nodes_and_edges(self, node_list, link_list):
        for node in node_list:
            self.graph.add_node(node.node_id)
        print('The number of nodes in the graph is:', self.graph.number_of_nodes())
        for link in link_list:
            self.graph.add_edge(link.from_node_id, link.to_node_id)
        print('The number of edges in the graph is:', self.graph.number_of_edges())

    def create_path_variable(self, demand_list, supply_list):
        for demand_point in demand_list:
            for supply_node in supply_list:
                source = supply_node.node_id
                target = demand_point.node_id
                path_node_seq_set = list(nx.all_simple_paths(self.graph, source=source, target=target))
                for path_node_seq in path_node_seq_set:
                    demand_point.path_list.append(path_node_seq[0])  # Store the first path found for each demand node
                    path_seq = len(g_path_list)
                    path = Path(path_seq, path_node_seq)
                    g_path_list.append(path)
                    path_var = tf.Variable(
                        initial_value=tf.random.uniform([1], minval=0.0, maxval=1.0, dtype=tf.float64),
                        trainable=True,
                        name=f"path_flow_{path_seq}")
                    g_path_variable_list.append(path_var)


class Path:
    def __init__(self, path_seq, node_seq):
        self.path_seq = path_seq
        self.node_seq = node_seq
        self.link_seq = []
        self.demand_point_id = self.node_seq[-1]  # The last node in the path is the demand point
        self.demand_point_decay_rate = 1.0  # Initialize decay rate for the demand point
        self.generate_link_seq()
        self.calculate_demand_point_decay_rate()

    def generate_link_seq(self):
        """
        Generate the link sequence for the path.
        :param link_seq: The sequence of links in the path.
        """
        for i in range(len(self.node_seq) - 1):
            from_node_id = self.node_seq[i]
            to_node_id = self.node_seq[i + 1]
            link_pair = (from_node_id, to_node_id)
            link_id = g_link_pair_id_dict[link_pair]
            self.link_seq.append(link_id)

    def calculate_demand_point_decay_rate(self):
        """
        Calculate the decay rate for the demand point based on the links in the path.
        :return: The decay rate for the demand point.
        """
        decay_rate = 1.0  # Initialize decay rate
        for link_id in self.link_seq:
            decay_rate *= g_link_parameter_dict[link_id]['decay_rate']
        self.demand_point_decay_rate = decay_rate


# =================Data Reading and Processing================
def network_reading():
    # ==============Read the input data================
    with open('node.csv', 'r') as file:
        csv_reader = csv.DictReader(file)
        node_seq = 0
        for row in csv_reader:
            # mandatory field, so we can raise an error if it's missing
            node_id = int(row['node_id'])
            upper_bound = row['ub_demand']
            lower_bound = row['lb_demand']
            penalty_surplus = row['penalty_surplus']
            penalty_shortage = row['penalty_shortage']
            node_type = row['node_type']

            # Optional fields with default values
            node_name = row.get('node_name', '')  # Default to empty string if missing
            x_coord = row.get('x_coord', '0')  # Default to 0 if missing
            y_coord = row.get('y_coord', '0')  # Default to 0 if missing
            geometry = row.get('geometry', 'POINT (0.0 0.0)')  # Default to 'POINT (0.0 0.0)' if missing

            node = Node(node_seq, node_name, node_id, node_type, x_coord, y_coord, geometry, upper_bound,
                        lower_bound, penalty_shortage, penalty_surplus)
            g_node_list.append(node)
            g_internal_node_seq_dict[int(node_id)] = node_seq
            g_external_node_id_dict[node_seq] = int(node_id)
            node_seq += 1
            if node_seq % 1 == 0:
                print(f"Processed {node_seq} nodes...")

        demand_nodes = [node for node in g_node_list if node.node_type == 'demand']
        supply_nodes = [node for node in g_node_list if node.node_type == 'supply']
        g_demand_list.extend(demand_nodes)
        g_supply_list.extend(supply_nodes)
        # if there are no supply nodes or demand nodes, raise an error
        if len(g_supply_list) == 0:
            raise ValueError("No supply nodes found in the input data.")
        else:
            print('the list of supply nodes is:', [node.node_id for node in g_supply_list])
        if len(g_demand_list) == 0:
            raise ValueError("No demand nodes found in the input data.")
        else:
            print('the list of demand nodes is:', [node.node_id for node in g_demand_list])

    with open('link.csv', 'r') as file:
        csv_reader = csv.DictReader(file)
        link_seq = 0
        for row in csv_reader:
            link_id = int(row['link_id'])
            from_node_id = int(row['from_node_id'])
            to_node_id = int(row['to_node_id'])

            link_name = str(row.get('link_name', ''))  # Default to empty string if missing
            dir_flag = int(row.get('dir_flag', 1))

            cost_coeff2 = float(row.get('cost_coeff2', 0))  # second-order cost coefficient
            cost_coeff1 = float(row.get('cost_coeff1', 0))  # first-order cost coefficient
            risk_coeff2 = float(row.get('risk_coeff2', 0))  # second-order risk coefficient
            risk_coeff1 = float(row.get('risk_coeff1', 0))  # first-order risk coefficient
            discard_coeff2 = float(row.get('discard_coeff2', 0))  # second-order discard coefficient
            discard_coeff1 = float(row.get('discard_coeff1', 0))  # first-order discard coefficient
            decay_rate = float(row.get('decay_rate', 1))  # decay rate at each link

            link = Link(link_name, link_id, from_node_id, to_node_id,
                        g_internal_node_seq_dict[from_node_id],
                        g_internal_node_seq_dict[to_node_id],
                        link_seq, dir_flag, cost_coeff1, cost_coeff2, risk_coeff1, risk_coeff2,
                        discard_coeff1, discard_coeff2, decay_rate)
            g_link_list.append(link)
            g_internal_link_seq_dict[link_id] = link_seq
            g_link_pair_id_dict[(from_node_id, to_node_id)] = link_id
            link_seq += 1
            if link_seq % 1 == 0:
                print(f"Processed {link_seq} links...")


def demand_reading(time_index=0):
    demand_df = pd.read_csv('demand.csv')
    demand_df = demand_df[demand_df['time_step'] == time_index]
    for demand_point in g_demand_list:
        demand_point_seq = demand_point.node_seq
        demand_point_df = demand_df[demand_df['demand_node_seq'] == demand_point_seq]
        demand_volume_array = demand_point_df['volume'].values

        g_demand_data_list.append(demand_volume_array)


# =================Generate Random Demand================

def generate_random_empirical_demand():
    # set random seed for reproducibility
    nb_samples = 800
    nb_time_steps = 7
    # generate random demand for each demand node using uniform distribution bounded by ub_demand and lb_demand
    total_demand_list = []
    for time_step in range(nb_time_steps):
        for demand_node in g_demand_list:
            demand_node_id = demand_node.node_id
            demand_node_seq = demand_node.node_seq
            demand_upper_bound = demand_node.ub_demand
            demand_lower_bound = demand_node.lb_demand
            mean_value = np.random.uniform(demand_lower_bound, demand_upper_bound)
            scale = (demand_upper_bound - demand_lower_bound) / np.random.uniform(4, 8)
            for sample_id in range(nb_samples):
                # if random > 0.5 , use the first mean value, otherwise use the second mean value
                if np.random.rand() > 0.7:
                    mean_value = np.random.uniform(demand_lower_bound, demand_upper_bound)
                    scale = (demand_upper_bound - demand_lower_bound) / np.random.uniform(4, 8)
                # Generate a random demand using normal distribution
                volume = np.random.normal(loc=mean_value, scale=scale)
                volume = max(volume, 0)  # Ensure volume is non-negative

                total_demand_list.append([time_step, sample_id, demand_node_id, demand_node_seq, volume])

    # Convert the total demand list to a DataFrame
    df = pd.DataFrame(total_demand_list, columns=['time_step', 'sample_id',
                                                  'demand_node_id', 'demand_node_seq', 'volume'])
    # Save the random demand to a CSV file
    df.to_csv('demand.csv', index=False)


def generate_random_uniform_demand():
    # set random seed for reproducibility
    nb_time_steps = 1
    # generate random demand for each demand node using uniform distribution bounded by ub_demand and lb_demand
    total_demand_list = []
    for time_step in range(nb_time_steps):
        for demand_node in g_demand_list:
            demand_node_id = demand_node.node_id
            demand_node_seq = demand_node.node_seq
            demand_upper_bound = demand_node.ub_demand
            demand_lower_bound = demand_node.lb_demand
            for sample_id in range(nb_samples):
                if 0.0 <= demand_lower_bound < demand_upper_bound and demand_upper_bound > 0:
                    # Generate a random demand using uniform distribution
                    volume = np.random.uniform(demand_lower_bound, demand_upper_bound)
                else:
                    volume = np.random.uniform(0, 1)
                total_demand_list.append([time_step, sample_id, demand_node_id, demand_node_seq, volume])

    # Convert the total demand list to a DataFrame
    df = pd.DataFrame(total_demand_list, columns=['time_step', 'sample_id',
                                                  'demand_node_id', 'demand_node_seq', 'volume'])
    # Save the random demand to a CSV file
    df.to_csv('demand.csv', index=False)


# =================Create Incidence Matrices================
def create_path_flow_incidence_matrices():
    number_of_paths = len(g_path_list)
    number_of_links = len(g_link_list)
    path_link_inc_mat = np.zeros((number_of_paths, number_of_links), dtype=np.float64)
    path_link_decay_inc_mat = np.zeros((number_of_paths, number_of_links), dtype=np.float64)
    path_demand_id_dict = {}
    path_demand_decay_dict = {}
    for path_index, path in enumerate(g_path_list):
        path_link_decay_rate = 1
        path_node_seq = path.node_seq
        for i in range(len(path_node_seq) - 1):
            from_node_id = path_node_seq[i]
            to_node_id = path_node_seq[i + 1]
            link_pair = (from_node_id, to_node_id)
            link_id = g_link_pair_id_dict[link_pair]
            link_decay_rate = g_link_parameter_dict[link_id]['decay_rate']
            path_link_inc_mat[path_index, g_internal_link_seq_dict[link_id]] = 1.0
            path_link_decay_inc_mat[path_index, g_internal_link_seq_dict[link_id]] = path_link_decay_rate
            path_link_decay_rate *= link_decay_rate

    return path_link_inc_mat, path_link_decay_inc_mat


def create_path_demand_point_incidence_matrices():
    number_of_paths = len(g_path_list)
    number_of_demand_points = len(g_demand_list)
    path_demand_inc_mat = np.zeros((number_of_paths, number_of_demand_points), dtype=np.float64)
    path_demand_decay_inc_mat = np.zeros((number_of_paths, number_of_demand_points), dtype=np.float64)

    for path_index, path in enumerate(g_path_list):
        for demand_index, demand_node in enumerate(g_demand_list):
            if path.demand_point_id == demand_node.node_id:
                path_demand_inc_mat[path_index, demand_index] = 1.0
                path_demand_decay_inc_mat[path_index, demand_index] = path.demand_point_decay_rate
    return path_demand_inc_mat, path_demand_decay_inc_mat


# =================Model Training================
@tf.function
def train_step(path_link_decay_mat, path_demand_decay_mat,
               path_link_inc_mat, path_demand_inc_mat):
    with tf.GradientTape() as tape:
        est_path_flow_tensor = g_path_variable_list
        # link_flow = path_flows * path_link_decay_mat
        estimation_link_flow_tensor = tf.matmul(tf.transpose(est_path_flow_tensor), path_link_decay_mat,
                                                name='link_flow_tensor')
        estimation_demand_flow_tensor = tf.matmul(tf.transpose(est_path_flow_tensor), path_demand_decay_mat,
                                                  name='demand_flow_tensor')
        # calculate the operational cost, risk, and discard cost
        cost_coeff1_tensor = [link.cost_coeff1 for link in g_link_list]
        cost_coeff2_tensor = [link.cost_coeff2 for link in g_link_list]
        risk_coeff1_tensor = [link.risk_coeff1 for link in g_link_list]
        risk_coeff2_tensor = [link.risk_coeff2 for link in g_link_list]
        discard_coeff1_tensor = [link.discard_coeff1 for link in g_link_list]
        discard_coeff2_tensor = [link.discard_coeff2 for link in g_link_list]

        link_cost_tensor = tf.square(
            estimation_link_flow_tensor) * cost_coeff2_tensor + estimation_link_flow_tensor * cost_coeff1_tensor
        link_risk_tensor = tf.square(
            estimation_link_flow_tensor) * risk_coeff2_tensor + estimation_link_flow_tensor * risk_coeff1_tensor
        link_discard_tensor = tf.square(
            estimation_link_flow_tensor) * discard_coeff2_tensor + estimation_link_flow_tensor * discard_coeff1_tensor
        # Calculate the total cost, risk, and discard cost
        loss_value_cost = tf.reduce_sum(link_cost_tensor, axis=1)
        loss_value_risk = tf.reduce_sum(link_risk_tensor, axis=1)
        loss_value_discard = tf.reduce_sum(link_discard_tensor, axis=1)

        # calculate shortage and surplus penalties
        # step 1: draw batch_size random samples from the demand data

        sampled_random_arrays = np.zeros((len(g_demand_list), batch_size), dtype=np.float64)
        for i in range(len(g_demand_data_list)):
            demand_data = g_demand_data_list[i]
            # shuffle the demand data and pick batch_size samples
            np.random.shuffle(demand_data)
            sampled_random_arrays[i] = demand_data[:batch_size]
        sampled_random_arrays = tf.constant(sampled_random_arrays, dtype=tf.float64)
        # calculate the difference between the demand flow tensor and the sampled random arrays
        penalty_surplus_tensor = [demand.p_surplus for demand in g_demand_list]
        penalty_shortage_tensor = [demand.p_shortage for demand in g_demand_list]

        difference_tensor = tf.subtract(tf.transpose(estimation_demand_flow_tensor), sampled_random_arrays,
                                        name='difference_tensor')
        # surplus is the positive part of the difference tensor
        surplus_tensor = tf.maximum(difference_tensor, 0.0, name='surplus_tensor')
        # surplus_tensor = tf.square(surplus_tensor, name='surplus_tensor_squared')
        # shortage is the negative part of the difference tensor
        shortage_tensor = tf.maximum(-difference_tensor, 0.0, name='shortage_tensor')
        # shortage_tensor = tf.square(shortage_tensor, name='shortage_tensor_squared')
        # calculate the surplus and shortage penalties
        # change penalty_surplus_tensor to tensor
        penalty_surplus_tensor = tf.constant(penalty_surplus_tensor, dtype=tf.float64)
        penalty_surplus_tensor = tf.reshape(penalty_surplus_tensor, (1, -1))  # Reshape to match dimensions
        penalty_shortage_tensor = tf.constant(penalty_shortage_tensor, dtype=tf.float64)
        penalty_shortage_tensor = tf.reshape(penalty_shortage_tensor, (1, -1))  # Reshape to match dimensions
        surplus_penalty_tensor = tf.matmul(penalty_surplus_tensor, surplus_tensor)
        shortage_penalty_tensor = tf.matmul(penalty_shortage_tensor, shortage_tensor)
        loss_value_surplus = tf.reduce_sum(surplus_penalty_tensor, axis=1) / batch_size
        loss_value_shortage = tf.reduce_sum(shortage_penalty_tensor, axis=1) / batch_size
        # calculate the total loss
        total_loss_value = loss_value_cost + risk_aversion * loss_value_risk + \
                           loss_value_discard + loss_value_surplus + loss_value_shortage
        #
        # print((f"Loss Cost: {loss_value_cost.numpy()[0]}, "
        #        f"Loss Risk: {loss_value_risk.numpy()[0]}, "
        #        f"Loss Discard: {loss_value_discard.numpy()[0]},"
        #        f"Loss Surplus: {loss_value_surplus.numpy()[0]}, "
        #        f"Loss Shortage: {loss_value_shortage.numpy()[0]}"))
        # print(f"Total Loss: {total_loss_value.numpy()[0]}")
        # print(f"path_flow_variable_list: {[var.numpy()[0].item() for var in g_path_variable_list]}")


    # Compute gradients and apply the optimizer
    grads = tape.gradient(total_loss_value, g_path_variable_list)
    optimizer.apply_gradients(zip(grads, g_path_variable_list))
    # ensure the path variables are non-negative
    for path_var in g_path_variable_list:
        path_var.assign(tf.maximum(path_var, 0.0))
    return (total_loss_value, loss_value_cost, loss_value_risk,
            loss_value_discard, loss_value_surplus, loss_value_shortage,
            grads, g_path_variable_list, estimation_demand_flow_tensor,
            estimation_link_flow_tensor)


if __name__ == '__main__':
    print('-------- Step 1: reading input data --------', '\n')
    t0 = datetime.datetime.now()
    network_reading()
    # generate_random_empirical_demand()
    generate_random_uniform_demand()

    demand_reading(time_index=0)

    print('-------- Step 2: find all paths between supply nodes and demand nodes --------', '\n')
    net = Network()
    net.add_nodes_and_edges(g_node_list, g_link_list)
    net.create_path_variable(g_demand_list, g_supply_list)
    print('generated ' + str(len(g_path_list)) + ' paths between supply nodes and demand nodes.')

    print('-------- Step 3: create incidence matrices and coefficients --------', '\n')
    path_link_inc_matrix, path_link_decay_matrix = create_path_flow_incidence_matrices()
    path_demand_inc_matrix, path_demand_decay_matrix = create_path_demand_point_incidence_matrices()

    # Step 6: Creat the demand random array
    print('-------- Step 6: Create the demand random array --------', '\n')

    # Step 7: Training
    print('-------- Step 6: Training --------', '\n')

    learning_rate = 0.01
    decay_steps = 100
    decay_rate = 0.96

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=learning_rate,
        decay_steps=decay_steps,
        decay_rate=decay_rate, )

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        (total_loss, loss_cost, loss_risk, loss_discard, loss_surplus,
         loss_shortage, gradients, path_variable_list, est_demand_flow_tensor,
         est_link_flow_tensor) = train_step(path_link_decay_matrix, path_demand_decay_matrix,
                                            path_link_inc_matrix, path_demand_inc_matrix)

        total_loss1 = total_loss.numpy()[0].item()
        loss_cost1 = loss_cost.numpy()[0].item()
        loss_risk1 = loss_risk.numpy()[0].item()
        loss_discard1 = loss_discard.numpy()[0].item()
        loss_surplus1 = loss_surplus.numpy()[0].item()
        loss_shortage1 = loss_shortage.numpy()[0].item()

        file_mode = 'w' if epoch == 0 else 'a'
        with open('output_training_log.csv', file_mode, newline='') as f:
            # write head first
            writer = csv.writer(f)
            if epoch == 0:
                writer.writerow(['epoch', 'total_loss', 'loss_cost', 'loss_risk', 'loss_discard',
                                 'loss_surplus', 'loss_shortage'])
            # total_loss1 = total_loss.item()
            # loss_cost1 = loss_cost.item()
            # loss_risk1 = loss_risk.item()
            # loss_discard1 = loss_discard.item()
            # loss_surplus1 = loss_surplus.item()
            # loss_shortage1 = loss_shortage.item()

            # write the data
            writer.writerow([epoch + 1, total_loss1, loss_cost1, loss_risk1, loss_discard1,
                             loss_surplus1, loss_shortage1])

        with open('output_path_flow.csv', file_mode, newline='') as f:
            writer = csv.writer(f)
            if epoch == 0:
                writer.writerow(['epoch', 'path_id', 'path_flow', 'gradients', 'link_seq'])

            for path_index, path_variable in enumerate(path_variable_list):
                writer.writerow([epoch + 1, path_index, path_variable.numpy()[0], gradients[path_index].numpy()[0],
                                 g_path_list[path_index].link_seq])

        with open('output_demand_flow.csv', file_mode, newline='') as f:
            writer = csv.writer(f)
            if epoch == 0:
                writer.writerow(['epoch', 'demand_node_id', 'demand_flow'])
            for demand_index, demand_node in enumerate(g_demand_list):
                writer.writerow([epoch + 1, demand_node.node_id, est_demand_flow_tensor.numpy()[0][demand_index]])

        with open('output_link_flow.csv', file_mode, newline='') as f:
            writer = csv.writer(f)
            if epoch == 0:
                writer.writerow(['epoch', 'link_id', 'link_flow'])
            for link_index, link in enumerate(g_link_list):
                writer.writerow([epoch + 1, link.link_id, est_link_flow_tensor.numpy()[0][link_index]])
