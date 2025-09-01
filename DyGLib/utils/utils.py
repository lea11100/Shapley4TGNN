from typing import Optional, List, Tuple
import random
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from Config.config import CONFIG

from DyGLib.utils.DataLoader import Data

CONFIG = CONFIG()

def set_random_seed(seed: int = 0):
    """
    set random seed
    :param seed: int, random seed
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def convert_to_gpu(*data, device: str):
    """
    convert data from cpu to gpu, accelerate the running speed
    :param data: can be any type, including Tensor, Module, ...
    :param device: str
    """
    res = []
    for item in data:
        item = item.to(device)
        res.append(item)
    if len(res) > 1:
        res = tuple(res)
    else:
        res = res[0]
    return res


def get_parameter_sizes(model: nn.Module):
    """
    get parameter size of trainable parameters in model
    :param model: nn.Module
    :return:
    """
    return sum([p.numel() for p in model.parameters() if p.requires_grad])


def create_optimizer(model: nn.Module, optimizer_name: str, learning_rate: float, weight_decay: float = 0.0):
    """
    create optimizer
    :param model: nn.Module
    :param optimizer_name: str, optimizer name
    :param learning_rate: float, learning rate
    :param weight_decay: float, weight decay
    :return:
    """
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Wrong value for optimizer {optimizer_name}!")

    return optimizer


class NeighborSampler:

    def __init__(self, adj_list: list, edge_features: np.ndarray, edge_labels:Optional[np.ndarray] = None, sample_neighbor_strategy: str = 'uniform', time_scaling_factor: float = 0.0, seed = None):
        """
        Neighbor sampler.
        :param adj_list: list, list of list, where each element is a list of triple tuple (node_id, edge_id, timestamp)
        :param sample_neighbor_strategy: str, how to sample historical neighbors, 'uniform', 'recent', or 'time_interval_aware'
        :param time_scaling_factor: float, a hyper-parameter that controls the sampling preference with time interval,
        a large time_scaling_factor tends to sample more on recent links, this parameter works when sample_neighbor_strategy == 'time_interval_aware'
        :param seed: int, random seed
        """
        self.sample_neighbor_strategy = sample_neighbor_strategy
        self.seed = seed

        # list of each node's neighbor ids, edge ids and interaction times, which are sorted by interaction times
        self.nodes_neighbor_ids = []
        self.nodes_edge_ids = []
        self.nodes_neighbor_times = []

        self.edge_features = torch.from_numpy(edge_features)

        if(edge_labels is not None):
            self.edge_labels = edge_labels

        if self.sample_neighbor_strategy == 'time_interval_aware':
            self.nodes_neighbor_sampled_probabilities = []
            self.time_scaling_factor = time_scaling_factor

        # the list at the first position in adj_list is empty, hence, sorted() will return an empty list for the first position
        # its corresponding value in self.nodes_neighbor_ids, self.nodes_edge_ids, self.nodes_neighbor_times will also be empty with length 0
        for node_idx, per_node_neighbors in enumerate(adj_list):
            # per_node_neighbors is a list of tuples (neighbor_id, edge_id, timestamp)
            # sort the list based on timestamps, sorted() function is stable
            # Note that sort the list based on edge id is also correct, as the original data file ensures the interactions are chronological
            sorted_per_node_neighbors = sorted(per_node_neighbors, key=lambda x: x[2])
            self.nodes_neighbor_ids.append(np.array([x[0] for x in sorted_per_node_neighbors]))
            self.nodes_edge_ids.append(np.array([x[1] for x in sorted_per_node_neighbors]))
            self.nodes_neighbor_times.append(np.array([x[2] for x in sorted_per_node_neighbors]))

            # additional for time interval aware sampling strategy (proposed in CAWN paper)
            if self.sample_neighbor_strategy == 'time_interval_aware':
                self.nodes_neighbor_sampled_probabilities.append(self.compute_sampled_probabilities(np.array([x[2] for x in sorted_per_node_neighbors])))

        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def get_edge_features(self, edge_ids:np.ndarray):
        return self.edge_features[edge_ids]
    
    def get_edge_labels(self, edge_ids:np.ndarray):
        return self.edge_labels[edge_ids]
    
    def get_edge_labels_for_multi_hop(self, edge_ids:List[np.ndarray]):
        result = [self.get_edge_labels(x) for x in edge_ids]
        return result
    
    def get_edge_features_for_multi_hop(self, edge_ids:List[np.ndarray]):
        result = [self.get_edge_features(x) for x in edge_ids]
        return result

    def compute_sampled_probabilities(self, node_neighbor_times: np.ndarray):
        """
        compute the sampled probabilities of historical neighbors based on their interaction times
        :param node_neighbor_times: ndarray, shape (num_historical_neighbors, )
        :return:
        """
        if len(node_neighbor_times) == 0:
            return np.array([])
        # compute the time delta with regard to the last time in node_neighbor_times
        node_neighbor_times = node_neighbor_times - np.max(node_neighbor_times)
        # compute the normalized sampled probabilities of historical neighbors
        exp_node_neighbor_times = np.exp(self.time_scaling_factor * node_neighbor_times)
        sampled_probabilities = exp_node_neighbor_times / np.cumsum(exp_node_neighbor_times)
        # note that the first few values in exp_node_neighbor_times may be all zero, which make the corresponding values in sampled_probabilities
        # become nan (divided by zero), so we replace the nan by a very large negative number -1e10 to denote the sampled probabilities
        sampled_probabilities[np.isnan(sampled_probabilities)] = -1e10
        return sampled_probabilities

    def find_neighbors_before(self, node_id: int, interact_time: float, kept_edge_ids: Optional[np.ndarray] = None, return_sampled_probabilities: bool = False):
        """
        extracts all the interactions happening before interact_time (less than interact_time) for node_id in the overall interaction graph
        the returned interactions are sorted by time.
        :param node_id: int, node id
        :param interact_time: float, interaction time
        :param return_sampled_probabilities: boolean, whether return the sampled probabilities of neighbors
        :return: neighbors, edge_ids, timestamps and sampled_probabilities (if return_sampled_probabilities is True) with shape (historical_nodes_num, )
        """
        # return index i, which satisfies list[i - 1] < v <= list[i]
        # return 0 for the first position in self.nodes_neighbor_times since the value at the first position is empty
        i = np.searchsorted(self.nodes_neighbor_times[node_id], interact_time)

        if(kept_edge_ids is not None):
            mask = np.isin(self.nodes_edge_ids[node_id][:i], kept_edge_ids)
        else:
            mask = np.full_like(self.nodes_edge_ids[node_id][:i], True, dtype="bool")

        if return_sampled_probabilities:
            return self.nodes_neighbor_ids[node_id][:i][mask], self.nodes_edge_ids[node_id][:i][mask], self.nodes_neighbor_times[node_id][:i][mask], \
                   self.nodes_neighbor_sampled_probabilities[node_id][:i][mask]
        else:
            return self.nodes_neighbor_ids[node_id][:i][mask], self.nodes_edge_ids[node_id][:i][mask], self.nodes_neighbor_times[node_id][:i][mask], None

    def get_historical_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 20, kept_edge_ids: Optional[np.ndarray] = None):
        """
        get historical neighbors of nodes in node_ids with interactions before the corresponding time in node_interact_times
        :param node_ids: ndarray, shape (batch_size, ) or (*, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ) or (*, ), node interaction times
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        assert num_neighbors > 0, 'Number of sampled neighbors for each node should be greater than 0!'
        # All interactions described in the following three matrices are sorted in each row by time
        # each entry in position (i,j) represents the id of the j-th dst node of src node node_ids[i] with an interaction before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_ids = np.zeros((len(node_ids), num_neighbors)).astype(np.longlong)
        # each entry in position (i,j) represents the id of the edge with src node node_ids[i] and dst node nodes_neighbor_ids[i][j] with an interaction before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_edge_ids = np.zeros((len(node_ids), num_neighbors)).astype(np.longlong)
        # each entry in position (i,j) represents the interaction time between src node node_ids[i] and dst node nodes_neighbor_ids[i][j], before node_interact_times[i]
        # ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_times = np.zeros((len(node_ids), num_neighbors)).astype(np.float32)

        # extracts all neighbors ids, edge ids and interaction times of nodes in node_ids, which happened before the corresponding time in node_interact_times
        for idx, (node_id, node_interact_time) in enumerate(zip(node_ids, node_interact_times)):
            kept_edges = kept_edge_ids[idx] if kept_edge_ids is not None else None
            # find neighbors that interacted with node_id before time node_interact_time
            node_neighbor_ids, node_edge_ids, node_neighbor_times, node_neighbor_sampled_probabilities = \
                self.find_neighbors_before(node_id=node_id, interact_time=node_interact_time, kept_edge_ids=kept_edges, return_sampled_probabilities=self.sample_neighbor_strategy == 'time_interval_aware')

            if len(node_neighbor_ids) > 0:
                if self.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
                    # when self.sample_neighbor_strategy == 'uniform', we shuffle the data before sampling with node_neighbor_sampled_probabilities as None
                    # when self.sample_neighbor_strategy == 'time_interval_aware', we sample neighbors based on node_neighbor_sampled_probabilities
                    # for time_interval_aware sampling strategy, we additionally use softmax to make the sum of sampled probabilities be 1
                    if node_neighbor_sampled_probabilities is not None:
                        # for extreme case that node_neighbor_sampled_probabilities only contains -1e10, which will make the denominator of softmax be zero,
                        # torch.softmax() function can tackle this case
                        node_neighbor_sampled_probabilities = torch.softmax(torch.from_numpy(node_neighbor_sampled_probabilities).float(), dim=0).numpy()
                    if self.seed is None:
                        sampled_indices = np.random.choice(a=len(node_neighbor_ids), size=num_neighbors, p=node_neighbor_sampled_probabilities)
                    else:
                        sampled_indices = self.random_state.choice(a=len(node_neighbor_ids), size=num_neighbors, p=node_neighbor_sampled_probabilities)

                    nodes_neighbor_ids[idx, :] = node_neighbor_ids[sampled_indices]
                    nodes_edge_ids[idx, :] = node_edge_ids[sampled_indices]
                    nodes_neighbor_times[idx, :] = node_neighbor_times[sampled_indices]

                    # resort based on timestamps, return the ids in sorted increasing order, note this maybe unstable when multiple edges happen at the same time
                    # (we still do this though this is unnecessary for TGAT or CAWN to guarantee the order of nodes,
                    # since TGAT computes in an order-agnostic manner with relative time encoding, and CAWN computes for each walk while the sampled nodes are in different walks)
                    sorted_position = nodes_neighbor_times[idx, :].argsort()
                    nodes_neighbor_ids[idx, :] = nodes_neighbor_ids[idx, :][sorted_position]
                    nodes_edge_ids[idx, :] = nodes_edge_ids[idx, :][sorted_position]
                    nodes_neighbor_times[idx, :] = nodes_neighbor_times[idx, :][sorted_position]
                elif self.sample_neighbor_strategy == 'recent':
                    # Take most recent interactions with number num_neighbors
                    node_neighbor_ids = node_neighbor_ids[-num_neighbors:]
                    node_edge_ids = node_edge_ids[-num_neighbors:]
                    node_neighbor_times = node_neighbor_times[-num_neighbors:]

                    # put the neighbors' information at the back positions
                    nodes_neighbor_ids[idx, num_neighbors - len(node_neighbor_ids):] = node_neighbor_ids
                    nodes_edge_ids[idx, num_neighbors - len(node_edge_ids):] = node_edge_ids
                    nodes_neighbor_times[idx, num_neighbors - len(node_neighbor_times):] = node_neighbor_times
                else:
                    raise ValueError(f'Not implemented error for sample_neighbor_strategy {self.sample_neighbor_strategy}!')

        # three ndarrays, with shape (batch_size, num_neighbors)
        return nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times

    def get_multi_hop_neighbors(self, num_hops: int, node_ids: np.ndarray, node_interact_times: np.ndarray, num_neighbors: int = 20, kept_edge_ids: Optional[np.ndarray] = None):
        """
        get historical neighbors of nodes in node_ids within num_hops hops
        :param num_hops: int, number of sampled hops
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :param num_neighbors: int, number of neighbors to sample for each node
        :return:
        """
        assert num_hops > 0, 'Number of sampled hops should be greater than 0!'

        # get the temporal neighbors at the first hop
        # nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times -> ndarray, shape (batch_size, num_neighbors)
        nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = self.get_historical_neighbors(node_ids=node_ids,
                                                                                                 node_interact_times=node_interact_times,
                                                                                                 num_neighbors=num_neighbors, kept_edge_ids = kept_edge_ids)
        # three lists to store the neighbor ids, edge ids and interaction timestamp information
        nodes_neighbor_ids_list = [nodes_neighbor_ids]
        nodes_edge_ids_list = [nodes_edge_ids]
        nodes_neighbor_times_list = [nodes_neighbor_times]
        kept_edge_ids = kept_edge_ids.repeat(num_neighbors, axis=0) if kept_edge_ids is not None else None
        for hop in range(1, num_hops):
            # get information of neighbors sampled at the current hop
            # three ndarrays, with shape (batch_size * num_neighbors ** hop, num_neighbors)
            nodes_neighbor_ids, nodes_edge_ids, nodes_neighbor_times = self.get_historical_neighbors(node_ids=nodes_neighbor_ids_list[-1].flatten(),
                                                                                                     node_interact_times=nodes_neighbor_times_list[-1].flatten(),
                                                                                                     num_neighbors=num_neighbors, kept_edge_ids = kept_edge_ids)
            # three ndarrays with shape (batch_size, num_neighbors ** (hop + 1))
            nodes_neighbor_ids = nodes_neighbor_ids.reshape(len(node_ids), -1)
            nodes_edge_ids = nodes_edge_ids.reshape(len(node_ids), -1)
            nodes_neighbor_times = nodes_neighbor_times.reshape(len(node_ids), -1)

            nodes_neighbor_ids_list.append(nodes_neighbor_ids)
            nodes_edge_ids_list.append(nodes_edge_ids)
            nodes_neighbor_times_list.append(nodes_neighbor_times)

        # tuple, each element in the tuple is a list of num_hops ndarrays, each with shape (batch_size, num_neighbors ** current_hop)
        return nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list

    def get_all_first_hop_neighbors(self, node_ids: np.ndarray, node_interact_times: np.ndarray, kept_edge_ids: Optional[np.ndarray] = None):
        """
        get historical neighbors of nodes in node_ids at the first hop with max_num_neighbors as the maximal number of neighbors (make the computation feasible)
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        :return:
        """
        # three lists to store the first-hop neighbor ids, edge ids and interaction timestamp information, with batch_size as the list length
        nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list = [], [], []
        # get the temporal neighbors at the first hop
        for idx, (node_id, node_interact_time) in enumerate(zip(node_ids, node_interact_times)):
            kept_edges = kept_edge_ids[idx] if kept_edge_ids is not None else None
            # find neighbors that interacted with node_id before time node_interact_time
            node_neighbor_ids, node_edge_ids, node_neighbor_times, _ = self.find_neighbors_before(node_id=node_id,
                                                                                                  interact_time=node_interact_time,
                                                                                                  return_sampled_probabilities=False,
                                                                                                  kept_edge_ids=kept_edges)
            nodes_neighbor_ids_list.append(node_neighbor_ids)
            nodes_edge_ids_list.append(node_edge_ids)
            nodes_neighbor_times_list.append(node_neighbor_times)

        return nodes_neighbor_ids_list, nodes_edge_ids_list, nodes_neighbor_times_list
        """
        Params
        ------
        src_idx: int
        cut_time: float
        (optional) e_idx: can be used to perform look up by e_idx
        """
        node_idx_l = self.node_idx_l
        node_ts_l = self.node_ts_l
        edge_idx_l = self.edge_idx_l
        off_set_l = self.off_set_l
        binary_prob_l = self.binary_prob_l  # TODO: make it in preprocessing
        idx, e_idxs, ts, prob, sources = [], [], [], [], []
        for src_idx in src_idx_list:
            start = off_set_l[src_idx]
            end = off_set_l[src_idx + 1]
            neighbors_idx = node_idx_l[start: end]
            neighbors_ts = node_ts_l[start: end]
            neighbors_e_idx = edge_idx_l[start: end]

            assert (len(neighbors_idx) == len(neighbors_ts) and len(neighbors_idx) == len(neighbors_e_idx))  # check the next line validality
            if e_idx is None:
                cut_idx = bisect_left_adapt(neighbors_ts, cut_time)  # very crucial to exempt ties (so don't use bisect)
            else:
                cut_idx = self.nodeedge2idx[src_idx].get(e_idx) if src_idx > 0 else 0
                if cut_idx is None:
                    cut_idx = 0
            idx.append(neighbors_idx[:cut_idx])
            e_idxs.append(neighbors_e_idx[:cut_idx])
            ts.append(neighbors_ts[:cut_idx])
            source_ids = [src_idx] * len(neighbors_ts[:cut_idx])
            sources.extend(source_ids)
            if return_binary_prob:
                neighbors_binary_prob = binary_prob_l[start: end]
                prob.append(neighbors_binary_prob[:cut_idx])
        idx_array = np.concatenate(idx)   #[num possible targets]
        e_id_array = np.concatenate(e_idxs)
        ts_array = np.concatenate(ts)
        source_array = np.array(sources)
        if return_binary_prob:
            prob_array = np.concatenate(prob)
            result = (source_array, idx_array, e_id_array, ts_array, prob_array)
        else:
            result = (source_array, idx_array, e_id_array, ts_array, None)
        return result

    def reset_random_state(self):
        """
        reset the random state by self.seed
        :return:
        """
        self.random_state = np.random.RandomState(self.seed)


def get_neighbor_sampler(data: Data, edge_features: np.ndarray, sample_neighbor_strategy: str = 'uniform', time_scaling_factor: float = 0.0, seed: Optional[int] = None):
    """
    get neighbor sampler
    :param data: Data
    :param sample_neighbor_strategy: str, how to sample historical neighbors, 'uniform', 'recent', or 'time_interval_aware''
    :param time_scaling_factor: float, a hyper-parameter that controls the sampling preference with time interval,
    a large time_scaling_factor tends to sample more on recent links, this parameter works when sample_neighbor_strategy == 'time_interval_aware'
    :param seed: int, random seed
    :return:
    """
    max_node_id = max(data.src_node_ids.max(), data.dst_node_ids.max())
    # the adjacency vector stores edges for each node (source or destination), undirected
    # adj_list, list of list, where each element is a list of triple tuple (node_id, edge_id, timestamp)
    # the list at the first position in adj_list is empty
    adj_list = [[] for _ in range(max_node_id + 1)]
    for src_node_id, dst_node_id, edge_id, node_interact_time in zip(data.src_node_ids, data.dst_node_ids, data.edge_ids, data.node_interact_times):
        if(not CONFIG.data.is_directed):
            adj_list[src_node_id].append((dst_node_id, edge_id, node_interact_time))
        adj_list[dst_node_id].append((src_node_id, edge_id, node_interact_time))

    return NeighborSampler(adj_list=adj_list, edge_features=edge_features, edge_labels=data.types, sample_neighbor_strategy=sample_neighbor_strategy, time_scaling_factor=time_scaling_factor, seed=seed)


class NegativeEdgeSampler(object):

    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, interact_times: Optional[np.ndarray] = None, last_observed_time: Optional[float] = None,
                 negative_sample_strategy: str = 'random', seed: Optional[int] = None):
        """
        Negative Edge Sampler, which supports three strategies: "random", "historical", "inductive".
        :param src_node_ids: ndarray, (num_src_nodes, ), source node ids, num_src_nodes == num_dst_nodes
        :param dst_node_ids: ndarray, (num_dst_nodes, ), destination node ids
        :param interact_times: ndarray, (num_src_nodes, ), interaction timestamps
        :param last_observed_time: float, time of the last observation (for inductive negative sampling strategy)
        :param negative_sample_strategy: str, negative sampling strategy, can be "random", "historical", "inductive"
        :param seed: int, random seed
        """
        self.seed = seed
        self.negative_sample_strategy = negative_sample_strategy
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.interact_times = interact_times
        self.unique_src_node_ids = np.unique(src_node_ids)
        self.unique_dst_node_ids = np.unique(dst_node_ids)
        self.unique_interact_times = np.unique(interact_times)
        self.earliest_time = min(self.unique_interact_times)
        self.last_observed_time = last_observed_time

        if self.negative_sample_strategy != 'random':
            # all the possible edges that connect source nodes in self.unique_src_node_ids with destination nodes in self.unique_dst_node_ids
            self.possible_edges = set((src_node_id, dst_node_id) for src_node_id in self.unique_src_node_ids for dst_node_id in self.unique_dst_node_ids)

        if self.negative_sample_strategy == 'inductive':
            # set of observed edges
            self.observed_edges = self.get_unique_edges_between_start_end_time(self.earliest_time, self.last_observed_time)

        if self.seed is not None:
            self.random_state = np.random.RandomState(self.seed)

    def get_unique_edges_between_start_end_time(self, start_time: float, end_time: float):
        """
        get unique edges happened between start and end time
        :param start_time: float, start timestamp
        :param end_time: float, end timestamp
        :return: a set of edges, where each edge is a tuple of (src_node_id, dst_node_id)
        """
        selected_time_interval = np.logical_and(self.interact_times >= start_time, self.interact_times <= end_time)
        # return the unique select source and destination nodes in the selected time interval
        return set((src_node_id, dst_node_id) for src_node_id, dst_node_id in zip(self.src_node_ids[selected_time_interval], self.dst_node_ids[selected_time_interval]))

    def sample(self, size: int, batch_src_node_ids: Optional[np.ndarray] = None, batch_dst_node_ids: Optional[np.ndarray] = None,
               current_batch_start_time: float = 0.0, current_batch_end_time: float = 0.0):
        """
        sample negative edges, support random, historical and inductive sampling strategy
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :param current_batch_start_time: float, start time in the current batch
        :param current_batch_end_time: float, end time in the current batch
        :return:
        """
        if self.negative_sample_strategy == 'random':
            negative_src_node_ids, negative_dst_node_ids = self.random_sample(size=size)
        elif self.negative_sample_strategy == 'historical':
            negative_src_node_ids, negative_dst_node_ids = self.historical_sample(size=size, batch_src_node_ids=batch_src_node_ids,
                                                                                  batch_dst_node_ids=batch_dst_node_ids,
                                                                                  current_batch_start_time=current_batch_start_time,
                                                                                  current_batch_end_time=current_batch_end_time)
        elif self.negative_sample_strategy == 'inductive':
            negative_src_node_ids, negative_dst_node_ids = self.inductive_sample(size=size, batch_src_node_ids=batch_src_node_ids,
                                                                                 batch_dst_node_ids=batch_dst_node_ids,
                                                                                 current_batch_start_time=current_batch_start_time,
                                                                                 current_batch_end_time=current_batch_end_time)
        else:
            raise ValueError(f'Not implemented error for negative_sample_strategy {self.negative_sample_strategy}!')
        return negative_src_node_ids, negative_dst_node_ids

    def random_sample(self, size: int):
        """
        random sampling strategy, which is used by previous works
        :param size: int, number of sampled negative edges
        :return:
        """
        if self.seed is None:
            random_sample_edge_src_node_indices = np.random.randint(0, len(self.unique_src_node_ids), size)
            random_sample_edge_dst_node_indices = np.random.randint(0, len(self.unique_dst_node_ids), size)
        else:
            random_sample_edge_src_node_indices = self.random_state.randint(0, len(self.unique_src_node_ids), size)
            random_sample_edge_dst_node_indices = self.random_state.randint(0, len(self.unique_dst_node_ids), size)
        return self.unique_src_node_ids[random_sample_edge_src_node_indices], self.unique_dst_node_ids[random_sample_edge_dst_node_indices]

    def random_sample_with_collision_check(self, size: int, batch_src_node_ids: np.ndarray, batch_dst_node_ids: np.ndarray):
        """
        random sampling strategy with collision check, which guarantees that the sampled edges do not appear in the current batch,
        used for historical and inductive sampling strategy
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :return:
        """
        assert batch_src_node_ids is not None and batch_dst_node_ids is not None
        batch_edges = set((batch_src_node_id, batch_dst_node_id) for batch_src_node_id, batch_dst_node_id in zip(batch_src_node_ids, batch_dst_node_ids))
        possible_random_edges = list(self.possible_edges - batch_edges)
        assert len(possible_random_edges) > 0
        # if replace is True, then a value in the list can be selected multiple times, otherwise, a value can be selected only once at most
        random_edge_indices = self.random_state.choice(len(possible_random_edges), size=size, replace=len(possible_random_edges) < size)
        return np.array([possible_random_edges[random_edge_idx][0] for random_edge_idx in random_edge_indices]), \
               np.array([possible_random_edges[random_edge_idx][1] for random_edge_idx in random_edge_indices])

    def historical_sample(self, size: int, batch_src_node_ids: np.ndarray, batch_dst_node_ids: np.ndarray,
                          current_batch_start_time: float, current_batch_end_time: float):
        """
        historical sampling strategy, first randomly samples among historical edges that are not in the current batch,
        if number of historical edges is smaller than size, then fill in remaining edges with randomly sampled edges
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :param current_batch_start_time: float, start time in the current batch
        :param current_batch_end_time: float, end time in the current batch
        :return:
        """
        assert self.seed is not None
        # get historical edges up to current_batch_start_time
        historical_edges = self.get_unique_edges_between_start_end_time(start_time=self.earliest_time, end_time=current_batch_start_time)
        # get edges in the current batch
        current_batch_edges = self.get_unique_edges_between_start_end_time(start_time=current_batch_start_time, end_time=current_batch_end_time)
        # get source and destination node ids of unique historical edges
        unique_historical_edges = historical_edges - current_batch_edges
        unique_historical_edges_src_node_ids = np.array([edge[0] for edge in unique_historical_edges])
        unique_historical_edges_dst_node_ids = np.array([edge[1] for edge in unique_historical_edges])

        # if sample size is larger than number of unique historical edges, then fill in remaining edges with randomly sampled edges with collision check
        if size > len(unique_historical_edges):
            num_random_sample_edges = size - len(unique_historical_edges)
            random_sample_src_node_ids, random_sample_dst_node_ids = self.random_sample_with_collision_check(size=num_random_sample_edges,
                                                                                                             batch_src_node_ids=batch_src_node_ids,
                                                                                                             batch_dst_node_ids=batch_dst_node_ids)

            negative_src_node_ids = np.concatenate([random_sample_src_node_ids, unique_historical_edges_src_node_ids])
            negative_dst_node_ids = np.concatenate([random_sample_dst_node_ids, unique_historical_edges_dst_node_ids])
        else:
            historical_sample_edge_node_indices = self.random_state.choice(len(unique_historical_edges), size=size, replace=False)
            negative_src_node_ids = unique_historical_edges_src_node_ids[historical_sample_edge_node_indices]
            negative_dst_node_ids = unique_historical_edges_dst_node_ids[historical_sample_edge_node_indices]

        # Note that if one of the input of np.concatenate is empty, the output will be composed of floats.
        # Hence, convert the type to long to guarantee valid index
        return negative_src_node_ids.astype(np.longlong), negative_dst_node_ids.astype(np.longlong)

    def inductive_sample(self, size: int, batch_src_node_ids: np.ndarray, batch_dst_node_ids: np.ndarray,
                         current_batch_start_time: float, current_batch_end_time: float):
        """
        inductive sampling strategy, first randomly samples among inductive edges that are not in self.observed_edges and the current batch,
        if number of inductive edges is smaller than size, then fill in remaining edges with randomly sampled edges
        :param size: int, number of sampled negative edges
        :param batch_src_node_ids: ndarray, shape (batch_size, ), source node ids in the current batch
        :param batch_dst_node_ids: ndarray, shape (batch_size, ), destination node ids in the current batch
        :param current_batch_start_time: float, start time in the current batch
        :param current_batch_end_time: float, end time in the current batch
        :return:
        """
        assert self.seed is not None
        # get historical edges up to current_batch_start_time
        historical_edges = self.get_unique_edges_between_start_end_time(start_time=self.earliest_time, end_time=current_batch_start_time)
        # get edges in the current batch
        current_batch_edges = self.get_unique_edges_between_start_end_time(start_time=current_batch_start_time, end_time=current_batch_end_time)
        # get source and destination node ids of historical edges but 1) not in self.observed_edges; 2) not in the current batch
        unique_inductive_edges = historical_edges - self.observed_edges - current_batch_edges
        unique_inductive_edges_src_node_ids = np.array([edge[0] for edge in unique_inductive_edges])
        unique_inductive_edges_dst_node_ids = np.array([edge[1] for edge in unique_inductive_edges])

        # if sample size is larger than number of unique inductive edges, then fill in remaining edges with randomly sampled edges
        if size > len(unique_inductive_edges):
            num_random_sample_edges = size - len(unique_inductive_edges)
            random_sample_src_node_ids, random_sample_dst_node_ids = self.random_sample_with_collision_check(size=num_random_sample_edges,
                                                                                                             batch_src_node_ids=batch_src_node_ids,
                                                                                                             batch_dst_node_ids=batch_dst_node_ids)

            negative_src_node_ids = np.concatenate([random_sample_src_node_ids, unique_inductive_edges_src_node_ids])
            negative_dst_node_ids = np.concatenate([random_sample_dst_node_ids, unique_inductive_edges_dst_node_ids])
        else:
            inductive_sample_edge_node_indices = self.random_state.choice(len(unique_inductive_edges), size=size, replace=False)
            negative_src_node_ids = unique_inductive_edges_src_node_ids[inductive_sample_edge_node_indices]
            negative_dst_node_ids = unique_inductive_edges_dst_node_ids[inductive_sample_edge_node_indices]

        # Note that if one of the input of np.concatenate is empty, the output will be composed of floats.
        # Hence, convert the type to long to guarantee valid index
        return negative_src_node_ids.astype(np.longlong), negative_dst_node_ids.astype(np.longlong)

    def reset_random_state(self):
        """
        reset the random state by self.seed
        :return:
        """
        self.random_state = np.random.RandomState(self.seed)

def compute_stats(data: Data):
    init_time = np.min(data.node_interact_times)

    last_timestamp_src = dict()
    last_timestamp_dst = dict()
    last_timestamp = dict()
    all_timediffs_src = []
    all_timediffs_dst = []
    all_timediffs = []
    for src, dst, t in zip(data.src_node_ids, data.dst_node_ids, data.node_interact_times):
        src, dst, t = src.item(), dst.item(), t.item()

        all_timediffs_src.append(t - last_timestamp_src.get(src, init_time))
        all_timediffs_dst.append(t - last_timestamp_dst.get(dst, init_time))
        all_timediffs.append(t - last_timestamp.get(src, init_time))
        all_timediffs.append(t - last_timestamp.get(dst, init_time))

        last_timestamp_src[src] = t
        last_timestamp_dst[dst] = t
        last_timestamp[src] = t
        last_timestamp[dst] = t

    src_and_dst = all_timediffs_src + all_timediffs_dst
    mean_delta_t = np.mean(all_timediffs)
    std_delta_t = np.std(all_timediffs)

    print(f'avg delta_t(src): {np.mean(all_timediffs_src)} +/- {np.std(all_timediffs_src)}')
    print(f'avg delta_t(dst): {np.mean(all_timediffs_dst)} +/- {np.std(all_timediffs_dst)}')
    print(f'avg delta_t(src+dst): {np.mean(src_and_dst)} +/- {np.std(src_and_dst)}')
    print(f'avg delta_t(all): {mean_delta_t} +/- {std_delta_t}')

    return mean_delta_t, std_delta_t, init_time

class BatchSubgraphs:
    """
    A container for storing and manipulating a batch of temporal subgraphs 
    across multiple layers.

    Each subgraph contains:
        - nodes: Node IDs for the subgraph.
        - events: Event IDs associated with edges/nodes.
        - timestamps: Timestamps for each event.
        - event_features: Feature tensors describing events.
        - event_attention: Attention weights for events.
        - node_attention: Attention weights for nodes.
        - timing_attention: Attention weights for timestamps.

    This data is typically structured layer-wise for models that 
    process temporal multi-hop subgraphs.

    Attributes
    ----------
    nodes : List[np.ndarray]
        A list of arrays of node IDs per layer. Shape: (batch_size, neighbors).
    events : List[np.ndarray]
        A list of arrays of event IDs per layer. Shape: (batch_size, neighbors).
    timestamps : List[np.ndarray]
        A list of arrays of event timestamps per layer. Shape: (batch_size, neighbors).
    event_features : List[torch.Tensor]
        A list of tensors containing event features. 
        Shape: (batch_size, neighbors, feature_dim).
    event_attention : List[torch.Tensor]
        A list of attention masks for events.
    node_attention : List[torch.Tensor]
        A list of attention masks for nodes.
    timing_attention : List[torch.Tensor]
        A list of attention masks for timing.

    Raises
    ------
    AssertionError
        If layer counts or shape dimensions between provided parameters do not match.
    """

    def __init__(self, nodes:List[np.ndarray], events:List[np.ndarray], timestamps:List[np.ndarray], event_features:List[torch.Tensor], event_attention:Optional[List[torch.Tensor]] = None, node_attention:Optional[List[torch.Tensor]]=None, timing_attention:Optional[List[torch.Tensor]]=None):
        """
        Initialize a batch of multi-layer subgraphs.

        Parameters
        ----------
        nodes : List[np.ndarray]
            List of node ID arrays for each layer.
        events : List[np.ndarray]
            List of event ID arrays for each layer.
        timestamps : List[np.ndarray]
            List of timestamp arrays for each layer.
        event_features : List[torch.Tensor]
            List of event feature tensors per layer.
        event_attention : Optional[List[torch.Tensor]], default=None
            Optional list of attention masks for events. 
            Defaults to tensors of ones if not provided.
        node_attention : Optional[List[torch.Tensor]], default=None
            Optional list of attention masks for nodes.
            Defaults to tensors of ones if not provided.
        timing_attention : Optional[List[torch.Tensor]], default=None
            Optional list of attention masks for timestamps.
            Defaults to tensors of ones if not provided.

        Raises
        ------
        AssertionError
            If input lists are not of the same length or 
            their shapes do not match in the first two dimensions.
        """
        assert len(nodes)==len(events)==len(timestamps)==len(event_features), f"All parameters must have the same number of layers. Found: Nodes: {len(nodes)}, events: {len(events)} timestamps: {len(timestamps)}, event features:{len(event_features)}"
        for layer in range(len(nodes)):
            assert nodes[layer].shape == events[layer].shape == timestamps[layer].shape == event_features[layer].shape[:2], f"The first two dimensions of all parameters must be the same. Found in layer {layer}: Nodes: {nodes[layer].shape}, events: {events[layer].shape} timestamps: {timestamps[layer].shape}, event features: {event_features[layer].shape}"

        self.nodes = [x.astype("int64") for x in nodes]
        self.events = [x.astype("int64") for x in events]
        self.timestamps = timestamps
        self.event_features = event_features

        if event_attention == None:
            self.event_attention = [torch.ones(x.shape) for x in events]
        else:
            assert len(event_attention)==len(timestamps), f"All parameters must have the same number of layers. Found: Timestamps: {len(timestamps)}, event attention: {len(event_attention)}"
            for layer in range(len(nodes)):
                assert timestamps[layer].shape == event_attention[layer].shape, f"The first two dimensions of all parameters must be the same. Found in layer {layer}: Timestamps: {timestamps[layer].shape}, event attention: {event_attention[layer].shape}"
            self.event_attention = event_attention 

        if node_attention == None:
            self.node_attention = [torch.ones(x.shape) for x in events]
        else:
            assert len(node_attention)==len(timestamps), f"All parameters must have the same number of layers. Found: Timestamps: {len(timestamps)}, node attention: {len(node_attention)}"
            for layer in range(len(nodes)):
                assert timestamps[layer].shape == node_attention[layer].shape, f"The first two dimensions of all parameters must be the same. Found in layer {layer}: Timestamps: {timestamps[layer].shape}, node attention: {node_attention[layer].shape}"
            self.node_attention = node_attention      

        if timing_attention == None:
            self.timing_attention = [torch.ones(x.shape) for x in events]
        else:
            assert len(timing_attention)==len(timestamps), f"All parameters must have the same number of layers. Found: Timestamps: {len(timestamps)}, timing attention: {len(timing_attention)}"
            for layer in range(len(nodes)):
                assert timestamps[layer].shape == timing_attention[layer].shape, f"The first two dimensions of all parameters must be the same. Found in layer {layer}: Timestamps: {timestamps[layer].shape}, timining attention: {timing_attention[layer].shape}"
            self.timing_attention = timing_attention  

    def __getitem__(self, indices):
        """
        Slice the batch along the first dimension.

        Parameters
        ----------
        indices : slice
            Slice object used to extract a subset of the batch.

        Returns
        -------
        BatchSubgraphs
            A new `BatchSubgraphs` instance containing the sliced data.

        Raises
        ------
        AssertionError
            If `indices` is not a slice object.
        """
        assert isinstance(indices, slice), "Only slices are supported!"
        nodes = [x[indices] for x in self.nodes]
        events = [x[indices] for x in self.events]
        timestamps = [x[indices] for x in self.timestamps]
        event_features = [x[indices] for x in self.event_features]
        event_attention = [x[indices] for x in self.event_attention]
        node_attention = [x[indices] for x in self.node_attention]
        timing_attention = [x[indices] for x in self.timing_attention]

        return BatchSubgraphs(nodes,events, timestamps, event_features, event_attention, node_attention, timing_attention)
    
    def __eq__(self, other):
        """
        Check if two `BatchSubgraphs` objects contain identical data.

        Parameters
        ----------
        other : BatchSubgraphs
            Another instance to compare to.

        Returns
        -------
        bool
            True if all attributes match exactly, otherwise False.
        """
        if self.get_num_layers() != other.get_num_layers():
            return False
        
        if self.get_num_instances() != other.get_num_instances():
            return False
        
        for l in range(self.get_num_layers()):
            if (self.nodes[l] != other.nodes[l]).any():
               return False
            if (self.events[l] != other.events[l]).any():
               return False
            if (self.timestamps[l] != other.timestamps[l]).any():
               return False
            if (self.event_features[l] != other.event_features[l]).any():
               return False
            if (self.event_attention[l] != other.event_attention[l]).any():
               return False
            if (self.node_attention[l] != other.node_attention[l]).any():
               return False
            if (self.timing_attention[l] != other.timing_attention[l]).any():
               return False

        return True


    def get_num_layers(self):
        """
        Returns
        -------
        int
            Number of layers in the subgraph batch.
        """
        return len(self.events)
    
    def get_num_instances(self):
        """
        Returns
        -------
        int
            Number of instances (batch size).
        """
        return self.events[0].shape[0]
    
    def get_num_events(self):
        """
        Compute the number of nonzero events for each instance across layers.

        Returns
        -------
        np.ndarray
            Array of shape (batch_size,) containing counts of events per instance.
        """
        result = np.zeros((self.get_num_instances(),))
        for e in self.events:
            result += (e != 0).sum(axis=1)
        return result
    
    def get_num_neighbors(self):
        """
        Returns
        -------
        int
            The number of neighbors at the first or last layer 
            depending on processing direction.
        """
        return min(self.nodes[0].shape[1], self.nodes[-1].shape[1]) #First or last depending on reversed
    
    def get_num_features(self):
        """
        Returns
        -------
        int
            Dimensionality of event features.
        """
        return self.event_features[0].shape[2]
    
    def get_events(self):
        """
        Concatenate events across all layers.

        Returns
        -------
        np.ndarray
            Concatenated events array of shape (batch_size, total_neighbors).
        """
        return np.concat(self.events, axis=1)
    
    def get_timings(self):
        """
        Concatenate timestamps across all layers.

        Returns
        -------
        np.ndarray
            Concatenated timestamps array.
        """
        return np.concat(self.timestamps, axis=1)
    
    def set_event_attention(self, attention):
        """
        Set the event attention values for all layers.

        Parameters
        ----------
        attention : List[torch.Tensor]
            List of attention masks with the same shapes as current event_attention.
        
        Raises
        ------
        AssertionError
            If shapes do not match existing attention tensors.
        """
        for i, _ in enumerate(self.event_attention):
            assert self.event_attention[i].shape == attention[i].shape, f"Dimensions do not match: Found {attention[i].shape} at layer {i}, expected {self.event_attention[i].shape}"
            self.event_attention[i] = attention[i]

    def chop_layers(self, new_num_layers):
        """
        Keep only the first `new_num_layers` layers.

        Parameters
        ----------
        new_num_layers : int
            Number of layers to keep.
        """
        self.nodes = self.nodes[:new_num_layers]
        self.events = self.events[:new_num_layers]
        self.timestamps = self.timestamps[:new_num_layers]
        self.event_features = self.event_features[:new_num_layers]
        self.event_attention = self.event_attention[:new_num_layers]
        self.node_attention = self.node_attention[:new_num_layers]
        self.timing_attention = self.timing_attention[:new_num_layers]
    
    def reverse_layers(self):
        """
        Reverse the order of layers.
        """
        self.nodes = self.nodes[::-1]
        self.events = self.events[::-1]
        self.timestamps = self.timestamps[::-1]
        self.event_features = self.event_features[::-1]
        self.event_attention = self.event_attention[::-1]
        self.node_attention = self.node_attention[::-1]
        self.timing_attention = self.timing_attention[::-1]

    def get_split(self, ignore_event_attention=False):
        """
        Get a tuple of core fields for all layers.

        Parameters
        ----------
        ignore_event_attention : bool, default=False
            If True, event_attention is excluded.
        
        Returns
        -------
        tuple
            Data split depending on `ignore_event_attention`.
        """
        if(ignore_event_attention):
            return (self.nodes, self.events, self.timestamps)
        return (self.nodes, self.events, self.timestamps, self.event_attention)
        
    def get_split_for_layer(self, layer:int, flat_to_node=False):
        """
        Retrieve all attributes for a specific layer.

        Parameters
        ----------
        layer : int
            Layer index.
        flat_to_node : bool, default=False
            If True, flatten the batch dimension into the node dimension.

        Returns
        -------
        tuple
            Layer-specific data (nodes, events, timestamps, features, 
            event attention, node attention).
        """
        if(flat_to_node):
            return (self.nodes[layer].reshape((-1,self.get_num_neighbors())),
                    self.events[layer].reshape((-1,self.get_num_neighbors())), 
                    self.timestamps[layer].reshape((-1,self.get_num_neighbors())), 
                    self.event_features[layer].reshape(-1,self.get_num_neighbors(),self.event_features[layer].shape[2]), 
                    self.event_attention[layer].reshape((-1,self.get_num_neighbors())),
                    self.node_attention[layer].reshape((-1,self.get_num_neighbors())))
        return (self.nodes[layer], self.events[layer], self.timestamps[layer], self.event_features[layer], self.event_attention[layer], self.node_attention[layer])
    
    def to(self, device):
        """
        Move tensor attributes to a given device.

        Parameters
        ----------
        device : torch.device or str
            The target device.
        """
        self.event_features = [x.to(device) for x in self.event_features]
        self.event_attention = [x.to(device) for x in self.event_attention]
        self.node_attention = [x.to(device) for x in self.node_attention]
        self.timing_attention = [x.to(device) for x in self.timing_attention]

    def get_event_masks(self, event_id: int):
        """
        Create boolean masks where events match a given ID.

        Parameters
        ----------
        event_id : int
            Event ID to match.

        Returns
        -------
        List[np.ndarray]
            Boolean masks per layer.
        """
        result = []
        for i, e in enumerate(self.events):
            result.append(e==event_id)
        return result

    def replace_event(self, masks: list, node_attention: torch.Tensor, timing: np.ndarray, event_features: torch.Tensor):
        """
        Replace event data where boolean masks are True. Event features are not individual.
        Parameters
        ----------
        masks : list of np.ndarray
            List of boolean masks indicating which events to replace.
        node_attention : torch.Tensor
            Node attention values to assign to masked events.
        timing : np.ndarray
            Timing values to assign to masked events.
        event_features : torch.Tensor
            Event feature vectors to assign to masked events.
        """
        for i, m in enumerate(masks):
            if(m.any()):
                self.event_features[i][m, :] = event_features
                self.node_attention[i][m] = node_attention
                self.timestamps[i][m] = timing

    def replace_event_2D(self, masks: list, node_attention: torch.Tensor, timing: np.ndarray, event_features: torch.Tensor):
        """
        Replace event data where boolean masks are True. Event features are individual.
        Parameters
        ----------
        masks : list of np.ndarray
            List of boolean masks indicating which events to replace.
        node_attention : torch.Tensor
            Node attention values per instance to assign to masked events.
        timing : np.ndarray
            Timing values per instance to assign to masked events.
        event_features : torch.Tensor
            Event feature tensors per instance to assign to masked events.
        """
        for i, m in enumerate(masks):
            if(m.any()):
                for j, row in enumerate(m):

                    self.event_features[i][j, row, :] = event_features[j]
                    self.node_attention[i][j, row] = node_attention[j]
                    self.timestamps[i][j, row] = timing[j]


    def mask_event_features(self, event_id: np.ndarray, event_features: torch.Tensor):
        """
        Mask and replace event features matching a specific ID.
        Parameters
        ----------
        event_id : np.ndarray
            ID of the event to mask.
        event_features : torch.Tensor
            Event feature tensor to replace matched events' features.
        """
        for i, x in enumerate(self.event_features):
            mask = self.events[i]==event_id
            if(mask.any()):
                x[mask, :] = event_features

    def mask_event_timing(self, event_id:np.ndarray, timing):
        """
        Mask and replace event timings for a given ID, and reset timing attention to 1.0.
        Parameters
        ----------
        event_id : np.ndarray
            ID of the event to mask.
        timing : np.ndarray
            Timing values to assign to masked events.
        """
        for i, (ts, a) in enumerate(zip(self.timestamps, self.timing_attention)):
            mask = self.events[i]==event_id
            if(mask.any()):
                ts[mask] = timing
                a[mask] = 1.0

    def mask_node_attention(self, event_id:np.ndarray, attention: torch.Tensor):
        """
        Mask and replace node attention for a given event ID.
        Parameters
        ----------
        event_id : np.ndarray
            ID of the event to mask.
        attention : torch.Tensor
            Node attention values to assign.
        """
        for i, a in enumerate(self.node_attention):
            mask = self.events[i]==event_id
            if(mask.any()):
                a[mask] = attention

    def mask_events(self, event_ids:np.ndarray, event_mask: np.ndarray, data_per_event: dict):
        """
        Mask multiple events and replace related data.
        Parameters
        ----------
        event_ids : np.ndarray
            Array of event IDs to mask.
        event_mask : np.ndarray
            Boolean mask per row/instance indicating relevant positions.
        data_per_event : dict
            Mapping from event ID -> data tuple containing timing, features, 
            and possibly other metadata.
        """
        for i, x in enumerate(self.events):
            for j, id in enumerate(event_ids):
                col = event_mask[:,j]
                mask = np.zeros_like(x,dtype="bool")
                mask[col!=0] = x[col!=0] == id
                if(mask.any()):
                    #self.nodes[i][mask] = 0
                    self.timestamps[i][mask] = data_per_event[id][1].cpu().numpy()
                    self.event_features[i][mask, :] = data_per_event[id][2:]
                    #self.event_attention[i][mask] = 0
                    self.node_attention[i][mask] = 0
                    self.timing_attention[i][mask] = 1.0
                    #self.events[i][mask] = 0

    def _get_default_event_array(self, layer:int, data_per_event: dict):
        """
        Generate default features and timings arrays for given events in a layer.
        Parameters
        ----------
        layer : int
            Layer index.
        data_per_event : dict
            Mapping from event ID to a tensor with event data, where:
                - index 1 contains timing,
                - remaining indices contain features.
        Returns
        -------
        tuple
            default_features : torch.Tensor
                Feature tensor for replacing masked events.
            default_timings : np.ndarray
                Timing array for replacing masked events.
        """
        num_features = next(iter(data_per_event.values())).shape[0]
        device = next(iter(data_per_event.values())).device
        data_per_event[0] = torch.zeros(num_features)

        default_features = torch.zeros((self.events[layer].shape[0], self.events[layer].shape[1], num_features-2), device=device)
        default_timings = torch.zeros((self.events[layer].shape[0], self.events[layer].shape[1]), device=device)

        for k, row in enumerate(self.events[layer]):
            for j, col in enumerate(row):
                default_features[k,j,:] = data_per_event[col][2:]
                default_timings[k,j] = data_per_event[col][1]
        default_timings = default_timings.cpu().numpy()

        return default_features, default_timings



    def keep_events(self, event_ids:np.ndarray, data_per_event: Optional[dict] = None):
        """
        Keep only specified event IDs, replacing or zeroing out other events.
        Parameters
        ----------
        event_ids : np.ndarray
            Array of event IDs to keep.
        data_per_event : dict, optional
            Mapping of event ID -> replacement data.
            If provided, default timing and features are used for removed events.
        """
        for i, x in enumerate(self.events):
            mask = ~((x[:,:,None] == event_ids[:, None,:]).any(axis=-1))
            if data_per_event is not None:
                default_features, default_timings = self._get_default_event_array(i, data_per_event)
                if mask.any():
                    self.timestamps[i][mask] = default_timings[mask]
                    self.event_features[i][mask, :] = default_features[mask, :]
                    self.node_attention[i][mask] = 0
            elif(mask.any()):
                    self.nodes[i][mask] = 0
                    self.timestamps[i][mask] = 0
                    self.event_features[i][mask, :] = 0
                    self.event_attention[i][mask] = 0
                    self.node_attention[i][mask] = 0
                    self.timing_attention[i][mask] = 0
                    self.events[i][mask] = 0

    def keep_features(self, kept_features: List[np.ndarray], data_per_event: Optional[dict] = None):
        """
        Keep a subset of features for events, replacing or resetting others.
        Parameters
        ----------
        kept_features : List[np.ndarray]
            Per instance and layer array specifying which (event_id, feature_idx) pairs to keep.
        data_per_event : dict, optional
            Mapping from event ID to default event data, including timings.
        """
        for i, feat in enumerate(self.event_features):
            default_features = None
            default_timings = None
            if data_per_event is not None:
                default_features, default_timings = self._get_default_event_array(i, data_per_event)

            mask_feat = torch.zeros_like(feat, dtype=torch.bool)
            mask_timing = np.zeros_like(self.events[i], dtype=bool)
            mask_node = np.zeros_like(self.events[i], dtype=bool)
            for j,l in enumerate(kept_features):
                #rows = np.array([np.where(self.events[i][j, :] == x[0])[0] if (self.events[i][j, :] == x[0]).any() else [-1] for x in l], dtype=int).reshape(-1)
                cells = []
                for cell in l:
                    c = np.where(self.events[i][j, :] == cell[0])[0].reshape((-1,1))
                    c = np.concat([c, np.full_like(c, cell[1])], axis=1)
                    cells.extend(c)
                if len(cells) == 0:
                    continue
                cells = np.array(cells, dtype=int)
                rows = cells[:,0]
                cols = cells[:,1]-2
                mask_feat[j, rows[(cols>=0)], cols[(cols>=0)]] = True
                mask_timing[j, rows[cols == -1]] = True
                mask_node[j, rows[cols == -2]] = True
            mask_feat = ~mask_feat
            mask_timing = ~mask_timing
            mask_node = ~mask_node

            if(mask_feat.any()):
                if(default_features is not None):
                    feat[mask_feat] = default_features[mask_feat]
                else:
                    feat[mask_feat] = 0
            if(mask_node.any()):
                self.node_attention[i][mask_node] = 0
            
            if(mask_timing.any()):
                if(default_timings is not None):
                    self.timestamps[i][mask_timing] = default_timings[mask_timing]
                else:
                    self.timestamps[i][mask_timing] = 0
                    self.timing_attention[i][mask_timing] = 0

    def repeat_nodes(self, n_times):
        """
        Repeat all instances in the batch `n_times` along the batch dimension.
        Parameters
        ----------
        n_times : int
            Number of repetitions.
        """
        for i, x in enumerate(self.events):
            self.nodes[i] = self.nodes[i].repeat(n_times,axis=0)
            self.events[i] = self.events[i].repeat(n_times,axis=0)
            self.timestamps[i] = self.timestamps[i].repeat(n_times,axis=0)
            self.event_features[i] = self.event_features[i].repeat(n_times,1,1)
            self.event_attention[i] = self.event_attention[i].repeat(n_times,1)
            self.node_attention[i] = self.node_attention[i].repeat(n_times,1)
            self.timing_attention[i] = self.timing_attention[i].repeat(n_times,1)

    def split_batch(self):
        """
        Split the batch into individual `BatchSubgraphs` of size 1.
        Returns
        -------
        List[BatchSubgraphs]
            List of single-instance subgraph batches.
        """
        result:List[BatchSubgraphs] = []
        for i in range(self.nodes[0].shape[0]):
            sg = BatchSubgraphs([x[[i]] for x in self.nodes],
                                [x[[i]] for x in self.events],
                                [x[[i]] for x in self.timestamps],
                                [x[[i]] for x in self.event_features],
                                [x[[i]] for x in self.event_attention],
                                [x[[i]] for x in self.node_attention],
                                [x[[i]] for x in self.timing_attention])
            result.append(sg)
        return result
    

def concat_subgraphs(subgraphs: List[BatchSubgraphs]):
    """
    Concatenate multiple `BatchSubgraphs` into one along the batch dimension.
    Parameters
    ----------
    subgraphs : List[BatchSubgraphs]
        List of batch subgraph objects with identical structure to concatenate.
    Returns
    -------
    BatchSubgraphs
        A new `BatchSubgraphs` instance containing concatenated data from input subgraphs.
    """
    nodes, events, timestamps, event_features, event_attention, node_attention, timing_attention = [], [], [], [], [], [], []
    for l in range(subgraphs[0].get_num_layers()):
        nodes.append(np.concat([x.nodes[l] for x in subgraphs]))
        events.append(np.concat([x.events[l] for x in subgraphs]))
        timestamps.append(np.concat([x.timestamps[l] for x in subgraphs]))
        event_features.append(torch.concat([x.event_features[l] for x in subgraphs]))
        event_attention.append(torch.concat([x.event_attention[l] for x in subgraphs]))
        node_attention.append(torch.concat([x.node_attention[l] for x in subgraphs]))
        timing_attention.append(torch.concat([x.timing_attention[l] for x in subgraphs]))

    result = BatchSubgraphs(nodes, events, timestamps, event_features, event_attention, node_attention, timing_attention)
    return result