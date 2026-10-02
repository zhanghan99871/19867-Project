from node import Node, Clusters, Network
from data_loader import CaseDataLoader, FlowDataLoader
import pandas as pd
import numpy as np 
from constants import * 

def fit():
    data_loader = CaseDataLoader(root="../data/case_data")
    # county_df, metadata = data_loader.load_county("42003")
    for state in ["Pennsylvania", "New York", "California", "Texas", "Florida"]:
        county_df, metadata = data_loader.load_state(state)
        print(metadata)
        population = metadata["population"]
        # 0.1% of total population
        threshold = 0.001 * population
        valid_rows = county_df[
            county_df["confirmed"] >= threshold
        ]
        if valid_rows.empty:
            print(
                f"{state}: cases never reached "
                f"0.1% of population"
            )
            continue
        start = valid_rows.index[0]
        end = start + 90
        example = Node(id=STATE_FIPS[state], name=state, model_type="SIR", case_data=county_df, total_population=metadata["population"])

        example.fit_model(start = start, end = end) 
        states = example.predict(start = start, end = start + 180)
        example.plot_results(states, start = start, end = start + 180, save_path="../results")

def fit_with_flow():
    data_loader = CaseDataLoader(root="../data/case_data")
    flow_loader = FlowDataLoader(root="../data/flow_data/state")
    network_start_date = "2020-04-01"
    # county_df, metadata = data_loader.load_county("42003")
    nodes = []
    start = None 
    network_start_date = pd.Timestamp("2020-05-01")
    network_end_date = network_start_date + pd.Timedelta(days=180)
    for state in STATE_FIPS.keys():
        county_df, metadata = data_loader.load_state(state)
        print(metadata)
        valid_rows = county_df[
            (county_df["date"] >= network_start_date) & (county_df["date"] <= network_end_date)
        ]
        start = valid_rows.index[0]
        example = Node(id=STATE_FIPS[state], name=state, model_type="SIR", case_data=county_df, total_population=metadata["population"])

        example.fit_model(start = start, end = start + 90) 
        states = example.predict(start = start, end = start + 180)
        example.plot_results(states, start = start, end = start + 180, save_path="../results")
        nodes.append(example) 
    
    
    flow_matrix_dict = flow_loader.flow_matrix_range(
        start_date=network_start_date,
        end_date=network_end_date,
        flow_type="pop_flows",
        include_self=False,
    )

    network = Network(
        nodes=nodes,
        flow_matrix_dict=flow_matrix_dict,
        mode="fit",
        method="euler",
    )
    
    network.fit_model(start=start, end=start + 90) 
    pred = network.predict(start=start, end=start + 180)
    network.plot_all(pred, start=start, end=start + 180, save_path="../results")

def simulate():
    example = Node("toy", model_type="SIR", mode="simulate", case_data=None, total_population=1000000)
    states = example.predict(start = 0, end = 180, initial_state=[999000, 1000, 0], theta=[0.5, 0.1])
    example.plot_results(states, start = 0, end = 180, save_path="results")

def test_flow():
    flow_loader = FlowDataLoader(root="../data/flow_data/state")
    flow_matrix = flow_loader.flow_matrix("2020-01-01", flow_type="pop_flows", include_self=False)
    print(flow_loader.flow_summary(flow_matrix))
    # flow_matrix_range = flow_loader.flow_matrix_range("2020-01-01", "2020-01-10", flow_type="pop_flows", include_self=True)

def simulate_flow():
    np.random.seed(7)
    node_ids = ["A", "B", "C", "D"]
    populations = [100_000, 80_000, 120_000, 60_000]
    T = 100

    nodes = [
        Node(
            id=node_id,
            mode="simulate",
            model_type="SIR",
            total_population=N,
        )
        for node_id, N in zip(node_ids, populations)
    ]

    n = len(nodes)

    N = np.asarray(populations, dtype=float)

    # 0.1% - 1% initially infected
    # infected_fraction = np.random.uniform(0.001, 0.01, n)
    
    # only first node has initially infected individuals
    infected_fraction = np.zeros(n)
    infected_fraction[0] = 0.05  # 1% of the first node is initially infected

    I0 = N * infected_fraction
    R0 = np.zeros(n)
    S0 = N - I0 - R0

    # Network expects shape (num_nodes, 3)
    initial_state = np.column_stack([
        S0,
        I0,
        R0,
    ])
    # simulate isolated version 
    clusters = Clusters(nodes)
    clusters.random_initializer(beta_range=(0.2, 0.3), gamma_range=(0.05, 0.1))
    cluster_theta = clusters.get_all()
    print(cluster_theta)
    cluster_initial_state = {node.id: initial_state[i] for i, node in enumerate(nodes)}
    cluster_predictions = clusters.predict_all(start=0, end=T-1, thetas=cluster_theta, initial_states=cluster_initial_state)
    clusters.plot_all(cluster_predictions, start=0, end=T-1, save_path="../results/clusters")
    flow_strength = 1.0
    flow_matrix = np.array([
                [0,    1000,    0,  500],
                [1000,    0,  800,    0],
                [0,     800,    0, 1200],
                [500,     0, 1200,    0],
            ])
    
    F = pd.DataFrame(
        flow_matrix * flow_strength,
        index=node_ids,
        columns=node_ids,
        dtype=float,
    )
    dates = pd.date_range("2020-01-01", periods=T - 1, freq="D")

    flow_matrix_dict = {
        date: F.copy()
        for date in dates
    }
    network = Network(
        nodes=nodes,
        flow_matrix_dict=flow_matrix_dict,
        mode="simulate",
        method="euler",
    )
    network.set_beta_travel(1.0)
    theta = network.model.get()
    states = network.predict(
        start=0,
        end=T - 1,
        initial_state=initial_state,
        theta=theta,
        flow_keys=list(dates),
    )
    
    network.plot_all(states, 0, T-1, save_path="../results/network")
    
    

if __name__ == "__main__":
    # fit()
    fit_with_flow()
    # simulate()
    # test_flow()
    # simulate_flow()