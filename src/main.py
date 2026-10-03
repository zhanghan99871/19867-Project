from node import Node, Clusters, Network
from data_loader import CaseDataLoader, FlowDataLoader
import pandas as pd
import numpy as np 
from constants import * 
from optimizer import * 
import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description="SIR network travel optimization experiment"
    )

    parser.add_argument(
        "--level",
        type=str,
        choices=["state", "county"],
        default="state",
        help="Geographic level: state or county (default: state)",
    )

    parser.add_argument(
        "--problem_size",
        type=str,
        default="small",
        choices=["small", "full"],
        help="whether to use full problem size or small problem size (default: small)",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["edge", "edge_continuous", "node", "cluster"],
        default="cluster",
        help="Optimizer type (default: cluster)",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="../results",
        help="Directory for experiment outputs (default: ../results)",
    )
    
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data",
        help="Directory for input data (default: ../data)",
    )

    parser.add_argument(
        "--travel_reduce_ratio",
        type=float,
        default=0.2,
        help="Maximum fraction of total travel that can be removed (default: 0.2)",
    )

    parser.add_argument(
        "--clusters",
        type=int,
        default=3,
        help="Maximum/number of clusters for cluster optimizer (default: 3)",
    )

    parser.add_argument(
        "--start_date",
        type=str,
        default="2020-05-01",
        help="Network start date (default: 2020-05-01)",
    )

    parser.add_argument(
        "--fit_days",
        type=int,
        default=90,
        help="Number of days used for model fitting (default: 90)",
    )

    parser.add_argument(
        "--opt_days",
        type=int,
        default=30,
        help="Number of days used for optimization after fitting (default: 30)",
    )
    
    parser.add_argument(
        "--method", 
        type=str,
        choices=["euler", "rk4"],
        default="euler",
        help="Numerical integration method for the network model (default: euler)",
    )
    
    parser.add_argument(
        "--model", 
        type=str,
        choices=["SIR", "SIRD"],
        default="SIR",
        help="Epidemiological model type for the network nodes (default: SIR)",
    )

    return parser.parse_args()
    
def get_optimizer(name, network):
    if name == "edge":
        return BinaryReduceEdgeOptimizer(network)
    elif name == "edge_continuous":
        return ContinuousReduceEdgeOptimizer(network)
    elif name == "node":
        return BinaryReduceNodeOptimizer(network)
    elif name == "cluster":
        return BinaryClusterOptimizer(network)
    else:
        raise ValueError(f"Unknown optimizer: {name}")
    

def run(parser):
    data_loader = CaseDataLoader(root=parser.data_dir + "/case_data")
    flow_loader = FlowDataLoader(root=parser.data_dir + "/flow_data/state")
    network_start_date = parser.start_date
    # county_df, metadata = data_loader.load_county("42003")
    nodes = []
    start = None 
    network_start_date = pd.Timestamp(parser.start_date)
    network_end_date = network_start_date + pd.Timedelta(days=180)
    for state in STATE_FIPS_SMALL.keys():
        county_df, metadata = data_loader.load_state(state)
        print(metadata)
        valid_rows = county_df[
            (county_df["date"] >= network_start_date) & (county_df["date"] <= network_end_date)
        ]
        start = valid_rows.index[0]
        example = Node(id=STATE_FIPS_SMALL[state], name=state, model_type=parser.model, 
                       case_data=county_df, total_population=metadata["population"], method=parser.method)

        example.fit_model(start = start, end = start + parser.fit_days) 
        states = example.predict(start = start + parser.fit_days, end = start + parser.fit_days + parser.opt_days)
        example.plot_results(states, start = start + parser.fit_days, end = start + parser.fit_days + parser.opt_days, save_path=parser.output_dir + "/single")
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
        method=parser.method,
    )


    network.fit_model(start=start, end=start + parser.fit_days) 
    pred = network.predict(start=start+parser.fit_days, end=start + parser.fit_days + parser.opt_days)
    baseline_infected = network.sum_infected(pred)
    network.plot_all(pred, start=start+parser.fit_days, end=start + parser.fit_days + parser.opt_days, save_path=parser.output_dir + "/network/baseline")

    original_flow = {
        date: F.copy()
        for date, F in network.flow_matrix_dict.items()
    }

    reduced_flow = {
        date: F * (1 - parser.travel_reduce_ratio)
        for date, F in original_flow.items()
    }
    network.flow_matrix_dict = reduced_flow
    network.model.flow_matrix_dict = reduced_flow
    even_reduce_pred = network.predict(start=start+parser.fit_days, end=start + parser.fit_days + parser.opt_days)
    even_reduce_infected = network.sum_infected(even_reduce_pred)
    network.flow_matrix_dict = original_flow
    network.model.flow_matrix_dict = original_flow

    optimizer = get_optimizer(parser.optimizer, network)
    if parser.optimizer == "cluster":
        result = optimizer.optimize(
            start=start + parser.fit_days,
            end=start + parser.fit_days + parser.opt_days,
            travel_reduce_ratio=None,
            K=parser.clusters,
            time_limit=600,
            mip_gap=0.05,
        )
    else:
        result = optimizer.optimize(
            start=start + parser.fit_days,
            end=start + parser.fit_days + parser.opt_days,
            travel_reduce_ratio=parser.travel_reduce_ratio,
            time_limit=600,
            mip_gap=0.05,
        )

    if result is not None:
        if parser.optimizer == "cluster":
            optimizer.visualize_clusters(
                result,
                start=start + parser.fit_days,
                end=start + parser.fit_days + parser.opt_days,
                save_path=parser.output_dir + "/network/clusters.png",
            )
            optimizer.visualize_cluster_population(
                result,
                save_path=parser.output_dir + "/network/cluster_population.png",
            )
        optimized_flow = optimizer.build_optimized_flow_matrices(result, start=start + parser.fit_days, end=start + parser.fit_days + parser.opt_days)

        optimizer.update_network_flow(optimized_flow)

        optimized_pred = network.predict(
            start=start+parser.fit_days,
            end=start + parser.fit_days + parser.opt_days
        )
        optimized_infected = network.sum_infected(optimized_pred)
        network.plot_all(
            optimized_pred,
            start=start+parser.fit_days,
            end=start + parser.fit_days + parser.opt_days,
            save_path=parser.output_dir + "/network/optimized"
        )
    else:
        print("No feasible optimization result found.") 
    with open(parser.output_dir + "/network/infected_summary.txt", "a") as f:
        f.write(f"Baseline infected: {baseline_infected}\n")
        f.write(f"Evenly reduced infected: {even_reduce_infected}\n")
        if result is not None:
            f.write(f"{optimizer.name} optimized infected: {optimized_infected}\n")
        else:
            f.write(f"{optimizer.name} optimized infected: N/A\n")
        f.write("\n")

if __name__ == "__main__":
    parser = parse_args()
    run(parser)