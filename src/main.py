from node import Node 
from data_loader import CaseDataLoader
import pandas as pd

def fit():
    data_loader = CaseDataLoader(root="data/case_data")
    # county_df, metadata = data_loader.load_county("42003")
    for state in ["Pennsylvania", "New York", "California", "Texas", "Florida"]:
        county_df, metadata = data_loader.load_state(state)
        print(metadata)
        population = metadata["population"]
        # 0.1% of total population
        threshold = 0.002 * population
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
        example = Node(state, model_type="SIR", case_data=county_df, total_population=metadata["population"])

        example.fit_model(start = start, end = end) 
        states = example.predict(start = start, end = start + 180)
        example.plot_results(states, start = start, end = start + 180, save_path="results")
        
def simulate():
    example = Node("toy", model_type="SIR", mode="simulate", case_data=None, total_population=1000000)
    states = example.predict(start = 0, end = 180, initial_state=[999000, 1000, 0], theta=[0.5, 0.1])
    example.plot_results(states, start = 0, end = 180, save_path="results")

if __name__ == "__main__":
    # fit()
    simulate()