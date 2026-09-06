from pathlib import Path
import pandas as pd
import re
import matplotlib.pyplot as plt
from node import Node
import numpy as np
ROOT = Path("data/csse_covid_19_time_series")

confirmed = pd.read_csv(
    ROOT / "time_series_covid19_confirmed_US.csv", dtype={"FIPS": str}
)
confirmed["FIPS"] = (
    pd.to_numeric(confirmed["FIPS"], errors="coerce")
    .astype("Int64")
    .astype("string")
    .str.zfill(5)
)

deaths = pd.read_csv(
    ROOT / "time_series_covid19_deaths_US.csv", dtype={"FIPS": str}
)
deaths["FIPS"] = (
    pd.to_numeric(deaths["FIPS"], errors="coerce")
    .astype("Int64")
    .astype("string")
    .str.zfill(5)
)

FIPS = "42003"

confirmed_county = confirmed[
    confirmed["FIPS"] == FIPS
].iloc[0]

deaths_county = deaths[
    deaths["FIPS"] == FIPS
].iloc[0]

def is_date_column(name):
    return re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2}", str(name)) is not None

date_cols = [
    col for col in confirmed.columns
    if is_date_column(col)
]

county_df = pd.DataFrame({
    "date": pd.to_datetime(date_cols, format="%m/%d/%y"),
    "confirmed": [
        confirmed_county[d] for d in date_cols
    ],
    "deaths": [
        deaths_county[d] for d in date_cols
    ]
})

county_df["confirmed"] = county_df["confirmed"].astype(float)
county_df["deaths"] = county_df["deaths"].astype(float)

county_df = county_df.sort_values("date").reset_index(drop=True)

population = int(deaths_county["Population"])

county_df["date"] = pd.to_datetime(county_df["date"])

# plt.figure(figsize=(12, 6))

# plt.plot(county_df["date"], county_df["confirmed"], label="Confirmed")
# plt.plot(county_df["date"], county_df["deaths"], label="Deaths")

# plt.xlabel("Date")
# plt.ylabel("Cases")
# plt.title("COVID-19 Cases Over Time")

# plt.legend()
# plt.grid(True)
# plt.tight_layout()

# plt.show()
start_date = pd.to_datetime("2020-03-14") 
end_date = pd.to_datetime("2020-04-01")
example = Node("pittsburgh", case_data=county_df[(county_df["date"] >= start_date) & (county_df["date"] <= end_date)], total_population=population)

example.fit_model() 
example.plot_results()