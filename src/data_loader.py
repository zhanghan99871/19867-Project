from pathlib import Path
import pandas as pd
import re

class CaseDataLoader:
    def __init__(self, root):
        self.root = Path(root)

        self.confirmed = None
        self.deaths = None

    @staticmethod
    def _normalize_fips(series):
        return (
            pd.to_numeric(series, errors="coerce")
            .astype("Int64")
            .astype("string")
            .str.zfill(5)
        )

    @staticmethod
    def _is_date_column(name):
        return (
            re.fullmatch(
                r"\d{1,2}/\d{1,2}/\d{2}",
                str(name)
            )
            is not None
        )

    def load_raw_data(self):
        confirmed_path = (
            self.root
            / "time_series_covid19_confirmed_US.csv"
        )

        deaths_path = (
            self.root
            / "time_series_covid19_deaths_US.csv"
        )

        self.confirmed = pd.read_csv(
            confirmed_path,
            dtype={"FIPS": str}
        )

        self.deaths = pd.read_csv(
            deaths_path,
            dtype={"FIPS": str}
        )

        self.confirmed["FIPS"] = self._normalize_fips(
            self.confirmed["FIPS"]
        )

        self.deaths["FIPS"] = self._normalize_fips(
            self.deaths["FIPS"]
        )

    def load_county(self, fips):
        if self.confirmed is None or self.deaths is None:
            self.load_raw_data()

        fips = str(fips).zfill(5)

        confirmed_match = self.confirmed[
            self.confirmed["FIPS"] == fips
        ]

        deaths_match = self.deaths[
            self.deaths["FIPS"] == fips
        ]

        if confirmed_match.empty:
            raise ValueError(
                f"FIPS {fips} not found in confirmed data."
            )

        if deaths_match.empty:
            raise ValueError(
                f"FIPS {fips} not found in deaths data."
            )

        confirmed_county = confirmed_match.iloc[0]
        deaths_county = deaths_match.iloc[0]

        date_cols = [
            col
            for col in self.confirmed.columns
            if self._is_date_column(col)
        ]

        county_df = pd.DataFrame({
            "date": pd.to_datetime(
                date_cols,
                format="%m/%d/%y"
            ),
            "confirmed": [
                confirmed_county[d]
                for d in date_cols
            ],
            "deaths": [
                deaths_county[d]
                for d in date_cols
            ]
        })

        county_df["confirmed"] = (
            county_df["confirmed"].astype(float)
        )

        county_df["deaths"] = (
            county_df["deaths"].astype(float)
        )

        county_df = (
            county_df
            .sort_values("date")
            .reset_index(drop=True)
        )

        population = int(
            deaths_county["Population"]
        )

        metadata = {
            "fips": fips,
            "county": confirmed_county["Admin2"],
            "state": confirmed_county["Province_State"],
            "population": population
        }

        return county_df, metadata

    def load_state(self, state):
        if self.confirmed is None or self.deaths is None:
            self.load_raw_data()

        state = str(state).strip().lower()

        confirmed_match = self.confirmed[
            self.confirmed["Province_State"].str.lower() == state
        ]

        deaths_match = self.deaths[
            self.deaths["Province_State"].str.lower() == state
        ]

        if confirmed_match.empty:
            raise ValueError(
                f"State '{state}' not found in confirmed data."
            )

        if deaths_match.empty:
            raise ValueError(
                f"State '{state}' not found in deaths data."
            )

        date_cols = [
            col
            for col in self.confirmed.columns
            if self._is_date_column(col)
        ]

        state_df = pd.DataFrame({
            "date": pd.to_datetime(
                date_cols,
                format="%m/%d/%y"
            ),
            "confirmed": [
                confirmed_match[d].sum()
                for d in date_cols
            ],
            "deaths": [
                deaths_match[d].sum()
                for d in date_cols
            ]
        })

        state_df["confirmed"] = (
            state_df["confirmed"].astype(float)
        )

        state_df["deaths"] = (
            state_df["deaths"].astype(float)
        )

        state_df = (
            state_df
            .sort_values("date")
            .reset_index(drop=True)
        )

        population = int(
            deaths_match["Population"].sum()
        )

        metadata = {
            "state": state,
            "population": population
        }

        return state_df, metadata


class FlowDataLoader:
    def __init__(self, root="data/flow_data/state"):
        self.root = Path(root)

    def _get_path(self, date):
        """
        Convert a date into the corresponding CSV filename.
        """
        date = pd.to_datetime(date)

        filename = (
            f"daily_state2state_"
            f"{date.year:04d}_{date.month:02d}_{date.day:02d}.csv"
        )

        return self.root / filename

    def load_day(self, date):
        """
        Load flow data for one day.

        Returns only:
            geoid_o, geoid_d, visitor_flows, pop_flows
        """
        path = self._get_path(date)

        if not path.exists():
            raise FileNotFoundError(f"Flow data not found: {path}")

        df = pd.read_csv(
            path,
            usecols=["geoid_o", "geoid_d", "pop_flows", "visitor_flows"],
            dtype={
                "geoid_o": str,
                "geoid_d": str,
            },
        )

        df["geoid_o"] = df["geoid_o"].str.zfill(2)
        df["geoid_d"] = df["geoid_d"].str.zfill(2)

        return df

    def load_range(self, start_date, end_date):
        """
        Load and concatenate all daily flow files
        between start_date and end_date (inclusive).
        """
        dates = pd.date_range(start_date, end_date, freq="D")

        frames = []

        for date in dates:
            path = self._get_path(date)

            if not path.exists():
                print(f"Warning: missing {path.name}")
                continue

            day_df = self.load_day(date)
            day_df["date"] = pd.Timestamp(date)
            frames.append(day_df)

        if not frames:
            raise ValueError(
                f"No flow data found between {start_date} and {end_date}"
            )

        return pd.concat(frames, ignore_index=True)

    def flow_matrix(
        self,
        date,
        flow_type="pop_flows",
        include_self=False,
    ):
        """
        Return a matrix F where

            F[i, j] = flow from state i -> state j

        Parameters
        ----------
        date : str or datetime
            Date to load.

        flow_type : str
            "pop_flows" or "visitor_flows".

        include_self : bool
            Whether to include within-state flows.
        """
        if flow_type not in {"pop_flows", "visitor_flows"}:
            raise ValueError(
                "flow_type must be 'pop_flows' or 'visitor_flows'"
            )

        df = self.load_day(date)

        if not include_self:
            df = df[df["geoid_o"] != df["geoid_d"]]

        matrix = df.pivot_table(
            index="geoid_o",
            columns="geoid_d",
            values=flow_type,
            aggfunc="sum",
            fill_value=0,
        )

        # Make origin/destination state sets identical
        states = sorted(
            set(matrix.index).union(matrix.columns)
        )

        matrix = matrix.reindex(
            index=states,
            columns=states,
            fill_value=0,
        )

        return matrix

    def flow_matrix_range(
        self,
        start_date,
        end_date,
        flow_type="pop_flows",
        include_self=False,
    ):
        """
        Return a dictionary of flow matrices for each day
        between start_date and end_date (inclusive).
        """
        dates = pd.date_range(start_date, end_date, freq="D")

        matrices = {}

        for date in dates:
            try:
                matrices[date] = self.flow_matrix(
                    date,
                    flow_type=flow_type,
                    include_self=include_self,
                )
            except FileNotFoundError:
                print(f"Warning: missing flow data for {date}")

        return matrices
    
    def flow_summary(self, flow_matrix):
        outflow = flow_matrix.sum(axis=1)
        inflow = flow_matrix.sum(axis=0)

        summary = pd.DataFrame({
            "inflow": inflow,
            "outflow": outflow,
            "net_flow": inflow - outflow,
        })

        return summary