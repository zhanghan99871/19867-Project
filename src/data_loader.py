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