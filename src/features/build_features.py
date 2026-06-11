import pandas as pd
import polars as pl

import os
import joblib
import numpy as np
from datetime import datetime
import pathlib
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler


from src.utils import read_csv_file


class PreprocessDataset:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def save_scaler(self, scaler, name):
        path = os.path.join("src/models/scalers", f"{name}.pkl")
        joblib.dump(scaler, path)

    def split_datetime_string(self, dt_str, name, time=True):
        """
        Divide una stringa datetime nel formato 'dd/mm/yyyy HH:MM:SS'
        in componenti singoli e restituisce encoding ciclico.

        Args:
            dt_str (str): stringa data e ora nel formato specificato
            name (str): prefisso per le chiavi restitute
            time (bool): se True considera anche tempo, altrimenti solo data

        Returns:
            dict: con componenti e encoding ciclico sin e cos
        """
        if not time:
            date_part = dt_str.split(" ")[0]
            day, month, year = map(int, date_part.split("/"))

            dt_obj = datetime(year, month, day)

            return {
                f"{name}_day_sin": np.sin(2 * np.pi * (day - 1) / 31),
                f"{name}_day_cos": np.cos(2 * np.pi * (day - 1) / 31),
                f"{name}_month_sin": np.sin(2 * np.pi * (month - 1) / 12),
                f"{name}_month_cos": np.cos(2 * np.pi * (month - 1) / 12),
                f"{name}_dayofweek_sin": np.sin(2 * np.pi * dt_obj.weekday() / 7),
                f"{name}_dayofweek_cos": np.cos(2 * np.pi * dt_obj.weekday() / 7),
                f"{name}_year": year,
            }

        else:
            date_part, time_part = dt_str.split(" ")
            day, month, year = map(int, date_part.split("/"))
            hour, minute, second = map(int, time_part.split(":"))

            dt_obj = datetime(year, month, day)

            minutes_day = hour * 60 + minute + second / 60

            return {
                f"{name}_day_sin": np.sin(2 * np.pi * (day - 1) / 31),
                f"{name}_day_cos": np.cos(2 * np.pi * (day - 1) / 31),
                f"{name}_month_sin": np.sin(2 * np.pi * (month - 1) / 12),
                f"{name}_month_cos": np.cos(2 * np.pi * (month - 1) / 12),
                f"{name}_dayofweek_sin": np.sin(2 * np.pi * dt_obj.weekday() / 7),
                f"{name}_dayofweek_cos": np.cos(2 * np.pi * dt_obj.weekday() / 7),
                f"{name}_year": year,
                f"{name}_hour_sin": np.sin(2 * np.pi * hour / 24),
                f"{name}_hour_cos": np.cos(2 * np.pi * hour / 24),
                f"{name}_minute_sin": np.sin(2 * np.pi * minute / 60),
                f"{name}_minute_cos": np.cos(2 * np.pi * minute / 60),
                f"{name}_second_sin": np.sin(2 * np.pi * second / 60),
                f"{name}_second_cos": np.cos(2 * np.pi * second / 60),
                f"{name}_minutes_day_sin": np.sin(2 * np.pi * minutes_day / (24 * 60)),
                f"{name}_minutes_day_cos": np.cos(2 * np.pi * minutes_day / (24 * 60)),
            }

    def preprocess_consumi(self):
        consumi_path = os.path.join(self.dataset_path, "CONSUMI.csv")
        output_path = os.path.join("data/processed", "CONSUMI.csv")

        schema = {
            "Supply_ID": pl.Utf8,
            "meas_ym": pl.Utf8,
            "meas_dd": pl.Utf8,
            "id": pl.Utf8,
            "val": pl.Utf8,
            "magnitude": pl.Utf8,
        }

        print("Loadding consumi dataset...")  # Debugging message

        # Read with polars
        df = pl.read_csv(
            consumi_path, schema=schema, separator="\t", encoding="utf-16-le"
        )

        # Drop rows with empty Supply_ID then convert to int
        df = df.filter(pl.col("Supply_ID") != "")
        df = df.with_columns([pl.col("Supply_ID").str.slice(6).cast(pl.Int32)])

        # Replace "" with None in 'val' and convert comma to dot
        df = df.with_columns(
            [
                pl.when(pl.col("val") == "")
                .then(None)
                .otherwise(pl.col("val"))
                .alias("val"),
            ]
        )

        df = df.with_columns([pl.col("val").str.replace(",", ".").cast(pl.Float64)])

        # Parsing meas_ym
        df = df.with_columns(
            [
                pl.col("meas_ym").str.slice(0, 4).cast(pl.Int32).alias("year"),
                pl.col("meas_ym").str.slice(4, 2).cast(pl.Int32).alias("month"),
                pl.col("meas_dd").cast(pl.Int32).alias("day"),
                pl.col("id").cast(pl.Int32),
            ]
        )

        print("Parsing completed")  # Debugging message
        print(
            "Creating cyclic encoding for month, day, and time..."
        )  # Debugging message

        # Add cyclic encoding
        df = df.with_columns(
            [
                (2 * np.pi * (pl.col("month") - 1) / 12).sin().alias("month_sin"),
                (2 * np.pi * (pl.col("month") - 1) / 12).cos().alias("month_cos"),
                (2 * np.pi * (pl.col("day") - 1) / 31).sin().alias("day_sin"),
                (2 * np.pi * (pl.col("day") - 1) / 31).cos().alias("day_cos"),
                (2 * np.pi * (pl.col("id") - 1) / 96).sin().alias("time_sin"),
                (2 * np.pi * (pl.col("id") - 1) / 96).cos().alias("time_cos"),
            ]
        )

        # magnitude as integer (category encoding)
        df = df.with_columns([pl.col("magnitude").cast(pl.Categorical).cast(pl.Int32)])

        # Handle NaN in val column
        val_mean = df.select(pl.col("val")).mean().item()
        df = df.with_columns([pl.col("val").fill_null(val_mean)])

        # Extract year and val to scale
        year_scaler = StandardScaler()
        val_scaler = StandardScaler()

        df_pd = df.select(["year", "val"]).to_pandas()
        df_scaled = df.with_columns(
            [
                pl.Series("year", year_scaler.fit_transform(df_pd[["year"]]).flatten()),
                pl.Series("val", val_scaler.fit_transform(df_pd[["val"]]).flatten()),
            ]
        )

        self.save_scaler(val_scaler, "consumi_val_scaler")
        self.save_scaler(year_scaler, "consumi_year_scaler")

        # Final selection
        df_scaled = df_scaled.select(
            [
                "Supply_ID",
                "year",
                "month_sin",
                "month_cos",
                "day_sin",
                "day_cos",
                "time_sin",
                "time_cos",
                "val",
                "magnitude",
            ]
        )

        # Write CSV with utf-16 encoding and tab separator
        df_pandas = df_scaled.to_pandas()

        # Save to CSV with UTF-16 encoding and tab separator
        df_pandas.to_csv(output_path, sep="\t", encoding="utf-16", index=False)

    def preprocess_anagrafica(self):
        anagrafica_path = os.path.join(self.dataset_path, "ANAGRAFICA.csv")
        output_path = os.path.join("data/processed", "ANAGRAFICA.csv")

        header, lines = read_csv_file(anagrafica_path)
        pandas_df = pd.DataFrame(lines, columns=header)

        pandas_df["Supply_ID"] = pandas_df["Supply_ID"].str[6:].astype(int)

        date_col_begin = (
            pandas_df["begin_date_ref"]
            .apply(lambda x: self.split_datetime_string(x, "begin_date_ref"))
            .apply(pd.Series)
        )
        date_col_end = (
            pandas_df["end_date_ref"]
            .apply(lambda x: self.split_datetime_string(x, "end_date_ref"))
            .apply(pd.Series)
        )

        pandas_df.drop(columns=["begin_date_ref", "end_date_ref"], inplace=True)
        pandas_df = pd.concat([pandas_df, date_col_begin, date_col_end], axis=1)

        pandas_df["available_power"] = pandas_df["available_power"].replace("", np.nan)
        pandas_df["available_power"] = pandas_df.groupby("Supply_ID")[
            "available_power"
        ].transform(lambda x: x.ffill().bfill())

        pandas_df["supply_status"] = (
            pandas_df["supply_status"].astype("category").cat.codes
        )
        pandas_df["available_power"] = (
            pandas_df["available_power"].str.replace(",", ".").astype(float)
        )

        scaler_av = StandardScaler()
        scaler_by = StandardScaler()
        scaler_ey = StandardScaler()

        pandas_df["available_power"] = scaler_av.fit_transform(
            pandas_df[["available_power"]]
        )
        pandas_df["begin_date_ref_year"] = scaler_by.fit_transform(
            pandas_df[["begin_date_ref_year"]]
        )
        pandas_df["end_date_ref_year"] = scaler_ey.fit_transform(
            pandas_df[["end_date_ref_year"]]
        )

        self.save_scaler(scaler_av, "anagrafica_available_power_scaler")
        self.save_scaler(scaler_by, "anagrafica_begin_year_scaler")
        self.save_scaler(scaler_ey, "anagrafica_end_year_scaler")

        pandas_df.to_csv(output_path, index=False, encoding="utf-16", sep="\t")

    def preprocess_lavori(self):
        lavori_path = os.path.join(self.dataset_path, "LAVORI.csv")
        output_path = os.path.join("data/processed", "LAVORI.csv")

        header, lines = read_csv_file(lavori_path)
        pandas_df = pd.DataFrame(lines, columns=header)

        pandas_df["woe_dt_execution"] = pandas_df["woe_dt_execution"].replace(
            "", np.nan
        )
        pandas_df["woa_activity_subtype"] = pandas_df["woa_activity_subtype"].replace(
            "", np.nan
        )
        pandas_df["woa_activity_type"] = pandas_df["woa_activity_type"].replace(
            "", np.nan
        )

        pandas_df["Supply_ID"] = pandas_df["Supply_ID"].str[6:].astype(int)
        pandas_df.dropna(subset=["woe_dt_execution"], inplace=True)

        date_col = (
            pandas_df["woe_dt_execution"]
            .apply(lambda x: self.split_datetime_string(str(x), "woe_dt_execution"))
            .apply(pd.Series)
        )

        pandas_df.drop(columns=["woe_dt_execution"], inplace=True)
        pandas_df = pd.concat([pandas_df, date_col], axis=1)
        pandas_df.dropna(subset=["woa_activity_type"], inplace=True)

        pandas_df["woa_activity_subtype"] = pandas_df["woa_activity_subtype"].fillna(
            pandas_df["woa_activity_subtype"].mode()[0]
        )

        pandas_df["woa_activity_type"] = (
            pandas_df["woa_activity_type"].astype("category").cat.codes
        )
        pandas_df["woa_activity_subtype"] = (
            pandas_df["woa_activity_subtype"].astype("category").cat.codes
        )

        scaler = StandardScaler()
        pandas_df["woe_dt_execution_year"] = scaler.fit_transform(
            pandas_df[["woe_dt_execution_year"]]
        )
        self.save_scaler(scaler, "lavori_year_scaler")

        pandas_df.to_csv(output_path, index=False, encoding="utf-16", sep="\t")

    def preprocess_interruzioni(self):
        interruzioni_path = os.path.join(self.dataset_path, "INTERRUZIONI.csv")
        output_path = os.path.join("data/processed", "INTERRUZIONI.csv")

        header, lines = read_csv_file(interruzioni_path)
        pandas_df = pd.DataFrame(lines, columns=header)

        pandas_df["Supply_ID"] = pandas_df["Supply_ID"].str[6:].astype(int)

        date_col_start = (
            pandas_df["start_date"]
            .apply(lambda x: self.split_datetime_string(x, "start_date", time=False))
            .apply(pd.Series)
        )
        date_col_end = (
            pandas_df["end_date"]
            .apply(lambda x: self.split_datetime_string(x, "end_date", time=False))
            .apply(pd.Series)
        )

        pandas_df.drop(columns=["start_date", "end_date"], inplace=True)
        pandas_df = pd.concat([pandas_df, date_col_start, date_col_end], axis=1)

        pandas_df["tipologia_interruzione"] = (
            pandas_df["tipologia_interruzione"].astype("category").cat.codes
        )

        scaler_sy = StandardScaler()
        scaler_ey = StandardScaler()
        scaler_dur = StandardScaler()

        pandas_df["start_date_year"] = scaler_sy.fit_transform(
            pandas_df[["start_date_year"]]
        )
        pandas_df["end_date_year"] = scaler_ey.fit_transform(
            pandas_df[["end_date_year"]]
        )
        pandas_df["durata_netta"] = scaler_dur.fit_transform(
            pandas_df[["durata_netta"]]
        )

        self.save_scaler(scaler_sy, "interruzioni_start_year_scaler")
        self.save_scaler(scaler_ey, "interruzioni_end_year_scaler")
        self.save_scaler(scaler_dur, "interruzioni_durata_scaler")

        pandas_df.to_csv(output_path, index=False, encoding="utf-16", sep="\t")

    def preprocess_parole_di_stato(self):
        parole_di_stato_path = os.path.join(self.dataset_path, "PAROLE_DI_STATO.csv")
        output_path = os.path.join("data/processed", "PAROLE_DI_STATO.csv")

        header, lines = read_csv_file(parole_di_stato_path)
        pandas_df = pd.DataFrame(lines, columns=header)

        pandas_df["Supply_ID"] = pandas_df["Supply_ID"].str[6:].astype(int)

        date_col = (
            pandas_df["meas_ts"]
            .apply(lambda x: self.split_datetime_string(x, "meas_ts"))
            .apply(pd.Series)
        )

        pandas_df.drop(columns=["meas_ts"], inplace=True)
        pandas_df = pd.concat([pandas_df, date_col], axis=1)

        pandas_df["ListaParole"] = pandas_df["ListaParole"].str.split(",")
        mlb = MultiLabelBinarizer()
        lista_parole = pd.DataFrame(
            mlb.fit_transform(pandas_df["ListaParole"]), columns=mlb.classes_
        )

        pandas_df.drop(columns=["ListaParole"], inplace=True)
        pandas_df = pd.concat([pandas_df, lista_parole], axis=1)

        scaler = StandardScaler()
        pandas_df["meas_ts_year"] = scaler.fit_transform(pandas_df[["meas_ts_year"]])
        self.save_scaler(scaler, "parole_meas_year_scaler")

        pandas_df.to_csv(output_path, index=False, encoding="utf-16", sep="\t")

    def preprocess_labels(self):
        labels_path = os.path.join(self.dataset_path, "LABELS.csv")
        output_path = os.path.join("data/processed", "LABELS.csv")

        header, lines = read_csv_file(labels_path)
        pandas_df = pd.DataFrame(lines, columns=header)

        # Transform supply_id to integer
        pandas_df["Supply_ID"] = pandas_df["Supply_ID"].str[6:].astype(int)

        # modify categories to integers
        pandas_df["CLUSTER"] = pandas_df["CLUSTER"].astype("category").cat.codes

        pandas_df.to_csv(output_path, index=False, encoding="utf-16", sep="\t")

    def preprocess_all(self):
        self.preprocess_consumi()
        self.preprocess_anagrafica()
        self.preprocess_lavori()
        self.preprocess_interruzioni()
        self.preprocess_parole_di_stato()
        self.preprocess_labels()


if __name__ == "__main__":
    DATASET_PATH = pathlib.Path("data/raw")

    preporcessor = PreprocessDataset(DATASET_PATH)

    preporcessor.preprocess_all()
