import os
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "P_WageEUR"
SEED = 28  # 42

TOP_LEAGUES = {
    "English Premier League (1)",
    "Spanish Primera División (1)",
    "German 1. Bundesliga (1)",
    "Italian Serie A (1)",
    "French Ligue 1 (1)",
}

GOOD_FEATURES = [
    # игрок
    "P_Age",
    "P_Overall",
    "P_Potential",
    "P_IntReputation",
    "P_WeakFoot",
    "P_SkillMoves",
    # статы
    "P_PaceTotal",
    "P_ShootingTotal",
    "P_PassingTotal",
    "P_DribblingTotal",
    "P_DefendingTotal",
    "P_PhysicalityTotal",
    # клуб
    "C_Overall",
    "C_TransferBudget",
    "C_DomesticPrestige",
    "C_IntPrestige",
    # FE
    "Growth",
    "OverallSquared",
    "ContractYearsLeft",
    "Club_Strength",
    "LeagueRank",
    "IsTopLeague",
]


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)


def load_data(path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    print(f"Загружено: {df.shape[0]:,} строк × {df.shape[1]} столбцов")
    return df


def merge_data(df_players: pd.DataFrame,
               df_teams: pd.DataFrame) -> pd.DataFrame:
    """Добавляет префиксы и мержит игроков с командами по названию клуба."""
    df_players = df_players.add_prefix("P_")
    df_teams = df_teams.add_prefix("C_")
    df = df_players.merge(
        df_teams, how="left",
        left_on="P_Club", right_on="C_Name"
    )
    print(f"После merge: {df.shape[0]:,} строк × {df.shape[1]} столбцов")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df["P_Club"] != "Free agent"]
    df = df[df[TARGET] > 0]
    df = df.dropna(subset=["C_Overall"])

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].fillna("Unknown")

    print(f"После очистки: {df.shape[0]:,} строк × {df.shape[1]} столбцов")
    return df


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[df[TARGET] < df[TARGET].quantile(0.99)]
    print(f"После удаления выбросов: {df.shape[0]:,} строк")
    return df


def raw_split(df: pd.DataFrame):
    """
    Сплит ДО feature engineering — чтобы LeagueRank считался только на трейне.
    Возвращает сырые (без FE) train/val/test.
    """
    df = df.copy()
    df["_strat"] = pd.qcut(
        np.log1p(df[TARGET]), q=5, labels=False, duplicates="drop"
    )
    train, temp = train_test_split(
        df, test_size=0.3, random_state=SEED, stratify=df["_strat"]
    )
    val, test = train_test_split(
        temp, test_size=0.5, random_state=SEED, stratify=temp["_strat"]
    )
    for d in [train, val, test]:
        d.drop(columns=["_strat"], inplace=True)

    print(f"Сплит: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def build_league_rank(train_df: pd.DataFrame) -> dict:
    """
    Считаем LeagueRank ТОЛЬКО на трейне — вызывается после raw_split,
    чтобы медиана по лиге не видела val/test зарплаты.
    """
    ranks = train_df.groupby("C_League")[TARGET].median().rank()
    return ranks.to_dict()


def feature_engineering(df: pd.DataFrame,
                        league_rank_dict: dict) -> pd.DataFrame:
    """Применяется к каждому сплиту отдельно
      с одним и тем же league_rank_dict."""
    df = df.copy()

    # нелинейная связь рейтинга с зарплатой:
    # звёзды получают непропорционально больше
    df["OverallSquared"] = df["P_Overall"] ** 2

    # нереализованный потенциал — важен при переговорах о контракте
    df["Growth"] = df["P_Potential"] - df["P_Overall"]

    # сколько лет осталось до конца контракта (FIFA 21 — год 2021)
    if "P_ContractUntil" in df.columns:
        df["ContractYearsLeft"] = df["P_ContractUntil"] - 2021
    else:
        df["ContractYearsLeft"] = 0

    # средняя сила клуба по трём линиям
    df["Club_Strength"] = (
        df["C_Attack"] + df["C_Midfield"] + df["C_Defence"]
    ) / 3

    # ранг лиги по медианной зарплате — словарь построен только на трейне
    df["LeagueRank"] = df["C_League"].map(league_rank_dict).fillna(0)

    # флаг топ-5 лиги
    df["IsTopLeague"] = df["C_League"].isin(TOP_LEAGUES).astype(int)

    df.drop(columns=["C_League"], inplace=True)
    return df


def encode_data(df: pd.DataFrame) -> pd.DataFrame:
    """Оставляем только нужные фичи + таргет."""
    df = df.copy()
    cols = [c for c in GOOD_FEATURES if c in df.columns] + [TARGET]
    return df[cols]


def save_splits(train, val, test, out_dir: str = "../data/processed"):
    os.makedirs(out_dir, exist_ok=True)
    train.to_csv(f"{out_dir}/train.csv", index=False)
    val.to_csv(f"{out_dir}/val.csv", index=False)
    test.to_csv(f"{out_dir}/test.csv", index=False)
    print(f"Сохранено в {out_dir}/")


def prepare_target(df: pd.DataFrame):
    y = np.log1p(df[TARGET])
    X = df.drop(columns=[TARGET])
    return X, y


def run_cp1(players_path: str, teams_path: str,
            out_dir: str = "../data/processed"):
    """
    Полный пайплайн без утечки:
    загрузка → merge → очистка → сплит → FE только на трейне → сохранение
    """
    df_players = load_data(players_path)
    df_teams = load_data(teams_path)
    df = merge_data(df_players, df_teams)
    df = clean_data(df)
    df = remove_outliers(df)
    train_raw, val_raw, test_raw = raw_split(df)
    league_rank_dict = build_league_rank(train_raw)
    train = encode_data(feature_engineering(train_raw, league_rank_dict))
    val = encode_data(feature_engineering(val_raw,   league_rank_dict))
    test = encode_data(feature_engineering(test_raw,  league_rank_dict))

    save_splits(train, val, test, out_dir)
    return train, val, test
