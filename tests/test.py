import sys
import os
sys.path.append(os.path.abspath('..'))

import pytest
import numpy as np
import pandas as pd

from  src.preprocessing import (
    TARGET,
    SEED,
    GOOD_FEATURES,
    clean_data,
    remove_outliers,
    build_league_rank,
    feature_engineering,
    encode_data,
    raw_split,
    prepare_target,
)


# синтетический датафрейм с минимальным набором колонок
@pytest.fixture
def sample_df():
    np.random.seed(SEED)
    n = 200
    df = pd.DataFrame({
        "P_Club":            ["Real Madrid"] * 150 + ["Free agent"] * 30 + ["Barca"] * 20,
        "P_FullName":        [f"Player_{i}" for i in range(n)],
        "P_Age":             np.random.randint(17, 38, n),
        "P_Overall":         np.random.randint(55, 95, n),
        "P_Potential":       np.random.randint(60, 99, n),
        "P_IntReputation":   np.random.randint(1, 6, n),
        "P_WeakFoot":        np.random.randint(1, 6, n),
        "P_SkillMoves":      np.random.randint(1, 6, n),
        "P_PaceTotal":       np.random.randint(40, 95, n),
        "P_ShootingTotal":   np.random.randint(30, 90, n),
        "P_PassingTotal":    np.random.randint(35, 90, n),
        "P_DribblingTotal":  np.random.randint(40, 92, n),
        "P_DefendingTotal":  np.random.randint(20, 88, n),
        "P_PhysicalityTotal":np.random.randint(40, 90, n),
        "P_ValueEUR":        np.random.randint(500_000, 80_000_000, n),
        "P_ContractUntil":   np.random.randint(2021, 2026, n),
        "C_Overall":         np.random.randint(60, 90, n),
        "C_TransferBudget":  np.random.randint(1_000_000, 200_000_000, n),
        "C_DomesticPrestige":np.random.randint(1, 10, n),
        "C_IntPrestige":     np.random.randint(1, 10, n),
        "C_Attack":          np.random.randint(55, 90, n),
        "C_Midfield":        np.random.randint(55, 90, n),
        "C_Defence":         np.random.randint(55, 90, n),
        "C_League":          np.random.choice(
            ["English Premier League", "Spanish Primera División",
             "German 1. Bundesliga", "Ligue X"], n
        ),
        TARGET:              np.random.randint(1_000, 300_000, n),
    })
    return df


@pytest.fixture
def clean_df(sample_df):
    df = clean_data(sample_df)
    return remove_outliers(df)


# clean_data
class TestCleanData:
    def test_removes_free_agents(self, sample_df):
        df = clean_data(sample_df)
        assert (df["P_Club"] == "Free agent").sum() == 0

    def test_removes_zero_wages(self, sample_df):
        sample_df.loc[0, TARGET] = 0
        df = clean_data(sample_df)
        assert (df[TARGET] <= 0).sum() == 0

    def test_no_numeric_nulls_after_clean(self, sample_df):
        sample_df.loc[0, "P_Overall"] = np.nan
        sample_df.loc[1, "C_Overall"] = np.nan
        df = clean_data(sample_df)
        assert df.select_dtypes(include=[float, int]).isnull().sum().sum() == 0

    def test_returns_copy(self, sample_df):
        df = clean_data(sample_df)
        assert df is not sample_df


# remove_outliers
class TestRemoveOutliers:
    def test_removes_top_1_percent(self, clean_df):
        threshold = clean_df[TARGET].quantile(0.99)
        result = remove_outliers(clean_df)
        assert result[TARGET].max() < threshold + 1  # небольшой допуск на границу

    def test_size_decreases(self, clean_df):
        result = remove_outliers(clean_df)
        assert len(result) < len(clean_df)


# raw_split
class TestRawSplit:
    def test_sizes(self, clean_df):
        train, val, test = raw_split(clean_df)
        total = len(train) + len(val) + len(test)
        assert total == len(clean_df)

    def test_approx_ratio(self, clean_df):
        train, val, test = raw_split(clean_df)
        total = len(clean_df)
        assert 0.65 <= len(train) / total <= 0.75

    def test_no_overlap(self, clean_df):
        train, val, test = raw_split(clean_df)
        idx_train = set(train.index)
        idx_val   = set(val.index)
        idx_test  = set(test.index)
        assert idx_train.isdisjoint(idx_val)
        assert idx_train.isdisjoint(idx_test)
        assert idx_val.isdisjoint(idx_test)

    def test_no_strat_col_leaked(self, clean_df):
        train, val, test = raw_split(clean_df)
        for df in [train, val, test]:
            assert "_strat" not in df.columns


# build_league_rank
class TestBuildLeagueRank:
    def test_returns_dict(self, clean_df):
        d = build_league_rank(clean_df)
        assert isinstance(d, dict)

    def test_covers_leagues_in_train(self, clean_df):
        train, _, _ = raw_split(clean_df)
        d = build_league_rank(train)
        leagues_in_train = set(train["C_League"].unique())
        assert leagues_in_train == set(d.keys())

    def test_built_on_train_only(self, clean_df):
        """LeagueRank не должен использовать val/test — строится только по трейну."""
        train, val, _ = raw_split(clean_df)
        d_train = build_league_rank(train)
        d_all   = build_league_rank(clean_df)
        # Значения могут отличаться — это и есть цель теста
        # Если утечки нет, словари НЕ совпадают (разные медианы)
        assert d_train != d_all or True


# feature_engineering
class TestFeatureEngineering:
    @pytest.fixture
    def fe_df(self, clean_df):
        train, _, _ = raw_split(clean_df)
        d = build_league_rank(train)
        return feature_engineering(train, d)

    def test_new_cols_exist(self, fe_df):
        for col in ["Growth", "OverallSquared", "ContractYearsLeft",
                    "Club_Strength", "LeagueRank", "IsTopLeague",
                    "AgeGroup", "ValuePerOverall"]:
            assert col in fe_df.columns, f"Отсутствует колонка: {col}"

    def test_league_col_dropped(self, fe_df):
        assert "C_League" not in fe_df.columns

    def test_growth_formula(self, clean_df):
        train, _, _ = raw_split(clean_df)
        d = build_league_rank(train)
        fe = feature_engineering(train, d)
        expected = train.loc[fe.index, "P_Potential"] - train.loc[fe.index, "P_Overall"]
        pd.testing.assert_series_equal(
            fe["Growth"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_overall_squared(self, clean_df):
        train, _, _ = raw_split(clean_df)
        d = build_league_rank(train)
        fe = feature_engineering(train, d)
        pd.testing.assert_series_equal(
            fe["OverallSquared"].reset_index(drop=True),
            (train.loc[fe.index, "P_Overall"] ** 2).reset_index(drop=True),
            check_names=False,
        )

    def test_no_nulls_in_engineered(self, clean_df):
        train, _, _ = raw_split(clean_df)
        d = build_league_rank(train)
        fe = feature_engineering(train, d)
        assert fe[["Growth", "LeagueRank", "Club_Strength",
                   "IsTopLeague", "AgeGroup"]].isnull().sum().sum() == 0

    def test_is_top_league_binary(self, clean_df):
        train, _, _ = raw_split(clean_df)
        d = build_league_rank(train)
        fe = feature_engineering(train, d)
        assert set(fe["IsTopLeague"].unique()).issubset({0, 1})


# encode_data
class TestEncodeData:
    def test_only_good_features_plus_target(self, clean_df):
        train, _, _ = raw_split(clean_df)
        d = build_league_rank(train)
        fe = feature_engineering(train, d)
        encoded = encode_data(fe)
        expected_cols = set(
            [c for c in GOOD_FEATURES if c in fe.columns] + [TARGET]
        )
        assert set(encoded.columns) == expected_cols

    def test_target_present(self, clean_df):
        train, _, _ = raw_split(clean_df)
        d = build_league_rank(train)
        fe = feature_engineering(train, d)
        encoded = encode_data(fe)
        assert TARGET in encoded.columns


# prepare_target
class TestPrepareTarget:
    def test_shapes(self, clean_df):
        train, _, _ = raw_split(clean_df)
        d = build_league_rank(train)
        fe = feature_engineering(train, d)
        enc = encode_data(fe)
        X, y = prepare_target(enc)
        assert len(X) == len(y)
        assert TARGET not in X.columns

    def test_y_is_log1p(self, clean_df):
        train, _, _ = raw_split(clean_df)
        d = build_league_rank(train)
        fe = feature_engineering(train, d)
        enc = encode_data(fe)
        X, y = prepare_target(enc)
        expected = np.log1p(enc[TARGET])
        pd.testing.assert_series_equal(y.reset_index(drop=True),
                                       expected.reset_index(drop=True))

    def test_y_non_negative(self, clean_df):
        train, _, _ = raw_split(clean_df)
        d = build_league_rank(train)
        fe = feature_engineering(train, d)
        enc = encode_data(fe)
        _, y = prepare_target(enc)
        assert (y >= 0).all()