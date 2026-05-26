import sys
import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from src.preprocessing import TOP_LEAGUES, GOOD_FEATURES

# Load model and league_rank_dict on startup
sys.path.append(os.path.abspath('..'))
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.joblib")
LEAGUE_RANK_PATH = os.getenv("LEAGUE_RANK_PATH",
                             "models/league_rank_dict.joblib")

try:
    model = joblib.load(MODEL_PATH)
    league_rank_dict: dict = joblib.load(LEAGUE_RANK_PATH)
except FileNotFoundError as e:
    raise RuntimeError(
        f"Model not found: {e}. "
        "Please run 03_experiments.ipynb and ensure the model is saved."
    )

app = FastAPI(
    title="FIFA 21 Wage Predictor",
    description=(
        "Predicts a football player's wage based on "
        "player and club statistics"
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PlayerInput(BaseModel):
    P_Age: int = Field(..., ge=15, le=45, example=24)
    P_Overall: int = Field(..., ge=40, le=99, example=82)
    P_Potential: int = Field(..., ge=40, le=99, example=88)
    P_IntReputation: int = Field(..., ge=1, le=5, example=2)
    P_WeakFoot: int = Field(..., ge=1, le=5, example=3)
    P_SkillMoves: int = Field(..., ge=1, le=5, example=4)
    P_PaceTotal: int = Field(..., ge=1, le=99, example=78)
    P_ShootingTotal: int = Field(..., ge=1, le=99, example=72)
    P_PassingTotal: int = Field(..., ge=1, le=99, example=80)
    P_DribblingTotal: int = Field(..., ge=1, le=99, example=81)
    P_DefendingTotal: int = Field(..., ge=1, le=99, example=35)
    P_PhysicalityTotal: int = Field(..., ge=1, le=99, example=70)
    P_ValueEUR: int = Field(..., ge=0, example=15_000_000)
    P_ContractUntil: int = Field(..., ge=2021, le=2029, example=2024)
    # Club
    C_Overall: int = Field(..., ge=40, le=99, example=78)
    C_TransferBudget: int = Field(..., ge=0, example=50_000_000)
    C_DomesticPrestige: int = Field(..., ge=1, le=10, example=6)
    C_IntPrestige: int = Field(..., ge=1, le=10, example=5)
    C_Attack: int = Field(..., ge=40, le=99, example=75)
    C_Midfield: int = Field(..., ge=40, le=99, example=74)
    C_Defence: int = Field(..., ge=40, le=99, example=72)
    C_League: str = Field(..., example="English Premier League")


class PredictionResponse(BaseModel):
    wage_eur_per_week: int
    wage_eur_per_year: int
    log_wage: float
    features_used: dict


# Feature Engineering (replicates preprocessing.py logic)
def apply_feature_engineering(data: PlayerInput) -> pd.DataFrame:
    row = data.model_dump()

    row["Growth"] = row["P_Potential"] - row["P_Overall"]
    row["OverallSquared"] = row["P_Overall"] ** 2
    row["ContractYearsLeft"] = row["P_ContractUntil"] - 2021
    row["Club_Strength"] = (row["C_Attack"] + row["C_Midfield"]
                            + row["C_Defence"]) / 3
    row["LeagueRank"] = league_rank_dict.get(row["C_League"], 0)
    row["IsTopLeague"] = int(row["C_League"] in TOP_LEAGUES)
    row["ValuePerOverall"] = row["P_ValueEUR"] / (row["P_Overall"] + 1)

    age = row["P_Age"]
    if age <= 21:
        row["AgeGroup"] = 0
    elif age <= 27:
        row["AgeGroup"] = 1
    elif age <= 32:
        row["AgeGroup"] = 2
    else:
        row["AgeGroup"] = 3

    df = pd.DataFrame([row])
    # Keep only model features in the correct order
    missing = [c for c in GOOD_FEATURES if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Missing features: {missing}")

    return df[GOOD_FEATURES]


@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_PATH}


@app.get("/leagues")
def get_leagues():
    """Returns a list of leagues from the training dataset."""
    return {"leagues": sorted(league_rank_dict.keys())}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: PlayerInput):
    df = apply_feature_engineering(data)
    log_wage = float(model.predict(df)[0])
    wage_eur = int(np.expm1(log_wage))

    return PredictionResponse(
        wage_eur_per_week=wage_eur,
        wage_eur_per_year=wage_eur * 52,
        log_wage=round(log_wage, 4),
        features_used=df.iloc[0].to_dict(),
    )
