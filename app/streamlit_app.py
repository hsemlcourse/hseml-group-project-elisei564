import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="FIFA 21 Wage Predictor",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ FIFA 21 — Player Wage Predictor")
st.caption(
    "Enter player and club characteristics "
    "to predict the weekly wage in EUR."
)


@st.cache_data
def get_leagues():
    try:
        resp = requests.get(f"{API_URL}/leagues", timeout=5)
        return resp.json()["leagues"]
    except Exception:
        return [
            "English Premier League", "Spanish Primera División",
            "German 1. Bundesliga", "Italian Serie A", "French Ligue 1",
        ]


leagues = get_leagues()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Player Info")
    age = st.slider("Age", 15, 45, 24)
    overall = st.slider("Overall", 40, 99, 82)
    potential = st.slider("Potential", 40, 99, 88)
    int_rep = st.slider("Int. Reputation", 1, 5, 2)
    weak_foot = st.slider("Weak Foot", 1, 5, 3)
    skill_moves = st.slider("Skill Moves", 1, 5, 4)
    value_eur = st.number_input("Value EUR", 0, 200_000_000,
                                15_000_000, step=500_000)
    contract = st.slider("Contract Until", 2021, 2029, 2024)

with col2:
    st.subheader("🏃 Stats")
    pace = st.slider("Pace", 1, 99, 78)
    shooting = st.slider("Shooting", 1, 99, 72)
    passing = st.slider("Passing", 1, 99, 80)
    dribbling = st.slider("Dribbling", 1, 99, 81)
    defending = st.slider("Defending", 1, 99, 35)
    physicality = st.slider("Physicality", 1, 99, 70)

with col3:
    st.subheader("🏟️ Club Info")
    league = st.selectbox("League", leagues)
    c_overall = st.slider("Club Overall", 40, 99, 78)
    c_budget = st.number_input("Transfer Budget EUR", 0, 500_000_000,
                               50_000_000, step=1_000_000)
    c_domestic = st.slider("Domestic Prestige", 1, 10, 6)
    c_int = st.slider("Int. Prestige", 1, 10, 5)
    c_attack = st.slider("Club Attack", 40, 99, 75)
    c_midfield = st.slider("Club Midfield", 40, 99, 74)
    c_defence = st.slider("Club Defence", 40, 99, 72)

st.divider()

if st.button("💰 Predict Wage", type="primary", use_container_width=True):
    payload = {
        "P_Age": age, "P_Overall": overall, "P_Potential": potential,
        "P_IntReputation": int_rep, "P_WeakFoot": weak_foot,
        "P_SkillMoves": skill_moves,
        "P_PaceTotal": pace, "P_ShootingTotal": shooting,
        "P_PassingTotal": passing, "P_DribblingTotal": dribbling,
        "P_DefendingTotal": defending, "P_PhysicalityTotal": physicality,
        "P_ValueEUR": value_eur, "P_ContractUntil": contract,
        "C_Overall": c_overall, "C_TransferBudget": c_budget,
        "C_DomesticPrestige": c_domestic, "C_IntPrestige": c_int,
        "C_Attack": c_attack, "C_Midfield": c_midfield, "C_Defence": c_defence,
        "C_League": league,
    }

    with st.spinner("Calculating..."):
        try:
            resp = requests.post(f"{API_URL}/predict",
                                 json=payload, timeout=10)
            resp.raise_for_status()
            result = resp.json()

            st.success("✅ Done!")
            r1, r2 = st.columns(2)
            r1.metric("Wage / Week", f"€{result['wage_eur_per_week']:,.0f}")
            r2.metric("Wage / Year", f"€{result['wage_eur_per_year']:,.0f}")

            with st.expander("🔧 Feature Engineering (Inputs passed to model)"):
                fe_keys = ["Growth", "OverallSquared", "ContractYearsLeft",
                           "Club_Strength", "LeagueRank", "IsTopLeague",
                           "AgeGroup", "ValuePerOverall"]
                st.json({k: v for k, v in result["features_used"].items()
                         if k in fe_keys})

        except requests.exceptions.ConnectionError:
            st.error(
                f"❌ Failed to connect to API ({API_URL}). "
                "Please make sure the FastAPI server is running."
            )
        except Exception as e:
            st.error(f"❌ Error: {e}")
