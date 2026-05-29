import streamlit as st
import pandas as pd
import joblib

model = joblib.load(
    "ada_boost_house_price.pkl"
)

model_columns = joblib.load(
    "model_columns.pkl"
)

st.title("California House Price Prediction")

longitude = st.number_input(
    "Longitude",
    value=-122.23
)

latitude = st.number_input(
    "Latitude",
    value=37.88
)

housing_median_age = st.slider(
    "Housing Median Age",
    1,
    60,
    25
)

total_rooms = st.number_input(
    "Total Rooms",
    min_value=1.0,
    value=1500.0
)

total_bedrooms = st.number_input(
    "Total Bedrooms",
    min_value=1.0,
    value=300.0
)

population = st.number_input(
    "Population",
    min_value=1.0,
    value=1000.0
)

households = st.number_input(
    "Households",
    min_value=1.0,
    value=400.0
)

median_income = st.slider(
    "Median Income",
    0.0,
    20.0,
    5.0
)

ocean_proximity = st.selectbox(
    "Ocean Proximity",
    [
        "<1H OCEAN",
        "INLAND",
        "ISLAND",
        "NEAR BAY",
        "NEAR OCEAN"
    ]
)

if st.button("Predict"):

    rooms_per_household = (
        total_rooms / households
    )

    bedrooms_per_room = (
        total_bedrooms / total_rooms
    )

    population_per_household = (
        population / households
    )

    data = {
        "longitude": longitude,
        "latitude": latitude,
        "housing_median_age": housing_median_age,
        "total_rooms": total_rooms,
        "total_bedrooms": total_bedrooms,
        "population": population,
        "households": households,
        "median_income": median_income,
        "rooms_per_household": rooms_per_household,
        "bedrooms_per_room": bedrooms_per_room,
        "population_per_household": population_per_household,
        "ocean_proximity": ocean_proximity
    }

    df = pd.DataFrame([data])

    df = pd.get_dummies(
        df,
        columns=["ocean_proximity"],
        drop_first=True
    )

    df = df.reindex(
        columns=model_columns,
        fill_value=0
    )

    prediction = model.predict(df)[0]

    st.success(
        f"Predicted House Value: ${prediction:,.2f}"
    )