from pathlib import Path

APP_TITLE = "AI Crime Intelligence System"
DEFAULT_DATA_PATH = Path("data/sample_crime_data.csv")
MODEL_PATH = Path("models/artifacts/crime_risk_model.joblib")
RANDOM_STATE = 42

REQUIRED_COLUMNS = {
    "crime_type",
    "timestamp",
    "latitude",
    "longitude",
    "district",
}

TN_CITY_CENTERS = {
    "Chennai": (13.0827, 80.2707),
    "Coimbatore": (11.0168, 76.9558),
    "Madurai": (9.9252, 78.1198),
    "Tiruchirappalli": (10.7905, 78.7047),
    "Salem": (11.6643, 78.1460),
    "Tirunelveli": (8.7139, 77.7567),
    "Erode": (11.3410, 77.7172),
}
