import os

# System One Model configuration
MODEL_NAME = os.getenv("MODEL_NAME", os.getenv("APP_MODEL_NAME", "/model"))
SYSTEM_ONE_DECISION_OPTIONS = ["LOW_RISK", "MEDIUM_RISK", "HIGH_RISK_FRAUD"]

# KServe Service Endpoint configuration
KSERVE_ENDPOINT_URL = os.getenv(
    "KSERVE_ENDPOINT_URL",
    os.getenv("APP_KSERVE_ENDPOINT_URL", "http://10.64.140.44:8080/v1/completions")
)

