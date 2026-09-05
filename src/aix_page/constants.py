"""Project-wide contracts."""

KDD_COLUMNS = (
    "Click",
    "Impression",
    "DisplayURL",
    "AdID",
    "AdvertiserID",
    "Depth",
    "Position",
    "QueryID",
    "KeywordID",
    "TitleID",
    "DescriptionID",
    "UserID",
)
PHASES = (
    "P00_BOOTSTRAP",
    "P01_ENVIRONMENT",
    "P02_DATA_ACCESS",
    "P03_DATA_INGESTION",
    "P04_DATA_QUALITY",
    "P05_FEATURE_PIPELINE",
    "P06_MODEL_DEVELOPMENT",
    "P07_MODEL_QUALIFICATION",
    "P07A_MODEL_PRELOCK_AUDIT",
    "P08_EXPERIMENT_DESIGN",
    "P08A_PRELOCK_PILOT",
    "P09_AA_QUALIFICATION",
    "P10_POWER_QUALIFICATION",
    "P10A_PRELOCK_READINESS_AUDIT",
    "P11_PROTOCOL_FREEZE",
    "P12_LOCKED_EXPERIMENT",
    "P13_PRIMARY_ANALYSIS",
    "P14_HTE_ANALYSIS",
    "P15_POLICY_ANALYSIS",
    "P16_ORACLE_ANALYSIS",
    "P17_ROBUSTNESS",
    "P18_REPORTING",
    "P19_FINAL_REVIEW",
    "COMPLETE",
)
BLOCKER_CLASSES = (
    "BLOCKED_EXTERNAL",
    "BLOCKED_ENVIRONMENT",
    "INVALID_SCIENTIFIC_RESULT",
    "NOT_SCIENTIFICALLY_SUPPORTED",
    "OPTIONAL_NOT_REQUIRED",
)
FORMATS = ("classic_text", "rich_product", "conversational_sponsored")
POSITIONS = ("top", "inline")
DENSITIES = ("one", "two")
TREATMENTS = tuple(f"{f}__{p}__{d}" for f in FORMATS for p in POSITIONS for d in DENSITIES)
CONTROL = "classic_text__top__one"
