from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import IsolationForest


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config" / "model_config.yaml"
FEATURE_COLUMNS = ["Amount", "Posting_Hour", "Account_Code", "User_ID", "Approval_Level"]


def load_config(path: Path = CONFIG_PATH) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model_features(df: pd.DataFrame, feature_columns: List[str] | None = None) -> pd.DataFrame:
    working = df.copy()
    selected = feature_columns or FEATURE_COLUMNS
    for column in selected:
        if column not in working.columns:
            raise KeyError(f"Missing required feature column: {column}")
    working["Amount"] = np.log1p(working["Amount"].clip(lower=0.01))
    return working[selected].copy()


class GLAnomalyModel:
    def __init__(self, config_path: Path = CONFIG_PATH):
        self.config = load_config(config_path)
        self.model = None
        self.feature_columns = self.config.get("feature_columns", FEATURE_COLUMNS)
        self.thresholds = self.config.get("thresholds", {})
        self.high_risk_threshold = float(self.thresholds.get("high_risk_score", 0.70))
        self.critical_risk_threshold = float(self.thresholds.get("critical_risk_score", 0.85))

    def fit(self, df: pd.DataFrame):
        model_cfg = self.config.get("model", {})
        X = build_model_features(df, self.feature_columns)
        self.model = IsolationForest(
            n_estimators=int(model_cfg.get("n_estimators", 200)),
            contamination=float(model_cfg.get("contamination", 0.05)),
            random_state=int(model_cfg.get("random_state", 42)),
            max_samples=model_cfg.get("max_samples", "auto"),
        )
        self.model.fit(X)
        return self

    def score_records(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None:
            raise ValueError("Model has not been fit yet.")

        X = build_model_features(df, self.feature_columns)
        decision_scores = self.model.decision_function(X)
        raw_risk = -decision_scores

        risk_min = float(raw_risk.min())
        risk_max = float(raw_risk.max())
        normalized = (raw_risk - risk_min) / (risk_max - risk_min + 1e-8)
        risk_score = np.clip(normalized, 0.0, 1.0)

        scored = df.copy()
        scored["Risk_Score"] = np.round(risk_score, 4)
        scored["Anomaly_Flag"] = self.model.predict(X) == -1
        scored["Decision_Score"] = np.round(decision_scores, 4)

        scored["Action_Required"] = scored.apply(
            lambda row: self._classify_action(row), axis=1
        )
        scored["Primary_Risk_Driver"] = scored.apply(
            lambda row: self._primary_risk_driver(row), axis=1
        )
        return scored

    def _classify_action(self, row: pd.Series) -> str:
        if row["Risk_Score"] >= self.critical_risk_threshold:
            return "Escalate: Potential Fraud"
        if row["Posting_Hour"] in {0, 1, 2, 3, 4, 5, 23}:
            return "Investigate: Off-Hours Posting"
        if row["Amount"] > 5000:
            return "Investigate: High-Value Exception"
        if row["Approval_Level"] <= 2:
            return "Review: Low-Approval Exception"
        return "Review: Manual Validation"

    def _primary_risk_driver(self, row: pd.Series) -> str:
        drivers = {
            "Amount": row["Amount"],
            "Posting_Hour": row["Posting_Hour"],
            "Account_Code": row["Account_Code"],
            "User_ID": row["User_ID"],
            "Approval_Level": row["Approval_Level"],
        }
        return max(drivers, key=drivers.get)

    def get_flagged_queue(self, df: pd.DataFrame, queue_limit: int | None = None) -> pd.DataFrame:
        threshold = self.high_risk_threshold
        queue = df[df["Risk_Score"] >= threshold].copy().sort_values("Risk_Score", ascending=False)
        if queue_limit is not None:
            queue = queue.head(queue_limit)
        return queue.reset_index(drop=True)

    def get_entry_for_explanation(self, entry_id: str, df: pd.DataFrame) -> pd.Series:
        match = df[df["Entry_ID"] == entry_id]
        if match.empty:
            raise ValueError(f"Entry {entry_id} not found.")
        return match.iloc[0]
