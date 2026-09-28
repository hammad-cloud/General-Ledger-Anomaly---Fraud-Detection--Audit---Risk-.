from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd


def generate_synthetic_gl_data(
    n_records: int = 5000,
    anomaly_ratio: float = 0.06,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    account_codes = [1000, 1100, 1200, 1300, 1400, 2100, 2200, 3100, 4100, 5100]
    user_ids = list(range(100, 220))
    approval_levels = [1, 2, 3, 4]

    amount = rng.lognormal(mean=3.2, sigma=0.8, size=n_records)
    posting_hour = rng.integers(0, 24, size=n_records)
    account_code = rng.choice(account_codes, size=n_records)
    user_id = rng.choice(user_ids, size=n_records)
    approval_level = rng.choice(
        approval_levels,
        p=[0.45, 0.30, 0.18, 0.07],
        size=n_records,
    )

    df = pd.DataFrame(
        {
            "Entry_ID": [f"GL-{idx:06d}" for idx in range(1, n_records + 1)],
            "Amount": np.round(amount, 2),
            "Posting_Hour": posting_hour,
            "Account_Code": account_code,
            "User_ID": user_id,
            "Approval_Level": approval_level,
        }
    )

    anomaly_count = max(1, int(round(n_records * anomaly_ratio)))
    anomaly_index = rng.choice(n_records, size=anomaly_count, replace=False)

    df.loc[anomaly_index, "Amount"] = rng.lognormal(mean=6.0, sigma=0.9, size=anomaly_count)
    df.loc[anomaly_index, "Posting_Hour"] = rng.choice([0, 1, 2, 3, 4, 5, 23], size=anomaly_count)
    df.loc[anomaly_index, "Account_Code"] = rng.choice([9000, 9100, 9200], size=anomaly_count)
    df.loc[anomaly_index, "Approval_Level"] = rng.choice([1, 2, 3, 4], p=[0.20, 0.25, 0.25, 0.30], size=anomaly_count)
    df.loc[anomaly_index, "User_ID"] = rng.choice([155, 160, 177, 182, 190], size=anomaly_count)

    df["Amount"] = df["Amount"].round(2)
    df["Posting_Hour"] = df["Posting_Hour"].astype(int)
    df["Account_Code"] = df["Account_Code"].astype(int)
    df["User_ID"] = df["User_ID"].astype(int)
    df["Approval_Level"] = df["Approval_Level"].astype(int)

    return df


def describe_generated_data(df: pd.DataFrame) -> List[str]:
    return [
        "Entry_ID, Amount, Posting_Hour, Account_Code, User_ID, Approval_Level",
        f"Rows generated: {len(df):,}",
        f"Anomaly injection rate: {df['Amount'].quantile(0.99):,.2f} and off-hours patterns included.",
    ]
