from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd
import shap


def explain_entry(model, entry: pd.Series, feature_columns: Iterable[str]):
    features = [column for column in feature_columns if column in entry.index]
    sample = entry[features].to_frame().T.copy()

    for column in features:
        sample[column] = pd.to_numeric(sample[column], errors="coerce").fillna(0.0)

    if "Amount" in sample.columns:
        sample["Amount"] = np.log1p(sample["Amount"].clip(lower=0.01))

    sample = sample.astype(float)

    explainer = shap.TreeExplainer(model.model)
    values = explainer.shap_values(sample)

    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]

    values = np.asarray(values)
    if values.ndim == 3:
        values = values[0]
    values = np.asarray(values).reshape(len(features),)

    expected_value = explainer.expected_value
    if isinstance(expected_value, list):
        expected_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
    expected_value = np.asarray(expected_value)
    if expected_value.ndim > 0:
        expected_value = expected_value.reshape(-1)[0]

    explanation = shap.Explanation(
        values=values,
        base_values=float(expected_value),
        data=sample.iloc[0].to_numpy(),
        feature_names=list(features),
    )
    return explanation


def describe_shap_contributors(explanation: shap.Explanation) -> Dict[str, float]:
    values = explanation.values
    if isinstance(values, np.ndarray):
        values = values.tolist()
    outputs = {}
    for feature_name, contribution in zip(explanation.feature_names, values):
        outputs[str(feature_name)] = float(contribution)
    return outputs
