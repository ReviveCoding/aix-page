from __future__ import annotations

import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import xgboost as xgb
from torch import nn

from aix_page.calibration.calibrators import Isotonic, Platt
from aix_page.models.baselines import HashedLinear
from aix_page.models.binomial import sigmoid, weighted_metrics
from aix_page.models.dcnv2 import DCNv2
from aix_page.models.xgb_binomial import aggregated_binomial_objective
from aix_page.utils.hashing import sha256_file

ROOT = Path(__file__).resolve().parents[1]
FEATURE_ROOT = ROOT / "data/processed/kdd_features_v7"
ARTIFACTS = ROOT / "artifacts/qualification/models_v7"
CATEGORICAL = [
    "display_url_id",
    "ad_id",
    "advertiser_id",
    "query_id",
    "keyword_id",
    "title_id",
    "description_id",
    "user_id",
]
DENSE = [
    "position",
    "depth",
    "normalized_position",
    "query_token_count",
    "keyword_token_count",
    "title_token_count",
    "description_token_count",
    "query_keyword_overlap",
    "query_title_overlap",
    "query_description_overlap",
    "query_keyword_jaccard",
    "hist_ad_ctr",
    "hist_advertiser_ctr",
    "hist_display_url_ctr",
    "hist_query_ctr",
    "hist_user_ctr",
    "hist_position_depth_ctr",
    "hist_ad_query_ctr",
    "cold_query",
    "cold_ad",
    "cold_advertiser",
    "cold_user",
    "tail_query",
    "low_support",
    "high_depth",
    "position_extreme",
]


def role_path(role: str) -> Path:
    return FEATURE_ROOT / f"data_role={role}/data_0.parquet"


def load(role: str, rows: int, columns: list[str] | None = None, *, namespace: str) -> pd.DataFrame:
    if columns is None:
        columns = pq.ParquetFile(role_path(role)).schema.names
    selected = ", ".join(f'"{column}"' for column in columns)
    path = role_path(role).as_posix().replace("'", "''")
    # Hash eligibility makes the retained rows independent of physical source order.
    query = (
        f"SELECT {selected} FROM read_parquet('{path}') "
        f"WHERE hash(setting_hash_v7, '{namespace}') % 1000000 < 600000 "
        f"LIMIT {int(rows)}"
    )
    frame = duckdb.connect().execute(query).fetchdf()
    if len(frame) != rows:
        raise RuntimeError(f"hash sample for {role}/{namespace} returned {len(frame)} of {rows}")
    return frame


def dense_matrix(frame: pd.DataFrame) -> np.ndarray:
    values = frame[DENSE].to_numpy(dtype=np.float32)
    for index, column in enumerate(DENSE):
        if "support" in column or "token_count" in column or "overlap" in column:
            values[:, index] = np.log1p(np.maximum(values[:, index], 0))
    return np.nan_to_num(values, copy=False)


def xgb_matrix(frame: pd.DataFrame) -> np.ndarray:
    dense = dense_matrix(frame)
    categorical = np.column_stack(
        [(frame[column].to_numpy(dtype=np.uint64) % 65_521) / 65_521 for column in CATEGORICAL]
    ).astype(np.float32)
    return np.column_stack([dense, categorical]).astype(np.float32)


def metric_row(model: str, frame: pd.DataFrame, probability: np.ndarray) -> dict[str, object]:
    return {
        "model": model,
        "split": "model_validation",
        "rows": len(frame),
        **weighted_metrics(frame["click"].to_numpy(), frame["impression"].to_numpy(), probability),
    }


def train_dcn(
    train: pd.DataFrame, validation: pd.DataFrame, seed: int
) -> tuple[DCNv2, np.ndarray, dict[str, object]]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    buckets = 65_521
    train_cat = torch.from_numpy(
        np.column_stack(
            [train[column].to_numpy(dtype=np.uint64) % buckets for column in CATEGORICAL]
        ).astype(np.int64)
    )
    train_dense_np = dense_matrix(train)
    mean = train_dense_np.mean(axis=0)
    std = np.maximum(train_dense_np.std(axis=0), 1e-4)
    train_dense = torch.from_numpy((train_dense_np - mean) / std)
    fraction = torch.from_numpy((train["click"] / train["impression"]).to_numpy(dtype=np.float32))
    impression = torch.from_numpy(train["impression"].to_numpy(dtype=np.float32))
    model = DCNv2(len(CATEGORICAL), buckets, 8, len(DENSE)).to(device)
    empirical_prior = float(train["click"].sum() / train["impression"].sum())
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.fill_(math.log(empirical_prior / (1.0 - empirical_prior)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    generator = torch.Generator().manual_seed(seed)
    batch_size = 8192
    history: list[dict[str, float]] = []
    best_loss = math.inf
    best_state: dict[str, torch.Tensor] | None = None
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(6):
        permutation = torch.randperm(len(train), generator=generator)
        model.train()
        for start in range(0, len(train), batch_size):
            index = permutation[start : start + batch_size]
            cat = train_cat[index].to(device, non_blocking=True)
            dense = train_dense[index].to(device, non_blocking=True)
            target = fraction[index].to(device, non_blocking=True)
            weight = impression[index].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                margin = model(cat, dense)
                loss = (
                    nn.functional.binary_cross_entropy_with_logits(
                        margin, target, weight=weight, reduction="sum"
                    )
                    / weight.sum()
                )
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
        probability = predict_dcn(model, validation, mean, std)
        validation_loss = weighted_metrics(
            validation["click"].to_numpy(),
            validation["impression"].to_numpy(),
            probability,
        )["weighted_log_loss"]
        history.append({"epoch": epoch + 1, "validation_weighted_log_loss": validation_loss})
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            break
    if best_state is None:
        raise RuntimeError("DCNv2 early stopping did not produce a checkpoint")
    model.load_state_dict(best_state)
    probability = predict_dcn(model, validation, mean, std)
    runtime = time.perf_counter() - started
    metadata = {
        "runtime_seconds": runtime,
        "training_rows": len(train),
        "rows_per_second": len(train) * len(history) / runtime,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "validation_history": history,
        "dense_mean": mean.tolist(),
        "dense_std": std.tolist(),
    }
    return model, probability, metadata


def predict_dcn(model: DCNv2, frame: pd.DataFrame, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    model.eval()
    buckets = 65_521
    categorical = np.column_stack(
        [frame[column].to_numpy(dtype=np.uint64) % buckets for column in CATEGORICAL]
    ).astype(np.int64)
    dense = (dense_matrix(frame) - mean) / std
    outputs = []
    with torch.no_grad():
        for start in range(0, len(frame), 16_384):
            cat = torch.from_numpy(categorical[start : start + 16_384]).cuda()
            values = torch.from_numpy(dense[start : start + 16_384]).cuda()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs.append(torch.sigmoid(model(cat, values)).float().cpu().numpy())
    return np.concatenate(outputs)


def main() -> None:
    if (ARTIFACTS / "model_selection_frozen.json").exists():
        raise FileExistsError("model selection already frozen; no silent rerun")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    common_columns = ["click", "impression", *CATEGORICAL, *DENSE]
    train_m1 = load("model_train", 300_000, common_columns, namespace="m1_v7")
    train_gpu = load("model_train", 2_000_000, common_columns, namespace="gpu_v7")
    train_dcn_frame = train_gpu.iloc[:1_000_000].copy()
    validation = load("model_validation", 300_000, common_columns, namespace="validation_v7")
    calibration = load("model_calibration", 300_000, common_columns, namespace="calibration_v7")
    challenge = load("challenge_pool", 300_000, common_columns, namespace="challenge_v7")
    prior = float(train_gpu["click"].sum() / train_gpu["impression"].sum())
    results = [metric_row("M0_smoothed_prior", validation, np.full(len(validation), prior))]

    rename = {
        "click": "Click",
        "impression": "Impression",
        "position": "Position",
        "depth": "Depth",
        "ad_id": "AdID",
        "advertiser_id": "AdvertiserID",
        "display_url_id": "DisplayURL",
        "query_id": "QueryID",
    }
    linear_train = train_m1.rename(columns=rename)
    linear_validation = validation.rename(columns=rename)
    started = time.perf_counter()
    linear = HashedLinear(seed=20260905).fit(linear_train)
    linear_probability = linear.predict(linear_validation)
    linear_runtime = time.perf_counter() - started
    results.append(metric_row("M1_hashed_linear", validation, linear_probability))
    joblib.dump(linear, ARTIFACTS / "m1_hashed_linear.joblib")

    x_train = xgb_matrix(train_gpu)
    x_validation = xgb_matrix(validation)
    dtrain = xgb.QuantileDMatrix(
        x_train,
        label=(train_gpu["click"] / train_gpu["impression"]).to_numpy(),
        weight=train_gpu["impression"].to_numpy(),
        max_bin=256,
    )
    dvalidation = xgb.QuantileDMatrix(
        x_validation,
        label=(validation["click"] / validation["impression"]).to_numpy(),
        weight=validation["impression"].to_numpy(),
        ref=dtrain,
        max_bin=256,
    )
    xgb_started = time.perf_counter()
    booster = xgb.train(
        {
            "tree_method": "hist",
            "device": "cuda",
            "max_depth": 7,
            "eta": 0.08,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 40,
            "base_score": float(np.log(prior / (1 - prior))),
            "seed": 20260905,
            "max_bin": 256,
        },
        dtrain,
        num_boost_round=120,
        obj=aggregated_binomial_objective,
        evals=[(dvalidation, "validation")],
        verbose_eval=False,
    )
    xgb_runtime = time.perf_counter() - xgb_started
    xgb_probability = sigmoid(booster.predict(dvalidation, output_margin=True))
    results.append(metric_row("M2_xgboost_cuda", validation, xgb_probability))
    booster.save_model(ARTIFACTS / "m2_xgboost_cuda.ubj")

    dcn, dcn_probability, dcn_metadata = train_dcn(train_dcn_frame, validation, 20260905)
    results.append(metric_row("M3_dcnv2_cuda", validation, dcn_probability))
    torch.save(dcn.state_dict(), ARTIFACTS / "m3_dcnv2_cuda.pt")
    (ARTIFACTS / "m3_metadata.json").write_text(
        json.dumps(dcn_metadata, indent=2) + "\n", encoding="utf-8"
    )
    benchmark = pd.DataFrame(results).sort_values("weighted_log_loss")
    benchmark["runtime_seconds"] = benchmark["model"].map(
        {
            "M0_smoothed_prior": 0.0,
            "M1_hashed_linear": linear_runtime,
            "M2_xgboost_cuda": xgb_runtime,
            "M3_dcnv2_cuda": dcn_metadata["runtime_seconds"],
        }
    )
    benchmark.to_csv(ARTIFACTS / "model_benchmark_validation.csv", index=False)
    winner = str(benchmark.iloc[0]["model"])

    def predict_selected(frame: pd.DataFrame) -> np.ndarray:
        if winner == "M0_smoothed_prior":
            return np.full(len(frame), prior)
        if winner == "M1_hashed_linear":
            return linear.predict(frame.rename(columns=rename))
        if winner == "M2_xgboost_cuda":
            matrix = xgb.DMatrix(xgb_matrix(frame))
            return sigmoid(booster.predict(matrix, output_margin=True))
        mean = np.asarray(dcn_metadata["dense_mean"], dtype=np.float32)
        std = np.asarray(dcn_metadata["dense_std"], dtype=np.float32)
        return predict_dcn(dcn, frame, mean, std)

    calibration_raw = predict_selected(calibration)
    calibrators = {
        "raw": None,
        "platt": Platt().fit(
            calibration["click"].to_numpy(),
            calibration["impression"].to_numpy(),
            calibration_raw,
        ),
        "isotonic": Isotonic().fit(
            calibration["click"].to_numpy(),
            calibration["impression"].to_numpy(),
            calibration_raw,
        ),
    }
    calibration_rows = []
    for name, calibrator in calibrators.items():
        probability = calibration_raw if calibrator is None else calibrator.predict(calibration_raw)
        calibration_rows.append(
            {
                "calibrator": name,
                "split": "model_calibration",
                **weighted_metrics(
                    calibration["click"].to_numpy(),
                    calibration["impression"].to_numpy(),
                    probability,
                ),
            }
        )
    calibration_metrics = pd.DataFrame(calibration_rows).sort_values(["weighted_log_loss", "ece"])
    calibration_metrics.to_csv(ARTIFACTS / "calibration_selection.csv", index=False)
    selected_calibrator = str(calibration_metrics.iloc[0]["calibrator"])
    if selected_calibrator != "raw":
        joblib.dump(calibrators[selected_calibrator], ARTIFACTS / "calibrator.joblib")
    selection = {
        "phase": "P07_MODEL_QUALIFICATION",
        "generated_at": datetime.now(UTC).isoformat(),
        "selection_rule": "lowest validation weighted log loss; material threshold 0.0001; calibration chosen only on model_calibration",
        "partition_version": "v7",
        "sampling_contract": "setting_hash_v7 eligibility with distinct role-purpose namespaces; never physical prefixes",
        "selected_model": winner,
        "selected_calibrator": selected_calibrator,
        "validation_rows": len(validation),
        "calibration_rows": len(calibration),
        "locked_test_open_count": 0,
        "training_rows": {
            "M1": len(train_m1),
            "M2": len(train_gpu),
            "M3": len(train_dcn_frame),
        },
        "challenge_rows_prelock": len(challenge),
        "artifacts": {
            path.name: sha256_file(path) for path in ARTIFACTS.iterdir() if path.is_file()
        },
        "frozen": True,
    }
    (ARTIFACTS / "model_selection_frozen.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )

    locked = load("model_locked_test", 500_000, common_columns, namespace="locked_test_once_v7")
    locked_raw = predict_selected(locked)
    selected_object = calibrators[selected_calibrator]
    locked_probability = (
        locked_raw if selected_object is None else selected_object.predict(locked_raw)
    )
    locked_overall = weighted_metrics(
        locked["click"].to_numpy(),
        locked["impression"].to_numpy(),
        locked_probability,
    )
    slice_rows = []
    for column in [
        "cold_query",
        "cold_ad",
        "cold_advertiser",
        "tail_query",
        "low_support",
        "high_depth",
        "position_extreme",
    ]:
        mask = locked[column].astype(bool).to_numpy()
        if mask.sum() < 100:
            continue
        slice_rows.append(
            {
                "slice": column,
                "rows": int(mask.sum()),
                **weighted_metrics(
                    locked.loc[mask, "click"].to_numpy(),
                    locked.loc[mask, "impression"].to_numpy(),
                    locked_probability[mask],
                ),
            }
        )
    pd.DataFrame(slice_rows).to_csv(ARTIFACTS / "locked_challenge_metrics.csv", index=False)
    selection["locked_test_open_count"] = 1
    selection["locked_test_rows"] = len(locked)
    selection["locked_metrics"] = locked_overall
    selection["locked_test_opened_at"] = datetime.now(UTC).isoformat()
    (ARTIFACTS / "model_selection_result.json").write_text(
        json.dumps(selection, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
