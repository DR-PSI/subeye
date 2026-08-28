#!/usr/bin/env python3
"""
SubEye · Daily ML — เทรน LSTM ต่ออุปกรณ์ พยากรณ์ 24 ชั่วโมงข้างหน้า เขียนคะแนนกลับ Firestore

ออกแบบให้ปลอดภัยกับข้อมูลจริง:
  · ถ้าข้อมูลน้อย จะถอยไปใช้ Holt-Winters เอง ไม่ยัด LSTM ให้ overfit
  · เทียบ MAE ของ LSTM กับ baseline ทุกครั้ง ตัวไหนแม่นกว่าใช้ตัวนั้น (เขียนไว้ใน ml.method)
  · เขียนกลับเฉพาะฟิลด์ ml เท่านั้น ไม่แตะ value / bl / hourly ที่ระบบอื่นเป็นเจ้าของ

อ่าน:  stations/{stationKey}/devices/{deviceId}.hourly   (ค่าเฉลี่ยรายชั่วโมง สูงสุด 168 จุด)
เขียน: เอกสารเดิม ฟิลด์ ml
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests
from google.cloud import firestore
from google.oauth2 import service_account

# ---------------------------------------------------------------- config

ARTIFACTS = Path(__file__).parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(ARTIFACTS / "run.log", mode="w")],
)
log = logging.getLogger("subeye")

PROJECT_ID = "inspection-asset"

# เกณฑ์เตือนต่อ template — ต้องตรงกับ index.html และ workflow n8n
TEMPLATES = {
    "xfmr":    {"unit": "°C",  "warn": 70, "alarm": 85},
    "temp":    {"unit": "°C",  "warn": 42, "alarm": 48},
    "humid":   {"unit": "%RH", "warn": 75, "alarm": 82},
    "dxfmr":   {"unit": "°C",  "warn": 70, "alarm": 85},
    "lbs":     {"unit": "°C",  "warn": 45, "alarm": 60},
    "netxfmr": {"unit": "°C",  "warn": 70, "alarm": 85},
    "unitsub": {"unit": "°C",  "warn": 60, "alarm": 75},
}

SEASON = 24          # 24 จุด = 1 วัน (hourly เก็บ 1 จุด/ชั่วโมง)
HORIZON = 24         # พยากรณ์ล่วงหน้า 24 ชั่วโมง
LOOKBACK = 48        # LSTM ดูย้อนหลัง 48 ชั่วโมงเพื่อทำนายจุดถัดไป
MIN_FOR_STATS = 48   # 2 วัน — เริ่มใช้ Holt-Winters ได้
MIN_FOR_LSTM = 336   # 14 วัน — น้อยกว่านี้ LSTM ไม่คุ้ม
DRY_RUN = os.environ.get("DRY_RUN", "").lower() == "true"


def default_scopes() -> list[dict]:
    """ตั้งค่าที่ repo variable SUBEYE_SCOPES เป็น JSON ได้ ถ้าไม่ตั้งใช้ค่านี้"""
    raw = os.environ.get("SUBEYE_SCOPES", "").strip()
    if raw:
        return json.loads(raw)
    return [
        {"level": "substation", "station": "คลองเตย", "substation": "WTK"},
        {"level": "substation", "station": "คลองเตย", "substation": "BPT"},
        {"level": "district",   "station": "คลองเตย"},
    ]


def sanitize(s: str) -> str:
    """ต้องให้ผลเหมือน sanitizeKey ใน index.html"""
    return "".join(
        c if (c.isalnum() or c == "_" or c == "-" or "\u0e00" <= c <= "\u0e7f") else "_"
        for c in str(s)
    )


def station_key(scope: dict) -> str:
    if scope.get("level") == "district":
        return f"district__{sanitize(scope['station'])}"
    return f"sub__{sanitize(scope['station'])}__{sanitize(scope['substation'])}"


# ---------------------------------------------------------------- models

@dataclass
class Result:
    score: int | None
    level: str
    method: str
    trend: float | None = None
    season_amp: float | None = None
    sigma: float | None = None
    z_robust: float | None = None
    forecast24: list[float] | None = None
    days_to_warn: float | None = None
    days_to_alarm: float | None = None
    mae: float | None = None
    mae_baseline: float | None = None
    note: str | None = None


def holt_winters(y: np.ndarray, m: int, alpha: float, beta: float, gamma: float):
    """Holt-Winters additive — ใช้เป็นทั้ง fallback และ baseline วัดว่า LSTM ดีจริงไหม"""
    n = len(y)
    cycles = n // m
    if cycles < 2:
        return None

    means = np.array([y[c * m:(c + 1) * m].mean() for c in range(cycles)])
    level = float(means[0])
    trend = float((means[1] - means[0]) / m)
    season = np.array([
        np.mean([y[c * m + i] - means[c] for c in range(cycles)]) for i in range(m)
    ], dtype=float)

    resid = np.empty(n)
    for t in range(n):
        si = t % m
        fitted = level + trend + season[si]
        resid[t] = y[t] - fitted
        last = level
        level = alpha * (y[t] - season[si]) + (1 - alpha) * (level + trend)
        trend = beta * (level - last) + (1 - beta) * trend
        season[si] = gamma * (y[t] - level) + (1 - gamma) * season[si]

    return {"level": level, "trend": trend, "season": season,
            "resid": resid, "rmse": float(np.sqrt((resid ** 2).mean()))}


def fit_holt_winters(y: np.ndarray, m: int = SEASON):
    best = None
    for a in (0.1, 0.25, 0.4, 0.6, 0.8):
        for b in (0.02, 0.08, 0.2):
            for g in (0.05, 0.2, 0.4):
                fit = holt_winters(y, m, a, b, g)
                if fit and (best is None or fit["rmse"] < best["rmse"]):
                    fit.update(alpha=a, beta=b, gamma=g)
                    best = fit
    return best


def robust_sigma(resid: np.ndarray) -> tuple[float, float]:
    """median + MAD — ทนค่าโดดจากสัญญาณรบกวน RS485 ดีกว่า mean + sd"""
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)))
    return med, mad * 1.4826


def hw_forecast(fit: dict, n_hist: int, horizon: int) -> list[float]:
    return [
        float(fit["level"] + h * fit["trend"] + fit["season"][(n_hist + h - 1) % SEASON])
        for h in range(1, horizon + 1)
    ]


def build_lstm(lookback: int):
    """สร้างทีหลังเพื่อไม่ให้ import tensorflow ตอนไม่ได้ใช้ (ประหยัดเวลา runner)"""
    import tensorflow as tf

    tf.random.set_seed(42)
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(lookback, 1)),
        tf.keras.layers.LSTM(32, return_sequences=True),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="huber", metrics=["mae"])
    return model


def windows(y: np.ndarray, lookback: int):
    X = np.array([y[i:i + lookback] for i in range(len(y) - lookback)])
    Y = np.array([y[i + lookback] for i in range(len(y) - lookback)])
    return X[..., None], Y


def run_lstm(y: np.ndarray, horizon: int, device_id: str):
    """คืน (forecast, mae_on_holdout) หรือ None ถ้าเทรนไม่ได้"""
    import tensorflow as tf

    lo, hi = float(y.min()), float(y.max())
    span = hi - lo if hi - lo > 1e-6 else 1.0
    yn = (y - lo) / span

    split = int(len(yn) * 0.85)
    X_tr, Y_tr = windows(yn[:split], LOOKBACK)
    X_va, Y_va = windows(yn[split - LOOKBACK:], LOOKBACK)
    if len(X_tr) < 40 or len(X_va) < 8:
        return None

    model = build_lstm(LOOKBACK)
    ckpt = ARTIFACTS / f"{device_id}.weights.h5"
    if ckpt.exists():
        try:
            model.load_weights(ckpt)          # เทรนต่อจากรอบก่อน ประหยัดเวลา
            log.info("      โหลดน้ำหนักเดิมของ %s", device_id)
        except Exception:
            pass

    model.fit(
        X_tr, Y_tr,
        validation_data=(X_va, Y_va),
        epochs=60, batch_size=16, shuffle=True, verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True)],
    )
    model.save_weights(ckpt)

    mae = float(np.mean(np.abs(model.predict(X_va, verbose=0).ravel() - Y_va)) * span)

    # พยากรณ์แบบ recursive — ป้อนค่าที่ทำนายได้กลับเข้าไป
    seq = list(yn[-LOOKBACK:])
    forecast = []
    for _ in range(horizon):
        nxt = float(model.predict(np.array(seq[-LOOKBACK:])[None, ..., None], verbose=0)[0, 0])
        seq.append(nxt)
        forecast.append(nxt * span + lo)

    return [round(v, 2) for v in forecast], mae


# ---------------------------------------------------------------- scoring

def score_device(hourly: list[float], tpl_id: str, value: float, device_id: str,
                 warn: float | None = None, alarm: float | None = None) -> Result:
    base = TEMPLATES.get(tpl_id, TEMPLATES["temp"])
    # เกณฑ์ต่ออุปกรณ์ชนะค่า template ถ้ามีตั้งไว้ในเอกสาร
    tpl = {
        "unit": base["unit"],
        "warn": float(warn) if warn is not None else base["warn"],
        "alarm": float(alarm) if alarm is not None else base["alarm"],
    }
    y = np.array([v for v in hourly if v is not None and np.isfinite(v)], dtype=float)

    if len(y) < MIN_FOR_STATS:
        return Result(score=None, level="learning", method="none",
                      note=f"ข้อมูลไม่พอ ({len(y)}/{MIN_FOR_STATS})")

    fit = fit_holt_winters(y)
    if fit is None:
        return Result(score=None, level="learning", method="none", note="fit ไม่สำเร็จ")

    med, sigma = robust_sigma(fit["resid"])
    z = (fit["resid"][-1] - med) / sigma if sigma > 1e-6 else 0.0

    # baseline MAE จาก in-sample residual ของช่วงท้าย เอาไว้เทียบกับ LSTM
    mae_baseline = float(np.mean(np.abs(fit["resid"][-max(8, len(y) // 8):])))
    forecast = hw_forecast(fit, len(y), HORIZON)
    method = "holt-winters"
    mae = mae_baseline

    if len(y) >= MIN_FOR_LSTM:
        log.info("      ข้อมูลพอสำหรับ LSTM (%d จุด) — เริ่มเทรน", len(y))
        try:
            res = run_lstm(y, HORIZON, device_id)
            if res:
                lstm_forecast, lstm_mae = res
                log.info("      LSTM MAE %.3f vs Holt-Winters %.3f", lstm_mae, mae_baseline)
                if lstm_mae < mae_baseline:      # ใช้เฉพาะเมื่อชนะจริง
                    forecast, mae, method = lstm_forecast, lstm_mae, "lstm"
        except Exception as e:
            log.warning("      LSTM ล้มเหลว ใช้ Holt-Winters แทน: %s", e)

    # แนวโน้มต่อวัน + อีกกี่วันถึงเกณฑ์ (ใช้ trend ล้วน ไม่รวมรอบวัน)
    trend_per_day = fit["trend"] * SEASON

    def days_to(threshold: float) -> float | None:
        if value >= threshold:
            return 0.0
        if trend_per_day <= 1e-3:
            return None
        return round((threshold - value) / trend_per_day, 1)

    d_warn, d_alarm = days_to(tpl["warn"]), days_to(tpl["alarm"])

    z_pen = min(46.0, max(0.0, abs(z) - 1.5) * 12)
    f_pen = 0.0 if d_warn is None else max(0.0, 30 - d_warn) / 30 * 34
    rel_err = mae / max(1e-6, float(np.mean(np.abs(y)))) * 100
    m_pen = min(10.0, max(0.0, rel_err - 8))
    score = int(round(max(0, min(100, 100 - z_pen - f_pen - m_pen))))

    return Result(
        score=score,
        level="ok" if score >= 85 else "watch" if score >= 70 else "risk",
        method=method,
        trend=round(float(trend_per_day), 3),
        season_amp=round(float(fit["season"].max() - fit["season"].min()), 2),
        sigma=round(float(sigma), 3),
        z_robust=round(float(z), 2),
        forecast24=forecast,
        days_to_warn=d_warn,
        days_to_alarm=d_alarm,
        mae=round(float(mae), 3),
        mae_baseline=round(float(mae_baseline), 3),
    )


# ---------------------------------------------------------------- main

def firestore_client() -> firestore.Client:
    raw = os.environ.get("FIREBASE_SERVICE_ACCOUNT")
    if not raw:
        raise SystemExit("ไม่พบ secret FIREBASE_SERVICE_ACCOUNT")
    creds = service_account.Credentials.from_service_account_info(json.loads(raw))
    return firestore.Client(project=PROJECT_ID, credentials=creds)


def push_line(summary: dict) -> None:
    url = os.environ.get("LINE_WEBHOOK_URL")
    if not url:
        log.info("ไม่ได้ตั้ง LINE_WEBHOOK_URL — ข้ามการส่ง")
        return

    risk = summary["risk"]
    watch = summary["watch"]
    if not risk and not watch:
        log.info("ทุกอุปกรณ์ปกติ — ไม่ส่งเข้า LINE")
        return

    def line_of(r: dict) -> str:
        eta = ("แนวโน้มคงที่" if r["days_to_warn"] is None
               else "เกินเกณฑ์แล้ว" if r["days_to_warn"] <= 0
               else f"ถึงเกณฑ์ ~{r['days_to_warn']:.0f} วัน")
        return f"· {r['label']} — คะแนน {r['score']} · {r['value']}{r['unit']} · {eta}"

    lines = ["📊 SubEye · สรุปสุขภาพอุปกรณ์รายวัน",
             datetime.now(timezone.utc).astimezone().strftime("%d/%m/%Y"), ""]
    if risk:
        lines += [f"🔴 เสี่ยงสูง {len(risk)} รายการ"] + [line_of(r) for r in risk[:5]] + [""]
    if watch:
        lines += [f"🟡 เฝ้าระวัง {len(watch)} รายการ"] + [line_of(r) for r in watch[:5]] + [""]
    lines.append(f"โมเดล: {summary['methods']} · {summary['total']} อุปกรณ์")

    try:
        res = requests.post(url, json={
            "source": "github-actions-ml",
            "kind": "summary",
            "message": "\n".join(lines),
            "riskCount": len(risk),
            "watchCount": len(watch),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, timeout=20)
        log.info("ส่งเข้า LINE: HTTP %s", res.status_code)
    except Exception as e:
        log.error("ส่งเข้า LINE ไม่สำเร็จ: %s", e)


def main() -> int:
    if DRY_RUN:
        log.info("โหมด DRY RUN — คำนวณแต่ไม่เขียนกลับ Firestore")

    db = firestore_client()
    scopes = default_scopes()
    log.info("ขอบเขตที่จะประมวลผล: %d รายการ", len(scopes))

    rows, methods = [], {}

    for scope in scopes:
        key = station_key(scope)
        log.info("── %s", key)
        docs = list(db.collection("stations").document(key).collection("devices").stream())
        if not docs:
            log.info("   ไม่มีอุปกรณ์")
            continue

        for doc in docs:
            d = doc.to_dict() or {}
            tpl_id = d.get("tplId", "temp")
            if tpl_id not in TEMPLATES:
                continue                       # ข้ามอุปกรณ์เปิด/ปิด ไม่มี baseline ตัวเลข

            hourly = d.get("hourly") or []
            value = float(d.get("value") or 0)
            label = d.get("label") or doc.id
            base_unit = TEMPLATES[tpl_id]["unit"]
            log.info("   %-34s hourly=%d", label[:34], len(hourly))

            def num(x):
                try:
                    return float(x) if x is not None else None
                except (TypeError, ValueError):
                    return None

            r = score_device(hourly, tpl_id, value, doc.id,
                             warn=num(d.get("warn")), alarm=num(d.get("alarm")))
            methods[r.method] = methods.get(r.method, 0) + 1

            payload = {k: v for k, v in asdict(r).items() if v is not None}
            payload["computedAt"] = datetime.now(timezone.utc).isoformat()
            payload["source"] = "github-actions"

            if not DRY_RUN:
                doc.reference.set({"ml": payload}, merge=True)

            if r.score is not None:
                log.info("      → คะแนน %d (%s) · %s", r.score, r.level, r.method)
                rows.append({"label": label, "value": round(value, 1),
                             "unit": base_unit, "score": r.score,
                             "level": r.level, "days_to_warn": r.days_to_warn,
                             "method": r.method})
            else:
                log.info("      → %s", r.note)

    rows.sort(key=lambda r: r["score"])
    summary = {
        "total": len(rows),
        "risk": [r for r in rows if r["level"] == "risk"],
        "watch": [r for r in rows if r["level"] == "watch"],
        "methods": ", ".join(f"{k}×{v}" for k, v in methods.items()) or "none",
        "computedAt": datetime.now(timezone.utc).isoformat(),
        "dryRun": DRY_RUN,
    }
    (ARTIFACTS / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    log.info("เสร็จ — ประเมินได้ %d อุปกรณ์ · เสี่ยงสูง %d · เฝ้าระวัง %d",
             summary["total"], len(summary["risk"]), len(summary["watch"]))

    if not DRY_RUN:
        push_line(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
