import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import argparse
from datetime import datetime
import os

#从Excel真实数据中加载真实海淀房价数据
def load_real_excel_data(excel_path: str) -> pd.DataFrame:
    """从Excel加载海淀房价数据（A列=年月，K列=每平米价格，日期格式如2025.08.01）"""
    df = pd.read_excel(excel_path)
    possible_date_cols = ["年月", "date", "时间", "month"]
    possible_price_cols = ["每平方米的价格", "price_per_sqm", "单价", "每平米价格"]

    date_col = next((c for c in possible_date_cols if c in df.columns), df.columns[0])
    price_col = next((c for c in possible_price_cols if c in df.columns), df.columns[-1])

    # ---- 🔧 日期解析增强 ----
    df["date_raw"] = df[date_col].astype(str).str.strip()
    # 统一替换各种符号并尝试多种解析格式
    df["date_clean"] = (
        df["date_raw"]
        .str.replace("年", "-")
        .str.replace("月", "-")
        .str.replace("日", "")
        .str.replace("/", "-")
        .str.replace(".", "-")
    )

    # 主动声明格式 "%Y-%m-%d" 并忽略无效值
    df["date"] = pd.to_datetime(df["date_clean"], format="%Y-%m-%d", errors="coerce")

    # 去除无效日期
    df = df.dropna(subset=["date"])

    # ---- 🔢 价格列 ----
    df["price_per_sqm"] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna(subset=["price_per_sqm"])

    # ---- 📆 按月聚合 ----
    df["year_month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    monthly = df.groupby("year_month")["price_per_sqm"].mean().reset_index()
    monthly = monthly.rename(columns={"year_month": "date"})

    print(f"✅ 成功加载Excel数据，共 {len(monthly)} 个月")
    print(f"时间范围：{monthly['date'].min().date()} - {monthly['date'].max().date()}")
    print("前5个月样例：")
    print(monthly.head())

    return monthly


# 数据生成（模拟海淀区房价数据）- 精确控制时间范围
def generate_synthetic_data(n_samples=2000, random_state=42):
    np.random.seed(random_state)
    area = np.random.normal(90, 20, n_samples).clip(30, 200)
    rooms = np.random.randint(1, 5, n_samples)
    floor = np.random.randint(1, 25, n_samples)
    year_built = np.random.randint(1990, 2023, n_samples)
    dist_metro_m = np.random.exponential(300, n_samples)
    school_rank = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
    lat = np.random.normal(39.96, 0.02, n_samples)
    lon = np.random.normal(116.33, 0.02, n_samples)
    neigh_price = np.random.normal(90000, 8000, n_samples)
    
    # 精确控制日期范围：2019-01-01 到 2025-08-01
    start_date = "2019-01-01"
    end_date = "2025-08-01"
    date = pd.date_range(start=start_date, end=end_date, periods=n_samples)

    # price 模型
    price_per_sqm = (
        30000
        + area * 300
        + rooms * 8000
        - dist_metro_m * 8
        + school_rank * 25000
        + (year_built - 2000) * 1000
        + np.random.normal(0, 12000, n_samples)
    )

    data = pd.DataFrame(
        {
            "area": area,
            "rooms": rooms,
            "floor": floor,
            #"year_built": year_built,
            "dist_metro_m": dist_metro_m,
            "school_rank": school_rank,
            "lat": lat,
            "lon": lon,
            "neigh_price": neigh_price,
            "price_per_sqm": price_per_sqm,
            "date": date,
        }
    )
    return data


# 月度均价加载（来自 CSV year_month, avg_price_per_sqm）
def load_monthly_prices(csv_path: str) -> pd.DataFrame:
    """将月度均价CSV转换为包含 date 与 price_per_sqm 的DataFrame。

    - 输入列: year_month, avg_price_per_sqm
    - 输出列: date(pd.Timestamp 月初), price_per_sqm
    """
    monthly = pd.read_csv(csv_path)
    if "year_month" not in monthly.columns or "avg_price_per_sqm" not in monthly.columns:
        raise ValueError("CSV需包含列: year_month, avg_price_per_sqm")
    # 解析为月初日期
    monthly["date"] = pd.to_datetime(monthly["year_month"], format="%Y-%m")
    monthly = monthly.rename(columns={"avg_price_per_sqm": "price_per_sqm"})
    return monthly[["date", "price_per_sqm"]]

# 模型训练与评估函数
def train_and_evaluate_models(df):
    X = df[["area", "rooms", "floor", "year_built", "dist_metro_m", "school_rank", "lat", "lon", "neigh_price"]]
    y = df["price_per_sqm"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": XGBRegressor(n_estimators=200, random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        rmse = mean_squared_error(y_test, preds) ** 0.5
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        results[name] = (rmse, mae, r2)
        print(f"{name} -> RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.3f}")

        # 可视化前 200 个样本预测效果
        if name == "LinearRegression":
            plt.figure(figsize=(10, 5))
            plt.plot(y_test.values[:200], label="Actual (first 200 test samples)")
            plt.plot(preds[:200], label="Predicted")
            plt.title("Actual vs Predicted (LinearRegression) - sample")
            plt.legend()
            plt.tight_layout()
            plt.show()

    best_model = min(results.items(), key=lambda x: x[1][0])[0]
    print(f"Best model by RMSE: {best_model}")
    return results, best_model


# 优化后的时间序列房价预测函数 - 精确控制预测起点
def forecast_future_prices(
    df,
    years_forward=3,
    piecewise=True,
    damped=True,
    phi=0.9,
    smooth_ma=None,
    ci=False,
    ci_level=0.9,
    floor_pct=None,
    mc_sims=0,
    upturn_horizon=12,
):
    data = df.copy()
    data = data.sort_values("date")
    data["year_month"] = data["date"].dt.to_period("M").dt.to_timestamp()
    monthly = data.groupby("year_month")["price_per_sqm"].mean().reset_index()
    # 可选：移动平均平滑，降低月度噪声
    if smooth_ma is not None and isinstance(smooth_ma, int) and smooth_ma > 1:
        monthly["price_per_sqm"] = (
            monthly["price_per_sqm"].rolling(window=smooth_ma, min_periods=smooth_ma).mean()
        )
        monthly = monthly.dropna(subset=["price_per_sqm"]).reset_index(drop=True)

    # 确保历史数据截止到2025年
    historical_cutoff = "2025-12-31"
    recent = monthly[monthly["year_month"] <= historical_cutoff]
    
    # 如果数据不足，使用所有可用数据
    if len(recent) < 12:
        recent = monthly[monthly["year_month"] <= historical_cutoff]
    
    recent["t"] = np.arange(len(recent))

    # === 基于最近趋势的多模型候选与验证选择 ===
    # 使用最近12个月作为验证集，其他作为训练集
    val_months = 12 if len(recent) > 24 else max(3, len(recent) // 5)
    train_end = max(1, len(recent) - val_months)

    def exp_weights(t_values: np.ndarray, halflife: float) -> np.ndarray:
        if halflife <= 0:
            return np.ones_like(t_values, dtype=float)
        t_max = float(t_values.max())
        # 衰减：每 halflife 月权重减半
        return np.power(0.5, (t_max - t_values) / halflife)

    candidates = []
    # 线性回归 - 全历史
    candidates.append({"kind": "linear_all", "window": None, "halflife": None, "label": "Linear (all)"})
    # 线性回归 - 滑动窗口
    for w in [12, 18, 24, 36]:
        if w < train_end:
            candidates.append({"kind": "linear_window", "window": w, "halflife": None, "label": f"Linear (last {w}m)"})
    # 线性回归 - 指数加权
    for hf in [6, 9, 12, 18]:
        candidates.append({"kind": "linear_ew", "window": None, "halflife": hf, "label": f"Linear EW (hf={hf}m)"})
    # Huber 稳健回归 - 窗口
    for w in [18, 24]:
        if w < train_end:
            candidates.append({"kind": "huber_window", "window": w, "halflife": None, "label": f"Huber (last {w}m)"})
    # 分段线性 - 在窗口内搜索断点（仅开启piecewise时纳入候选）
    if piecewise:
        for w in [24, 36]:
            if w < train_end:
                candidates.append({"kind": "piecewise_window", "window": w, "halflife": None, "label": f"Piecewise (last {w}m)"})

    best = None
    best_rmse = float("inf")

    for c in candidates:
        # 划分训练子集（禁止未来信息泄漏）
        if c["window"] is None:
            start_idx = 0
        else:
            start_idx = max(0, train_end - c["window"])
        train_slice = slice(start_idx, train_end)
        X_tr = recent.iloc[train_slice][["t"]].values
        y_tr = recent.iloc[train_slice]["price_per_sqm"].values
        X_val = recent.iloc[train_end:][["t"]].values
        y_val = recent.iloc[train_end:]["price_per_sqm"].values

        if c["kind"].startswith("linear"):
            lr = LinearRegression()
            if c["halflife"] is not None:
                w = exp_weights(recent.iloc[train_slice]["t"].values.astype(float), float(c["halflife"]))
                lr.fit(X_tr, y_tr, sample_weight=w)
            else:
                lr.fit(X_tr, y_tr)
            preds = lr.predict(X_val)
            rmse = float(np.sqrt(((preds - y_val) ** 2).mean()))
            if rmse < best_rmse:
                best_rmse = rmse
                best = {"cfg": c, "est": lr}
        elif c["kind"].startswith("huber"):
            hb = HuberRegressor()
            hb.fit(X_tr, y_tr)
            preds = hb.predict(X_val)
            rmse = float(np.sqrt(((preds - y_val) ** 2).mean()))
            if rmse < best_rmse:
                best_rmse = rmse
                best = {"cfg": c, "est": hb}
        elif c["kind"].startswith("piecewise"):
            # 在训练片段内搜索断点，使用两段线性，验证集上用第二段线性外推
            min_seg = 6  # 每段至少6个月
            start_idx = max(0, train_end - c["window"]) if c["window"] is not None else 0
            best_local_rmse = float("inf")
            best_break_t = None
            best_lr1 = None
            best_lr2 = None
            for b in range(start_idx + min_seg, train_end - min_seg):
                X1 = recent.iloc[start_idx:b][["t"]].values
                y1 = recent.iloc[start_idx:b]["price_per_sqm"].values
                X2 = recent.iloc[b:train_end][["t"]].values
                y2 = recent.iloc[b:train_end]["price_per_sqm"].values
                if len(X1) < min_seg or len(X2) < min_seg:
                    continue
                lr1 = LinearRegression()
                lr2 = LinearRegression()
                lr1.fit(X1, y1)
                lr2.fit(X2, y2)
                preds = lr2.predict(X_val)
                rmse = float(np.sqrt(((preds - y_val) ** 2).mean()))
                if rmse < best_local_rmse:
                    best_local_rmse = rmse
                    best_break_t = float(recent.iloc[b]["t"])
                    best_lr1 = lr1
                    best_lr2 = lr2
            if best_break_t is not None and best_local_rmse < best_rmse:
                best_rmse = best_local_rmse
                best = {"cfg": {**c, "break_t": best_break_t}, "est": (best_lr1, best_lr2)}

    # 以最优配置在全部可用训练范围上重训（包含验证期之前的所有数据或窗口期）
    assert best is not None
    cfg = best["cfg"]
    if cfg["window"] is None:
        final_start = 0
    else:
        final_start = max(0, len(recent) - cfg["window"])  # 对最终拟合，窗口覆盖到最近
    final_slice = slice(final_start, len(recent))
    X_final = recent.iloc[final_slice][["t"]].values
    y_final = recent.iloc[final_slice]["price_per_sqm"].values

    if cfg["kind"].startswith("linear"):
        model = LinearRegression()
        if cfg["halflife"] is not None:
            w = exp_weights(recent.iloc[final_slice]["t"].values.astype(float), float(cfg["halflife"]))
            model.fit(X_final, y_final, sample_weight=w)
        else:
            model.fit(X_final, y_final)
    elif cfg["kind"].startswith("huber"):
        model = HuberRegressor()
        model.fit(X_final, y_final)
    else:
        # 分段线性：在最终片段内重搜断点并训练两段模型
        min_seg = 6
        start_idx = final_start
        best_local_rmse = float("inf")
        best_break_t = None
        best_lr1 = None
        best_lr2 = None
        # 使用全部 final 片段进行内部交叉验证选择断点（在训练范围内，用简单BIC替代，这里用残差RMSE最小）
        for b in range(start_idx + min_seg, len(recent) - min_seg):
            X1 = recent.iloc[start_idx:b][["t"]].values
            y1 = recent.iloc[start_idx:b]["price_per_sqm"].values
            X2 = recent.iloc[b:len(recent)][["t"]].values
            y2 = recent.iloc[b:len(recent)]["price_per_sqm"].values
            if len(X1) < min_seg or len(X2) < min_seg:
                continue
            lr1 = LinearRegression()
            lr2 = LinearRegression()
            lr1.fit(X1, y1)
            lr2.fit(X2, y2)
            # 在训练范围内用第二段残差作简化评分
            preds2 = lr2.predict(X2)
            rmse2 = float(np.sqrt(((preds2 - y2) ** 2).mean()))
            if rmse2 < best_local_rmse:
                best_local_rmse = rmse2
                best_break_t = float(recent.iloc[b]["t"])
                best_lr1 = lr1
                best_lr2 = lr2
        model = (best_lr1, best_lr2)
        cfg["break_t"] = best_break_t

    # 历史拟合（对整段 recent 可视化）
    if isinstance(model, tuple):
        lr1, lr2 = model
        bt = cfg.get("break_t", recent["t"].iloc[-6])
        t_values = recent["t"].values.reshape(-1, 1)
        historical_fit = np.where(
            recent["t"].values <= bt,
            lr1.predict(t_values).ravel(),
            lr2.predict(t_values).ravel(),
        )
    else:
        historical_fit = model.predict(recent[["t"]])

    # 未来预测：从2026年1月开始
    future_months = years_forward * 12
    future_t = np.arange(len(recent), len(recent) + future_months)
    if isinstance(model, tuple):
        lr1, lr2 = model
        base_pred = lr2.predict(future_t.reshape(-1, 1))
        slope = float(lr2.coef_[0])
        y_last = float(recent["price_per_sqm"].iloc[-1])
        if damped:
            pred = []
            prev = y_last
            for k in range(1, future_months + 1):
                prev = prev + slope * (phi ** k)
                pred.append(prev)
            future_pred = np.array(pred, dtype=float)
        else:
            future_pred = base_pred
    else:
        base_pred = model.predict(future_t.reshape(-1, 1))
        slope = float(model.coef_[0]) if hasattr(model, "coef_") else float((model.predict([[recent["t"].iloc[-1] + 1]]) - model.predict([[recent["t"].iloc[-1]]]))[0])
        y_last = float(recent["price_per_sqm"].iloc[-1])
        if damped:
            pred = []
            prev = y_last
            for k in range(1, future_months + 1):
                prev = prev + slope * (phi ** k)
                pred.append(prev)
            future_pred = np.array(pred, dtype=float)
        else:
            future_pred = base_pred

    # 未来日期从2026年1月开始
    future_start_date = "2026-01-01"
    future_dates = pd.date_range(
        start=future_start_date,
        periods=future_months, 
        freq="MS"
    )

    # 可选：设置远期最低安全边界（按最近12个月分位数）
    if floor_pct is not None:
        tail = recent["price_per_sqm"].tail(12).dropna().values
        if len(tail) > 0:
            floor_val = float(np.percentile(tail, float(floor_pct)))
            future_pred = np.maximum(future_pred, floor_val)

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "predicted_price_per_sqm": future_pred
    })

    # 绘制优化后的趋势图
    plt.figure(figsize=(12, 6))
    
    # 绘制历史实际数据点（到2025年）
    plt.scatter(recent["year_month"], recent["price_per_sqm"], 
                color="blue", alpha=0.6, s=30, label="Historical Data (2015-2025)")
    
    # 绘制历史数据的拟合线（标注所选模型）
    plt.plot(
        recent["year_month"],
        historical_fit,
        color="red",
        linewidth=2,
        label=f"Selected Fit: {cfg['label']}"
    )
    
    # 绘制未来预测线（从2026年开始）
    plt.plot(
        forecast_df["date"],
        forecast_df["predicted_price_per_sqm"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Forecast (2026-2028)"
    )

    # 可选：预测区间（基于训练残差近似，按地平线扩张）
    if ci:
        # 计算训练残差标准差
        if isinstance(model, tuple):
            lr1, lr2 = model
            preds_train = np.where(
                recent["t"].iloc[final_slice].values <= cfg.get("break_t", recent["t"].iloc[-6]),
                lr1.predict(X_final).ravel(),
                lr2.predict(X_final).ravel(),
            )
        else:
            preds_train = model.predict(X_final)
        resid = y_final - preds_train
        sigma = float(np.std(resid, ddof=1)) if len(resid) > 1 else float(np.std(resid))
        # z 值查表（常见置信水平），默认近似 0.9 -> 1.645
        z_map = {0.8: 1.282, 0.85: 1.440, 0.9: 1.645, 0.95: 1.960, 0.98: 2.326, 0.99: 2.576}
        z = float(z_map.get(round(ci_level, 2), 1.645))
        # 随预测步数放大不确定性（sqrt(h/12)）
        h_scale = np.sqrt(np.arange(1, future_months + 1) / 12.0)
        ci_band = z * sigma * h_scale
        ci_lower = future_pred - ci_band
        ci_upper = future_pred + ci_band
        plt.fill_between(
            forecast_df["date"], ci_lower, ci_upper, color="red", alpha=0.12, label=f"{int(ci_level*100)}% CI"
        )

    # 可选：蒙特卡洛模拟 -> 回升概率 + 扇形图
    upturn_prob = None
    if isinstance(mc_sims, int) and mc_sims > 0:
        # 残差标准差（若上面未算过，补算一次）
        if not ci:
            if isinstance(model, tuple):
                lr1, lr2 = model
                preds_train = np.where(
                    recent["t"].iloc[final_slice].values <= cfg.get("break_t", recent["t"].iloc[-6]),
                    lr1.predict(X_final).ravel(),
                    lr2.predict(X_final).ravel(),
                )
            else:
                preds_train = model.predict(X_final)
            resid = y_final - preds_train
            sigma = float(np.std(resid, ddof=1)) if len(resid) > 1 else float(np.std(resid))
        sigma = max(1e-6, float(sigma))

        rng = np.random.default_rng(42)
        y_last = float(recent["price_per_sqm"].iloc[-1])
        slope_ref = float(model[1].coef_[0]) if isinstance(model, tuple) else (
            float(model.coef_[0]) if hasattr(model, "coef_") else 0.0
        )

        sims = np.empty((mc_sims, future_months), dtype=float)
        for i in range(mc_sims):
            prev = y_last
            for k in range(1, future_months + 1):
                step = slope_ref if not damped else slope_ref * (phi ** k)
                noise = rng.normal(0.0, sigma)
                prev = prev + step + noise
                if floor_pct is not None:
                    # 下限在模拟层面也生效
                    tail = recent["price_per_sqm"].tail(12).dropna().values
                    if len(tail) > 0:
                        floor_val = float(np.percentile(tail, float(floor_pct)))
                        if prev < floor_val:
                            prev = floor_val
                sims[i, k - 1] = prev

        # 扇形：10/25/50/75/90 分位
        qs = np.percentile(sims, [10, 25, 50, 75, 90], axis=0)
        q10, q25, q50, q75, q90 = qs
        plt.fill_between(forecast_df["date"], q10, q90, color="green", alpha=0.10, label="MC 80% band")
        plt.fill_between(forecast_df["date"], q25, q75, color="green", alpha=0.15, label="MC 50% band")
        plt.plot(forecast_df["date"], q50, color="green", linestyle=":", linewidth=2, label="MC median")

        # 回升概率：预测 horizon 月价格 >= 当前价格
        h = max(1, min(upturn_horizon, future_months))
        upturn_prob = float(np.mean(sims[:, h - 1] >= y_last))
        plt.axhline(y=y_last, color="#666666", linestyle=":", alpha=0.6)
        plt.text(
            forecast_df["date"].iloc[min(h, len(forecast_df)-1)],
            y_last * 1.005,
            f"Upturn Prob @{h}m: {upturn_prob*100:.1f}%",
            color="#224422",
        )
    
    # 添加当前年份标记线（2025年底）
    current_year_line = pd.Timestamp("2025-12-31")
    plt.axvline(x=current_year_line, color='green', linestyle=':', alpha=0.8, 
                label='Current Time (End of 2025)')
    
    # 添加预测起点标记线（2026年初）
    forecast_start_line = pd.Timestamp("2026-01-01")
    plt.axvline(x=forecast_start_line, color='orange', linestyle=':', alpha=0.6, 
                label='Forecast Start (2026)')
    
    plt.title("Haidian District: House Price Trend Analysis\nHistorical (2015-2025) & Forecast (2026-2028)")
    plt.xlabel("Date")
    plt.ylabel("Average price per sqm (¥)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    # 设置x轴范围，让图表更清晰
    x_min = recent["year_month"].min() - pd.DateOffset(months=6)
    x_max = forecast_df["date"].max() + pd.DateOffset(months=6)
    plt.xlim(x_min, x_max)
    
    plt.tight_layout()
    plt.show()

    # 输出回归模型信息
    if isinstance(model, tuple):
        monthly_change = float(model[1].coef_[0])
    else:
        monthly_change = float(model.coef_[0]) if hasattr(model, "coef_") else float((model.predict([[recent["t"].iloc[-1] + 1]]) - model.predict([[recent["t"].iloc[-1]]]))[0])
    yearly_change = monthly_change * 12
    
    print(f"\n=== 房价趋势分析报告 ===")
    print(f"历史数据期间: {recent['year_month'].min().strftime('%Y-%m')} 到 {recent['year_month'].max().strftime('%Y-%m')}")
    print(f"预测期间: 2026-01 到 2028-12")
    print(f"\n线性回归模型:")
    print(f"月度变化率: {monthly_change:+.2f} ¥/平方米")
    print(f"年度变化率: {yearly_change:+.2f} ¥/平方米")
    print(f"选择的模型: {cfg['label']}")
    # 一些分段模型返回 (model, breakpoint)，需取第一个元素
    if isinstance(model, tuple):
        fitted_model = model[0]
    else:
        fitted_model = model
    try:
        r2 = fitted_model.score(recent[['t']], recent['price_per_sqm'])
        print(f"模型拟合优度 (R²): {r2:.3f}")
    except Exception as e:
        print(f"⚠️ 无法计算R²：{e}")

    
    # 预测总结
    current_avg_price = recent["price_per_sqm"].iloc[-1]  # 2025年底价格
    predicted_2028_price = future_pred[-1]  # 2028年底预测价格
    total_change = predicted_2028_price - current_avg_price
    change_percentage = (total_change / current_avg_price) * 100
    
    print(f"\n价格预测总结:")
    print(f"当前价格 (2025年底): {current_avg_price:,.0f} ¥/平方米")
    print(f"预测价格 (2028年底): {predicted_2028_price:,.0f} ¥/平方米")
    print(f"三年总变化: {total_change:+,.0f} ¥/平方米 ({change_percentage:+.1f}%)")
    print(f"年均变化: {yearly_change:+,.0f} ¥/平方米 ({yearly_change/current_avg_price*100:+.1f}%)")
    return forecast_df

# 主函数
def main_demo():
    print("生成2015-2025年海淀区房价模拟数据...")
    df = generate_synthetic_data()
    print(f"数据时间范围: {df['date'].min().date()} 到 {df['date'].max().date()}")
    print(f"训练样本: {int(len(df)*0.8)}, 测试样本: {int(len(df)*0.2)}")

    results, best_model = train_and_evaluate_models(df)

    print("\n模型评估结果:")
    for k, v in results.items():
        print(f"{k}: RMSE={v[0]:.2f}, MAE={v[1]:.2f}, R2={v[2]:.3f}")
    print(f"\n最佳模型 (按RMSE): {best_model}")

    # 预测未来房价趋势
    forecast_years = 3
    print(f"\n基于2015-2025年历史数据，预测2026-2028年房价趋势...")
    forecast_df = forecast_future_prices(df, years_forward=forecast_years)
    print(f"\n未来三年价格预测:")
    print(forecast_df.head(6))  # 显示前6个月预测

    print(f"\n演示完成。当前分析基于真实的2025年时间线。")


# 入口
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run demo with synthetic data")
    parser.add_argument("--file", type=str, help="Path to real CSV data (date, price_per_sqm required)")
    parser.add_argument("--excel", type=str, help="Path to Excel file with Haidian housing data")  # ✅ 新增
    parser.add_argument("--use-builtin", action="store_true", help="Use builtin monthly CSV data")
    parser.add_argument("--monthly-file", type=str, help="Path to monthly CSV (year_month, avg_price_per_sqm)")
    parser.add_argument("--no-piecewise", action="store_true", help="禁用分段线性候选")
    parser.add_argument("--no-damped", action="store_true", help="禁用阻尼外推")
    parser.add_argument("--phi", type=float, default=0.9, help="阻尼外推参数phi，默认0.9")
    parser.add_argument("--smooth-ma", type=int, default=None, help="移动平均窗口（整数月），如 3/6")
    parser.add_argument("--ci", action="store_true", help="绘制预测区间带")
    parser.add_argument("--ci-level", type=float, default=0.9, help="预测区间置信水平，默认0.9")
    parser.add_argument("--floor-pct", type=float, default=None, help="设置未来价格不低于最近12月的该分位数，如 10 表示P10")
    parser.add_argument("--mc-sims", type=int, default=0, help="蒙特卡洛模拟次数，>0 则绘制扇形并计算回升概率")
    parser.add_argument("--upturn-horizon", type=int, default=12, help="回升概率评估的月份，默认 12")
    args = parser.parse_args()

    # ---------------------
    # 模式1：演示模式（使用模拟数据）
    # ---------------------
    if args.demo:
        main_demo()

    # ---------------------
    # 模式2：CSV 文件（包含 date, price_per_sqm）
    # ---------------------
    elif args.file:
        df = pd.read_csv(args.file, parse_dates=["date"])
        results, best_model = train_and_evaluate_models(df)
        forecast_future_prices(
            df,
            piecewise=not args.no_piecewise,
            damped=not args.no_damped,
            phi=args.phi,
            smooth_ma=args.smooth_ma,
            ci=args.ci,
            ci_level=args.ci_level,
            floor_pct=args.floor_pct,
            mc_sims=args.mc_sims,
            upturn_horizon=args.upturn_horizon,
        )

        # ---------------------
    # ✅ 模式3：Excel 文件（如 haidianfangjia.xlsx）
    # ---------------------
    elif args.excel:
        def load_real_excel_data(excel_path: str) -> pd.DataFrame:
            """从Excel加载海淀房价数据（A列=年月, K列=每平米价格，日期格式形如2025.08.01）"""
            df = pd.read_excel(excel_path)
            
            # 明确A列和K列
            date_col = df.columns[0]   # 假定A列为年月
            price_col = df.columns[10] # 假定K列为每平方米价格（从0计数）

            # ---- 🔧 日期解析 ----
            df["date_str"] = df[date_col].astype(str).str.strip()
            # 将 "2025.08.01" 转换成 pandas 可识别格式
            df["date_str"] = df["date_str"].str.replace(".", "-", regex=False)
            df["date"] = pd.to_datetime(df["date_str"], format="%Y-%m-%d", errors="coerce")
            df = df.dropna(subset=["date"])

            # ---- 🔢 价格列 ----
            df["price_per_sqm"] = pd.to_numeric(df[price_col], errors="coerce")
            df = df.dropna(subset=["price_per_sqm"])

            # ---- 📆 按月聚合 ----
            df["year_month"] = df["date"].dt.to_period("M").dt.to_timestamp()
            monthly = df.groupby("year_month")["price_per_sqm"].mean().reset_index()
            monthly = monthly.rename(columns={"year_month": "date"})

            print(f"✅ 成功加载Excel数据，共 {len(monthly)} 个月")
            if len(monthly) > 0:
                print(f"时间范围：{monthly['date'].min().date()} - {monthly['date'].max().date()}")
                print("前5个月样例：")
                print(monthly.head())
            else:
                print("⚠️ 未解析出有效数据，请检查A列是否为日期，K列是否为价格")
            return monthly

        excel_df = load_real_excel_data(args.excel)
        if len(excel_df) == 0:
            print("❌ 无法继续：未从Excel中读取到有效数据。")
        else:
            forecast_future_prices(
                excel_df,
                piecewise=not args.no_piecewise,
                damped=not args.no_damped,
                phi=args.phi,
                smooth_ma=args.smooth_ma,
                ci=args.ci,
                ci_level=args.ci_level,
                floor_pct=args.floor_pct,
                mc_sims=args.mc_sims,
                upturn_horizon=args.upturn_horizon,
            )


    # ---------------------
    # 模式4：内置月度CSV
    # ---------------------
    elif args.use_builtin or args.monthly_file:
        csv_path = args.monthly_file
        if args.use_builtin and not csv_path:
            csv_path = os.path.join(os.path.dirname(__file__), "data", "haidian_monthly_prices_2015_2025.csv")
        monthly_df = load_monthly_prices(csv_path)
        monthly_df = monthly_df.sort_values("date")
        monthly_df["year_month"] = monthly_df["date"]
        df_for_forecast = monthly_df.rename(columns={"price_per_sqm": "price_per_sqm"})
        forecast_future_prices(
            df_for_forecast,
            piecewise=not args.no_piecewise,
            damped=not args.no_damped,
            phi=args.phi,
            smooth_ma=args.smooth_ma,
            ci=args.ci,
            ci_level=args.ci_level,
            floor_pct=args.floor_pct,
            mc_sims=args.mc_sims,
            upturn_horizon=args.upturn_horizon,
        )

    else:
        print("未提供数据源。请使用以下任一方式运行：")
        print("  python haidian_price_pipeline_v3.py --demo")
        print("  python haidian_price_pipeline_v3.py --excel haidianfangjia.xlsx")
        print("  python haidian_price_pipeline_v3.py --file data.csv")