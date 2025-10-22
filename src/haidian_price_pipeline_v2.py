import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import argparse
from datetime import datetime

# -----------------------------
# 数据生成（模拟海淀区房价数据）- 精确控制时间范围
# -----------------------------
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
    
    # 精确控制日期范围：2015-01-01 到 2025-12-31
    start_date = "2015-01-01"
    end_date = "2025-12-31"
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
            "year_built": year_built,
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

# -----------------------------
# 模型训练与评估函数
# -----------------------------
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

# -----------------------------
# 优化后的时间序列房价预测函数 - 精确控制预测起点
# -----------------------------
def forecast_future_prices(df, years_forward=3):
    data = df.copy()
    data = data.sort_values("date")
    data["year_month"] = data["date"].dt.to_period("M").dt.to_timestamp()
    monthly = data.groupby("year_month")["price_per_sqm"].mean().reset_index()

    # 确保历史数据截止到2025年
    historical_cutoff = "2025-12-31"
    recent = monthly[monthly["year_month"] <= historical_cutoff]
    
    # 如果数据不足，使用所有可用数据
    if len(recent) < 12:
        recent = monthly[monthly["year_month"] <= historical_cutoff]
    
    recent["t"] = np.arange(len(recent))

    model = LinearRegression()
    model.fit(recent[["t"]], recent["price_per_sqm"])

    # 计算历史数据的拟合值
    historical_fit = model.predict(recent[["t"]])

    # 未来预测：从2026年1月开始
    future_months = years_forward * 12
    future_t = np.arange(len(recent), len(recent) + future_months)
    future_pred = model.predict(future_t.reshape(-1, 1))

    # 未来日期从2026年1月开始
    future_start_date = "2026-01-01"
    future_dates = pd.date_range(
        start=future_start_date,
        periods=future_months, 
        freq="MS"
    )

    forecast_df = pd.DataFrame({
        "date": future_dates,
        "predicted_price_per_sqm": future_pred
    })

    # 绘制优化后的趋势图
    plt.figure(figsize=(12, 6))
    
    # 绘制历史实际数据点（到2025年）
    plt.scatter(recent["year_month"], recent["price_per_sqm"], 
                color="blue", alpha=0.6, s=30, label="Historical Data (2015-2025)")
    
    # 绘制历史数据的线性拟合线
    plt.plot(recent["year_month"], historical_fit, 
             color="red", linewidth=2, label="Linear Regression Fit")
    
    # 绘制未来预测线（从2026年开始）
    plt.plot(forecast_df["date"], forecast_df["predicted_price_per_sqm"], 
             color="red", linestyle="--", linewidth=2, label=f"Forecast (2026-2028)")
    
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
    
    # 设置x轴范围，让图表更清晰
    x_min = recent["year_month"].min() - pd.DateOffset(months=6)
    x_max = forecast_df["date"].max() + pd.DateOffset(months=6)
    plt.xlim(x_min, x_max)
    
    plt.tight_layout()
    plt.show()

    # 输出回归模型信息
    monthly_change = model.coef_[0]
    yearly_change = monthly_change * 12
    
    print(f"\n=== 房价趋势分析报告 ===")
    print(f"历史数据期间: {recent['year_month'].min().strftime('%Y-%m')} 到 {recent['year_month'].max().strftime('%Y-%m')}")
    print(f"预测期间: 2026-01 到 2028-12")
    print(f"\n线性回归模型:")
    print(f"月度变化率: {monthly_change:+.2f} ¥/平方米")
    print(f"年度变化率: {yearly_change:+.2f} ¥/平方米")
    print(f"模型拟合优度 (R²): {model.score(recent[['t']], recent['price_per_sqm']):.3f}")
    
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

# -----------------------------
# 主函数
# -----------------------------
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

# -----------------------------
# 入口
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run demo with synthetic data")
    parser.add_argument("--file", type=str, help="Path to real CSV data (date, price_per_sqm required)")
    args = parser.parse_args()

    if args.demo:
        main_demo()
    elif args.file:
        df = pd.read_csv(args.file, parse_dates=["date"])
        results, best_model = train_and_evaluate_models(df)
        forecast_future_prices(df)
    else:
        print("No arguments provided. Run with --demo to execute demo or --file <csv>")