from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "datasets" / "02_bike_sharing_dataset" / "hour.csv"
CHART_DIR = BASE_DIR / "图表输出"
OUTPUT_DIR = BASE_DIR / "数据输出"

CHART_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False


def save_chart(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(CHART_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close()


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    df["日期"] = pd.to_datetime(df["dteday"])
    df["月份"] = df["日期"].dt.to_period("M").astype(str)
    df["日期_日"] = df["日期"].dt.date
    df["日期序号"] = (df["日期"] - df["日期"].min()).dt.days

    season_map = {1: "春季", 2: "夏季", 3: "秋季", 4: "冬季"}
    weather_map = {
        1: "晴天/少云",
        2: "多云/薄雾",
        3: "小雨/小雪",
        4: "恶劣天气",
    }
    weekday_map = {0: "周日", 1: "周一", 2: "周二", 3: "周三", 4: "周四", 5: "周五", 6: "周六"}

    df["季节"] = df["season"].map(season_map)
    df["天气"] = df["weathersit"].map(weather_map)
    df["星期"] = df["weekday"].map(weekday_map)
    df["是否工作日"] = np.where(df["workingday"] == 1, "工作日", "非工作日")
    df["是否高峰小时"] = df["hr"].isin([7, 8, 17, 18]).astype(int)
    df["小时_sin"] = np.sin(2 * np.pi * df["hr"] / 24)
    df["小时_cos"] = np.cos(2 * np.pi * df["hr"] / 24)
    df["温度_摄氏度"] = df["temp"] * 41
    df["体感温度_摄氏度"] = df["atemp"] * 50
    df["湿度"] = df["hum"] * 100
    df["风速"] = df["windspeed"] * 67

    overview = pd.DataFrame(
        {
            "指标": [
                "记录数",
                "日期范围",
                "总租赁量",
                "日均租赁量",
                "小时均租赁量",
                "注册用户租赁占比",
                "非注册用户租赁占比",
                "最高小时租赁量",
            ],
            "数值": [
                len(df),
                f"{df['日期'].min().date()} 至 {df['日期'].max().date()}",
                int(df["cnt"].sum()),
                round(df.groupby("日期_日")["cnt"].sum().mean(), 2),
                round(df["cnt"].mean(), 2),
                f"{df['registered'].sum() / df['cnt'].sum():.2%}",
                f"{df['casual'].sum() / df['cnt'].sum():.2%}",
                int(df["cnt"].max()),
            ],
        }
    )
    overview.to_csv(OUTPUT_DIR / "项目基础概览.csv", index=False, encoding="utf-8-sig")

    monthly = df.groupby("月份", as_index=False)["cnt"].sum()
    hourly = df.groupby("hr", as_index=False)["cnt"].mean()
    weekday_hour = df.pivot_table(index="hr", columns="是否工作日", values="cnt", aggfunc="mean")
    weather = df.groupby("天气", as_index=False)["cnt"].mean().sort_values("cnt", ascending=False)
    season = df.groupby("季节", as_index=False)["cnt"].mean().sort_values("cnt", ascending=False)
    corr_cols = ["cnt", "温度_摄氏度", "体感温度_摄氏度", "湿度", "风速"]
    corr = df[corr_cols].corr()

    monthly.to_csv(OUTPUT_DIR / "月度租赁量.csv", index=False, encoding="utf-8-sig")
    hourly.to_csv(OUTPUT_DIR / "小时平均租赁量.csv", index=False, encoding="utf-8-sig")
    weather.to_csv(OUTPUT_DIR / "天气平均租赁量.csv", index=False, encoding="utf-8-sig")
    season.to_csv(OUTPUT_DIR / "季节平均租赁量.csv", index=False, encoding="utf-8-sig")
    corr.to_csv(OUTPUT_DIR / "相关系数矩阵.csv", encoding="utf-8-sig")

    plt.figure(figsize=(12, 5))
    sns.lineplot(data=monthly, x="月份", y="cnt", marker="o", color="#2563eb")
    plt.xticks(rotation=45)
    plt.title("月度租赁量趋势")
    plt.xlabel("月份")
    plt.ylabel("总租赁量")
    save_chart("01_月度租赁量趋势.png")

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=hourly, x="hr", y="cnt", marker="o", color="#0f766e")
    plt.xticks(range(0, 24))
    plt.title("24小时平均租赁量")
    plt.xlabel("小时")
    plt.ylabel("平均租赁量")
    save_chart("02_24小时平均租赁量.png")

    plt.figure(figsize=(10, 5))
    weekday_hour.plot(kind="line", marker="o", figsize=(10, 5))
    plt.xticks(range(0, 24))
    plt.title("工作日与非工作日的小时需求差异")
    plt.xlabel("小时")
    plt.ylabel("平均租赁量")
    save_chart("03_工作日与非工作日小时差异.png")

    plt.figure(figsize=(9, 5))
    sns.barplot(data=weather, x="天气", y="cnt", hue="天气", palette="Set2", legend=False)
    plt.title("不同天气下的平均租赁量")
    plt.xlabel("天气")
    plt.ylabel("平均租赁量")
    save_chart("04_天气平均租赁量.png")

    plt.figure(figsize=(8, 5))
    sns.barplot(data=season, x="季节", y="cnt", hue="季节", palette="Set3", legend=False)
    plt.title("不同季节的平均租赁量")
    plt.xlabel("季节")
    plt.ylabel("平均租赁量")
    save_chart("05_季节平均租赁量.png")

    sample = df.sample(n=min(2500, len(df)), random_state=42)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=sample, x="温度_摄氏度", y="cnt", alpha=0.35, color="#dc2626")
    sns.regplot(data=sample, x="温度_摄氏度", y="cnt", scatter=False, color="#111827")
    plt.title("温度与租赁量关系")
    plt.xlabel("温度（摄氏度）")
    plt.ylabel("租赁量")
    save_chart("06_温度与租赁量关系.png")

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="RdBu_r", center=0, fmt=".2f")
    plt.title("核心数值变量相关系数")
    save_chart("07_相关系数热力图.png")

    # 简单可解释的基线预测：用历史同小时、同工作日类型、同天气的平均需求预测。
    train = df[df["日期"] < pd.Timestamp("2012-10-01")].copy()
    test = df[df["日期"] >= pd.Timestamp("2012-10-01")].copy()
    key_cols = ["hr", "workingday", "weathersit", "season"]
    group_mean = train.groupby(key_cols)["cnt"].mean().rename("预测租赁量").reset_index()
    global_mean = train["cnt"].mean()
    pred = test.merge(group_mean, on=key_cols, how="left")
    pred["预测租赁量"] = pred["预测租赁量"].fillna(global_mean)
    baseline_metrics = regression_metrics(pred["cnt"], pred["预测租赁量"].to_numpy())

    feature_cols = [
        "season",
        "yr",
        "mnth",
        "hr",
        "holiday",
        "weekday",
        "workingday",
        "weathersit",
        "日期序号",
        "是否高峰小时",
        "小时_sin",
        "小时_cos",
        "temp",
        "atemp",
        "hum",
        "windspeed",
    ]
    categorical_cols = ["season", "yr", "mnth", "hr", "holiday", "weekday", "workingday", "weathersit", "是否高峰小时"]
    numeric_cols = ["日期序号", "小时_sin", "小时_cos", "temp", "atemp", "hum", "windspeed"]

    x_train = train[feature_cols]
    y_train = train["cnt"]
    x_test = test[feature_cols]
    y_test = test["cnt"]

    linear_preprocess = ColumnTransformer(
        transformers=[
            ("category", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("number", StandardScaler(), numeric_cols),
        ]
    )
    tree_preprocess = ColumnTransformer(
        transformers=[
            ("category", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("number", "passthrough", numeric_cols),
        ]
    )

    linear_model = Pipeline(
        steps=[
            ("preprocess", linear_preprocess),
            ("model", LinearRegression()),
        ]
    )
    rf_model = Pipeline(
        steps=[
            ("preprocess", tree_preprocess),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=120,
                    max_depth=20,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )

    linear_model.fit(x_train, y_train)
    rf_model.fit(x_train, y_train)

    pred["线性回归预测"] = linear_model.predict(x_test)
    pred["随机森林预测"] = rf_model.predict(x_test)

    model_rows = []
    for name, values in {
        "历史均值基线": pred["预测租赁量"].to_numpy(),
        "线性回归": pred["线性回归预测"].to_numpy(),
        "随机森林": pred["随机森林预测"].to_numpy(),
    }.items():
        metric = regression_metrics(y_test, values)
        model_rows.append({"模型": name, **metric})

    metrics_df = pd.DataFrame(model_rows)
    metrics_df[["MAE", "RMSE", "R2"]] = metrics_df[["MAE", "RMSE", "R2"]].round(4)
    metrics_df.to_csv(OUTPUT_DIR / "模型评估对比.csv", index=False, encoding="utf-8-sig")

    encoded_feature_names = rf_model.named_steps["preprocess"].get_feature_names_out()
    importances = rf_model.named_steps["model"].feature_importances_
    feature_importance = (
        pd.DataFrame({"特征": encoded_feature_names, "重要性": importances})
        .sort_values("重要性", ascending=False)
        .head(20)
    )
    feature_importance.to_csv(OUTPUT_DIR / "随机森林特征重要性Top20.csv", index=False, encoding="utf-8-sig")

    pred_output = pred[["日期", "hr", "cnt", "预测租赁量", "线性回归预测", "随机森林预测", "季节", "天气", "是否工作日"]].copy()
    for col in ["预测租赁量", "线性回归预测", "随机森林预测"]:
        pred_output[col] = pred_output[col].round(2)
    pred_output.to_csv(OUTPUT_DIR / "测试集预测结果.csv", index=False, encoding="utf-8-sig")

    day_pred = pred.assign(日=pred["日期"].dt.date).groupby("日", as_index=False)[["cnt", "预测租赁量", "随机森林预测"]].sum()
    plt.figure(figsize=(12, 5))
    plt.plot(day_pred["日"], day_pred["cnt"], label="实际租赁量", color="#2563eb")
    plt.plot(day_pred["日"], day_pred["预测租赁量"], label="历史均值基线", color="#9ca3af")
    plt.plot(day_pred["日"], day_pred["随机森林预测"], label="随机森林预测", color="#f97316")
    plt.xticks(rotation=45)
    plt.title("测试集日度实际值与预测值对比")
    plt.xlabel("日期")
    plt.ylabel("租赁量")
    plt.legend()
    save_chart("08_预测值与实际值对比.png")

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x="重要性", y="特征", color="#2563eb")
    plt.title("随机森林特征重要性 Top20")
    plt.xlabel("重要性")
    plt.ylabel("特征")
    save_chart("09_随机森林特征重要性Top20.png")

    best_row = metrics_df.sort_values("RMSE").iloc[0]

    summary_lines = [
        "共享单车需求预测与运营分析 - 自动分析摘要",
        f"数据时间范围：{df['日期'].min().date()} 至 {df['日期'].max().date()}",
        f"总记录数：{len(df):,}",
        f"总租赁量：{int(df['cnt'].sum()):,}",
        f"小时平均租赁量：{df['cnt'].mean():.2f}",
        f"租赁量最高小时：{int(df.loc[df['cnt'].idxmax(), 'cnt'])}",
        f"平均需求最高时段：{int(hourly.loc[hourly['cnt'].idxmax(), 'hr'])} 点",
        f"平均需求最高天气：{weather.iloc[0]['天气']}",
        f"平均需求最高季节：{season.iloc[0]['季节']}",
        f"表现最好模型：{best_row['模型']}",
        f"最好模型 MAE：{best_row['MAE']:.2f}",
        f"最好模型 RMSE：{best_row['RMSE']:.2f}",
        f"最好模型 R2：{best_row['R2']:.4f}",
    ]
    (OUTPUT_DIR / "自动分析摘要.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()
