import os

from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__)


def lcm_random(n, seed=123):
    a = 1664525
    c = 1013904223
    m = 2 ** 32
    random_numbers = []
    zi = seed
    for _ in range(n):
        zi = (a * zi + c) % m
        scaled_num = int((zi / m) * 1000)
        random_numbers.append(scaled_num)
    return random_numbers


def load_and_prepare_data():
    df = pd.read_csv("crop_production.csv")
    df_rice = df[df["Crop"] == "Rice"].copy()
    grouped = (
        df_rice.groupby("Crop_Year")["Production"]
        .sum()
        .reset_index()
        .sort_values("Crop_Year")
    )
    base = grouped.tail(10).reset_index(drop=True)
    total_prod = base["Production"].sum()
    base["Probability"] = base["Production"] / total_prod
    base["Cumulative_Probability"] = base["Probability"].cumsum()

    # Intervals in 0-1000 range
    lower_limits = []
    upper_limits = []
    prev_upper = 0
    for cp in base["Cumulative_Probability"]:
        upper = int(round(cp * 1000))
        lower_limits.append(prev_upper)
        upper_limits.append(upper - 1 if upper > 0 else 0)
        prev_upper = upper

    # Ensure last upper is 999
    if upper_limits:
        upper_limits[-1] = 999

    base["Lower_Limit"] = lower_limits
    base["Upper_Limit"] = upper_limits
    return base


def map_random_to_production(interval_df, random_numbers):
    results = []
    years = interval_df["Crop_Year"].tolist()
    productions = interval_df["Production"].tolist()
    lowers = interval_df["Lower_Limit"].tolist()
    uppers = interval_df["Upper_Limit"].tolist()

    for idx, rn in enumerate(random_numbers, start=1):
        prod_value = None
        ref_year = None
        for year, prod, low, up in zip(years, productions, lowers, uppers):
            if low <= rn <= up:
                prod_value = prod
                ref_year = year
                break
        results.append(
            {
                "year_index": idx,
                "random_number": rn,
                "predicted_production": float(prod_value) if prod_value is not None else None,
                "reference_year": int(ref_year) if ref_year is not None else None,
            }
        )

    return results


base_data_cache = None


@app.before_request
def ensure_base_data_loaded():
    global base_data_cache
    if base_data_cache is None:
        base_data_cache = load_and_prepare_data()


@app.route("/", methods=["GET", "POST"])
def index():
    global base_data_cache
    prediction_results = []
    avg_predicted = None
    years_to_predict = None

    if request.method == "POST":
        try:
            years_to_predict = int(request.form.get("years_to_predict", "0").strip())
        except (ValueError, AttributeError):
            years_to_predict = 0

        if years_to_predict > 0:
            random_numbers = lcm_random(years_to_predict)
            prediction_results = map_random_to_production(base_data_cache, random_numbers)
            valid_values = [r["predicted_production"] for r in prediction_results if r["predicted_production"] is not None]
            if valid_values:
                avg_predicted = sum(valid_values) / len(valid_values)

    base_json = base_data_cache.to_dict(orient="records") if base_data_cache is not None else []

    return render_template(
        "index.html",
        base_data=base_json,
        prediction_results=prediction_results,
        avg_predicted=avg_predicted,
        years_to_predict=years_to_predict,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    global base_data_cache
    data = request.get_json(silent=True) or {}
    years_to_predict = int(data.get("years_to_predict", 0))
    if years_to_predict <= 0:
        return jsonify({"error": "Nilai years_to_predict harus lebih dari 0"}), 400

    random_numbers = lcm_random(years_to_predict)
    prediction_results = map_random_to_production(base_data_cache, random_numbers)
    valid_values = [r["predicted_production"] for r in prediction_results if r["predicted_production"] is not None]
    avg_predicted = sum(valid_values) / len(valid_values) if valid_values else None

    return jsonify(
        {
            "results": prediction_results,
            "average_predicted": avg_predicted,
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
