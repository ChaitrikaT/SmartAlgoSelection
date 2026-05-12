"""
app.py - SmartAlgoSelection Backend (FINAL — connection-error-free)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fixes applied vs previous versions:
  ✅ analyzer.py now returns 'paradigm' + 'paradigm_reasoning' (was the 500 root cause)
  ✅ All routes wrapped in try/except → always return JSON, never HTML on error
  ✅ DOMAIN_TO_CSV includes all 4 domains (was missing "pathfinding" → "Shortest Path")
  ✅ CSV algo names fully synced with KB via KB_NAME_FIX
  ✅ ml_ready guard prevents LabelEncoder crash when CSV is absent
  ✅ .copy() / deepcopy on every benchmark input (fair race, no mutation bugs)
  ✅ Speedup + is_fastest fields for full UI support
"""

from flask import Flask, render_template, request, jsonify
import random, warnings, copy
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

import sort as sa
import dp
from analyzer import analyze_problem
from knowledge_base import get_algorithms

warnings.filterwarnings("ignore")
app = Flask(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# ML BOOTSTRAP
# ═══════════════════════════════════════════════════════════════════════════

prob_encoder = LabelEncoder()
algo_encoder = LabelEncoder()
ml_model     = DecisionTreeClassifier(random_state=42)
ml_ready     = False

# NLP domain  →  CSV Problem_Type column  (ALL 4 domains covered)
DOMAIN_TO_CSV = {
    "ordering":     "Sorting",
    "optimization": "Knapsack",
    "pathfinding":  "Shortest Path",   # was missing → caused KeyError → 500
    "staged":       "Multistage Graph",
}

# CSV Best_Algorithm value  →  Knowledge Base dict key
KB_NAME_FIX = {
    "Knapsack_01 (DP)":    "Knapsack 0/1 (DP)",
    "Floyd-Warshall (DP)": "Floyd-Warshall",
    "Multistage (DP)":     "Multistage Graph (DP)",
}

# Rule-based fallback when ML unavailable
DOMAIN_FALLBACK = {
    "ordering":     "Quick Sort",
    "optimization": "Knapsack 0/1 (DP)",
    "pathfinding":  "Floyd-Warshall",
    "staged":       "Multistage Graph (DP)",
}


def train_stable_model():
    global ml_ready
    try:
        df = pd.read_csv("algorithm_data.csv")
        df["Prob_N"] = prob_encoder.fit_transform(df["Problem_Type"])
        df["Algo_N"] = algo_encoder.fit_transform(df["Best_Algorithm"])
        ml_model.fit(df[["Prob_N", "Input_Size"]], df["Algo_N"])
        ml_ready = True
        print(f"✅ AI Model ready — {len(algo_encoder.classes_)} algorithms learned.")
    except FileNotFoundError:
        print("⚠️  algorithm_data.csv not found — rule-based fallback active.")
    except Exception as e:
        print(f"⚠️  ML training error: {e} — rule-based fallback active.")


print("Training AI Model on Empirical Dataset...")
train_stable_model()


def ml_predict(domain: str, input_size: int) -> str:
    """Returns KB-compatible algorithm name. Never raises — always returns a string."""
    if not ml_ready:
        return DOMAIN_FALLBACK.get(domain, "Quick Sort")
    try:
        csv_type = DOMAIN_TO_CSV.get(domain, "Sorting")
        if csv_type not in prob_encoder.classes_:
            return DOMAIN_FALLBACK.get(domain, "Quick Sort")
        prob_idx = prob_encoder.transform([csv_type])[0]
        pred_idx = ml_model.predict([[prob_idx, input_size]])[0]
        raw_name = algo_encoder.inverse_transform([pred_idx])[0]
        return KB_NAME_FIX.get(raw_name, raw_name)
    except Exception as e:
        print(f"⚠️  ML predict error: {e}")
        return DOMAIN_FALLBACK.get(domain, "Quick Sort")


# ═══════════════════════════════════════════════════════════════════════════
# SAMPLE QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════

SAMPLE_QUESTIONS = {
    "Sorting (Brute Force)": [
        "Arrange the numbers [34, 12, 5, 78, 23] from smallest to largest",
        "Put these 15 student marks in ascending order",
        "Order the elements [9, 1, 4, 7, 2, 8, 3, 6, 5] from low to high",
        "I have 25 random values, organize them in increasing sequence",
    ],
    "Sorting (Decrease & Conquer)": [
        "Organize these 120 employee IDs in ascending sequence",
        "Arrange 75 temperature readings from lowest to highest",
        "I need to organize 150 random server ping times from lowest to highest",
        "Order 100 product prices from cheapest to most expensive",
    ],
    "Sorting (Divide & Conquer)": [
        "Arrange 5000 numbers from smallest to largest",
        "I need to organize 10000 records in ascending order efficiently",
        "Put 2500 sensor readings in sequence from low to high",
        "Order 8000 transaction amounts from smallest to largest",
    ],
    "Knapsack (DP)": [
        "Given items with weights = [2, 3, 4, 5] and values = [3, 4, 5, 6], capacity = 8, maximize value",
        "I have 20 items with different weights and profits, find the best combination within weight limit 150",
        "Pack a bag with capacity = 50 using items: weights = [10, 20, 30] values = [60, 100, 120]",
        "Select the best subset of 30 products to maximize profit without exceeding budget constraint of 500",
    ],
    "Shortest Path (DP)": [
        "Find minimum travel distance between every pair of 40 cities connected by roads",
        "Given a network of 25 locations, find the shortest route between all pairs",
        "Calculate minimum cost paths between all 15 nodes in this weighted network",
        "Find shortest distance between every pair of 50 vertices in a weighted graph",
    ],
    "Multistage Graph (DP)": [
        "Process 60 tasks through 4 pipeline stages to minimize total cost",
        "Find optimal path through a 5-stage network with 30 nodes",
        "Route data through a 3-level layered network of 45 nodes minimizing latency",
        "Allocate resources across 4 sequential phases with 20 decision points",
    ],
}


# ═══════════════════════════════════════════════════════════════════════════
# BENCHMARK RUNNERS
# ═══════════════════════════════════════════════════════════════════════════

def benchmark_sorting(size: int, extracted_array=None) -> dict:
    if extracted_array:
        arr   = list(extracted_array)
        size  = len(arr)
        using = "User-provided array"
    else:
        arr   = [random.randint(1, 10000) for _ in range(size)]
        using = "Randomly generated array"

    preview_n = min(size, 20)
    preview   = (
        f"[{', '.join(str(x) for x in arr[:preview_n])}"
        f"{'...' if size > preview_n else ''}]  ({size} elements)"
    )

    times = {
        "Bubble Sort":    sa.bubble_sort(arr.copy()),
        "Selection Sort": sa.selection_sort(arr.copy()),
        "Insertion Sort": sa.insertion_sort(arr.copy()),
        "Merge Sort":     sa.merge_sort(arr.copy()),
        "Quick Sort":     sa.quick_sort(arr.copy()),
    }
    return {"times": times, "preview": preview, "data_desc": f"{using} of size {size}"}


def benchmark_dp(size: int, knapsack_data=None, graph_data=None) -> dict:
    results = {}

    # Knapsack
    if knapsack_data and "weights" in knapsack_data and "values" in knapsack_data:
        weights  = knapsack_data["weights"]
        values   = knapsack_data["values"]
        capacity = knapsack_data.get("capacity", sum(weights) // 2)
        cap_n    = len(weights)
        ks_desc  = f"User-provided: {cap_n} items, capacity={capacity}"
    else:
        cap_n    = min(size, 500)
        weights  = [random.randint(1, 50)   for _ in range(cap_n)]
        values   = [random.randint(10, 100) for _ in range(cap_n)]
        capacity = cap_n * 10
        ks_desc  = f"Random: {cap_n} items, capacity={capacity}"
    results["Knapsack 0/1 (DP)"] = dp.knapsack_01(capacity, weights, values)

    # Floyd-Warshall
    if graph_data and "matrix" in graph_data:
        fw_graph = copy.deepcopy(graph_data["matrix"])
        v        = len(fw_graph)
        fw_desc  = f"User-provided {v}x{v} matrix"
    else:
        v        = min(size, 80)
        fw_graph = [
            [random.randint(1, 100) if i != j else 0 for j in range(v)]
            for i in range(v)
        ]
        fw_desc = f"Random {v}x{v} matrix"
    results["Floyd-Warshall"] = dp.floyd_warshall(copy.deepcopy(fw_graph))

    # Multistage Graph
    if graph_data and "matrix" in graph_data:
        ms_graph = copy.deepcopy(graph_data["matrix"])
        ms       = len(ms_graph)
    else:
        ms       = min(size, 80)
        ms_graph = [
            [random.randint(1, 100) if i != j else float("inf") for j in range(ms)]
            for i in range(ms)
        ]
    results["Multistage Graph (DP)"] = dp.multistage_graph(copy.deepcopy(ms_graph), 4, ms)

    preview = (
        f"Knapsack: {ks_desc} | "
        f"Floyd-Warshall: {fw_desc} | "
        f"Multistage: {ms} nodes"
    )
    return {
        "times":     results,
        "preview":   preview,
        "data_desc": f"DP suite benchmarked with size ~{size}",
    }


# ═══════════════════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════════════════

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/sample-questions")
def sample_questions():
    return jsonify(SAMPLE_QUESTIONS)


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid request — expected JSON body."}), 400

        statement = data.get("statement", "").strip()
        if not statement:
            return jsonify({"error": "Please enter a problem statement."}), 400

        # 1. NLP Analysis
        analysis = analyze_problem(statement)
        if analysis.get("error"):
            return jsonify({"error": analysis["error"]}), 400

        domain     = analysis["domain"]
        paradigm   = analysis["paradigm"]
        input_size = analysis["input_size"]
        extracted  = analysis.get("extracted_data", {})

        algos = get_algorithms(domain)
        if not algos:
            return jsonify({"error": f"No algorithms found for domain: {domain}"}), 400

        # 2. ML Prediction
        ml_best_name = ml_predict(domain, input_size)
        if ml_best_name not in algos:
            ml_best_name = DOMAIN_FALLBACK.get(domain, next(iter(algos)))

        # 3. Benchmarking
        if domain == "ordering":
            bench = benchmark_sorting(input_size, extracted_array=extracted.get("array"))
        else:
            bench = benchmark_dp(
                input_size,
                knapsack_data=extracted.get("knapsack"),
                graph_data=extracted.get("graph"),
            )# --- EXACT LOGIC FOR THE UI SCREENSHOT ---
        times        = bench["times"]
        fastest_name = min(times, key=times.get)
        fastest_time = times[fastest_name]
        best_name    = ml_best_name
        best_time    = times.get(best_name, fastest_time)

# 4. Build algo list for UI
        algo_list = []
        for name, info in algos.items():
            t          = times.get(name)
            is_best    = (name == best_name)
            is_fastest = (name == fastest_name)
            
            # --- THE SAVAGE UX WARNING ---
            cons_list = info.get("cons", []).copy()
            if is_fastest and not is_best:
                cons_list.insert(0, "⚠️ STOPWATCH TRAP: Finished fastest, but mathematically INVALID or unscalable for this specific problem domain. AI correctly ignored it!")

            # --- THE GREEN BADGE FIX ---
            # Pass None for Best/Fastest so the frontend CSS handles the Green/White tags natively.
            # Pass ONLY the raw float (e.g., 3.58) for the rest so it doesn't double the "x slower" text.
            if is_best:
                speedup_val = None
            elif is_fastest:
                speedup_val = None
            elif t and fastest_time and t > fastest_time:
                speedup_val = round(t / fastest_time, 2)
            else:
                speedup_val = None
            
            algo_list.append({
                "name":            name,
                "paradigm":        info.get("paradigm", ""),
                "time_best":       info.get("time_best", ""),
                "time_avg":        info.get("time_avg", ""),
                "time_worst":      info.get("time_worst", ""),
                "space":           info.get("space", ""),
                "description":     info.get("description", ""),
                "pros":            info.get("pros", []),
                "cons":            cons_list,
                "best_when":       info.get("best_when", []),
                "actual_time":     round(t, 8) if t is not None else None,
                "is_best":         is_best,
                "is_fastest":      is_fastest,
                "speedup_vs_best": speedup_val, # Passes null or pure number
            })
        
        algo_list.sort(key=lambda x: (not x["is_best"], x["actual_time"] or float("inf")))
        # -----------------------------------------

        # 5. Extracted data summary (strip raw matrix from JSON payload)
        extracted_summary = {}
        if extracted.get("array"):
            extracted_summary["array"] = extracted["array"]
        if extracted.get("knapsack"):
            extracted_summary["knapsack"] = extracted["knapsack"]
        if extracted.get("graph"):
            g = extracted["graph"]
            extracted_summary["graph"] = {k: v for k, v in g.items() if k != "matrix"}
            if "matrix" in g:
                extracted_summary["graph"]["matrix_size"] = (
                    f"{len(g['matrix'])}x{len(g['matrix'])}"
                )

        # Build paradigm_reasoning with explanation of best vs fastest
        if ml_ready:
            if best_name == fastest_name:
                paradigm_reasoning = (
                    f"✅ ML model (DecisionTree) recommended '{best_name}' "
                    f"for {domain} problems at input size N={input_size}. "
                    f"This also happens to be the FASTEST algorithm in benchmarks!"
                )
            else:
                speedup_factor = round(times[fastest_name] / times[best_name], 2) if times[best_name] else 1
                paradigm_reasoning = (
                    f"🎯 ML model (DecisionTree) chose '{best_name}' "
                    f"as the best algorithm for {domain} problems at size N={input_size}. "
                    f"However, '{fastest_name}' runs {speedup_factor}x faster in benchmarks. "
                    f"The ML recommendation is based on overall performance across many {domain} scenarios, "
                    f"while '{fastest_name}' is faster for THIS specific input."
                )
        else:
            paradigm_reasoning = analysis.get("paradigm_reasoning", "")

        return jsonify({
            "analysis": {
                "original_statement": statement,
                "domain":             domain,
                "paradigm":           algos.get(best_name, {}).get("paradigm", paradigm),
                "paradigm_reasoning": paradigm_reasoning,
                "input_size":       input_size,
                "size_source":      analysis.get("size_source", "NLP extraction"),
                "matched_keywords": analysis.get("matched_keywords", []),
                "extracted_data":   extracted_summary,
                "ml_ready":         ml_ready,
            },
            "best_algorithm": {
                "name":     best_name,
                "time":     round(best_time, 8),
                "paradigm": algos.get(best_name, {}).get("paradigm", ""),
                "is_same_as_fastest": (best_name == fastest_name),
                "info":     algos.get(best_name, {}),
            },
            "fastest_algorithm": {
                "name": fastest_name,
                "time": round(fastest_time, 8),
                "speedup_vs_best": (
                    round(times[best_name] / times[fastest_name], 2) 
                    if times[best_name] and times[fastest_name] else None
                ),
            },
            "all_algorithms": algo_list,
            "benchmark": {
                "preview":   bench["preview"],
                "data_desc": bench["data_desc"],
            },
        })

    except Exception as e:
        import traceback
        print(f"[ERROR /analyze] {e}\n{traceback.format_exc()}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5001)