"""
analyzer.py - NLP Problem Statement Analyzer
Detects domain, extracts real data, and returns paradigm info needed by app.py.
"""
import re

DOMAIN_KEYWORDS = {
    "ordering":     ["arrange", "order", "ascending", "descending", "smallest to largest",
                     "largest to smallest", "organize", "rank", "sort"],
    "optimization": ["maximize", "minimize", "optimal", "best combination", "capacity",
                     "budget", "profit", "knapsack"],
    "pathfinding":  ["shortest", "minimum distance", "route", "path between", "travel",
                     "navigate", "floyd", "warshall", "dijkstra"],
    "staged":       ["stages", "pipeline", "levels", "phases", "sequential",
                     "multi-step", "multistage"],
}

# Maps domain → human-readable paradigm label (generic, used as fallback only)
DOMAIN_PARADIGM = {
    "ordering":     "Sorting",
    "optimization": "Dynamic Programming",
    "pathfinding":  "Dynamic Programming",
    "staged":       "Dynamic Programming",
}

# Maps domain → reasoning string shown in UI
DOMAIN_REASONING = {
    "ordering":     "Problem involves arranging/ordering elements — Sorting algorithms apply.",
    "optimization": "Problem involves maximizing/minimizing under a constraint — Knapsack DP applies.",
    "pathfinding":  "Problem involves finding shortest paths between nodes — Floyd-Warshall DP applies.",
    "staged":       "Problem has sequential stages/pipeline structure — Multistage Graph DP applies.",
}


def _normalize(text):
    return re.sub(r"\s+", " ", text.lower().strip())


def detect_domain(text):
    text = _normalize(text)
    scores  = {}
    matched = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        hits = [k for k in keywords if k in text]
        scores[domain]  = len(hits)
        matched[domain] = hits
    best = max(scores, key=scores.get) if scores else None
    if not best or scores[best] == 0:
        return None, [], "Could not detect problem domain. Please describe the problem more clearly."
    return best, matched[best], None


def extract_input_size(text):
    text = _normalize(text)
    patterns = [
        r"n\s*[=:]\s*(\d+)",
        r"(\d+)\s+(?:random\s+)?(?:elements?|numbers?|integers?|items?|values?|nodes?|vertices|cities|server\s+ping\s+times?|records?|readings?|scores?|prices?)",
        r"(?:size|capacity|limit)\s*(?:of|is|=|:)?\s*(\d+)",
    ]
    for p in patterns:
        m = re.search(p, text)
        if m:
            size = int(m.group(1))
            if 1 <= size <= 1_000_000:
                return size, m.group(0)
    return None, None


def extract_array(text):
    m = re.search(r'[\[({]([\d\s,]+)[\])}]', text)
    if m:
        nums = re.findall(r'\d+', m.group(1))
        if len(nums) >= 2:
            return [int(n) for n in nums]
    m = re.search(
        r'(?:numbers|elements|array|list|data|values)[:\s]+((?:\d+[\s,]+){2,}\d+)',
        _normalize(text)
    )
    if m:
        nums = re.findall(r'\d+', m.group(1))
        if len(nums) >= 2:
            return [int(n) for n in nums]
    return None


def extract_knapsack_data(text):
    t = _normalize(text)
    data = {}
    w = re.search(r'(?:weights?|w)\s*[=:]\s*[\[({]([\d\s,]+)[\])}]', t)
    if w: data['weights'] = [int(n) for n in re.findall(r'\d+', w.group(1))]
    v = re.search(r'(?:values?|profits?|v|p)\s*[=:]\s*[\[({]([\d\s,]+)[\])}]', t)
    if v: data['values']  = [int(n) for n in re.findall(r'\d+', v.group(1))]
    c = re.search(r'(?:capacity|cap|weight\s*limit|max\s*weight)\s*[=:]\s*(\d+)', t)
    if c: data['capacity'] = int(c.group(1))
    return data if data else None


def extract_graph_data(text):
    t = _normalize(text)
    data = {}
    n = re.search(r'(\d+)\s+(?:nodes?|vertices|vertex|cities|locations)', t)
    if n: data['nodes'] = int(n.group(1))
    edges = re.findall(r'[(]([\d]+)\s*[,\s]\s*([\d]+)\s*[,\s]\s*([\d]+)[)]', text)
    if edges: data['edges'] = [(int(a), int(b), int(w)) for a, b, w in edges]
    matrix_rows = re.findall(r'[\[({]([\d\s,infinf]+)[\])}]', t)
    if len(matrix_rows) >= 2:
        matrix = []
        for row in matrix_rows:
            vals = []
            for val in re.split(r'[,\s]+', row.strip()):
                if val in ('inf', 'infinity', '∞', '-'):
                    vals.append(float('inf'))
                elif val.isdigit():
                    vals.append(int(val))
            if vals:
                matrix.append(vals)
        if len(matrix) >= 2 and all(len(r) == len(matrix[0]) for r in matrix):
            data['matrix'] = matrix
    return data if data else None


def analyze_problem(statement: str) -> dict:
    """
    Returns a dict with ALL keys expected by app.py:
        original_statement, domain, paradigm, paradigm_reasoning,
        input_size, size_source, matched_keywords, extracted_data, error
    """
    domain, keywords, error = detect_domain(statement)
    if error:
        return {"error": error}

    # ── Data extraction ───────────────────────────────────────────────────
    arr           = extract_array(statement)
    knapsack_data = extract_knapsack_data(statement)
    graph_data    = extract_graph_data(statement)

    # ── Input size (most specific source wins) ────────────────────────────
    input_size  = None
    size_source = None

    if arr:
        input_size  = len(arr)
        size_source = f"counted array length: {input_size} elements"
    elif knapsack_data and 'weights' in knapsack_data:
        input_size  = len(knapsack_data['weights'])
        size_source = f"counted {input_size} items from weights"
    elif graph_data and 'nodes' in graph_data:
        input_size  = graph_data['nodes']
        size_source = f"extracted {input_size} nodes"
    elif graph_data and 'matrix' in graph_data:
        input_size  = len(graph_data['matrix'])
        size_source = f"extracted {input_size}×{input_size} matrix"

    if not input_size:
        input_size, size_source = extract_input_size(statement)
    if not input_size:
        input_size  = 100
        size_source = "default (100)"

    # ── Paradigm info (required by app.py) ───────────────────────────────
    paradigm          = DOMAIN_PARADIGM.get(domain, "Algorithm Design")
    paradigm_reasoning = DOMAIN_REASONING.get(domain, f"Domain detected: {domain}")

    return {
        "original_statement": statement,
        "domain":             domain,
        "paradigm":           paradigm,           # ← was missing; caused 500
        "paradigm_reasoning": paradigm_reasoning, # ← was missing; caused 500
        "input_size":         input_size,
        "size_source":        size_source,
        "matched_keywords":   keywords,
        "extracted_data": {
            "array":    arr,
            "knapsack": knapsack_data,
            "graph":    graph_data,
        },
        "error": None,
    }