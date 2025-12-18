from main import search_text_to_image
import json
TOP_K = 15

# ================= CONFIG =================
GENDERS = ['girls', 'boys', 'men', 'women']
COLOURS = [
    'white', 'black', 'blue', 'pink', 'red', 'olive', 'yellow', 'navy blue',
    'magenta', 'grey', 'green', 'orange', 'purple', 'turquoise blue', 'peach',
    'off white', 'teal', 'sea green', 'lime green', 'brown', 'lavender', 'beige',
    'khaki', 'multi', 'maroon', 'cream', 'rust', 'grey melange', 'silver', 'tan',
    'charcoal', 'mushroom brown', 'copper', 'gold', 'bronze', 'taupe', 'metallic',
    'mustard', 'nude'
]
PRODUCT_TYPES = [
    'tops', 'capris', 'dresses', 'shorts', 'tshirts', 'skirts', 'jeans', 'leggings',
    'vests', 'rompers', 'choli', 'salwar', 'booties', 'set', 'trousers', 'shirts',
    'jackets', 'kurtas', 'sweatshirts', 'sets', 'churidar', 'waistcoat', 'blazers',
    'shoes', 'flops', 'sandals', 'flats', 'heels'
]
USAGES = ['casual', 'ethnic', 'sports', 'formal', 'smart casual', 'party']

ATTRIBUTE_WEIGHTS = {
    "gender": 0.1,
    "colour": 0.2,
    "usage": 0.3,
    "product_type": 0.4
}

# ================= PARSING QUERY =================
def parse_query(query: str):
    q = query.lower()
    gender = next((g for g in GENDERS if g in q), None)
    colour = next((c for c in COLOURS if c in q), None)
    usage = next((u for u in USAGES if u in q), None)
    product_type = next((p for p in PRODUCT_TYPES if p in q), None)
    
    return {
        "gender": gender,
        "colour": colour,
        "usage": usage,
        "product_type": product_type
    }

# ================= SCORING =================
def weighted_attribute_score(hit, gt_attrs, weights):
    score = 0.0
    total_weight = 0.0

    for attr, weight in weights.items():
        gt_value = gt_attrs.get(attr)
        if gt_value is None:
            continue
        total_weight += weight
        if hit.payload.get(attr) == gt_value:
            score += weight

    if total_weight == 0:
        return 0.0

    return score / total_weight

# ================= EVALUATION =================
def evaluate_query(query: str, top_k: int = TOP_K):
    gt_attrs = parse_query(query)
    results = search_text_to_image(query, top_k)

    item_scores = [
        weighted_attribute_score(hit, gt_attrs, ATTRIBUTE_WEIGHTS)
        for hit in results
    ]

    return {
        "query": query,
        "top_k": top_k,
        "avg_score": round(sum(item_scores) / len(item_scores), 2) if item_scores else 0.0,
        "top1_score": round(item_scores[0], 2) if item_scores else 0.0,
        "full_match_ratio": round(
            sum(1 for s in item_scores if s == 1.0) / len(item_scores),
            2
        ) if item_scores else 0.0,
    }


def evaluate_queries(queries, top_k: int = TOP_K):
    results = []
    for q in queries:
        results.append(evaluate_query(q, top_k))
    return results
# ================= SAVE RESULTS =================
def save_results(results, json_path="eval_results.json"):
    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

# ================= MAIN =================
if __name__ == "__main__":
    queries = [
        "men white sports shoes",
        "boys white casual tshirts",
        "men black casual shoes",
        "women black casual heels",
        "boys red casual tshirts",
        "boys yellow casual tshirts",
        "boys green casual tshirts",
        "men brown casual shoes",
        "boys blue casual tshirts",
        "girls white casual tops"
    ]


    eval_results = evaluate_queries(queries, top_k=TOP_K)

    for r in eval_results:
        print(r)
    save_results(eval_results)
