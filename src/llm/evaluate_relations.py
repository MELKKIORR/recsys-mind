
import argparse
import json
from pathlib import Path

ASYMMETRIC_RELATIONS = {
    "plays_for",
    "works_for",
    "leads",
    "member_of",
    "supports",
    "located_in",
    "participates_in",
}

ROOT_DIR = Path(__file__).resolve().parents[2]


def load_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def normalize_relation(relation):
    if relation is None:
        return "unknown"

    relation = str(relation).strip().lower()

    if relation in {"", "none", "null", "unknown"}:
        return "unknown"

    return relation


def normalize_entity(entity):
    if entity is None:
        return ""

    return " ".join(str(entity).strip().lower().split())


def load_predictions(path):
    predictions = load_json(path)

    if not isinstance(predictions, list):
        raise ValueError("Le fichier de prédictions doit contenir une liste JSON.")

    return predictions


def load_annotations(path):
    annotations = load_json(path)

    if not isinstance(annotations, list):
        raise ValueError("Le fichier d'annotations doit contenir une liste JSON.")

    return annotations


def build_pair_key(entity_1, entity_2):
    return (
        normalize_entity(entity_1),
        normalize_entity(entity_2),
    )


def evaluate_relations(predictions, annotations):
    prediction_map = {}

    for prediction in predictions:
        key = build_pair_key(
            prediction.get("entity_1"),
            prediction.get("entity_2"),
        )
        prediction_map[key] = prediction

    total = 0
    correct = 0
    predicted_relations = 0
    unknown_count = 0
    direction_total = 0
    direction_correct = 0

    results = []

    for annotation in annotations:
        entity_1 = annotation.get("entity_1")
        entity_2 = annotation.get("entity_2")
        expected = normalize_relation(annotation.get("relation"))

        key = build_pair_key(entity_1, entity_2)
        prediction = prediction_map.get(key)

        if prediction is None:
            predicted = "unknown"
            source_entity = None
            target_entity = None
        else:
            predicted = normalize_relation(prediction.get("relation"))
            source_entity = prediction.get("source_entity")
            target_entity = prediction.get("target_entity")

        total += 1

        if predicted == "unknown":
            unknown_count += 1
        else:
            predicted_relations += 1

        is_correct = predicted == expected

        if is_correct:
            correct += 1

        if expected in ASYMMETRIC_RELATIONS and predicted == expected:
            direction_total += 1

            expected_source = normalize_entity(entity_1)
            expected_target = normalize_entity(entity_2)

            actual_source = normalize_entity(source_entity)
            actual_target = normalize_entity(target_entity)

            if (
                actual_source == expected_source
                and actual_target == expected_target
            ):
                direction_correct += 1

        results.append(
            {
                "entity_1": entity_1,
                "entity_2": entity_2,
                "expected_relation": expected,
                "predicted_relation": predicted,
                "correct": is_correct,
                "source_entity": source_entity,
                "target_entity": target_entity,
            }
        )

    accuracy = correct / total if total else 0
    coverage = predicted_relations / total if total else 0
    unknown_rate = unknown_count / total if total else 0

    direction_accuracy = (
        direction_correct / direction_total
        if direction_total
        else 0
    )

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "coverage": coverage,
        "unknown_count": unknown_count,
        "unknown_rate": unknown_rate,
        "direction_total": direction_total,
        "direction_correct": direction_correct,
        "direction_accuracy": direction_accuracy,
        "details": results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate extracted entity relations."
    )

    parser.add_argument(
        "--predictions",
        type=str,
        default="results/entity_relations.json",
        help="Path to the predictions JSON file.",
    )

    parser.add_argument(
        "--annotations",
        type=str,
        default="data/relation_annotations.json",
        help="Path to the manually annotated JSON file.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results/relation_evaluation.json",
        help="Path to save evaluation results.",
    )

    args = parser.parse_args()

    predictions_path = ROOT_DIR / args.predictions
    annotations_path = ROOT_DIR / args.annotations
    output_path = ROOT_DIR / args.output

    predictions = load_predictions(predictions_path)
    annotations = load_annotations(annotations_path)

    metrics = evaluate_relations(predictions, annotations)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4, ensure_ascii=False)

    print("\n========== RELATION EVALUATION ==========")
    print(f"Total pairs:        {metrics['total']}")
    print(f"Correct:            {metrics['correct']}")
    print(f"Accuracy:           {metrics['accuracy']:.2%}")
    print(f"Coverage:           {metrics['coverage']:.2%}")
    print(f"Unknown:            {metrics['unknown_count']}")
    print(f"Unknown rate:       {metrics['unknown_rate']:.2%}")
    print(f"Direction accuracy: {metrics['direction_accuracy']:.2%}")
    print("=========================================")
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
