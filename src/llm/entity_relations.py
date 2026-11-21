
import argparse
import json
import sqlite3
import urllib.request
from pathlib import Path

import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:3b"


def get_entity_pairs(db_path, limit, min_frequency):
    """Retrieve the most frequent pairs of entities from SQLite."""

    con = sqlite3.connect(db_path)

    query = """
        SELECT
            ae1.entity_id,
            ae2.entity_id,
            e1.label AS entity_1,
            e2.label AS entity_2,
            COUNT(DISTINCT ae1.article_id) AS frequency
        FROM ArticleEntities ae1
        JOIN ArticleEntities ae2
            ON ae1.article_id = ae2.article_id
            AND ae1.entity_id < ae2.entity_id
        JOIN Entities e1 ON ae1.entity_id = e1.entity_id
        JOIN Entities e2 ON ae2.entity_id = e2.entity_id
        GROUP BY ae1.entity_id, ae2.entity_id
        HAVING COUNT(DISTINCT ae1.article_id) >= ?
        ORDER BY frequency DESC
        LIMIT ?
    """

    rows = con.execute(query, (min_frequency, limit)).fetchall()

    pairs = []

    for row in rows:
        entity_id_1, entity_id_2, entity_1, entity_2, frequency = row

        context_query = """
            SELECT DISTINCT
                a.category,
                a.title,

                a.abstract
            FROM ArticleEntities ae1
            JOIN ArticleEntities ae2
                ON ae1.article_id = ae2.article_id
                AND ae1.entity_id < ae2.entity_id
            JOIN Articles a ON a.article_id = ae1.article_id
            WHERE ae1.entity_id = ?
              AND ae2.entity_id = ?
            LIMIT 5
        """

        contexts = con.execute(
            context_query,
            (entity_id_1, entity_id_2)
        ).fetchall()

        pairs.append({
            "entity_1": entity_1,
            "entity_2": entity_2,
            "frequency": frequency,
            "contexts": [
                {
                    "category": context[0],
                    "title": context[1],
                    "abstract": context[2]
                }
                for context in contexts
            ]
        })

    con.close()

    return pairs



def extract_relation(entity_1, entity_2, contexts, model):
    context_text = "\n\n".join(
        [
            f"Category: {context.get('category', '')}\n"
            f"Title: {context.get('title', '')}\n"
            f"Abstract: {context.get('abstract', '')}"
            for context in contexts
        ]
    )

    prompt = f"""
You are a semantic knowledge graph relation extraction system.

Identify the strongest meaningful relationship between these two
entities using ONLY the article contexts below.

ENTITY 1: {entity_1}
ENTITY 2: {entity_2}

ARTICLE CONTEXTS:
{context_text}

RULES:
1. Use only evidence from the provided contexts.
2. Co-occurrence alone is not a relationship.
3. Do not use external knowledge or assumptions.
4. Choose a precise relation in snake_case.
5. The relation can go in either direction:
   Entity 1 -> Entity 2, or Entity 2 -> Entity 1.
6. source_entity is the entity performing or originating the relation.
7. target_entity is the entity the relation points to.
8. Both source_entity and target_entity must exactly match
   the original entity names.
9. If the evidence is insufficient, use "unknown".
10. For unknown, set source_entity and target_entity to null,
    confidence to 0.0, and explain why.
11. Confidence must be a number between 0.0 and 1.0.
    Do not use words such as "high" or "medium".
source_entity and target_entity must contain the exact original
entity names, never placeholders.
You may reverse their order when the evidence supports it.

Return ONLY valid JSON with exactly these fields:
{{
    "entity_1": "{entity_1}",
    "source_entity": "{entity_2}",
    "relation": "meaningful_relation_in_snake_case",
    "target_entity": "{entity_1}",
    "entity_2": "{entity_2}",
    "confidence": 0.7,
    "explanation": "Evidence supporting the relationship"
}}

IMPORTANT:
- Do not automatically return unknown.
- Examine all article contexts together.
- A relationship can be strongly implied by the combined contexts,
  even if it is not expressed in one exact sentence.
- If several articles consistently connect the entities,
  identify their most precise meaningful relationship.
- Use unknown only when there is genuinely no meaningful evidence.
- Never invent a relationship just to avoid unknown.
"""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "format": "json",
        "stream": False,
        "options": {"temperature": 0}
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.loads(response.read().decode("utf-8"))

        relation = json.loads(result["message"]["content"])

        relation_label = relation.get("relation", "unknown")
        if not isinstance(relation_label, str):
            relation_label = "unknown"
        relation_label = relation_label.strip().lower()

        confidence = relation.get("confidence", 0.0)
        if (
            not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
        ):
            confidence = 0.0

        confidence = max(0.0, min(1.0, float(confidence)))

        source_entity = relation.get("source_entity")
        target_entity = relation.get("target_entity")

        valid_entities = {entity_1, entity_2}

        if (
            relation_label == "unknown"
            or source_entity not in valid_entities
            or target_entity not in valid_entities
            or source_entity == target_entity
        ):
            relation_label = "unknown"
            source_entity = None
            target_entity = None
            confidence = 0.0

        return {
            "entity_1": entity_1,
            "source_entity": source_entity,
            "relation": relation_label,
            "target_entity": target_entity,
            "entity_2": entity_2,
            "confidence": confidence,
            "explanation": relation.get("explanation", "")
        }

    except (urllib.error.URLError, KeyError, ValueError, TypeError) as error:
        print(f"Error extracting relation for {entity_1} -- {entity_2}: {error}")

        return {
            "entity_1": entity_1,
            "source_entity": None,
            "relation": "unknown",
            "target_entity": None,
            "entity_2": entity_2,
            "confidence": 0.0,
            "explanation": "Extraction failed."
        }



def main():
    parser = argparse.ArgumentParser(
        description="Extract semantic relations between frequent entities."
    )

    parser.add_argument(
        "--config",
        default="configs/test.yaml",
        help="Path to the YAML configuration file."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of entity pairs to process."
    )

    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum number of articles containing a pair."
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ollama model to use."
    )

    parser.add_argument(
        "--output",
        default="results/entity_relations.json",
        help="Output JSON file."
    )

    args = parser.parse_args()

    config_path = ROOT_DIR / args.config

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    db_path = ROOT_DIR / config["db_path"]

    print(f"Database: {db_path}")
    print(f"Model: {args.model}")

    pairs = get_entity_pairs(
        db_path,
        args.limit,
        args.min_frequency
    )

    print(f"\n{len(pairs)} entity pairs found.")

    results = []

    for index, pair in enumerate(pairs, start=1):
        print(
            f"\n[{index}/{len(pairs)}] "
            f"{pair['entity_1']} -- {pair['entity_2']}"
        )

        try:
            relation = extract_relation(
                pair["entity_1"],
                pair["entity_2"],
                pair["contexts"],
                args.model
            )

            relation["frequency"] = pair["frequency"]
            results.append(relation)

            print(f"Relation: {relation['relation']}")

        except Exception as error:
            print(f"Error: {error}")

    output_path = ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print(f"\nFinished! {len(results)} results saved to:")
    print(output_path)


if __name__ == "__main__":
    main()
