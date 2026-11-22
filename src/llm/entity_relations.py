
import argparse
import json
import re
import sqlite3
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import yaml


ROOT_DIR = Path(__file__).resolve().parents[2]
OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5:3b"

CONTEXT_POOL_SIZE = 30
MAX_EVIDENCE = 6
NUM_CTX = 4096
REQUIRE_JOINT_MENTION = True

KIND_NAMES = {
    "P": "person",
    "O": "organization/place/event",
    None: "type unknown",
}

RELATION_SCHEMA = {
    "plays_for": {
        "source": {"P"}, "target": {"O"}, "symmetric": False,
        "definition": "a person is a player or athlete on a sports team",
        "template": "{s} plays for {t}",
    },
    "works_for": {
        "source": {"P"}, "target": {"O"}, "symmetric": False,
        "definition": "a person is employed by, or holds a staff or official role at, an organization (not a team roster spot, not the top leader)",
        "template": "{s} works for {t}",
    },
    "leads": {
        "source": {"P"}, "target": {"O"}, "symmetric": False,
        "definition": "a person is the head or leader of an organization, team, city or government body",
        "template": "{s} leads {t}",
    },
    "member_of": {
        "source": {"P"}, "target": {"O"}, "symmetric": False,
        "definition": "a person is a formal member of a party, legislature, committee, board or similar body",
        "template": "{s} is a member of {t}",
    },
    "supports": {
        "source": {"P"}, "target": {"O"}, "symmetric": False,
        "definition": "a person publicly endorses or backs an organization, party or cause without being stated to be a member",
        "template": "{s} supports {t}",
    },
    "located_in": {
        "source": {"P", "O"}, "target": {"O"}, "symmetric": False,
        "definition": "the source is based in, or is part of, the geographic target",
        "template": "{s} is located in {t}",
    },
    "participates_in": {
        "source": {"P", "O"}, "target": {"O"}, "symmetric": False,
        "definition": "the source takes part in the target event, competition or tournament",
        "template": "{s} participates in {t}",
    },
    "competes_against": {
        "source": {"P", "O"}, "target": {"P", "O"}, "symmetric": True,
        "definition": "the two play or compete directly against each other",
        "template": "{s} competes against {t}",
    },
}

GENERIC_TOKENS = {
    "the", "of", "and", "new", "united", "party", "inc", "corp", "company",
    "university", "college", "state", "states", "city", "national", "league",
    "association", "committee", "department", "school", "club", "team", "group",
}
NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}

FEWSHOT_EXAMPLES = """EXAMPLES (fictional entities, for format only):

Entity 1: Harbor City Sharks (organization/place/event)
Entity 2: Dana Whitfield (person)
Evidence:
[1] Sharks goalkeeper Whitfield saved 31 shots in Friday's win.
Answer: {"evidence_ids": [1], "explanation": "Whitfield is described as the Sharks' goalkeeper.", "relation": "plays_for", "source_entity": "Dana Whitfield", "target_entity": "Harbor City Sharks"}

Entity 1: Lakeport (organization/place/event)
Entity 2: Elena Ruiz (person)
Evidence:
[1] Lakeport Mayor Elena Ruiz signed the budget on Monday.
Answer: {"evidence_ids": [1], "explanation": "Ruiz is described as the mayor of Lakeport.", "relation": "leads", "source_entity": "Elena Ruiz", "target_entity": "Lakeport"}

Entity 1: Elena Ruiz (person)
Entity 2: Orion Motors (organization/place/event)
Evidence:
[1] Ruiz and Orion Motors executives were both at the trade fair.
Answer: {"evidence_ids": [], "explanation": "They are only mentioned at the same event; no relation is stated.", "relation": "none", "source_entity": "Elena Ruiz", "target_entity": "Orion Motors"}"""


def get_entity_pairs(db_path, limit, min_frequency):
    """Retrieve the most frequent pairs of entities from SQLite."""

    con = sqlite3.connect(db_path)

    query = """
        SELECT
            ae1.entity_id,
            ae2.entity_id,
            e1.label AS entity_1,
            e2.label AS entity_2,
            COUNT(DISTINCT ae1.article_id) AS frequency,
            e1.type AS type_1,
            e2.type AS type_2
        FROM ArticleEntities ae1
        JOIN ArticleEntities ae2
            ON ae1.article_id = ae2.article_id
            AND ae1.entity_id < ae2.entity_id
        JOIN Entities e1 ON ae1.entity_id = e1.entity_id
        JOIN Entities e2 ON ae2.entity_id = e2.entity_id
        GROUP BY ae1.entity_id, ae2.entity_id
        HAVING COUNT(DISTINCT ae1.article_id) >= ?
        ORDER BY frequency DESC, ae1.entity_id, ae2.entity_id
        LIMIT ?
    """

    rows = con.execute(query, (min_frequency, limit)).fetchall()

    pairs = []

    for row in rows:
        (
            entity_id_1, entity_id_2, entity_1, entity_2,
            frequency, type_1, type_2
        ) = row

        context_query = """
            SELECT DISTINCT
                a.article_id,
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
            ORDER BY a.article_id
            LIMIT ?
        """

        contexts = con.execute(
            context_query,
            (entity_id_1, entity_id_2, CONTEXT_POOL_SIZE)
        ).fetchall()

        pairs.append({
            "entity_1": entity_1,
            "entity_2": entity_2,
            "type_1": type_1,
            "type_2": type_2,
            "frequency": frequency,
            "contexts": [
                {
                    "article_id": context[0],
                    "category": context[1],
                    "title": context[2],
                    "abstract": context[3]
                }
                for context in contexts
            ]
        })

    con.close()

    return pairs


def _kind(entity_type):
    """Map a raw DB entity type to 'P', 'O' or None."""
    if not entity_type:
        return None

    entity_type = str(entity_type).strip().upper()

    if entity_type in {"P", "O"}:
        return entity_type

    return None


def _norm(text):
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def _split_sentences(text):
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text or "")
        if sentence.strip()
    ]


def build_mention_patterns(label, kind):
    """Build patterns for full entity labels and one derived short form."""

    label = label.strip()
    patterns = [
        re.compile(r"\b" + re.escape(label) + r"\b", re.IGNORECASE)
    ]

    tokens = re.findall(r"[\w'’&-]+", label)

    if len(tokens) > 1 and tokens[0].lower() == "the":
        patterns.append(
            re.compile(
                r"\b" + re.escape(" ".join(tokens[1:])) + r"\b",
                re.IGNORECASE
            )
        )
        tokens = tokens[1:]

    partial = None

    if len(tokens) >= 2:
        if kind == "P":
            core = [token for token in tokens if token.lower() not in NAME_SUFFIXES]
            partial = core[-1] if core else None
        else:
            candidates = [
                token for token in tokens
                if token.lower() not in GENERIC_TOKENS and len(token) >= 4
            ]
            partial = candidates[-1] if candidates else None

    if partial and len(partial) >= 3:
        patterns.append(re.compile(r"\b" + re.escape(partial) + r"\b"))

    return patterns


def _mentions(text, patterns):
    return any(pattern.search(text) for pattern in patterns)


def select_evidence(
    entity_1, entity_2, kind_1, kind_2, contexts,
    max_items=MAX_EVIDENCE
):
    """Stage 0: select the most informative article snippets."""

    patterns_1 = build_mention_patterns(entity_1, kind_1)
    patterns_2 = build_mention_patterns(entity_2, kind_2)

    candidates = []

    for article_index, context in enumerate(contexts):
        title = (context.get("title") or "").strip()
        abstract = (context.get("abstract") or "").strip()

        title_text = (
            title if (not title or title[-1] in ".!?")
            else title + "."
        )

        full_text = " ".join(
            part for part in (title_text, abstract) if part
        )

        if not full_text:
            continue

        units = (
            ([title_text] if title_text else [])
            + _split_sentences(abstract)
        )

        best = None

        for sentence in units:
            if (
                _mentions(sentence, patterns_1)
                and _mentions(sentence, patterns_2)
            ):
                best = {"tier": 1, "text": sentence}
                break

        if best is None:
            both = (
                _mentions(full_text, patterns_1)
                and _mentions(full_text, patterns_2)
            )
            best = {
                "tier": 2 if both else 3,
                "text": full_text
            }

        best["article_index"] = article_index
        best["article_id"] = context.get("article_id")
        best["category"] = context.get("category") or ""

        candidates.append(best)

    candidates.sort(
        key=lambda item: (item["tier"], item["article_index"])
    )

    if any(item["tier"] <= 2 for item in candidates):
        candidates = [
            item for item in candidates if item["tier"] <= 2
        ]

    evidence = []
    seen_texts = set()

    for item in candidates:
        key = _norm(item["text"])

        if key in seen_texts:
            continue

        seen_texts.add(key)
        item["id"] = len(evidence) + 1
        evidence.append(item)

        if len(evidence) == max_items:
            break

    return evidence


def _signature(spec):
    if spec["symmetric"]:
        return "symmetric"

    source = "/".join(
        KIND_NAMES[kind] for kind in sorted(spec["source"])
    )
    target = "/".join(
        KIND_NAMES[kind] for kind in sorted(spec["target"])
    )

    return f"{source} -> {target}"



def build_selection_prompt(
    entity_1, entity_2, kind_1, kind_2, evidence
):
    relation_lines = "\n".join(
        f"- {label} ({_signature(spec)}): {spec['definition']}"
        for label, spec in RELATION_SCHEMA.items()
    )

    evidence_text = "\n".join(
        f"[{item['id']}] ({item['category']}) {item['text']}"
        for item in evidence
    )

    return f"""Your task is to identify a relationship explicitly supported by the news evidence.

Allowed relations, written as (source -> target):
{relation_lines}
- none: no allowed relationship is supported by the evidence.

STRICT RULES:
1. Use ONLY the supplied evidence. Do not use external knowledge, assumptions or general world knowledge.
2. Select a relationship only when the evidence explicitly states it or clearly and unambiguously implies it.
3. Merely mentioning two entities in the same article, sentence, event, topic or category is NOT evidence of a relationship.
4. Sharing an industry, domain, activity or category does NOT establish a relationship.
5. In particular, do NOT infer competes_against merely because two companies operate in the same industry. The evidence must explicitly identify them as competitors, rivals, opponents, or describe an actual competitive relationship.
6. Do NOT infer employment, membership, leadership, support or participation from co-occurrence alone.
7. If the relationship is ambiguous, indirect, or insufficiently supported, choose "none".
8. Do not invent facts to justify a relationship.

Evidence requirements:
- evidence_ids must contain ONLY the IDs that directly support the selected relationship.
- If relation is "none", evidence_ids must be an empty list.
- The explanation must describe what the evidence actually states, not an assumption.

Direction:
- source_entity and target_entity must follow the direction defined by the selected relation.
- This direction applies regardless of the order of Entity 1 and Entity 2.
- Entities may be mentioned using a surname, nickname or short form.

{FEWSHOT_EXAMPLES}

NOW ANSWER FOR THIS INPUT.

Entity 1: {entity_1} ({KIND_NAMES[kind_1]})
Entity 2: {entity_2} ({KIND_NAMES[kind_2]})
Evidence:

{evidence_text}
Answer:"""



def build_selection_schema(entity_1, entity_2):
    """JSON schema for Ollama structured outputs."""

    return {
        "type": "object",
        "properties": {
            "evidence_ids": {
                "type": "array",
                "items": {"type": "integer"}
            },
            "explanation": {"type": "string"},
            "relation": {
                "type": "string",
                "enum": list(RELATION_SCHEMA) + ["none"]
            },
            "source_entity": {
                "type": "string",
                "enum": [entity_1, entity_2]
            },
            "target_entity": {
                "type": "string",
                "enum": [entity_1, entity_2]
            },
        },
        "required": [
            "evidence_ids", "explanation", "relation",
            "source_entity", "target_entity"
        ],
    }


VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["yes", "no"]
        }
    },
    "required": ["verdict"],
}


def call_ollama(prompt, model, schema=None):
    """Single Ollama chat call."""

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "format": schema if schema is not None else "json",
        "stream": False,
        "options": {
            "temperature": 0,
            "seed": 0,
            "num_ctx": NUM_CTX,
            "num_predict": 300
        }
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))

    if result.get("prompt_eval_count", 0) >= NUM_CTX - 64:
        print(
            f"WARNING: prompt used {result['prompt_eval_count']} tokens; "
            "context may be truncated."
        )

    parsed = json.loads(result["message"]["content"])

    if not isinstance(parsed, dict):
        raise ValueError("Model output is not a JSON object.")

    return parsed


def validate_relation(
    relation_label, source_entity, target_entity,
    entity_1, entity_2, kind_1, kind_2
):
    """Stage 1b: validate relation, entities and direction."""

    spec = RELATION_SCHEMA.get(relation_label)

    names = {
        _norm(entity_1): entity_1,
        _norm(entity_2): entity_2
    }

    source = names.get(_norm(source_entity))
    target = names.get(_norm(target_entity))

    if spec is None:
        return None, None, None, False, "out_of_vocabulary"

    if source is None or target is None or source == target:
        return None, None, None, False, "invalid_entities"

    if spec["symmetric"]:
        return relation_label, entity_1, entity_2, False, None

    kinds = {
        entity_1: kind_1,
        entity_2: kind_2
    }

    def fits(source, target):
        return (
            (kinds[source] is None or kinds[source] in spec["source"])
            and
            (kinds[target] is None or kinds[target] in spec["target"])
        )

    if fits(source, target):
        return relation_label, source, target, False, None

    if fits(target, source):
        return relation_label, target, source, True, None

    return None, None, None, False, "type_signature_violation"



def verify_relation(
    source_entity, relation_label, target_entity,
    evidence_texts, model
):
    """Stage 2: verify that the evidence entails the proposed triple."""

    claim = RELATION_SCHEMA[relation_label]["template"].format(
        s=source_entity,
        t=target_entity
    )

    evidence_text = "\n".join(
        f"- {text}" for text in evidence_texts
    )

    prompt = f"""You are a strict fact-checker.

Evidence from news articles:
{evidence_text}

Claim: {claim}.

Decide whether the supplied evidence directly supports this exact claim.

Answer "yes" ONLY if:
- The evidence explicitly states the relationship between the EXACT source and target entities.
- The relationship has the correct direction.
- The evidence must directly connect these two entities, not infer their relationship through a third person or organization.
- Evidence about a third person's membership in an organization does NOT prove that the source entity is also a member.
- The evidence supports the actual relationship, not merely the presence of both entities.

Answer "no" if:
- Both entities are merely mentioned in the same article, sentence, event or topic.
- They share an industry, category or general activity.
- The relationship is based on external knowledge or assumptions.
- The evidence suggests a different relationship or the opposite direction.
- The evidence is ambiguous or insufficient.

IMPORTANT:
Two companies mentioned in the same industry are NOT necessarily competitors.
Do not infer competition without explicit evidence of rivalry or direct competition.
When in doubt, answer "no".

Return only the required JSON verdict."""

    verdict = call_ollama(prompt, model, VERDICT_SCHEMA)
    print("DEBUG VERIFICATION:", verdict)
    return (
        str(verdict.get("verdict", "no")).strip().lower() == "yes"
    )



def score_evidence(used_evidence):
    """Heuristic evidence score, not a calibrated probability."""

    articles = {
        item["article_index"] for item in used_evidence
    }

    has_joint_sentence = any(
        item["tier"] == 1 for item in used_evidence
    )

    score = (
        0.5
        + 0.2 * (min(len(articles), 3) / 3)
        + (0.3 if has_joint_sentence else 0.0)
    )

    return round(score, 2)


def extract_relation(
    entity_1, entity_2, contexts, model,
    type_1=None, type_2=None, verify_model=None
):
    """Extract a semantic relationship using a two-stage LLM pipeline."""

    kind_1, kind_2 = _kind(type_1), _kind(type_2)

    result = {
        "entity_1": entity_1,
        "source_entity": None,
        "relation": "unknown",
        "target_entity": None,
        "entity_2": entity_2,
        "confidence": 0.0,
        "explanation": "",
        "unknown_reason": None,
        "direction_corrected": False,
        "n_evidence": 0,
        "evidence": [],
        "candidate": None
    }

    def reject(reason, explanation=""):
        result["unknown_reason"] = reason
        result["explanation"] = explanation
        return result

    if _norm(entity_1) == _norm(entity_2):
        return reject("identical_labels")

    evidence = select_evidence(
        entity_1, entity_2, kind_1, kind_2, contexts
    )

    result["n_evidence"] = len(evidence)

    if not evidence:
        return reject("no_contexts")

    if (
        REQUIRE_JOINT_MENTION
        and all(item["tier"] == 3 for item in evidence)
    ):
        return reject("no_joint_mention")

    try:
        selection = call_ollama(
            build_selection_prompt(
                entity_1, entity_2, kind_1, kind_2, evidence
            ),
            model,
            build_selection_schema(entity_1, entity_2)
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(
            f"Error extracting relation for "
            f"{entity_1} -- {entity_2}: {error}"
        )
        return reject("request_failed", "Extraction failed.")

    relation_label = str(
        selection.get("relation", "none")
    ).strip().lower()

    explanation = selection.get("explanation", "")
    explanation = explanation if isinstance(explanation, str) else ""

    if relation_label == "none":
        return reject("model_none", explanation)

    (
        relation_label,
        source_entity,
        target_entity,
        direction_corrected,
        reason
    ) = validate_relation(
        relation_label,
        selection.get("source_entity"),
        selection.get("target_entity"),
        entity_1,
        entity_2,
        kind_1,
        kind_2
    )

    if reason:
        return reject(reason, explanation)

    result["candidate"] = {
        "source_entity": source_entity,
        "relation": relation_label,
        "target_entity": target_entity
    }

    cited = selection.get("evidence_ids")

    cited = (
        [
            item for item in cited
            if isinstance(item, int) and not isinstance(item, bool)
        ]
        if isinstance(cited, list)
        else []
    )
    used_evidence = [
        item for item in evidence if item["id"] in cited
    ]

    if not used_evidence:
        return reject("no_evidence_selected", explanation)

    try:
        verified = verify_relation(
            source_entity,
            relation_label,
            target_entity,
            [item["text"] for item in used_evidence],
            verify_model or model
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(
            f"Verification failed for "
            f"{entity_1} -- {entity_2}: {error}"
        )
        return reject("request_failed", "Verification failed.")

    if not verified:
        return reject("not_entailed", explanation)

    result.update({
        "source_entity": source_entity,
        "relation": relation_label,
        "target_entity": target_entity,
        "confidence": score_evidence(used_evidence),
        "explanation": explanation,
        "unknown_reason": None,
        "direction_corrected": direction_corrected,
        "evidence": [
            item["text"] for item in used_evidence
        ]
    })

    return result


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
        "--verify-model",
        default=None,
        help="Optional separate Ollama model for verification."
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
    print(f"Verification model: {args.verify_model or args.model}")

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
                args.model,
                pair["type_1"],
                pair["type_2"],
                args.verify_model
            )

        except Exception as error:
            print(f"Error: {error}")

            relation = {
                "entity_1": pair["entity_1"],
                "source_entity": None,
                "relation": "unknown",
                "target_entity": None,
                "entity_2": pair["entity_2"],
                "confidence": 0.0,
                "explanation": "Unexpected exception.",
                "unknown_reason": "exception"
            }

        relation["frequency"] = pair["frequency"]
        results.append(relation)

        print(
            f"Relation: {relation['relation']} "
            f"({relation.get('unknown_reason') or 'ok'})"
        )

    outcomes = Counter(
        result.get("unknown_reason") or "extracted"
        for result in results
    )

    print("\nOutcome summary:")

    for outcome, count in outcomes.most_common():
        print(f"  {outcome}: {count}")

    output_path = ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print(f"\nFinished! {len(results)} results saved to:")
    print(output_path)


if __name__ == "__main__":
    main()
