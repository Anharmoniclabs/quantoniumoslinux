"""Frozen V2 scoring for the bounded paired-data experiment.

A lexical knowledge grader recognizes a limited set of affirmative relations.
An unrecognized paraphrase scores zero and is marked for blind human review.
All raw answers are retained. Scoring is greedy pass@1 with a required EOS.
"""

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import coherence_quality as legacy
from chat_recovery_data import SOURCES
from paired_instruction_data import FAMILIES

VERSION = "phistate-paired-grader-v2"
GRADER_SHA = hashlib.sha256(Path(__file__).read_bytes() + Path(legacy.__file__).read_bytes()).hexdigest()
RETENTION_ROWS = 8
CONCEPTS = ("ram", "storage", "cpu", "python", "variable", "os", "photosynthesis", "gravity")
SCOPE = "Frozen lexical relations, exact strings/integers, finite safe Python AST cases. An unmatched paraphrase may be correct."

# The subject and predicate must belong to the same assertion. Distinct word
# boundaries prevent matches such as "compute" inside "computer".
CONTENT = r"(?:data|information|instructions?|programs?|values?|files?|code|bytes)"
HOLD = r"(?:hold(?:s)?|stor(?:e|es|ing)|keep(?:s)?|contain(?:s)?|retain(?:s)?|hous(?:e|es))"
LIVE = r"(?:temporar\w*|short[- ]term|work(?:ing)?|(?:currently|actively) (?:used|needed)|in (?:active )?use|\brunn?ing\b|while .{0,30}(?:runs?|open)|at the moment|right now|being used|quick access|while .{0,25}(?:powered on|in use))"
RELATIONS = {
    "ram": (
        r"(?:ram|random[- ]access memory|(?:main|working) memory|it)",
        [
            r"(?:is|acts as|serves as)(?: .{0,48})? (?:scratch ?pad|temporary workspace|working memory|short[- ]term memory|volatile memory)\b",
            rf"{HOLD} .{{0,36}}\b{CONTENT}\b.{{0,75}}\b{LIVE}\b",
            rf"(?:temporarily {HOLD}|{HOLD} (?:temporary|short[- ]term|active|current|working)) .{{0,24}}\b{CONTENT}\b",
            rf"{HOLD} what .{{0,30}}(?:currently needs|is working on|is using|is running)\b",
            r"(?:gives?|provides?) .{0,35}(?:scratch ?pad|temporary workspace|working memory)\b",
            r"(?:a |the )?(?:computer.s )?(?:scratch ?pad|working memory|temporary workspace) (?:for|that|where)\b",
            r"(?:is )?where (?:the )?(?:computer|cpu|processor) (?:keeps?|holds?) .{0,55}(?:working on|using|right now)\b",
        ],
        r"permanent\w*|\bpersist\w*|\bforever\b|\blong[- ]term\b|\bdelet\w*|\berases?\b|\bon (?:an? |the )?(?:ssd|hard drive|disk)\b",
    ),
    "storage": (
        r"(?:computer storage|storage|an? ssd|(?:a )?hard (?:drive|disk)|(?:ssd|disk|drive)s?|it)",
        [
            rf"{HOLD} .{{0,40}}\b{CONTENT}\b.{{0,80}}(?:permanent|persist|long[- ]term|later|even when|when .{{0,18}}(?:power|off)|after .{{0,18}}(?:shut|power)|between sessions)",
            rf"(?:persistently|permanently) {HOLD} .{{0,36}}{CONTENT}\b",
            r"(?:is|provides?) .{0,30}(?:long[- ]term|persistent|nonvolatile|non-volatile) .{0,28}(?:memory|storage|space|home)\b",
            r"(?:saves?|stores?|keeps?) (?:the |our |your )?(?:files|documents|programs|photos)\b",
        ],
        r"only temporarily|loses? .{0,25}(?:when|after) .{0,20}(?:power|shutdown)|delet\w*",
    ),
    "cpu": (
        r"(?:the )?(?:cpu|processor|central processing unit|it)",
        [
            r"(?:executes?|runs?|carries out|processes?|follows?|interprets?|performs?) .{0,45}\b(?:instructions?|programs?|commands?|calculations?|computations?)\b",
            r"(?:performs?|does?) .{0,40}(?:arithmetic|logical operations)\b",
        ],
        r"(?:stores?|deletes?|erases?) .{0,35}(?:instructions?|files?)\b",
    ),
    "python": (
        r"(?:python|it)",
        [
            r"is (?:an? |the )?(?:[\w-]+ ){0,5}(?:programming|coding|computer) language\b",
            r"is (?:an? |the )?language (?:used |that .{0,16})?(?:for|to) (?:write|writing|build|building|program)\b",
        ],
        r"\b(?:hardware device|operating system|snake)\b",
    ),
    "variable": (
        r"(?:a |the )?(?:programming variable|variable|identifier|it)",
        [
            r"(?:is|denotes|provides) (?:an? )?(?:name|label|identifier|named (?:place|location|container|reference)).{0,48}\b(?:values?|data|information)\b",
            r"(?:names?|refers to|points to|holds|stores|represents) (?:an? |the )?(?:current |stored )?(?:value|piece of data)\b",
        ],
        r"\b(?:always|must) (?:be |stay )?(?:constant|unchanged)\b",
    ),
    "os": (
        r"(?:an? |the )?(?:operating system|os|it)",
        [
            r"(?:manages?|controls?|coordinates?|allocates?|schedules?|mediates?) .{0,45}\b(?:hardware|resources?|memory|processes|programs|applications|devices?)\b",
            r"(?:lets?|allows?|enables?) (?:applications?|programs?) (?:to )?(?:run|use .{0,15}hardware)\b",
        ],
        r"\b(?:physically powers|supplies electricity|is (?:an? )?physical processor)\b",
    ),
    "photosynthesis": (
        r"(?:the plant|plants?|photosynthesis|it|they)",
        [
            r"(?:uses?|captures?|absorbs?|converts?|turns?) .{0,30}\b(?:light|sunlight|solar energy|light energy)\b.{0,58}\b(?:sugars?|glucose|food|chemical energy)\b",
            r"(?:makes?|produces?|synthesi[sz]es?) .{0,30}\b(?:sugars?|glucose|food)\b.{0,36}\b(?:sunlight|light|solar energy)\b",
        ],
        r"\b(?:destroys?|burns?)\b.{0,20}(?:glucose|sugar)",
    ),
    "gravity": (
        r"(?:gravity|gravitation|earth.s gravity|it)",
        [
            r"(?:pulls?|attracts?|draws?) .{0,45}\b(?:masses|objects?|things|bodies|earth|ground|each other)\b",
            r"is (?:an? |the )?(?:attractive |gravitational )?(?:force|attraction) .{0,45}\b(?:masses|mass|objects?|bodies|earth)\b",
        ],
        r"\b(?:repels?|pushes? .{0,30}away|away from earth|pulls? .{0,25}(?:upward|into the sky))\b",
    ),
}

# Safe qualifiers that do not contradict the positive information. These
# are intentionally narrow; other negated clauses go to review, not a pass.
QUALIFIERS = {
    "ram": [
        r"(?:(?:ram|it) )?(?:(?:is not|isn.t) permanent|(?:does not|doesn.t|cannot) (?:permanently )?(?:store|retain|keep) (?:files|data|information) (?:permanently|after (?:power loss|shutdown)))",
        r"(?:its )?(?:contents|data) (?:are |is )?lost (?:when|if|after) .{0,35}(?:off|shut down|lost)",
        r"(?:ram|it) loses (?:its )?(?:contents|data) (?:when|if|after) .{0,35}(?:off|shut down|lost)",
    ],
    "storage": [
        r"(?:its )?(?:files|data) (?:remain|persist|survive) .{0,40}(?:power loss|shutdown|off)"
    ],
}
NEGATION = r"\b(?:no|not|never|cannot|can't|doesn't|isn't|won't|don't|aren't|unable|fails? to)\b"
SPLIT = re.compile(
    r"[.;!?\n]+|,?\s+\b(?:but|however|whereas)\b\s+|,?\s+and\s+(?=(?:it|they|instead|(?:also )?(?:stores?|keeps?|retains?|deletes?|erases?|executes?|repels?|pulls?|never|does not))\b)"
)


def digest(value):
    serial = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(serial.encode()).hexdigest()


def knowledge_grade(item, text):
    index = item["id"].rsplit("-", 1)[-1]
    if not index.isdigit() or int(index) >= len(CONCEPTS):
        raise ValueError("Unknown frozen knowledge rubric: " + item["id"])
    concept = CONCEPTS[int(index)]
    subject, relations, contradiction = RELATIONS[concept]
    clean = text.strip().lower().replace("’", "'").replace("—", " ").replace("–", "-")
    clean = re.sub(r"\bit's\b", "it is", clean)
    clauses = [c.strip(" ,:") for c in SPLIT.split(clean) if c.strip(" ,:")]
    passed = False
    reason = "unrecognized_relation"
    all_supported = bool(clauses)
    for clause in clauses:
        if any(re.fullmatch(p, clause) for p in QUALIFIERS.get(concept, ())):
            continue
        if re.search(contradiction, clause) or re.search(NEGATION, clause):
            return {
                "concept": concept,
                "passed": False,
                "reason": "contradiction_or_negation",
                "review": True,
            }
        predicate = re.sub(r"^(?:" + subject + r")\s+", "", clause, count=1)
        recognized = any(re.match(r"(?:" + p + r")", predicate) for p in relations)
        if recognized:
            passed = True
        else:
            all_supported = False
    if passed and all_supported:
        reason = "affirmative_relation"
    return {
        "concept": concept,
        "passed": passed and all_supported,
        "reason": reason,
        "review": not (passed and all_supported),
    }


def grade_answer(item, text):
    if item["rubric"]["kind"] != "keywords":
        if item["rubric"]["kind"] not in ("exact", "code"):
            raise ValueError("Unknown rubric: " + str(item["rubric"]["kind"]))
        return legacy.grade_answer(item, text)
    sem = knowledge_grade(item, text)
    nonempty, repeated = bool(text.strip()), legacy.repetition_failure(text)
    return dict(
        **{**sem, "passed": sem["passed"] and nonempty and not repeated},
        nonempty=nonempty,
        repetition_failure=repeated,
    )


def _invalid_decode(exc):
    return {
        "text": "",
        "eos": False,
        "tokens": 0,
        "passed": False,
        "nonempty": False,
        "repetition_failure": False,
        "decode_error": f"{type(exc).__name__}: {exc}",
    }


def _grade_output(item, answer):
    if (
        not isinstance(answer, dict)
        or not isinstance(answer.get("text"), str)
        or type(answer.get("eos")) is not bool
        or type(answer.get("tokens")) is not int
        or not 0 <= answer["tokens"] <= item["max_tokens"]
    ):
        return _invalid_decode(ValueError("Invalid decoder result"))
    graded = grade_answer(item, answer["text"])
    graded["passed"] = bool(graded["passed"] and answer["eos"])
    return dict(**answer, **graded)


def _score_items(model, tokenizer, items):
    items = list(items)
    first = next(model.parameters(), None) if callable(getattr(model, "parameters", None)) else None
    if first is not None and first.device.type == "cuda":
        from a100_greedy_batch import generate_batch

        # On a sampled parity mismatch, regenerate the entire score set serially.
        outputs = generate_batch(
            model, tokenizer, items, batch_size=16, verify=2, serial_on_mismatch=True
        )
        if len(outputs) != len(items):
            raise RuntimeError("Batched decoder returned a different number of answers")
        backend = {"backend": getattr(outputs, "decode_backend", "batched_greedy")}
        if hasattr(outputs, "mismatch_index"):
            backend["batch_parity_mismatch_index"] = outputs.mismatch_index
        return [_grade_output(item, answer) for item, answer in zip(items, outputs)], backend
    scores = []
    for item in items:
        try:
            answer = legacy.generate_answer(
                model, tokenizer, item["prompt"], max_tokens=item["max_tokens"]
            )
        except (ValueError, FloatingPointError, RuntimeError) as exc:
            if isinstance(exc, RuntimeError) and not re.search(
                "nonfinite|invalid token|decode", str(exc), re.IGNORECASE
            ):
                raise
            scores.append(_invalid_decode(exc))
        else:
            scores.append(_grade_output(item, answer))
    return scores, {"backend": "serial_greedy"}


def _metric(rows):
    return {"passed": sum(bool(row["passed"]) for row in rows), "total": len(rows)}


def evaluate_pairs(model, tokenizer, probes):
    probes = list(probes)
    if not probes or any(len(p.get("items", [])) != 2 for p in probes):
        raise ValueError("Each V2 pair requires two items")
    if len({p["pair_id"] for p in probes}) != len(probes):
        raise ValueError("Duplicate pair IDs")
    items = [i for p in probes for i in p["items"]]
    splits = {i["split"] for i in items}
    if splits not in ({"dev"}, {"final"}) or len({i["id"] for i in items}) != len(items):
        raise ValueError("Mixed splits or duplicate item IDs")
    for p in probes:
        s = p["strata"]
        if {s.get("input_regime"), s.get("wording_regime")} - {"familiar", "novel"}:
            raise ValueError("Unknown pair transfer stratum")
        for i in p["items"]:
            if (
                i["pair_id"] != p["pair_id"]
                or i["bucket"] != p["family"]
                or i["strata"] != s
                or not i["prompt"].strip()
                or not 1 <= i["max_tokens"] <= 256
                or i["rubric"]["kind"] not in ("exact", "code", "keywords")
            ):
                raise ValueError("Malformed pair: " + p["pair_id"])
    results, families = [], defaultdict(list)
    groups, feature_groups, family_transfer = (defaultdict(list) for _ in range(3))
    scores, backend = _score_items(model, tokenizer, items)
    grades = iter(scores)
    for p in probes:
        answers = [
            dict(id=i["id"], prompt=i["prompt"], bucket=i["bucket"], **next(grades)) for i in p["items"]
        ]
        row = {
            "pair_id": p["pair_id"],
            "bucket": p["family"],
            "strata": p["strata"],
            "features": p.get("features", {}),
            "passed": all(a["passed"] for a in answers),
            "items": answers,
        }
        results.append(row)
        families[p["family"]].append(row)
        s = p["strata"]
        group = f"input={s['input_regime']},wording={s['wording_regime']}"
        groups[group].append(row)
        family_transfer[f"{p['family']}/{group}"].append(row)
        for feature in ("add_carry", "subtract_sign", "subtract_borrow"):
            value = row["features"].get(feature)
            if value is not None:
                feature_groups[f"{p['family']}/{feature}={value}"].append(row)
        for operation in sorted(set(row["features"].get("operations", []))):
            feature_groups[f"{p['family']}/includes_{operation}"].append(row)
    buckets = {}
    for family, rows in families.items():
        singles = [a for r in rows for a in r["items"]]
        buckets[family] = {
            "pairs_passed": _metric(rows)["passed"],
            "pairs_total": len(rows),
            "items_passed": _metric(singles)["passed"],
            "items_total": len(singles),
        }
    successes = _metric(results)["passed"]
    item_wins = sum(b["items_passed"] for b in buckets.values())
    return {
        "pairs_passed": successes,
        "pairs_total": len(results),
        "pair_rate": successes / len(results),
        "items_passed": item_wins,
        "items_total": len(items),
        "item_rate": item_wins / len(items),
        "buckets": buckets,
        "strata": {k: _metric(v) for k, v in sorted(groups.items())},
        "family_by_stratum": {k: _metric(v) for k, v in sorted(family_transfer.items())},
        "feature_pair_scores": {k: _metric(v) for k, v in sorted(feature_groups.items())},
        "rows": results,
        "split": next(iter(splits)),
        "suite_fingerprint": digest(probes),
        "grader_sha256": GRADER_SHA,
        "version": VERSION,
        "code_pass_at_1": (buckets["code"]["items_passed"] / buckets["code"]["items_total"])
        if "code" in buckets
        else None,
        "decode_errors": sum("decode_error" in i for r in results for i in r["items"]),
        "decoding": {
            **backend,
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "samples": 1,
            "require_eos": True,
        },
    }


def evaluate_suite(model, tokenizer, suite):
    if suite.get("fingerprint") != legacy.fingerprint(suite) or suite["split"] not in ("dev", "final"):
        raise ValueError("Malformed frozen broad suite")
    items = suite["items"]
    if Counter(i["bucket"] for i in items) != Counter({k: 8 for k in legacy.BUCKETS}):
        raise ValueError("Expected eight frozen questions per bucket")
    if len({i["id"] for i in items}) != len(items) or len({i["prompt"] for i in items}) != len(items):
        raise ValueError("Duplicate broad prompts or IDs")
    scores, backend = _score_items(model, tokenizer, items)
    rows = [
        dict(id=i["id"], bucket=i["bucket"], prompt=i["prompt"], **grade)
        for i, grade in zip(items, scores)
    ]
    buckets = {k: _metric([r for r in rows if r["bucket"] == k]) for k in legacy.BUCKETS}
    for b in buckets.values():
        b["rate"] = b["passed"] / b["total"]
    macro = sum(b["rate"] for b in buckets.values()) / len(buckets)
    repeated = sum(r["repetition_failure"] for r in rows) / len(rows)
    nonempty = sum(r["nonempty"] for r in rows) / len(rows)
    errors = sum("decode_error" in r for r in rows)
    c = legacy.CRITERIA
    qualified = (
        not errors
        and macro >= c["min_macro_score"]
        and min(b["rate"] for b in buckets.values()) >= c["min_bucket_score"]
        and repeated <= c["max_repetition_failure_rate"]
        and nonempty >= c["min_nonempty_rate"]
    )
    return {
        "macro_score": macro,
        "qualified": qualified,
        "buckets": buckets,
        "rows": rows,
        "nonempty_rate": nonempty,
        "repetition_failure_rate": repeated,
        "suite_fingerprint": suite["fingerprint"],
        "grader_sha256": GRADER_SHA,
        "version": VERSION,
        "split": suite["split"],
        "scope": SCOPE,
        "code_pass_at_1": buckets["code"]["rate"],
        "decode_errors": errors,
        "decoding": {
            **backend,
            "temperature": 0.0,
            "repetition_penalty": 1.0,
            "samples": 1,
            "require_eos": True,
        },
    }


def blind_review_packet(reports):
    """Deduplicate and blind all knowledge answers, including automatic passes.

    Share only packet with the reviewer. The key links responses back to arms.
    Human annotations never silently replace the preregistered automated gate.
    """
    packet, key = {}, defaultdict(list)
    for name, report in reports.items():
        for row in report["rows"]:
            if row["bucket"] != "knowledge":
                continue
            rid = digest([row["prompt"], row["text"]])
            packet[rid] = {"id": rid, "prompt": row["prompt"], "answer": row["text"]}
            key[rid].append({"arm": name, "item": row["id"], "auto_pass": row["passed"]})
    return {"packet": [packet[k] for k in sorted(packet)], "key": dict(key), "scope": SCOPE}


def evaluate_dev(model, tokenizer, probes, broad_suite, heldout_rows):
    if broad_suite["split"] != "dev":
        raise ValueError("Development evaluation requires dev broad suite")
    previous = model.training
    try:
        probes = list(probes)
        if not probes or any(i["split"] != "dev" for p in probes for i in p["items"]):
            raise ValueError("Development evaluation requires dev pairs")
        p = evaluate_pairs(model, tokenizer, probes)
        b = evaluate_suite(model, tokenizer, broad_suite)
        selected = {k: sorted(heldout_rows[k], key=lambda v: v["id"])[:RETENTION_ROWS] for k in SOURCES}
        if any(
            len(rows) != RETENTION_ROWS or len({r["id"] for r in rows}) != RETENTION_ROWS
            for rows in selected.values()
        ):
            raise ValueError("Missing or duplicate fixed answer-retention rows")
        from chat_recovery_train import evaluate as evaluate_answers

        ret = evaluate_answers(model, selected, RETENTION_ROWS)
        ret["rows_fingerprint"] = digest(
            {k: [(r["id"], r["input_ids"], r["labels"]) for r in rows] for k, rows in selected.items()}
        )
        return {"paired": p, "broad": b, "answer_validation": ret, "grader_sha256": GRADER_SHA}
    finally:
        model.train(previous)


def screen_candidate(baseline, control, candidate):
    reports = baseline, control, candidate
    for name in ("paired", "broad"):
        parts = [r[name] for r in reports]
        if len({r["suite_fingerprint"] for r in parts}) != 1 or any(
            r["split"] != "dev" or r["grader_sha256"] != GRADER_SHA for r in parts
        ):
            raise ValueError("Mismatched V2 development probes or grader")
    retention = [r["answer_validation"] for r in reports]
    if len({r["rows_fingerprint"] for r in retention}) != 1:
        raise ValueError("Different validation rows")
    margin = max(2, math.ceil(0.05 * candidate["paired"]["pairs_total"]))
    checks = {
        "valid_decodes": all(not r[k]["decode_errors"] for r in reports for k in ("paired", "broad")),
        "paired_improves_over_both": candidate["paired"]["pairs_passed"]
        >= margin + max(r["paired"]["pairs_passed"] for r in reports[:2]),
        "broad_buckets_retained": all(
            candidate["broad"]["buckets"][k]["passed"]
            >= max(r["broad"]["buckets"][k]["passed"] for r in reports[:2])
            for k in legacy.BUCKETS
        ),
        "answer_bpt_retained": all(math.isfinite(r["val_answer_bpt"]) for r in retention)
        and retention[2]["val_answer_bpt"] <= min(r["val_answer_bpt"] for r in retention[:2]) + 0.5,
    }
    return {
        "eligible": all(checks.values()),
        "checks": checks,
        "reasons": [k for k, v in checks.items() if not v],
        "min_pair_gain": margin,
        "max_answer_bpt_regression": 0.5,
        "grader_sha256": GRADER_SHA,
    }


def final_decision(screen, broad_dev, paired_final, broad_final=None):
    if screen["grader_sha256"] != GRADER_SHA:
        raise ValueError("Unknown development screen")
    if not screen["eligible"]:
        if paired_final is not None or broad_final is not None:
            raise ValueError("Final results supplied for an ineligible candidate")
        return {"status": "screen_failed", "eligible_for_native_export": False}
    if broad_dev["split"] != "dev" or broad_dev["grader_sha256"] != GRADER_SHA:
        raise ValueError("Wrong broad development report")
    if paired_final["split"] != "final" or paired_final["grader_sha256"] != GRADER_SHA:
        raise ValueError("Wrong final pair report")
    expected_groups = {
        f"input={i},wording={w}" for i in ("familiar", "novel") for w in ("familiar", "novel")
    }
    if (
        set(paired_final["buckets"]) != set(FAMILIES)
        or set(paired_final["strata"]) != expected_groups
        or any(v["pairs_total"] <= 0 for v in paired_final["buckets"].values())
        or any(v["total"] <= 0 for v in paired_final["strata"].values())
    ):
        raise ValueError("Final evaluation missing a family or transfer group")
    passed = (
        not paired_final["decode_errors"]
        and paired_final["pair_rate"] >= 2 / 3
        and all(v["pairs_passed"] * 2 >= v["pairs_total"] for v in paired_final["buckets"].values())
        and all(v["passed"] * 2 >= v["total"] for v in paired_final["strata"].values())
    )
    if broad_final is not None and (
        not passed
        or not broad_dev["qualified"]
        or broad_final["grader_sha256"] != GRADER_SHA
        or broad_final["split"] != "final"
    ):
        raise ValueError("Broad final report supplied outside the preregistered gate")
    if not passed:
        status = "final_pairs_failed"
    elif not broad_dev["qualified"]:
        status = "paired_transfer_only"
    elif broad_final is None:
        status = "broad_final_required"
    else:
        status = (
            "qualified"
            if broad_final["qualified"] and not broad_final["decode_errors"]
            else "paired_transfer_only"
        )
    return {
        "status": status,
        "paired_passed": bool(passed),
        "eligible_for_native_export": status == "qualified",
    }
