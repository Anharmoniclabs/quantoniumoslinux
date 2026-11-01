"""Deterministic query and assignment crossover cases shared by both CPU diagnostics."""

import itertools


def make_bank():
    """Every seen key/value token occurs in both splits, without sharing a binding."""
    bank = {}
    for split, count in (("train", 24), ("dev", 16)):
        groups = []
        for i in range(count):
            block, a = divmod(i, 8)
            b = (a + (2, 4, 6)[block]) % 8
            v = (a + 2 * block + int(split == "dev")) % 8
            w = (v + (2, 6, 4)[block]) % 8
            group = {
                "id": f"{split}/{i:02d}",
                "keys": (chr(65 + a), chr(65 + b)),
                "values": (str(v), str(w)),
            }
            groups.append(group)
        bank[split] = groups
    return bank


def cases(group):
    """Swap the values and query each key; only the last key decides what to return."""
    a, b = group["keys"]
    result = []
    for v, w in itertools.permutations(group["values"]):
        for query, answer, foil in ((a, v, w), (b, w, v)):
            result.append(
                {
                    "key1": a,
                    "value1": v,
                    "key2": b,
                    "value2": w,
                    "query": query,
                    "answer": answer,
                    "foil": foil,
                }
            )
    return result


def wording(row, regime="familiar"):
    if regime == "familiar":
        return "Records: {key1}={value1}; {key2}={value2}. Return the value of {query}. Reply with one digit.".format(
            **row
        )
    if regime == "new":
        return "For {key1} the digit is {value1}. For {key2} the digit is {value2}. Which digit goes with {query}? Reply with one digit.".format(
            **row
        )
    raise ValueError("regime must be familiar or new")
