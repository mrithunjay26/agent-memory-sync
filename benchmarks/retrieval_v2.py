"""Retrieval benchmark v2: built so lexical and semantic retrieval score differently.

v1 could not discriminate. Its "paraphrased" queries repeated the document's own
words (slug, subsystem, and the exact quality phrase), and every query carried a
unique slug, so plain keyword matching solved every category. A dense retriever
could not look better than BM25 because there was nothing for it to be better at.

v2 fixes that with two deliberately opposed categories plus distractors:

  exact_lookup
      The query contains a unique identifier that appears verbatim in the target.
      Lexical search should win here. Kept so a dense-only system cannot look
      good by accident.

  paraphrased_decision_recall
      The query and its target share NO content words at all. It describes the
      same situation in different vocabulary. Keyword search has nothing to match,
      so this isolates semantic recall.

  hard negatives
      For every scenario a distractor document repeats the query's salient words
      while answering something else. A system that only pattern-matches keywords
      ranks these above the real answer.

ZERO_OVERLAP is enforced, not asserted in prose: assert_zero_overlap() recomputes
the intersection from the dataset and raises if any paraphrase query shares a
content token with its target. tests/test_benchmark_dataset.py runs it.
"""
from __future__ import annotations

import re

# Words ignored when checking lexical overlap. Deliberately small: only true
# function words, so we never hide a real content-word collision.
STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has have
    how i if in into is it its of on or our so should that the their then there
    these they this to was we were what when where which while who why will with
    without you your does not no than them us about after before during between
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def content_tokens(text: str) -> set[str]:
    """Content words used for the overlap check, matching how retrieval tokenizes."""
    return {
        token
        for token in _TOKEN_RE.findall((text or "").casefold())
        if token not in STOPWORDS and len(token) > 1
    }


# Each scenario carries:
#   marker    unique id that appears only in the exact-lookup document
#   decision  the target document for the paraphrase query
#   question  paraphrase query, written to share no content word with `decision`
#   negative  distractor that echoes the question's words but answers nothing
SCENARIOS = (
    {
        "slug": "atlas",
        "marker": "AMS-001-ATLAS",
        "decision": (
            "Decision for the atlas scheduler: the team chose write-ahead "
            "journaling so queued entries survive an abrupt halt."
        ),
        "question": (
            "How does the timing service avoid losing pending tasks when a "
            "machine dies unexpectedly?"
        ),
        "negative": (
            "The timing service dashboard lists pending tasks per machine and "
            "highlights anything unexpectedly idle."
        ),
    },
    {
        "slug": "beacon",
        "marker": "AMS-002-BEACON",
        "decision": (
            "Decision for the beacon identity gateway: issue short-lived "
            "capability tokens so a leaked secret expires quickly."
        ),
        "question": (
            "What stops a stolen login credential from being useful forever?"
        ),
        "negative": (
            "Support runbook: if a user reports a stolen login credential, "
            "open a ticket and mark the account useful for audit."
        ),
    },
    {
        "slug": "cedar",
        "marker": "AMS-003-CEDAR",
        "decision": (
            "Decision for the cedar billing worker: idempotency keys guard "
            "every charge attempt so retries never bill twice."
        ),
        "question": (
            "How do we make sure a customer is not invoiced two times if the "
            "payment call repeats?"
        ),
        "negative": (
            "The finance report shows how often a customer is invoiced and "
            "whether the payment call repeats within a cycle."
        ),
    },
    {
        "slug": "delta",
        "marker": "AMS-004-DELTA",
        "decision": (
            "Decision for the delta event relay: bounded retry queues apply "
            "backpressure instead of flooding downstream consumers."
        ),
        "question": (
            "What keeps a slow subscriber from being overwhelmed by a burst of "
            "traffic?"
        ),
        "negative": (
            "Metrics note: a slow subscriber can be overwhelmed during a burst "
            "of traffic, and this chart plots that rate."
        ),
    },
    {
        "slug": "fjord",
        "marker": "AMS-006-FJORD",
        "decision": (
            "Decision for the fjord search indexer: content hash invalidation "
            "keeps results fresh without a full rebuild."
        ),
        "question": (
            "How does lookup stay current after a document changes, but skip "
            "reprocessing everything?"
        ),
        "negative": (
            "Runbook: lookup latency stays current in this graph even after a "
            "document changes, though reprocessing everything is slow."
        ),
    },
    {
        "slug": "grove",
        "marker": "AMS-007-GROVE",
        "decision": (
            "Decision for the grove audit service: append-only segments make "
            "tampering evident because history cannot be rewritten."
        ),
        "question": (
            "How can we prove nobody quietly altered an older record?"
        ),
        "negative": (
            "The compliance page explains who may prove access and which older "
            "record fields an operator altered last quarter."
        ),
    },
    {
        "slug": "harbor",
        "marker": "AMS-008-HARBOR",
        "decision": (
            "Decision for the harbor upload coordinator: multipart checkpoints "
            "let a transfer resume instead of starting over."
        ),
        "question": (
            "If a big file send drops halfway, how do we continue from where it "
            "stopped?"
        ),
        "negative": (
            "The status page shows whether a big file send drops and how often "
            "users continue from a stopped session."
        ),
    },
    {
        "slug": "juniper",
        "marker": "AMS-010-JUNIPER",
        "decision": (
            "Decision for the juniper job dispatcher: lease-based ownership "
            "guarantees a single worker holds a task at a time."
        ),
        "question": (
            "What prevents two machines from picking up the same unit of work "
            "simultaneously?"
        ),
        "negative": (
            "Capacity planning: two machines can be added to pick up more units "
            "of work simultaneously during peak hours."
        ),
    },
    {
        "slug": "kepler",
        "marker": "AMS-011-KEPLER",
        "decision": (
            "Decision for the kepler metrics collector: cardinality budgets cap "
            "distinct label values so memory stays stable."
        ),
        "question": (
            "How do we stop unbounded tag variety from exhausting RAM?"
        ),
        "negative": (
            "The tagging guide describes unbounded tag variety and how "
            "exhausting RAM appears in a profile."
        ),
    },
    {
        "slug": "mesa",
        "marker": "AMS-013-MESA",
        "decision": (
            "Decision for the mesa cache warmer: generation counters reject a "
            "stale write that arrives after a newer value."
        ),
        "question": (
            "How is an out-of-order update discarded once fresher data landed?"
        ),
        "negative": (
            "Incident notes: an out-of-order update was discarded manually "
            "before fresher data landed in the report."
        ),
    },
    {
        "slug": "northstar",
        "marker": "AMS-014-NORTHSTAR",
        "decision": (
            "Decision for the northstar release controller: two-phase rollout "
            "allows an immediate revert when errors climb."
        ),
        "question": (
            "How can a bad deploy be undone quickly after failures spike?"
        ),
        "negative": (
            "The deploy calendar marks when a bad deploy was undone and which "
            "failures spike during business hours."
        ),
    },
    {
        "slug": "onyx",
        "marker": "AMS-015-ONYX",
        "decision": (
            "Decision for the onyx secret broker: envelope encryption keeps the "
            "master key separate from encrypted payloads."
        ),
        "question": (
            "How is the top-level cipher material isolated from stored data?"
        ),
        "negative": (
            "Glossary entry: top-level cipher material is isolated in hardware, "
            "and stored data volumes are listed elsewhere."
        ),
    },
    {
        "slug": "quartz",
        "marker": "AMS-017-QUARTZ",
        "decision": (
            "Decision for the quartz query planner: cost-based routing chooses "
            "a plan that keeps latency predictable."
        ),
        "question": (
            "How does the engine pick an execution path so response times stay "
            "steady?"
        ),
        "negative": (
            "The engine pick list shows every execution path and the response "
            "times recorded, but it stays informational."
        ),
    },
    {
        "slug": "raven",
        "marker": "AMS-018-RAVEN",
        "decision": (
            "Decision for the raven webhook receiver: signature-first parsing "
            "rejects a forged payload before any field is read."
        ),
        "question": (
            "How do we confirm an inbound message really came from the sender "
            "prior to trusting its body?"
        ),
        "negative": (
            "The integration guide explains how an inbound message from the "
            "sender is shaped and what its body contains."
        ),
    },
    {
        "slug": "summit",
        "marker": "AMS-019-SUMMIT",
        "decision": (
            "Decision for the summit policy engine: deny-by-default rules mean "
            "an unmatched request is refused."
        ),
        "question": (
            "What happens to an action nobody explicitly permitted?"
        ),
        "negative": (
            "The audit log lists each action and who explicitly permitted it "
            "over the last month."
        ),
    },
    {
        "slug": "umbra",
        "marker": "AMS-021-UMBRA",
        "decision": (
            "Decision for the umbra session manager: rotating refresh families "
            "detect a replayed token and revoke the lineage."
        ),
        "question": (
            "How do we notice somebody reusing an old sign-in ticket and cut "
            "off that chain?"
        ),
        "negative": (
            "Support macro: if somebody reusing an old sign-in ticket calls in, "
            "cut off the chat and escalate."
        ),
    },
    {
        "slug": "vale",
        "marker": "AMS-022-VALE",
        "decision": (
            "Decision for the vale migration runner: advisory locks serialize "
            "schema changes so two deploys cannot collide."
        ),
        "question": (
            "What keeps concurrent database upgrades from running at the same "
            "moment?"
        ),
        "negative": (
            "The upgrade FAQ covers concurrent database upgrades and why "
            "running at the same moment feels slow."
        ),
    },
    {
        "slug": "xenon",
        "marker": "AMS-024-XENON",
        "decision": (
            "Decision for the xenon rate limiter: sliding window counters keep "
            "a burst fair across tenants."
        ),
        "question": (
            "How do we stop one noisy caller from consuming everyone else's "
            "quota in a short spike?"
        ),
        "negative": (
            "Billing note: one noisy caller consuming quota in a short spike "
            "will still be charged at the standard rate."
        ),
    },
    {
        "slug": "zephyr",
        "marker": "AMS-026-ZEPHYR",
        "decision": (
            "Decision for the zephyr edge proxy: hedged upstream requests trim "
            "tail latency by racing a duplicate."
        ),
        "question": (
            "How are the slowest few percent of responses sped up?"
        ),
        "negative": (
            "The SLO doc defines the slowest percent of responses and how they "
            "are sped up manually during incidents."
        ),
    },
    {
        "slug": "dune",
        "marker": "AMS-030-DUNE",
        "decision": (
            "Decision for the dune backup verifier: sampled restore drills "
            "prove an archive can actually be recovered."
        ),
        "question": (
            "How do we gain confidence that saved copies are usable before a "
            "real disaster?"
        ),
        "negative": (
            "The storage bill breaks down saved copies, their usable capacity, "
            "and confidence intervals before renewal."
        ),
    },
)


def _exact_document(scenario: dict) -> str:
    return (
        f"Lookup evidence for incident {scenario['marker']}. The failing "
        f"artifact is config/{scenario['slug']}-route.yaml."
    )


def _exact_query(scenario: dict) -> str:
    return f"Find incident {scenario['marker']}."


def load_dataset() -> dict:
    documents: list[dict] = []
    queries: list[dict] = []

    for number, scenario in enumerate(SCENARIOS, 1):
        slug = scenario["slug"]
        agent = "codex" if number % 2 else "claude-code"

        exact_id = f"{slug}:exact_lookup"
        decision_id = f"{slug}:paraphrased_decision_recall"
        negative_id = f"{slug}:hard_negative"

        for document_id, text in (
            (exact_id, _exact_document(scenario)),
            (decision_id, scenario["decision"]),
            (negative_id, scenario["negative"]),
        ):
            documents.append(
                {
                    "id": document_id,
                    "agent": agent,
                    "event_type": "history",
                    "summary": text,
                    "text": text,
                }
            )

        queries.append(
            {
                "id": f"q{number:03d}:exact_lookup",
                "category": "exact_lookup",
                "query": _exact_query(scenario),
                "relevance": {exact_id: 3},
            }
        )
        queries.append(
            {
                "id": f"q{number:03d}:paraphrased_decision_recall",
                "category": "paraphrased_decision_recall",
                "query": scenario["question"],
                "relevance": {decision_id: 3},
            }
        )

    return {
        "schema_version": 2,
        "name": "agentmemorysync-retrieval-v2",
        "description": (
            "Synthetic regression suite with an enforced zero-lexical-overlap "
            "paraphrase split and one hard negative per scenario. It measures "
            "retrieval only and is not evidence of real-world task completion."
        ),
        "documents": documents,
        "queries": queries,
    }


def overlap_report() -> list[dict]:
    """Content words shared between each paraphrase query and its target."""
    dataset = load_dataset()
    by_id = {document["id"]: document["text"] for document in dataset["documents"]}
    rows = []
    for query in dataset["queries"]:
        if query["category"] != "paraphrased_decision_recall":
            continue
        target_id = next(iter(query["relevance"]))
        shared = content_tokens(query["query"]) & content_tokens(by_id[target_id])
        rows.append({"query_id": query["id"], "target": target_id, "shared": sorted(shared)})
    return rows


def assert_zero_overlap() -> None:
    """Fail loudly if a paraphrase query reuses any content word from its target.

    This is what makes the paraphrase split meaningful: without it the category
    silently degrades back into keyword matching, which is exactly how v1 broke.
    """
    violations = [row for row in overlap_report() if row["shared"]]
    if violations:
        detail = "; ".join(
            f"{row['query_id']} shares {row['shared']} with {row['target']}"
            for row in violations
        )
        raise AssertionError(f"paraphrase queries must share no content words: {detail}")


if __name__ == "__main__":
    assert_zero_overlap()
    data = load_dataset()
    print(
        f"{data['name']}: {len(data['documents'])} documents, "
        f"{len(data['queries'])} queries, zero lexical overlap verified."
    )
