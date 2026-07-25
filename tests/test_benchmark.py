import json

import pytest

import benchmark
from benchmarks.retrieval_v2 import (
    assert_zero_overlap,
    content_tokens,
    load_dataset,
    overlap_report,
)


QUERY_COUNT = 40
DOCUMENT_COUNT = 60


def test_dataset_shape_and_categories():
    summary = benchmark.validate_dataset(load_dataset())

    assert summary["queries"] == QUERY_COUNT
    assert summary["documents"] == DOCUMENT_COUNT
    assert summary["categories"] == {
        "exact_lookup": 20,
        "paraphrased_decision_recall": 20,
    }


def test_paraphrase_queries_share_no_content_word_with_their_target():
    """The paraphrase split is only meaningful if keyword matching cannot solve
    it. v1 failed here: its 'paraphrases' repeated the document's own words."""
    assert_zero_overlap()
    assert all(not row["shared"] for row in overlap_report())


def test_every_scenario_has_a_keyword_sharing_hard_negative():
    """Each distractor must actually look attractive to a lexical retriever,
    otherwise it is not a hard negative."""
    dataset = load_dataset()
    by_id = {document["id"]: document["text"] for document in dataset["documents"]}
    paraphrase_queries = [
        query
        for query in dataset["queries"]
        if query["category"] == "paraphrased_decision_recall"
    ]

    assert paraphrase_queries
    for query in paraphrase_queries:
        slug = next(iter(query["relevance"])).split(":", 1)[0]
        negative = by_id[f"{slug}:hard_negative"]
        shared = content_tokens(query["query"]) & content_tokens(negative)
        assert len(shared) >= 3, f"{slug} negative shares too little with the query"


def test_dataset_rejects_unknown_relevance_document():
    dataset = load_dataset()
    dataset["queries"][0]["relevance"] = {"missing": 3}

    with pytest.raises(ValueError, match="unknown documents"):
        benchmark.validate_dataset(dataset)


def test_metric_calculation_uses_rank_and_relevance_grade():
    dataset = load_dataset()
    query = dataset["queries"][0]
    relevant = next(iter(query["relevance"]))
    distractor = next(
        document["id"]
        for document in dataset["documents"]
        if document["id"] != relevant
    )
    predictions = {item["id"]: [] for item in dataset["queries"]}
    predictions[query["id"]] = [distractor, relevant]

    result = benchmark.evaluate_predictions(dataset, predictions)["overall"]

    assert result["recall_at_5"] == pytest.approx(1 / QUERY_COUNT)
    assert result["mrr"] == pytest.approx(0.5 / QUERY_COUNT)
    assert result["task_completion"] is None
    assert result["citation_correctness"] is None


def test_external_predictions_file_is_scored(tmp_path):
    dataset = load_dataset()
    first = dataset["queries"][0]
    prediction_path = tmp_path / "predictions.json"
    prediction_path.write_text(
        json.dumps({"predictions": {first["id"]: list(first["relevance"])}}),
        encoding="utf-8",
    )

    name, result = benchmark._load_predictions(f"dense={prediction_path}", dataset)

    assert name == "dense"
    assert result["overall"]["recall_at_5"] == pytest.approx(1 / QUERY_COUNT)


def test_external_predictions_reject_unknown_document_ids():
    dataset = load_dataset()
    first = dataset["queries"][0]

    with pytest.raises(ValueError, match="unknown documents"):
        benchmark.evaluate_predictions(dataset, {first["id"]: ["typo"]})


def test_hybrid_beats_lexical_on_paraphrases_without_losing_exact_lookup():
    """The claim this project makes about semantic recall, asserted as a test.

    Lexical retrieval cannot answer a query that shares no words with its
    target, so the paraphrase split isolates what the embedding leg adds.
    Exact lookup is checked at the same time to catch a dense retriever that
    buys paraphrase recall by breaking literal identifier search.
    """
    report = benchmark.run_benchmark()

    assert set(report["systems"]) == {"like", "lexical", "hybrid"}
    assert report["indexing"]["document_count"] == DOCUMENT_COUNT

    lexical = report["systems"]["lexical"]["by_category"]
    hybrid = report["systems"]["hybrid"]["by_category"]

    lexical_paraphrase = lexical["paraphrased_decision_recall"]["recall_at_5"]
    hybrid_paraphrase = hybrid["paraphrased_decision_recall"]["recall_at_5"]

    # Keyword matching has nothing to match on this split.
    assert lexical_paraphrase <= 0.05
    # The embedding leg has to earn a large, not marginal, improvement.
    assert hybrid_paraphrase >= 0.40
    assert hybrid_paraphrase > lexical_paraphrase + 0.30

    # Literal identifier lookup must not regress.
    assert lexical["exact_lookup"]["recall_at_5"] == 1.0
    assert hybrid["exact_lookup"]["recall_at_5"] == 1.0
