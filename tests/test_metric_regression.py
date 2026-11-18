import numpy as np
import pytest
import torch

from utils.metric import evaluate_cmct, evaluate_task


TASK_KEYS = {
    "precisions": "precision_avg",
    "recalls": "recall_avg",
    "f1": "f1_avg",
    "accuracys": "accuracy_avg",
}
CONCEPT_KEYS = {
    "mses": "mse_avg",
    "rmses": "rmse_avg",
    "r2s": "r2_avg",
    "maes": "mae_avg",
}


class FixedPredictions(torch.nn.Module):
    def __init__(self, logits, concepts=None):
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits, dtype=torch.float32))
        self.register_buffer("concepts", concepts)

    def forward(self, images):
        indices = images[:, 0, 0, 0].long()
        logits = self.logits[indices]
        if self.concepts is None:
            return logits
        return self.concepts[indices], logits


def evaluate_fixture(logits, labels, concept_targets, concept_predictions, batch_size):
    images = torch.arange(len(labels), dtype=torch.float32).reshape(-1, 1, 1, 1)
    targets = torch.tensor(labels, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(images, concept_targets, targets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    model = FixedPredictions(logits, concept_predictions)
    return evaluate_cmct(model, torch.device("cpu"), loader, targets.shape[1])


@pytest.mark.parametrize(
    "positive_logit, expected",
    [(0.25, 1.0), (0.75, 1.0), (0.0, 0.0), (-0.25, 0.0)],
    ids=["small-positive", "large-positive", "zero", "negative"],
)
def test_cmct_task_logit_threshold(positive_logit, expected):
    concepts = torch.tensor([[1.0], [2.0]])
    result = evaluate_fixture(
        [[positive_logit], [-0.25]], [[1], [0]], concepts, concepts, batch_size=2
    )
    assert (result["f1_avg"], result["accuracy_avg"]) == (expected, (expected + 1.0) / 2)


@pytest.fixture
def multi_label_case():
    logits = [[0.25, -0.25], [0.75, 0.75], [0.75, 0.25], [0.0, 0.0]]
    labels = [[1, 0], [1, 1], [0, 1], [0, 0]]
    concept_targets = torch.tensor([[0.0, 2.0], [2.0, 4.0], [4.0, 6.0], [6.0, 8.0]])
    concept_predictions = torch.tensor([[0.0, 4.0], [1.0, 2.0], [5.0, 8.0], [6.0, 6.0]])
    return logits, labels, concept_targets, concept_predictions


def test_cmct_aggregates_per_label_task_metrics_across_batches(multi_label_case):
    result = evaluate_fixture(*multi_label_case, batch_size=2)
    # Label 0 has two true positives, one false positive and one true
    # negative. Label 1 is predicted perfectly.
    expected = {"precisions": 2 / 3, "recalls": 1.0, "f1": 0.8, "accuracys": 0.75}
    for metric, label_zero in expected.items():
        assert result[metric] == pytest.approx([label_zero, 1.0])
        assert result[TASK_KEYS[metric]] == pytest.approx((label_zero + 1.0) / 2)


def test_cmct_task_metrics_match_baseline_evaluator(multi_label_case):
    result = evaluate_fixture(*multi_label_case, batch_size=2)
    logits, labels, _, _ = multi_label_case
    images = torch.arange(len(labels), dtype=torch.float32).reshape(-1, 1, 1, 1)
    dataset = torch.utils.data.TensorDataset(images, torch.tensor(labels, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(dataset, batch_size=2)
    baseline = evaluate_task(FixedPredictions(logits), torch.device("cpu"), loader, num_labels=2)

    assert {key: result[key] for key in baseline} == baseline


def test_cmct_preserves_metric_schema_and_concept_scores(multi_label_case):
    result = evaluate_fixture(*multi_label_case, batch_size=2)
    assert result.keys() == {*TASK_KEYS, *TASK_KEYS.values(), *CONCEPT_KEYS, *CONCEPT_KEYS.values()}
    for key, avg_key in TASK_KEYS.items():
        assert isinstance(result[key], list)
        assert len(result[key]) == 2
        assert type(result[avg_key]) is float

    expected_concepts = {
        "mses": [0.5, 4.0],
        "rmses": [np.sqrt(0.5), 2.0],
        "r2s": [0.9, 0.2],
        "maes": [0.5, 2.0],
    }
    for key, expected in expected_concepts.items():
        assert isinstance(result[key], np.ndarray)
        assert result[key].shape == (2,)
        np.testing.assert_allclose(result[key], expected)
        avg_key = CONCEPT_KEYS[key]
        assert type(result[avg_key]) is float
        assert result[avg_key] == pytest.approx(np.mean(expected))
