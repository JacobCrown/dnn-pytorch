from src.core.metrics import BaseMetrics


class TestBaseMetrics:
    def test_base_metrics(self):
        losses_dict = {"train": [1, 2, 3, 4, 5], "val": [2, 3, 4, 5, 6], "test": [3.0, 4.0, 5.0, 6.0, 7.0]}
        base_metrics = BaseMetrics()
        base_metrics.losses = losses_dict
        assert base_metrics.losses == losses_dict
        base_metrics_dict = base_metrics.to_dict()
        assert base_metrics_dict["losses"] == losses_dict
        base_metrics_from_dict = BaseMetrics.from_dict(base_metrics_dict)
        assert base_metrics_from_dict.losses == losses_dict

