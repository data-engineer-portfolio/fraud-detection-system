# tests/test_features.py
import pytest
from unittest.mock import MagicMock
from fraud_detection.features.transformers import (
    AmountTransformer,
    TimeTransformer,
    PCAInteractionTransformer,
    VelocityTransformer,
)


class TestAmountTransformer:

    def test_instantiates(self):
        assert AmountTransformer() is not None

    def test_has_transform_method(self):
        assert hasattr(AmountTransformer(), "transform")

    def test_transform_returns_dataframe(self):
        """Transform must return a DataFrame not None."""
        t  = AmountTransformer()
        df = MagicMock()
        df.columns = (
            ["Time", "Amount", "Class"]
            + [f"V{i}" for i in range(1, 29)]
        )
        df.count.return_value = 1000
        df.withColumn.return_value = df
        df.select.return_value.collect.return_value = [
            MagicMock(**{"__getitem__.side_effect":
                lambda k: 100.0 if k == "mean" else 50.0})
        ]
        result = t.transform(df)
        assert result is not None


class TestTimeTransformer:

    def test_instantiates(self):
        assert TimeTransformer() is not None

    def test_seconds_constants_correct(self):
        t = TimeTransformer()
        assert t.SECONDS_IN_HOUR == 3600
        assert t.SECONDS_IN_DAY  == 86400

    def test_has_transform_method(self):
        assert hasattr(TimeTransformer(), "transform")


class TestPCAInteractionTransformer:

    def test_instantiates(self):
        assert PCAInteractionTransformer() is not None

    def test_top_features_contains_v14(self):
        """V14 must be in top features — strongest fraud signal."""
        t = PCAInteractionTransformer()
        assert "V14" in t.TOP_FEATURES

    def test_top_features_length(self):
        t = PCAInteractionTransformer()
        assert len(t.TOP_FEATURES) == 6

    def test_has_transform_method(self):
        assert hasattr(PCAInteractionTransformer(), "transform")


class TestVelocityTransformer:

    def test_instantiates(self):
        assert VelocityTransformer() is not None

    def test_windows_defined(self):
        t = VelocityTransformer()
        assert 100  in t.WINDOWS
        assert 500  in t.WINDOWS
        assert 1000 in t.WINDOWS

    def test_has_transform_method(self):
        assert hasattr(VelocityTransformer(), "transform")