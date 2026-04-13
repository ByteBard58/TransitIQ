"""
conftest.py – session-scoped fixtures for the TransitIQ test suite.

Strategy
--------
The FastAPI app loads ML model artifacts (pipe.pkl, column_names.pkl) during
its lifespan startup via ``initialize_artifacts()``. In CI those files are not
present, and downloading them from Hugging Face would be slow and fragile.

We therefore:
1. Stub out the ``models`` package so the bare import doesn't fail.
2. Patch ``initialize_artifacts`` to return lightweight MagicMock objects.
3. Expose a ``TestClient`` fixture whose lifespan uses those mocks.
"""

import sys
import io
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# ── 1. Stub heavy / optional dependencies before the app is imported ─────────
# Prevents import errors in CI where model artifacts or HF credentials may
# not be available.
_models_stub = MagicMock()
sys.modules.setdefault("models", _models_stub)
sys.modules.setdefault("models.download_from_hf", _models_stub)

# ── 2. Column order must exactly match app/schema/validate.py ────────────────
COLUMN_NAMES = np.array([
    "koi_period",
    "koi_time0bk",
    "koi_depth",
    "koi_prad",
    "koi_sma",
    "koi_incl",
    "koi_teq",
    "koi_insol",
    "koi_impact",
    "koi_ror",
    "koi_srho",
    "koi_dor",
    "koi_num_transits",
])


@pytest.fixture(scope="session")
def mock_pipe():
    """A minimal sklearn-compatible pipeline mock."""
    pipe = MagicMock()
    # Return a label and probability vector scaled by the number of rows so
    # both single-row (/predict) and multi-row (/predict/batch) calls work.
    pipe.predict.side_effect = lambda x: np.array([2] * len(x))
    pipe.predict_proba.side_effect = (
        lambda x: np.array([[0.05, 0.15, 0.80]] * len(x))
    )
    return pipe


@pytest.fixture(scope="session")
def client(mock_pipe):
    """
    Session-scoped TestClient whose lifespan uses the mock pipeline.
    ``scope="session"`` means the app boots once for the entire test run,
    which mirrors real usage and keeps the suite fast.
    """
    with patch("app.app.initialize_artifacts", return_value=(mock_pipe, COLUMN_NAMES)):
        # Import deferred so the sys.modules stubs are already in place.
        from app.app import app  # noqa: PLC0415

        with TestClient(app) as c:
            yield c
