"""
test_api.py – integration and schema tests for the TransitIQ FastAPI app.

Run from the project root:
    pytest tests/ -v

The TestClient and all mocking are set up in conftest.py.
"""

import io
import pytest
import pandas as pd
from pydantic import ValidationError

# ── Shared test data ──────────────────────────────────────────────────────────

#: Column order matches app/schema/validate.py and conftest.COLUMN_NAMES.
_COLUMNS = [
    "koi_period", "koi_time0bk", "koi_depth", "koi_prad", "koi_sma",
    "koi_incl", "koi_teq", "koi_insol", "koi_impact", "koi_ror",
    "koi_srho", "koi_dor", "koi_num_transits",
]

#: A valid, boundary-safe payload for /predict.
VALID_PAYLOAD: dict = {
    "koi_period": 10.0,
    "koi_time0bk": 2454834.0,
    "koi_depth": 500.0,
    "koi_prad": 1.5,
    "koi_sma": 0.1,
    "koi_incl": 85.0,
    "koi_teq": 400.0,
    "koi_insol": 2.0,
    "koi_impact": 0.3,
    "koi_ror": 0.05,
    "koi_srho": 1.2,
    "koi_dor": 10.0,
    "koi_num_transits": 5,
}

_VALID_LABELS = {"FALSE POSITIVE", "CANDIDATE", "CONFIRMED"}


def _make_csv(rows: list | None = None) -> bytes:
    """Build a well-formed CSV payload from a list of row dicts."""
    if rows is None:
        rows = [VALID_PAYLOAD]
    return pd.DataFrame(rows, columns=_COLUMNS).to_csv(index=False).encode()


# ── Health check ──────────────────────────────────────────────────────────────

class TestHealthRoute:
    def test_status_code(self, client):
        assert client.get("/health").status_code == 200

    def test_body_title(self, client):
        assert client.get("/health").json()["title"] == "TransitIQ"

    def test_body_status_message(self, client):
        assert client.get("/health").json()["status"] == "All systems operational"


# ── Static / HTML routes ──────────────────────────────────────────────────────

class TestStaticRoutes:
    def test_home_returns_200(self, client):
        assert client.get("/").status_code == 200

    def test_home_content_type(self, client):
        assert "text/html" in client.get("/").headers["content-type"]

    def test_about_returns_200(self, client):
        assert client.get("/about").status_code == 200

    def test_about_content_type(self, client):
        assert "text/html" in client.get("/about").headers["content-type"]


# ── POST /predict ─────────────────────────────────────────────────────────────

class TestPredictEndpoint:
    # --- Happy path -----------------------------------------------------------

    def test_valid_payload_returns_201(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 201

    def test_response_has_required_keys(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert {"status", "prediction", "probabilities"} <= data.keys()

    def test_status_is_success(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert data["status"] == "success"

    def test_prediction_is_valid_label(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert data["prediction"] in _VALID_LABELS

    def test_probabilities_keys_match_labels(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert set(data["probabilities"].keys()) == _VALID_LABELS

    def test_probabilities_sum_to_one(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        total = sum(data["probabilities"].values())
        assert abs(total - 1.0) < 0.01

    # --- Validation errors ----------------------------------------------------

    def test_missing_field_returns_422(self, client):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "koi_period"}
        assert client.post("/predict", json=payload).status_code == 422

    def test_koi_period_zero_returns_422(self, client):
        """koi_period must be > 0."""
        assert client.post(
            "/predict", json={**VALID_PAYLOAD, "koi_period": 0}
        ).status_code == 422

    def test_koi_period_negative_returns_422(self, client):
        assert client.post(
            "/predict", json={**VALID_PAYLOAD, "koi_period": -5.0}
        ).status_code == 422

    def test_koi_incl_above_90_returns_422(self, client):
        """koi_incl must be ≤ 90."""
        assert client.post(
            "/predict", json={**VALID_PAYLOAD, "koi_incl": 91.0}
        ).status_code == 422

    def test_koi_impact_at_one_returns_422(self, client):
        """koi_impact must be strictly < 1."""
        assert client.post(
            "/predict", json={**VALID_PAYLOAD, "koi_impact": 1.0}
        ).status_code == 422

    def test_koi_impact_above_one_returns_422(self, client):
        assert client.post(
            "/predict", json={**VALID_PAYLOAD, "koi_impact": 1.5}
        ).status_code == 422

    def test_koi_ror_at_one_returns_422(self, client):
        """koi_ror must be strictly < 1."""
        assert client.post(
            "/predict", json={**VALID_PAYLOAD, "koi_ror": 1.0}
        ).status_code == 422

    def test_koi_num_transits_zero_returns_422(self, client):
        """koi_num_transits must be ≥ 1."""
        assert client.post(
            "/predict", json={**VALID_PAYLOAD, "koi_num_transits": 0}
        ).status_code == 422

    def test_string_value_for_float_field_returns_422(self, client):
        assert client.post(
            "/predict", json={**VALID_PAYLOAD, "koi_period": "not-a-number"}
        ).status_code == 422

    def test_empty_body_returns_422(self, client):
        assert client.post("/predict", json={}).status_code == 422


# ── POST /predict/batch ───────────────────────────────────────────────────────

class TestBatchPredictEndpoint:
    # --- Happy path -----------------------------------------------------------

    def test_valid_csv_returns_201(self, client):
        resp = client.post(
            "/predict/batch",
            files={"file": ("data.csv", io.BytesIO(_make_csv()), "text/csv")},
        )
        assert resp.status_code == 201

    def test_response_has_required_keys(self, client):
        data = client.post(
            "/predict/batch",
            files={"file": ("data.csv", io.BytesIO(_make_csv()), "text/csv")},
        ).json()
        assert "status" in data
        assert "predicted_labels" in data
        assert "predction_probability" in data  # kept as-is (typo in app.py)

    def test_single_row_prediction_count(self, client):
        data = client.post(
            "/predict/batch",
            files={"file": ("data.csv", io.BytesIO(_make_csv()), "text/csv")},
        ).json()
        assert len(data["predicted_labels"]) == 1

    def test_multi_row_prediction_count(self, client):
        csv_bytes = _make_csv([VALID_PAYLOAD] * 3)
        data = client.post(
            "/predict/batch",
            files={"file": ("data.csv", io.BytesIO(csv_bytes), "text/csv")},
        ).json()
        assert len(data["predicted_labels"]) == 3

    def test_labels_are_valid(self, client):
        data = client.post(
            "/predict/batch",
            files={"file": ("data.csv", io.BytesIO(_make_csv()), "text/csv")},
        ).json()
        for label in data["predicted_labels"]:
            assert label in _VALID_LABELS

    # --- Validation errors ----------------------------------------------------

    def test_non_csv_extension_returns_422(self, client):
        resp = client.post(
            "/predict/batch",
            files={"file": ("data.txt", io.BytesIO(b"some text"), "text/plain")},
        )
        assert resp.status_code == 422

    def test_json_extension_returns_422(self, client):
        resp = client.post(
            "/predict/batch",
            files={"file": ("data.json", io.BytesIO(b'{"a":1}'), "application/json")},
        )
        assert resp.status_code == 422

    def test_wrong_column_names_returns_422(self, client):
        bad_csv = b"col_a,col_b,col_c\n1.0,2.0,3.0\n"
        resp = client.post(
            "/predict/batch",
            files={"file": ("data.csv", io.BytesIO(bad_csv), "text/csv")},
        )
        assert resp.status_code == 422

    def test_extra_columns_returns_422(self, client):
        """An extra column must be rejected even if valid columns are present."""
        extra = pd.DataFrame(
            [{**VALID_PAYLOAD, "extra_col": 99.0}],
            columns=[*_COLUMNS, "extra_col"]
        ).to_csv(index=False).encode()
        resp = client.post(
            "/predict/batch",
            files={"file": ("data.csv", io.BytesIO(extra), "text/csv")},
        )
        assert resp.status_code == 422

    def test_non_numeric_values_returns_422(self, client):
        """String cell values must fail numeric validation."""
        str_rows = [{k: "abc" for k in _COLUMNS}]
        bad_csv = pd.DataFrame(str_rows, columns=_COLUMNS).to_csv(index=False).encode()
        resp = client.post(
            "/predict/batch",
            files={"file": ("data.csv", io.BytesIO(bad_csv), "text/csv")},
        )
        assert resp.status_code == 422


# ── Pydantic schema unit tests ────────────────────────────────────────────────

class TestUserInputSchema:
    """Direct unit tests for the Pydantic model – no HTTP overhead."""

    def test_valid_input_creates_model(self):
        from app.schema.validate import UserInput
        obj = UserInput(**VALID_PAYLOAD)
        assert obj.koi_num_transits == 5

    @pytest.mark.parametrize("field,bad_value", [
        ("koi_period", 0),           # must be > 0
        ("koi_period", -1.0),
        ("koi_time0bk", 100.0),      # must be > 2_450_000
        ("koi_depth", 0),            # must be > 0
        ("koi_incl", 0),             # must be > 0
        ("koi_incl", 91.0),          # must be ≤ 90
        ("koi_impact", -0.1),        # must be ≥ 0
        ("koi_impact", 1.0),         # must be < 1
        ("koi_ror", 0),              # must be > 0
        ("koi_ror", 1.0),            # must be < 1
        ("koi_dor", 0.5),            # must be > 1
        ("koi_num_transits", 0),     # must be ≥ 1
    ])
    def test_invalid_field_raises_validation_error(self, field, bad_value):
        from app.schema.validate import UserInput
        with pytest.raises(ValidationError):
            UserInput(**{**VALID_PAYLOAD, field: bad_value})
