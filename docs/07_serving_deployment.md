# ClearML Serving Deployment

This project now supports deploying the promoted `champion` model to ClearML
Serving after the training pipeline promotes a candidate.

## Manual Deployment

Run this first before relying on the pipeline step:

```bash
python -m scripts.py.serving.deploy_clearml_serving \
  --service_id "$CLEARML_SERVING_SERVICE_ID"
```

If no service exists yet, set:

```bash
AUTO_CREATE_CLEARML_SERVING_SERVICE=true
```

The deploy command:

1. Resolves the published ClearML model tagged `champion`.
2. Adds it to the ClearML Serving service.
3. Uses `serving/clearml_preprocess.py` as the request adapter.
4. Stores endpoint metadata on the ClearML model.

## Verification

Check runtime reachability:

```bash
python -m scripts.py.serving.verify_clearml_serving
```

Check a model request:

```bash
python -m scripts.py.serving.verify_clearml_serving \
  --payload_json examples/serving_payload.json
```

Default endpoint URL:

```text
http://127.0.0.1:8082/serve/deposit_cashflow/<endpoint-version>
```

## Pipeline Wiring

The training pipeline now appends:

```text
promote_champion -> deploy_serving -> verify_endpoint
```

It also deploys published candidate models to a separate staging endpoint:

```text
register -> deploy_candidate_serving -> verify_candidate_endpoint
```

Candidate serving does not require champion promotion. If `register_model` does
not publish a candidate, candidate deployment is skipped. If staging verification
cannot reach the runtime, the verification result is recorded without failing the
training pipeline.

Register templates again before running the updated pipeline:

```bash
python register_templates.py
```

The registration script writes new template IDs to `.env`:

```text
TEMPLATE_DEPLOY_SERVING_ID=...
TEMPLATE_VERIFY_ENDPOINT_ID=...
TEMPLATE_DEPLOY_CANDIDATE_SERVING_ID=...
TEMPLATE_VERIFY_CANDIDATE_ENDPOINT_ID=...
```

Default endpoints are horizon-specific. With `FORECAST_HORIZONS=7,14,30`, the
training pipeline deploys separate candidate and champion endpoints per horizon:

```text
Champion h7 : http://127.0.0.1:8082/serve/deposit_cashflow_h7/<endpoint-version>
Champion h14: http://127.0.0.1:8082/serve/deposit_cashflow_h14/<endpoint-version>
Champion h30: http://127.0.0.1:8082/serve/deposit_cashflow_h30/<endpoint-version>
Candidate   : http://127.0.0.1:8082/serve/deposit_cashflow_candidate_h<days>/<model-id>
```

## Request Contract

The serving preprocess accepts:

```json
{
  "horizon": 7,
  "history": [
    {"date": "2026-01-01", "cashflow": 120000000}
  ]
}
```

It also accepts `rows` or `data` instead of `history`, and `ds`/`y` instead of
`date`/`cashflow`.
