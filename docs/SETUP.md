# Setup and Operations

## Local Docker demo

From a clone of the repository, run `docker compose up --build -d`. The stack starts local TiDB, applies migrations, loads fixtures, and scores them before launching the dashboard and chatbot. It doesn't read your host `.env`, publish the TiDB port, or mount your local data.

| Component | Default address |
| --- | --- |
| Dashboard | `http://127.0.0.1:8502` |
| Chatbot interface | `http://127.0.0.1:5002/chat` |
| Chatbot API | `http://127.0.0.1:5002` |

If the ports are occupied, use `DASHBOARD_PORT=8510 CHATBOT_PORT=5010 docker compose up --build -d` in a POSIX shell. If startup stalls, run `docker compose ps -a` and `docker compose logs bootstrap tidb`. `docker compose down` removes the local demo containers; it does not touch your research files or TiDB Cloud.

To confirm that reruns do not duplicate scores:

```bash
docker compose run --rm --no-deps bootstrap
```

The bootstrap should report zero new scores after the initial run.

## Local Python or TiDB Cloud

Use Python 3.13 and the pinned [runtime dependencies](../requirements.txt). Install pinned [research extras](../requirements-research.txt) to run notebooks (Matplotlib, seaborn, IPython kernel, and nbconvert).

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Configure `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASS`, and `TIDB_DEMO_DATABASE` in an ignored `.env` or your process environment. [.env.example](../.env.example) lists the settings. The database name must end in `_demo`. TLS verifies the certificate and hostname by default; use `SSL_CA` for a custom CA. Disabling TLS is allowed only for loopback development connections.

```bash
python -m src.database.tidb_admin migrate --create-database
python -m src.database.tidb_admin seed
python -m src.database.tidb_scoring --dataset synthetic_financial
python -m src.database.tidb_scoring --dataset ieee_cis
python -m src.database.tidb_admin check
```

Omit `--create-database` when an administrator has already created an empty database. The migration runner refuses an unmanaged schema. The scorer rechecks invalid rows after their inputs change and never overwrites an existing score. Use separate read-only database credentials for shared dashboard and chatbot instances. These development servers are not intended for public hosting.

Run `streamlit run src/dashboard/dashboard_app.py` and `python -m src.chatbot.chatbot_api` in separate shells. The chatbot works without Gemini. Optional general chat requires `python -m pip install -r requirements-gemini.txt` plus `GEMINI_API_KEY`, `GEMINI_MODEL`, and a request with `"mode": "general"`. Transaction lookup never sends scores or records to Gemini.

## Research dashboard data

Research Analytics uses aggregate JSON snapshots and verified evaluation reports included in the image. It does not load transaction-level research CSVs or private IEEE-CIS weights inside Docker. Confusion matrices are exact counts from the evaluation reports. ROC and precision–recall plots display up to 450 points from actual predictions per curve; rankings are computed before plot sampling. Dataset analytics cover the entire labeled source, while model metrics are scoped to the selected validation or test window.

To regenerate the snapshots locally, keep the original PaySim CSV, IEEE-CIS transactions and identity CSVs, and private IEEE-CIS release model available and run:

```bash
python -m src.dashboard.export_dashboard
```

The export checks transaction CSV checksums and replays the release metrics. It writes only aggregate counts, statistics, and ranking curve points. Run `docker compose up --build -d` after refreshing the snapshots.

## Verify

Run `python -m unittest discover -s tests -v` for unit and UI tests. Set `RUN_TIDB_INTEGRATION=1` and use a fresh, disposable `TIDB_DEMO_DATABASE` other than the default to include live integration tests. The [main README](../README.md#tests) shows the Docker command used by CI.
