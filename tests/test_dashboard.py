import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from src.dashboard.dashboard_data import read_release, read_snapshot


APP = Path(__file__).resolve().parents[1] / "src/dashboard/dashboard_app.py"
DATASETS = {
    "synthetic_financial": (6_362_620, 8_213, [[293_516, 2_476], [1_777, 815]]),
    "ieee_cis": (590_540, 20_663, [[87_212, 4_142], [2_520, 762]]),
}


def open_research():
    with patch("src.database.connection.load_demo_settings", side_effect=ValueError):
        app = AppTest.from_file(str(APP)).run()
    return app.sidebar.radio[0].set_value("Research Analytics").run()


def radio(app, label):
    return next(widget for widget in app.radio if widget.label == label)


def chart_spec(app, position):
    return json.loads(app.get("plotly_chart")[position].proto.spec)


class ResearchDashboardTest(unittest.TestCase):
    def test_snapshot_bins_cover_every_transaction_and_release_curves_match_report(self):
        for dataset, (total, fraud, matrix) in DATASETS.items():
            with self.subTest(dataset=dataset):
                manifest, report = read_release(dataset)
                snapshot = read_snapshot(dataset, manifest, report)
                for groups in (snapshot["by_day"], *snapshot["segments"].values()):
                    self.assertEqual(sum(group["rows"] for group in groups), total)
                    self.assertEqual(sum(group["fraud"] for group in groups), fraud)
                for distribution in snapshot["distributions"].values():
                    bins = distribution["bins"]
                    self.assertEqual(sum(group["not_fraud"] for group in bins), total - fraud)
                    self.assertEqual(sum(group["fraud"] for group in bins), fraud)
                for split, curves in snapshot["curves"].items():
                    for key, value in curves["ranking"].items():
                        if key in report[split]:
                            self.assertAlmostEqual(value, report[split][key], places=10)
                    for key in ("roc", "pr"):
                        x = [point[0] for point in curves[key]]
                        self.assertEqual(x, sorted(x))
                        self.assertEqual((x[0], x[-1]), (0, 1))
                self.assertEqual(sum(map(sum, matrix)), report["test"]["rows"])

    def test_all_research_pages_render_and_test_confusion_matrices_are_exact(self):
        app = open_research()
        for dataset, (total, fraud, matrix) in DATASETS.items():
            app.sidebar.selectbox[0].set_value(dataset)
            app.sidebar.radio[1].set_value("Overview").run()
            self.assertFalse(app.exception)
            self.assertEqual([metric.value for metric in app.metric][:2], [f"{total:,}", f"{fraud:,}"])
            self.assertEqual(len(app.get("plotly_chart")), 3)
            for page, charts in (("Segment Risk", 2), ("Distributions", 1), ("Model Performance", 2)):
                with self.subTest(dataset=dataset, page=page):
                    app.sidebar.radio[1].set_value(page).run()
                    self.assertFalse(app.exception)
                    self.assertFalse(app.warning)
                    self.assertEqual(len(app.get("plotly_chart")), charts)
            self.assertEqual(chart_spec(app, 0)["data"][0]["text"], [[f"{value:,}" for value in row] for row in matrix])
            radio(app, "Curve").set_value("ROC")
            radio(app, "Matrix values").set_value("Row %")
            radio(app, "Evaluation split").set_value("Validation").run()
            self.assertFalse(app.exception)
            self.assertEqual(chart_spec(app, 1)["data"][1]["name"], "ROC")
            report = read_release(dataset)[1]["validation"]
            self.assertEqual(app.metric[2].value, f"{report['recall']:.2%}")
            self.assertTrue(chart_spec(app, 0)["data"][0]["text"][1][1].endswith("%"))

    def test_missing_snapshot_leaves_model_report_available(self):
        with TemporaryDirectory() as folder, patch.dict("os.environ", {"DASHBOARD_ARTIFACT_DIR": folder}):
            app = open_research()
            self.assertTrue(app.warning)
            app.sidebar.radio[1].set_value("Model Performance").run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.get("plotly_chart")), 1)
            self.assertIn("Curve points are unavailable", app.info[0].value)

    def test_balance_and_device_controls_show_their_own_aggregates(self):
        app = open_research()
        app.sidebar.radio[1].set_value("Distributions").run()
        next(widget for widget in app.selectbox if widget.label == "Measure").set_value("Receiver balance change")
        radio(app, "Distribution scale").set_value("Counts").run()
        self.assertFalse(app.exception)
        chart = chart_spec(app, 0)
        self.assertEqual(chart["layout"]["yaxis"]["title"]["text"], "Transactions")
        self.assertIn("Receiver balance change", chart["layout"]["xaxis"]["title"]["text"])
        app.sidebar.selectbox[0].set_value("ieee_cis")
        app.sidebar.radio[1].set_value("Segment Risk").run()
        next(widget for widget in app.selectbox if widget.label == "Group by").set_value("Device type").run()
        self.assertFalse(app.exception)
        self.assertEqual(set(app.dataframe[0].value["Segment"]), {"Unknown", "desktop", "mobile"})
