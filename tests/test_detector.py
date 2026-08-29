import unittest

import pandas as pd

from equipment_trend_detector import DetectionConfig, fuse_equipment_results, run_resi_detector, run_temp_detector


CFG = DetectionConfig(baseline_min_wafers=10, baseline_max_history=100, drift_min_slope=0.1)


def make_resi(values, product="P1"):
    """Create a two-side wide fixture; values are (mean, std, max) for LEFT."""
    rows = []
    for index, (mean, std, maximum) in enumerate(values):
        jitter = (index % 3 - 1)
        rows.append({
            "eqp_masked": "EQP_1", "wafer_masked": f"W{index}", "product_masked": product,
            "write_time": pd.Timestamp("2026-01-01") + pd.Timedelta(minutes=index),
            "L_n": 100, "R_n": 100,
            "L_mean": mean, "L_std": std, "L_max": maximum,
            "R_mean": 5.0 + jitter * 0.02, "R_std": 0.9 + jitter * 0.01, "R_max": 9.0 + jitter * 0.10,
            "L_2SD": 1, "R_2SD": 1, "L_3SD": 1, "R_3SD": 1,
            "L_5SD": 0, "R_5SD": 0, "L_9SD": 0, "R_9SD": 0,
        })
    return pd.DataFrame(rows)


def stable(count=20):
    return [(5.0 + (i % 3 - 1) * .02, .4 + (i % 3 - 1) * .01, 7.0 + (i % 3 - 1) * .1) for i in range(count)]


class ResiTrendTests(unittest.TestCase):
    def test_stable_asymmetry_is_normal(self):
        detector = run_resi_detector(make_resi(stable()), CFG)
        result = detector["equipment"].iloc[0]
        self.assertEqual(result.RESI_LEFT_STATUS, "NORMAL")
        self.assertEqual(result.RESI_RIGHT_STATUS, "NORMAL")
        self.assertEqual(result.RESI_AFFECTED_SIDE, "NONE")
        self.assertEqual(detector["equipment_side"]["LR_CONTRAST_HIGH_COUNT_5"].sum(), 0)

    def test_persistent_right_lr_imbalance_is_detected(self):
        data = make_resi(stable(20) + stable(5))
        data.loc[data.index[-5:], ["R_std", "R_max"]] = [1.8, 15.0]
        right = run_resi_detector(data, CFG)["equipment_side"].set_index("SIDE").loc["RIGHT"]
        self.assertGreaterEqual(right.LR_CONTRAST_HIGH_COUNT_5, 3)
        self.assertEqual(right.LR_CONTRAST_DIRECTION, "RIGHT")
        self.assertIn(right.SIDE_EQUIPMENT_STATUS, {"WARNING", "ALARM"})

    def test_single_max_spike_stays_wafer_event(self):
        values = stable() + [(5.0, .4, 15.0)] + stable(2)
        side = run_resi_detector(make_resi(values), CFG)["equipment_side"].set_index("SIDE").loc["LEFT"]
        self.assertEqual(side.TREND_TYPE, "SPIKE")
        self.assertNotEqual(side.SIDE_EQUIPMENT_STATUS, "ALARM")

    def test_persistent_multi_group_step_alarms(self):
        values = stable() + [(7.0, 1.0, 10.0)] * 5
        side = run_resi_detector(make_resi(values), CFG)["equipment_side"].set_index("SIDE").loc["LEFT"]
        self.assertEqual(side.TREND_TYPE, "STEP_CHANGE")
        self.assertEqual(side.SIDE_EQUIPMENT_STATUS, "ALARM")

    def test_gradual_multi_group_drift(self):
        values = stable() + [(5 + d, .4 + d / 3, 7 + d * 2) for d in (.1, .2, .35, .5, .7, .9, 1.1)]
        side = run_resi_detector(make_resi(values), CFG)["equipment_side"].set_index("SIDE").loc["LEFT"]
        self.assertEqual(side.TREND_TYPE, "GRADUAL_DRIFT")
        self.assertIn(side.SIDE_EQUIPMENT_STATUS, {"WARNING", "ALARM"})

    def test_product_levels_do_not_create_raw_mean_alarm(self):
        # Each product has enough of its own causal history. Concatenating the
        # records then interleaving product rows must remain normal.
        a = make_resi(stable(14), "A")
        b = make_resi([(8 + (i % 3 - 1) * .02, .4 + (i % 3 - 1) * .01, 10 + (i % 3 - 1) * .1) for i in range(14)], "B")
        b["write_time"] += pd.Timedelta(seconds=30)
        result = run_resi_detector(pd.concat([a, b]).sort_values("write_time"), CFG)["equipment"]
        self.assertEqual(result.iloc[0].RESI_EQP_STATUS, "NORMAL")

    def test_both_side_shift_is_both(self):
        data = make_resi(stable() + [(7.0, 1.0, 10.0)] * 5)
        for feature in ("mean", "std", "max"):
            data[f"R_{feature}"] = data[f"L_{feature}"]
        result = run_resi_detector(data, CFG)["equipment"].iloc[0]
        self.assertEqual(result.RESI_AFFECTED_SIDE, "BOTH")


class FusionTests(unittest.TestCase):
    def test_temp_pipeline_handles_quantized_stable_data(self):
        rows = []
        for index in range(16):
            jitter = index % 2
            rows.append({
                "eqp_masked": "EQP_TEMP", "wafer_masked": f"T{index}", "product_masked": "P1",
                "write_time": pd.Timestamp("2026-02-01") + pd.Timedelta(minutes=index),
                "L_n": 100, "R_n": 100,
                "L_temp_mean": 60 + jitter, "R_temp_mean": 62 + jitter,
                "L_temp_max": 65 + jitter, "R_temp_max": 67 + jitter,
                "L_temp_p99_high_count": 1, "R_temp_p99_high_count": 1,
                "L_temp_p99_high_share": .01, "R_temp_p99_high_share": .01,
            })
        result = run_temp_detector(pd.DataFrame(rows), CFG)["equipment"].iloc[0]
        self.assertEqual(result.TEMP_EQP_STATUS, "NORMAL")

    def test_same_side_resi_temp_alarm_is_critical(self):
        resi = pd.DataFrame([{"EQP_ID": "E", "RESI_EQP_STATUS": "ALARM", "RESI_AFFECTED_SIDE": "LEFT"}])
        temp = pd.DataFrame([{"EQP_ID": "E", "TEMP_EQP_STATUS": "ALARM", "TEMP_AFFECTED_SIDE": "LEFT"}])
        self.assertEqual(fuse_equipment_results(resi, temp).iloc[0].FINAL_SEVERITY, "CRITICAL")

    def test_resi_alarm_is_not_cancelled_by_normal_temp(self):
        resi = pd.DataFrame([{"EQP_ID": "E", "RESI_EQP_STATUS": "ALARM", "RESI_AFFECTED_SIDE": "LEFT"}])
        temp = pd.DataFrame([{"EQP_ID": "E", "TEMP_EQP_STATUS": "NORMAL", "TEMP_AFFECTED_SIDE": "NONE"}])
        self.assertEqual(fuse_equipment_results(resi, temp).iloc[0].FINAL_SEVERITY, "RESI_ALARM")


if __name__ == "__main__":
    unittest.main()
