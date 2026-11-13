"""受控故障注入与异常诊断的核对测试。

测试原则：
- 故障由 faults 模块以已知真值注入，诊断必须把故障类型与注入对应起来，
  不允许把所有残差都归咎于噪声；
- 干净观测不得被误报为角度/曝光/运动故障；
- 角度单因、曝光单因、二者联合、局部运动四类对照分别断言；
- 诊断只读输入：不重排角度、不修改观测栈、不夹紧负值；
- 正常观测经诊断路径后分量结果与原引擎逐元素一致；
- 诊断包可在进程外重建同一结论（rebuild_bundle）。
"""

import os
import tempfile
import unittest

import numpy as np

from polanalyzer import faults
from polanalyzer.bundle import extract_bundle, rebuild_bundle
from polanalyzer.diagnostics import (DiagnosisConfig, HYP_AMBIGUOUS,
                                     HYP_ANGLE, HYP_CLEAN, HYP_EXPOSURE,
                                     HYP_JOINT, HYP_MOTION, HYP_UNFITTABLE,
                                     diagnose_pixel, diagnose_stack)
from polanalyzer.engine import FitConfig, fit_pixels
from polanalyzer.synthetic import demo_truth, make_scene

ANGLES = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]
PERM = (1, 0, 2, 3, 4, 5)
GAINS = (1.0, 1.0, 1.0, 1.6, 1.0, 1.0)


class DiagnosticFixture(unittest.TestCase):

    def setUp(self):
        self.A, self.B, self.C = demo_truth(48, 56)
        self.scene = make_scene(
            self.A, self.B, self.C, ANGLES, noise_sigma=8.0, rng_seed=42,
            occluder_cells=[(slice(6, 10), slice(8, 12), [0, 1])],
            motion_cells=[(slice(24, 30), slice(30, 38), [3])])
        self.cfg = FitConfig(self.scene.saturation, self.scene.noise_sigma)
        self.dcfg = DiagnosisConfig()
        self.motion_rows = slice(24, 30)
        self.motion_cols = slice(30, 38)

    def _fit(self, stack, mm=None, km=None):
        return fit_pixels(stack, self.scene.angles_rad, self.cfg,
                          motion_mask=mm, missing_mask=km)

    def _diag(self, stack, result):
        return diagnose_stack(stack, self.scene.angles_rad, result,
                              config=self.dcfg)


class CleanSceneTests(DiagnosticFixture):

    def test_clean_scene_not_false_positive(self):
        result = self._fit(self.scene.intensity,
                           self.scene.motion_mask, self.scene.missing_mask)
        report = self._diag(self.scene.intensity, result)
        total = sum(report.counts.values())
        # 干净观测绝大多数与模型一致，任何故障假设都不得成势
        self.assertGreaterEqual(
            report.counts.get(HYP_CLEAN, 0), 0.8 * total)
        for hyp in (HYP_ANGLE, HYP_EXPOSURE, HYP_JOINT, HYP_MOTION):
            self.assertLess(report.counts.get(hyp, 0), 0.05 * total)
        # 干净帧的全局增益必须围绕 1
        np.testing.assert_allclose(report.global_gains, 1.0, atol=0.05)

    def test_clean_pixels_components_equivalent(self):
        result = self._fit(self.scene.intensity,
                           self.scene.motion_mask, self.scene.missing_mask)
        report = self._diag(self.scene.intensity, result)
        eq = report.component_equivalence
        self.assertTrue(eq["equivalent"])
        self.assertEqual(eq["max_abs_beta_diff"], 0.0)
        self.assertGreater(eq["checked_clean_pixels"], 100)


class AngleFaultTests(DiagnosticFixture):

    def test_angle_only_detected_and_exposure_rejected(self):
        scenario = faults.inject_angle_permutation(
            self.scene.intensity, self.scene.angles_rad, permutation=PERM)
        result = self._fit(scenario.intensity)
        report = self._diag(scenario.intensity, result)
        # 角度错配成为主导假设
        self.assertEqual(
            max(report.counts, key=report.counts.get), HYP_ANGLE)
        # 角度单因下增益必须保持在 1 附近，不得误报曝光
        np.testing.assert_allclose(report.global_gains, 1.0, atol=0.05)
        self.assertEqual(report.counts.get(HYP_EXPOSURE, 0), 0)
        self.assertEqual(report.counts.get(HYP_JOINT, 0), 0)
        # 记录里保留重配排列证据
        angle_rec = next(r for r in report.records
                         if r.hypothesis == HYP_ANGLE)
        self.assertIsNotNone(angle_rec.angle_perm_best)
        self.assertTrue(any("重配" in e for e in angle_rec.evidence))

    def test_permutation_recorded_as_reproducible_evidence(self):
        scenario = faults.inject_angle_permutation(
            self.scene.intensity, self.scene.angles_rad, permutation=PERM)
        result = self._fit(scenario.intensity)
        report = self._diag(scenario.intensity, result)
        rec = next(r for r in report.records
                   if r.hypothesis == HYP_ANGLE)
        self.assertLess(rec.angle_chi2_best, rec.angle_chi2_base)
        self.assertGreaterEqual(
            rec.angle_relative_improvement, self.dcfg.angle_rel_gain)


class ExposureFaultTests(DiagnosticFixture):

    def test_exposure_only_detected_and_angle_rejected(self):
        scenario = faults.inject_exposure_gains(
            self.scene.intensity, self.scene.angles_rad, gains=GAINS)
        result = self._fit(scenario.intensity)
        report = self._diag(scenario.intensity, result)
        self.assertEqual(
            max(report.counts, key=report.counts.get), HYP_EXPOSURE)
        # 故障帧（索引 3，90°）增益显著偏离 1，其余帧围绕 1
        self.assertGreater(
            abs(report.global_gains[3] - 1.0),
            self.dcfg.exposure_gain_tol)
        # 曝光单因下不允许角度/联合假设成势
        total = sum(report.counts.values())
        self.assertLess(
            report.counts.get(HYP_ANGLE, 0) + report.counts.get(HYP_JOINT, 0),
            0.05 * total)
        rec = next(r for r in report.records
                   if r.hypothesis == HYP_EXPOSURE)
        self.assertTrue(any("增益" in e for e in rec.evidence))

    def test_gain_evidence_is_low_spread(self):
        scenario = faults.inject_exposure_gains(
            self.scene.intensity, self.scene.angles_rad, gains=GAINS)
        result = self._fit(scenario.intensity)
        report = self._diag(scenario.intensity, result)
        # 曝光是整帧常数增益：MAD 应远小于偏离量
        self.assertLess(report.global_gain_mad[3], 0.05)


class JointFaultTests(DiagnosticFixture):

    def test_joint_fault_keeps_both_candidates(self):
        scenario = faults.combine(
            faults.inject_angle_permutation(
                self.scene.intensity, self.scene.angles_rad,
                permutation=PERM),
            faults.inject_exposure_gains(
                self.scene.intensity, self.scene.angles_rad, gains=GAINS))
        result = self._fit(scenario.intensity)
        report = self._diag(scenario.intensity, result)
        total = sum(report.counts.values())
        # 联合场景：曝光证据与角度证据都必须成势，不能只报单因
        self.assertGreaterEqual(
            report.counts.get(HYP_EXPOSURE, 0), 0.2 * total)
        self.assertGreaterEqual(
            report.counts.get(HYP_JOINT, 0)
            + report.counts.get(HYP_ANGLE, 0), 0.3 * total)
        joint_rec = next((r for r in report.records
                          if r.hypothesis == HYP_JOINT), None)
        if joint_rec is not None:
            self.assertIn(HYP_ANGLE, joint_rec.candidates)
            self.assertIn(HYP_EXPOSURE, joint_rec.candidates)


class MotionFaultTests(DiagnosticFixture):

    def test_local_motion_localized_in_injected_region(self):
        scenario = faults.inject_local_motion(
            self.scene.intensity, self.scene.angles_rad,
            self.motion_rows, self.motion_cols, 3, shift=150.0)
        result = self._fit(scenario.intensity)
        report = self._diag(scenario.intensity, result)
        patch = report.hypothesis_map[self.motion_rows, self.motion_cols]
        self.assertGreaterEqual(
            int((patch == HYP_MOTION).sum()),
            self.dcfg.motion_min_pixels)
        # 运动是局部现象：全图绝大多数像素不得被报为运动
        total = report.hypothesis_map.size
        self.assertLess(report.counts.get(HYP_MOTION, 0), 0.1 * total)
        # 帧级增益在运动场景下保持 ~1
        np.testing.assert_allclose(report.global_gains, 1.0, atol=0.05)

    def test_isolated_hotspot_downgraded_to_ambiguous(self):
        # 只在单个像素注入运动，达不到连通块最小尺寸 -> 保留歧义而非断言
        scenario = faults.inject_local_motion(
            self.scene.intensity, self.scene.angles_rad,
            slice(20, 21), slice(20, 21), 2, shift=150.0)
        result = self._fit(scenario.intensity)
        report = self._diag(scenario.intensity, result)
        self.assertEqual(report.hypothesis_map[20, 20], HYP_AMBIGUOUS)


class ImmutabilityTests(DiagnosticFixture):

    def test_diagnostics_do_not_mutate_inputs(self):
        stack = self.scene.intensity.copy()
        angles = self.scene.angles_rad.copy()
        before = stack.copy()
        scenario = faults.inject_angle_permutation(
            stack, angles, permutation=PERM)
        # 注入器不修改原栈
        np.testing.assert_array_equal(stack, before)
        result = self._fit(scenario.intensity)
        A_before = result.A.copy()
        report = self._diag(scenario.intensity, result)
        np.testing.assert_array_equal(stack, before)
        np.testing.assert_array_equal(angles, self.scene.angles_rad)
        # 诊断不修改拟合结果，也不夹紧任何负值
        np.testing.assert_array_equal(result.A, A_before)
        # 注入器输出与输入的有限/缺失位置完全一致，无静默清洗
        np.testing.assert_array_equal(
            np.isfinite(scenario.intensity), np.isfinite(before))

    def test_no_automatic_reorder_or_clamp(self):
        scenario = faults.inject_angle_permutation(
            self.scene.intensity, self.scene.angles_rad, permutation=PERM)
        result = self._fit(scenario.intensity)
        report = self._diag(scenario.intensity, result)
        rec = next(r for r in report.records
                   if r.hypothesis == HYP_ANGLE)
        # 建议只要求人工核对角度声明，不包含自动重排/夹紧动作
        self.assertIn("核对", rec.suggestion)
        self.assertNotIn("自动", rec.suggestion)
        # 观测原样保留在记录中
        np.testing.assert_allclose(
            np.array([x for x in rec.observed if x is not None]),
            scenario.intensity[rec.row, rec.col][
                np.isfinite(scenario.intensity[rec.row, rec.col])])


class RecordContentTests(DiagnosticFixture):

    def test_pixel_record_contains_required_evidence(self):
        result = self._fit(self.scene.intensity,
                           self.scene.motion_mask, self.scene.missing_mask)
        report = self._diag(self.scene.intensity, result)
        rec = report.records[0]
        for attr in ("angles_deg", "observed", "predicted", "residual",
                     "standardized_residual", "valid", "exclusion_reasons",
                     "n_valid", "design_rank", "condition_number",
                     "gain_estimates", "evidence", "suggestion",
                     "hypothesis", "candidates"):
            self.assertTrue(hasattr(rec, attr), attr)
        self.assertEqual(len(rec.angles_deg), 6)
        self.assertEqual(len(rec.observed), 6)
        # 可 JSON 序列化
        import json
        json.dumps(rec.to_dict(), ensure_ascii=False)

    def test_record_volume_capped_for_clean_pixels(self):
        result = self._fit(self.scene.intensity,
                           self.scene.motion_mask, self.scene.missing_mask)
        report = diagnose_stack(
            self.scene.intensity, self.scene.angles_rad, result,
            config=self.dcfg, record_fraction=0.01)
        clean_records = [r for r in report.records
                         if r.hypothesis == HYP_CLEAN]
        # 正常对照记录远小于像素总数
        self.assertLess(len(clean_records), 0.05 * report.hypothesis_map.size)


class BundleTests(DiagnosticFixture):

    def test_bundle_rebuilds_same_conclusion(self):
        scenario = faults.inject_exposure_gains(
            self.scene.intensity, self.scene.angles_rad, gains=GAINS)
        result = self._fit(scenario.intensity)
        report = self._diag(scenario.intensity, result)
        with tempfile.TemporaryDirectory() as tmp:
            path = extract_bundle(report, result, scenario.intensity,
                                  tmp, name="exp_bundle")
            self.assertTrue(os.path.isfile(
                os.path.join(path, "manifest.json")))
            self.assertTrue(os.path.isfile(os.path.join(path, "data.npz")))
            self.assertTrue(os.path.isfile(
                os.path.join(path, "rebuild.py")))
            ok, rebuilt, sealed = rebuild_bundle(path)
            self.assertTrue(ok)
            self.assertEqual(rebuilt.counts, sealed["hypothesis_counts"])
            # 体积控制：manifest 不超过 256 KB
            size = os.path.getsize(os.path.join(path, "manifest.json"))
            self.assertLess(size, 256 * 1024)

    def test_bundle_locates_region_on_original_grid(self):
        scenario = faults.inject_local_motion(
            self.scene.intensity, self.scene.angles_rad,
            self.motion_rows, self.motion_cols, 3, shift=150.0)
        result = self._fit(scenario.intensity)
        report = self._diag(scenario.intensity, result)
        with tempfile.TemporaryDirectory() as tmp:
            path = extract_bundle(report, result, scenario.intensity,
                                  tmp, name="motion_bundle")
            import json
            with open(os.path.join(path, "manifest.json"),
                      encoding="utf-8") as fh:
                manifest = json.load(fh)
            bbox = manifest["region_bbox_rcrc"]
            # 包围盒必须覆盖注入的运动区域
            self.assertLessEqual(bbox[0], self.motion_rows.start)
            self.assertGreaterEqual(bbox[1], self.motion_rows.stop)
            self.assertLessEqual(bbox[2], self.motion_cols.start)
            self.assertGreaterEqual(bbox[3], self.motion_cols.stop)
            self.assertEqual(manifest["image_shape"],
                             list(report.hypothesis_map.shape))


class RegionDiagnosisTests(DiagnosticFixture):

    def test_region_diagnosis_records_window(self):
        result = self._fit(self.scene.intensity,
                           self.scene.motion_mask, self.scene.missing_mask)
        region = (20, 30, 25, 40)
        report = diagnose_stack(
            self.scene.intensity, self.scene.angles_rad, result,
            config=self.dcfg, region=region)
        self.assertEqual(report.region, region)
        self.assertEqual(report.hypothesis_map.shape, (10, 15))
        self.assertTrue(all(region[0] <= r.row < region[1]
                            and region[2] <= r.col < region[3]
                            for r in report.records))


class UnfittableTests(DiagnosticFixture):

    def test_too_few_angles_marked_not_estimable(self):
        # 手工构造只有两个有效角度的像素
        angles = self.scene.angles_rad
        I = np.full((4, 4, 6), 1000.0)
        valid = np.zeros((4, 4, 6), dtype=bool)
        valid[..., :2] = True
        rng = np.random.default_rng(1)
        d = diagnose_pixel(
            I[0, 0], angles, valid[0, 0], 8.0,
            np.ones(6), self.dcfg, rng, result_state="insufficient_angles")
        self.assertEqual(d["hypothesis"], HYP_UNFITTABLE)


if __name__ == "__main__":
    unittest.main()
