"""拟合引擎与成像模型的核对测试。

测试原则：
- 用已知真值的合成观测核对估计参数，而非用拟合结果自证；
- 显式覆盖饱和、运动、缺失角度被排除的行为；
- 核对预测强度与残差；
- 弱响应区必须给出 weak_response 而不是强行输出偏振角。
"""

import unittest

import numpy as np

from polanalyzer.engine import FitConfig, fit_pixels
from polanalyzer.masks import (STATE_ALL_INVALID, STATE_FEW_ANGLES, STATE_OK,
                               STATE_PHYSICAL, STATE_WEAK,
                               valid_sample_mask, exclusion_reasons)
from polanalyzer.model import design_matrix, derive_components, predict
from polanalyzer.synthetic import demo_truth, make_scene

ANGLES_DEG = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]


class ModelTests(unittest.TestCase):

    def test_design_matrix_orthogonality(self):
        X = design_matrix(np.deg2rad(ANGLES_DEG))
        self.assertEqual(X.shape, (6, 3))
        # 等间隔 30 度、6 个角时 cos/sin 列与常数列正交
        self.assertAlmostEqual(X[:, 0] @ X[:, 1], 0.0, places=10)
        self.assertAlmostEqual(X[:, 0] @ X[:, 2], 0.0, places=10)
        self.assertAlmostEqual(X[:, 1] @ X[:, 2], 0.0, places=10)

    def test_predict_and_components_roundtrip(self):
        angles = np.deg2rad(np.linspace(0, 180, 37))
        A, B, C = 500.0, 120.0, -70.0
        I = predict(angles, A, B, C)
        R, phi = derive_components(np.asarray(B), np.asarray(C))
        rebuilt = A + R * np.cos(2 * (angles - phi))
        np.testing.assert_allclose(I, rebuilt, atol=1e-10)
        self.assertAlmostEqual(R, np.hypot(B, C), places=10)

    def test_phi_undefined_when_zero_modulation(self):
        R, phi = derive_components(np.array(0.0), np.array(0.0))
        self.assertEqual(R, 0.0)
        self.assertTrue(np.isnan(phi))


class MaskTests(unittest.TestCase):

    def test_saturation_and_invalid_excluded(self):
        I = np.array([[10.0, 4095.0, np.nan, -3.0, 100.0]])
        valid = valid_sample_mask(I, saturation=4095.0)
        np.testing.assert_array_equal(valid[0],
                                      [True, False, False, False, True])

    def test_motion_and_missing_priority(self):
        I = np.array([[10.0, 10.0, 10.0]])
        motion = np.array([[True, False, False]])
        missing = np.array([[False, True, False]])
        valid = valid_sample_mask(I, 4095.0, motion, missing)
        np.testing.assert_array_equal(valid[0], [False, False, True])
        reasons = exclusion_reasons(I, 4095.0, motion, missing)
        self.assertEqual(reasons[0, 0], "motion")
        self.assertEqual(reasons[0, 1], "missing")
        self.assertEqual(reasons[0, 2], "")


class EngineTests(unittest.TestCase):

    def setUp(self):
        self.A, self.B, self.C = demo_truth(64, 80)
        self.scene = make_scene(
            self.A, self.B, self.C, ANGLES_DEG,
            noise_sigma=5.0, rng_seed=42,
            occluder_cells=[(slice(5, 9), slice(5, 9), [0, 1, 2, 3])],
            motion_cells=[(slice(30, 34), slice(30, 34), [4])])
        self.cfg = FitConfig(saturation=self.scene.saturation,
                             noise_sigma=self.scene.noise_sigma)
        self.result = fit_pixels(
            self.scene.intensity, self.scene.angles_rad, self.cfg,
            motion_mask=self.scene.motion_mask,
            missing_mask=self.scene.missing_mask)

    def test_shapes_preserved(self):
        h, w = self.A.shape
        for name in ("A", "B", "C", "R", "phi_rad", "dolp",
                     "residual_rms", "state"):
            self.assertEqual(getattr(self.result, name).shape, (h, w))
        self.assertEqual(self.result.predicted.shape, (h, w, 6))
        self.assertEqual(self.result.valid.shape, (h, w, 6))

    def test_estimates_match_known_truth(self):
        ok = self.result.state == STATE_OK
        self.assertTrue(ok.mean() > 0.5)
        np.testing.assert_allclose(
            self.result.A[ok], self.A[ok], atol=4 * self.scene.noise_sigma)
        R_true = np.hypot(self.B, self.C)
        np.testing.assert_allclose(
            self.result.R[ok], R_true[ok], atol=4 * self.scene.noise_sigma)

    def test_phi_matches_truth_with_pi_over_2_ambiguity(self):
        # 线偏振模型下 phi 与 phi + pi/2 在 cos(2(theta-phi)) 中等价。
        # 仅在高信噪比（R 远超噪声）的可靠像素上核对角度。
        R_true = np.hypot(self.B, self.C)
        ok = (self.result.state == STATE_OK) & (R_true > 10 * self.scene.noise_sigma)
        self.assertTrue(ok.sum() > 100)
        phi_true = 0.5 * np.arctan2(self.C, self.B)
        diff = np.abs((self.result.phi_rad[ok] - phi_true[ok])
                      % (np.pi / 2))
        diff = np.minimum(diff, np.pi / 2 - diff)
        np.testing.assert_array_less(diff, np.deg2rad(5))

    def test_residuals_consistent_with_noise(self):
        ok = self.result.state == STATE_OK
        # 6 个角度、3 个参数，残差 RMS 应与注入噪声同量级
        med = np.median(self.result.residual_rms[ok])
        self.assertLess(med, 1.5 * self.scene.noise_sigma)
        self.assertGreater(med, 0.3 * self.scene.noise_sigma)

    def test_predicted_intensity_matches_model(self):
        rebuilt = predict(self.scene.angles_rad,
                          self.result.A, self.result.B, self.result.C)
        finite = np.isfinite(self.result.predicted)
        np.testing.assert_allclose(
            self.result.predicted[finite], rebuilt[finite], atol=1e-8)

    def test_motion_samples_excluded_from_equations(self):
        # 运动像素仍可拟合（其余 5 个角度有效），且该运动样本不在 valid 中
        self.assertFalse(self.result.valid[30:34, 30:34, 4].any())
        self.assertTrue((self.result.n_valid[30:34, 30:34] == 5).all())

    def test_insufficient_and_all_invalid_states(self):
        self.assertEqual(self.result.state[6, 6], STATE_FEW_ANGLES)
        self.assertEqual(self.result.n_valid[6, 6], 2)
        scene = make_scene(
            self.A, self.B, self.C, ANGLES_DEG,
            occluder_cells=[(slice(0, 2), slice(0, 2), list(range(6)))])
        res = fit_pixels(scene.intensity, scene.angles_rad,
                         FitConfig(scene.saturation, scene.noise_sigma),
                         missing_mask=scene.missing_mask)
        self.assertEqual(res.state[0, 0], STATE_ALL_INVALID)
        self.assertTrue(np.isnan(res.A[0, 0]))

    def test_weak_response_flagged_not_angle_confidence(self):
        h, w = 64, 80
        yy, xx = np.mgrid[0:h, 0:w]
        A = np.full((h, w), 1000.0)
        B = np.full((h, w), 2.0)   # R 远小于噪声 sigma=30
        C = np.zeros((h, w))
        scene = make_scene(A, B, C, ANGLES_DEG, noise_sigma=30.0,
                           rng_seed=7)
        res = fit_pixels(scene.intensity, scene.angles_rad,
                         FitConfig(scene.saturation, scene.noise_sigma))
        self.assertTrue((res.state == STATE_WEAK).mean() > 0.8)
        # 弱响应下不允许声称高置信偏振度
        self.assertFalse(np.any(
            (res.state == STATE_WEAK) & (res.sigma_phi_rad < np.deg2rad(5))))

    def test_saturated_pixels_excluded(self):
        h, w = 16, 16
        A = np.full((h, w), 4000.0)
        B = np.full((h, w), 300.0)
        C = np.zeros((h, w))
        scene = make_scene(A, B, C, ANGLES_DEG, saturation=4095.0,
                           noise_sigma=1.0, rng_seed=11)
        res = fit_pixels(scene.intensity, scene.angles_rad,
                         FitConfig(4095.0, 1.0))
        # 高值角度必然饱和，被排除；像素仍由剩余角度拟合
        self.assertTrue(np.any(~res.valid))
        saturated_count = (~res.valid).sum(axis=-1)
        self.assertTrue((saturated_count >= 1).all())
        self.assertTrue(np.isfinite(res.A).all())

    def test_unphysical_fit_detected(self):
        # 手工构造 R > A 的观测（物理上不可能的负极小值）
        angles = np.deg2rad(ANGLES_DEG)
        I = predict(angles, np.array(50.0), np.array(120.0), np.array(0.0))
        I = np.broadcast_to(I, (4, 4, 6)).copy()
        res = fit_pixels(I, angles, FitConfig(saturation=1e9, noise_sigma=1.0))
        self.assertTrue((res.state == STATE_PHYSICAL).all())

    def test_dolp_in_unit_interval_when_reliable(self):
        ok = self.result.state == STATE_OK
        self.assertTrue(np.all(self.result.dolp[ok] >= 0.0))
        self.assertTrue(np.all(self.result.dolp[ok] <= 1.0 + 1e-9))

    def test_uncertainty_finite_for_reliable_pixels(self):
        ok = self.result.state == STATE_OK
        self.assertTrue(np.isfinite(self.result.sigma_R[ok]).all())
        self.assertTrue(np.isfinite(self.result.sigma_phi_rad[ok]).all())
        self.assertTrue(np.isfinite(self.result.sigma_dolp[ok]).all())


class ColorStackTests(unittest.TestCase):

    def test_multichannel_fit(self):
        A, B, C = demo_truth(32, 40)
        A3 = np.repeat(A[..., None], 3, axis=-1)
        B3 = np.repeat(B[..., None], 3, axis=-1)
        C3 = np.repeat(C[..., None], 3, axis=-1)
        scene = make_scene(A3, B3, C3, ANGLES_DEG, noise_sigma=5.0,
                           rng_seed=99)
        res = fit_pixels(scene.intensity, scene.angles_rad,
                         FitConfig(scene.saturation, scene.noise_sigma))
        self.assertEqual(res.A.shape, (32, 40, 3))
        self.assertEqual(res.predicted.shape, (32, 40, 3, 6))
        ok = res.state == STATE_OK
        np.testing.assert_allclose(
            res.A[ok], A3[ok], atol=4 * scene.noise_sigma)


if __name__ == "__main__":
    unittest.main()
