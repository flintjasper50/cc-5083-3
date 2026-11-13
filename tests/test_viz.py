"""可视化数据构造与可靠性着色测试。"""

import unittest

import numpy as np

from polanalyzer.engine import FitConfig, fit_pixels
from polanalyzer.masks import STATE_ALL_INVALID, STATE_OK, STATE_WEAK
from polanalyzer.synthetic import demo_truth, make_scene
from polanalyzer.viz import (angle_hsv_image, component_arrays,
                             pixel_curve_data, state_overlay)

ANGLES = [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]


class VizTests(unittest.TestCase):

    def setUp(self):
        A, B, C = demo_truth(48, 64)
        self.scene = make_scene(
            A, B, C, ANGLES, noise_sigma=8.0,
            occluder_cells=[(slice(8, 12), slice(8, 12), [0, 2])],
            motion_cells=[(slice(30, 34), slice(30, 34), [3])])
        self.result = fit_pixels(
            self.scene.intensity, self.scene.angles_rad,
            FitConfig(self.scene.saturation, self.scene.noise_sigma),
            motion_mask=self.scene.motion_mask,
            missing_mask=self.scene.missing_mask)

    def test_pixel_curve_uses_real_arrays(self):
        d = pixel_curve_data(self.result, 20, 20)
        self.assertEqual(d["angles_deg"].size, 6)
        np.testing.assert_array_equal(
            d["observed"], self.result.intensity[20, 20, :])
        np.testing.assert_array_equal(
            d["predicted"], self.result.predicted[20, 20, :])
        # 拟合曲线在观测角度上与预测一致
        dense_pred = np.interp(d["angles_deg"], d["fit_angles_deg"],
                               d["fit_curve"])
        np.testing.assert_allclose(dense_pred, d["predicted"], atol=1.0)

    def test_excluded_samples_annotated_with_reasons(self):
        d = pixel_curve_data(self.result, 9, 9)
        self.assertEqual(d["exclusion_reasons"][0], "missing")
        self.assertEqual(d["exclusion_reasons"][2], "missing")
        self.assertFalse(d["valid"][0])
        self.assertFalse(d["valid"][2])
        self.assertTrue(d["valid"][1])

        dm = pixel_curve_data(self.result, 31, 31)
        self.assertEqual(dm["exclusion_reasons"][3], "motion")
        self.assertFalse(dm["valid"][3])
        self.assertEqual(dm["n_valid"], 5)

    def test_uncertainty_band_encloses_curve(self):
        d = pixel_curve_data(self.result, 20, 20)
        self.assertTrue(np.all(d["band_upper"] >= d["fit_curve"] - 1e-9))
        self.assertTrue(np.all(d["band_lower"] <= d["fit_curve"] + 1e-9))

    def test_state_overlay_colors_all_states(self):
        img = state_overlay(self.result.state)
        self.assertEqual(img.shape, (*self.result.state.shape, 3))
        # 弱响应像素不应为可靠绿色
        weak = self.result.state == STATE_WEAK
        if weak.any():
            self.assertFalse(np.allclose(img[weak][0], (0.20, 0.65, 0.30)))

    def test_angle_image_desaturates_unreliable(self):
        img = angle_hsv_image(self.result.phi_rad, self.result.R,
                              self.result.state)
        weak = self.result.state == STATE_WEAK
        if weak.any():
            np.testing.assert_allclose(img[weak][0], 0.15)
        ok = self.result.state == STATE_OK
        self.assertTrue(img[ok].max() > 0.3)

    def test_component_arrays_keep_dimensions(self):
        comps = component_arrays(self.result)
        for name, arr in comps.items():
            self.assertEqual(arr.shape, self.result.A.shape, name)

    def test_all_invalid_pixel_curve_does_not_crash(self):
        A, B, C = demo_truth(16, 16)
        scene = make_scene(
            A, B, C, ANGLES,
            occluder_cells=[(slice(0, 2), slice(0, 2), list(range(6)))])
        res = fit_pixels(scene.intensity, scene.angles_rad,
                         FitConfig(scene.saturation, scene.noise_sigma),
                         missing_mask=scene.missing_mask)
        self.assertEqual(res.state[0, 0], STATE_ALL_INVALID)
        d = pixel_curve_data(res, 0, 0)
        self.assertFalse(d["valid"].any())
        self.assertTrue(np.isnan(d["A"]))


if __name__ == "__main__":
    unittest.main()
