"""配准、曝光归一与导出的测试。"""

import json
import os
import tempfile
import unittest

import numpy as np
from PIL import Image

from polanalyzer.engine import FitConfig, fit_pixels
from polanalyzer.io_stack import (estimate_translation, export_fit_npz,
                                  load_stack, register_stack, shift_image)
from polanalyzer.masks import STATE_OK
from polanalyzer.model import predict


class TranslationTests(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(0)
        self.base = rng.normal(size=(48, 64))

    def test_known_shift_estimate_and_correction(self):
        target = np.roll(self.base, 2, axis=0)
        dy, dx = estimate_translation(self.base, target)
        self.assertEqual((dy, dx), (-2, 0))
        recovered = shift_image(target, -dy, -dx)
        np.testing.assert_allclose(recovered[3:-3, 3:-3],
                                   self.base[3:-3, 3:-3])

    def test_horizontal_shift(self):
        target = np.roll(self.base, 3, axis=1)
        dy, dx = estimate_translation(self.base, target)
        self.assertEqual((dy, dx), (0, -3))
        recovered = shift_image(target, -dy, -dx)
        np.testing.assert_allclose(recovered[3:-3, 4:-4],
                                   self.base[3:-3, 4:-4])

    def test_register_stack_recovers_misaligned_frame(self):
        h, w, k = 32, 40, 4
        rng = np.random.default_rng(3)
        # 同一内容跨帧近似不变（仅加噪声），第三帧人为平移
        scene = np.repeat(rng.normal(100, 20, (h, w))[..., None], k, axis=-1)
        scene[..., 2] = np.roll(scene[..., 2], 2, axis=1)
        aligned, shifts, missing = register_stack(scene)
        self.assertEqual(tuple(shifts[2]), (0, -2))
        # 配准后第三帧内部区域与参考帧一致
        np.testing.assert_allclose(aligned[2:-2, 3:-2, 2],
                                   scene[2:-2, 3:-2, 0], atol=1e-8)
        # 越界区域标记为缺失（右移内容校正后右边缘为空）
        self.assertTrue(missing[:, -2:, 2].all())
        self.assertFalse(missing[:, 5:-5, 2].any())


class ExposureNormalizationTests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        h, w = 24, 32
        self.A = np.full((h, w), 1000.0)
        self.B = np.full((h, w), 120.0)
        self.C = np.full((h, w), -40.0)
        frames = []
        for k, deg in enumerate([0, 45, 90, 135]):
            I = predict(np.array([np.deg2rad(deg)]),
                        self.A, self.B, self.C)[..., 0]
            exposure = 1.0 if k % 2 == 0 else 2.0
            raw = np.clip(I * exposure, 0, 65535).astype(np.uint16)
            path = os.path.join(self.tmp, f"f{k}.tif")
            Image.fromarray(raw).save(path)
            frames.append({"path": path, "angle_deg": deg,
                           "exposure_s": exposure, "gain": 1.0,
                           "saturation": 65535})
        with open(os.path.join(self.tmp, "manifest.json"), "w") as fh:
            json.dump({"frames": frames}, fh)

    def test_load_normalizes_exposure(self):
        stack = load_stack(os.path.join(self.tmp, "manifest.json"))
        # 归一化后不同曝光帧在相同角度模型下应给出一致的 A 估计
        result = fit_pixels(stack.intensity, stack.angles_rad,
                            FitConfig(saturation=65535.0, noise_sigma=1.0))
        ok = result.state == STATE_OK
        self.assertTrue(ok.mean() > 0.95)
        np.testing.assert_allclose(result.A[ok], self.A[ok], rtol=0.02)
        np.testing.assert_allclose(
            result.R[ok], np.hypot(self.B, self.C)[ok], rtol=0.05)

    def test_relative_manifest_paths_resolve_from_manifest_dir(self):
        # 清单内使用相对路径时，必须相对清单目录解析，与当前工作目录无关
        sub = os.path.join(self.tmp, "nested")
        os.makedirs(sub, exist_ok=True)
        rel_frames = []
        for k, deg in enumerate([0, 45, 90, 135]):
            src = os.path.join(self.tmp, f"f{k}.tif")
            dst = os.path.join(sub, f"r{k}.tif")
            os.replace(src, dst)
            rel_frames.append({"path": f"r{k}.tif", "angle_deg": deg,
                               "exposure_s": 1.0, "gain": 1.0,
                               "saturation": 65535})
        rel_manifest = os.path.join(sub, "manifest.json")
        with open(rel_manifest, "w") as fh:
            json.dump({"frames": rel_frames}, fh)
        cwd = os.getcwd()
        try:
            os.chdir(os.path.dirname(self.tmp))  # 离开清单目录
            stack = load_stack(rel_manifest)
        finally:
            os.chdir(cwd)
        self.assertEqual(stack.intensity.shape, (24, 32, 4))
        self.assertTrue(np.isfinite(stack.intensity).all())

    def test_export_preserves_dimensions(self):
        stack = load_stack(os.path.join(self.tmp, "manifest.json"))
        result = fit_pixels(stack.intensity, stack.angles_rad,
                            FitConfig(saturation=65535.0, noise_sigma=1.0))
        out = os.path.join(self.tmp, "fit.npz")
        export_fit_npz(out, result, "normalized intensity")
        data = np.load(out, allow_pickle=True)
        self.assertEqual(data["A"].dtype, np.float64)
        self.assertEqual(str(data["unit"]), "normalized intensity")
        np.testing.assert_array_equal(data["angles_rad"], stack.angles_rad)
        self.assertIn("sigma_phi_rad", data.files)
        self.assertIn("valid", data.files)


if __name__ == "__main__":
    unittest.main()
