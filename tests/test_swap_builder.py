"""Tests for src/benchmark/swap_builder.py"""

import sys
from pathlib import Path
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1]))
from src.benchmark.swap_builder import is_valid_swap, pixel_mse, save_benchmark, load_benchmark


def make_rgb_image(color=(128, 64, 32), size=(224, 224)) -> Image.Image:
    arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    return Image.fromarray(arr)


class TestPixelMSE:
    def test_identical_images_zero(self):
        img = make_rgb_image((100, 100, 100))
        assert pixel_mse(img, img) < 1e-6

    def test_different_images_positive(self):
        a = make_rgb_image((0, 0, 0))
        b = make_rgb_image((255, 255, 255))
        mse = pixel_mse(a, b)
        assert mse > 0.9, f"Expected high MSE for black vs white, got {mse}"

    def test_similar_images_low(self):
        a = make_rgb_image((100, 100, 100))
        b = make_rgb_image((105, 100, 100))
        mse = pixel_mse(a, b)
        assert mse < 0.01


class TestIsValidSwap:
    def _make_rec(self, subject, answer, color):
        img = make_rgb_image(color)
        return {"subject": subject, "answer": answer, "_image": img}

    def test_valid_swap(self):
        orig = self._make_rec("biology", "A", (0, 0, 0))
        cand = self._make_rec("biology", "B", (255, 255, 255))
        assert is_valid_swap(orig, cand, min_pixel_diff=0.1)

    def test_fails_different_subject(self):
        orig = self._make_rec("biology", "A", (0, 0, 0))
        cand = self._make_rec("physics", "B", (255, 255, 255))
        assert not is_valid_swap(orig, cand)

    def test_fails_same_answer(self):
        orig = self._make_rec("biology", "A", (0, 0, 0))
        cand = self._make_rec("biology", "A", (255, 255, 255))
        assert not is_valid_swap(orig, cand)

    def test_fails_visually_similar(self):
        orig = self._make_rec("biology", "A", (100, 100, 100))
        cand = self._make_rec("biology", "B", (101, 101, 101))
        assert not is_valid_swap(orig, cand, min_pixel_diff=0.25)

    def test_fails_missing_image(self):
        orig = {"subject": "biology", "answer": "A", "_image": None}
        cand = self._make_rec("biology", "B", (255, 255, 255))
        assert not is_valid_swap(orig, cand)


class TestSaveBenchmark:
    def test_save_and_load_roundtrip(self, tmp_path):
        triplets = [
            {
                "triplet_id": "sqa_triplet_0001",
                "source": "scienceqa",
                "original": {"id": "scienceqa_0", "hf_index": 0, "question": "What is this?",
                              "answer": "A", "subject": "biology"},
                "swap": {"id": "scienceqa_1", "hf_index": 1, "question": "What is this?",
                         "answer": "B", "subject": "biology"},
                "pixel_mse": 0.5,
            }
        ]
        out_path = str(tmp_path / "benchmark_metadata.json")
        save_benchmark(triplets, out_path)
        loaded = load_benchmark(out_path)
        assert len(loaded) == 1
        assert loaded[0]["triplet_id"] == "sqa_triplet_0001"
        assert "_image" not in loaded[0]

    def test_save_strips_image_keys(self, tmp_path):
        img = make_rgb_image()
        triplets = [
            {
                "triplet_id": "t1",
                "source": "scienceqa",
                "original": {"id": "x", "hf_index": 0, "question": "q", "answer": "A",
                              "subject": "bio", "_image": img},
                "swap": {"id": "y", "hf_index": 1, "question": "q", "answer": "B",
                         "subject": "bio", "_image": img},
                "pixel_mse": 0.3,
            }
        ]
        out_path = str(tmp_path / "b.json")
        save_benchmark(triplets, out_path)
        loaded = load_benchmark(out_path)
        assert "_image" not in loaded[0]["original"]
        assert "_image" not in loaded[0]["swap"]
