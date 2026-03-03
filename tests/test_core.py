"""Tests for pixelduet core modules."""

import os

import numpy as np
import pytest

# ---------- Color ----------


class TestColor:
    def test_luminance_white(self):
        from pixelduet.color import luminance

        white = np.array([[255.0, 255.0, 255.0]])
        lum = luminance(white)
        assert lum.shape == (1,)
        assert pytest.approx(lum[0], rel=1e-3) == 255.0

    def test_luminance_black(self):
        from pixelduet.color import luminance

        black = np.array([[0.0, 0.0, 0.0]])
        assert luminance(black)[0] == 0.0

    def test_rgb_to_lab_shape(self):
        from pixelduet.color import rgb_to_lab

        rgb = np.random.randint(0, 256, (100, 3), dtype=np.uint8)
        lab = rgb_to_lab(rgb)
        assert lab.shape == (100, 3)
        assert lab.dtype == np.float32

    def test_rgb_to_lab_white(self):
        from pixelduet.color import rgb_to_lab

        white = np.array([[255, 255, 255]], dtype=np.uint8)
        lab = rgb_to_lab(white)
        # L should be ~100 for white
        assert lab[0, 0] > 95.0
        # a and b should be near 0 for neutral white
        assert abs(lab[0, 1]) < 1.0
        assert abs(lab[0, 2]) < 1.0

    def test_rgb_to_lab_black(self):
        from pixelduet.color import rgb_to_lab

        black = np.array([[0, 0, 0]], dtype=np.uint8)
        lab = rgb_to_lab(black)
        assert lab[0, 0] < 1.0  # L ~= 0


# ---------- Edges ----------


class TestEdges:
    def test_edges_shape(self):
        from pixelduet.edges import edges_from_image

        img = np.random.rand(20, 30, 3).astype(np.float32) * 255
        edge = edges_from_image(img)
        assert edge.shape == (20, 30)

    def test_edges_range(self):
        from pixelduet.edges import edges_from_image

        # Image with a clear vertical edge
        img = np.zeros((20, 30, 3), dtype=np.float32)
        img[:, 15:, :] = 255.0
        edge = edges_from_image(img)
        assert edge.min() >= 0.0
        assert edge.max() <= 1.0 + 1e-6

    def test_uniform_image_zero_edges(self):
        from pixelduet.edges import edges_from_image

        img = np.ones((10, 10, 3), dtype=np.float32) * 128.0
        edge = edges_from_image(img)
        assert edge.max() < 1e-6


# ---------- Utils ----------


class TestUtils:
    def test_grid_coords(self):
        from pixelduet.utils import grid_coords

        x, y = grid_coords(3, 4)
        assert x.shape == (12,)
        assert y.shape == (12,)
        assert x[0] == 0 and y[0] == 0
        assert x[4] == 0 and y[4] == 1  # second row, first col

    def test_permute_image_identity(self):
        from pixelduet.utils import permute_image

        img = np.random.rand(5, 5, 3).astype(np.float32) * 255
        M = np.arange(25, dtype=np.int64)  # identity mapping
        result = permute_image(img, M)
        np.testing.assert_array_equal(result, img.astype(np.uint8))


# ---------- Mapping ----------


class TestMapping:
    def test_mapping_global_is_bijection(self):
        from pixelduet.mapping import mapping_global

        A = np.random.rand(10, 10, 3).astype(np.float32) * 255
        B = np.random.rand(10, 10, 3).astype(np.float32) * 255
        M = mapping_global(A, B)
        assert M.shape == (100,)
        # Bijection: sorted mapping must be 0..N-1
        np.testing.assert_array_equal(np.sort(M), np.arange(100))

    def test_mapping_global_pairs_brightness(self):
        from pixelduet.mapping import mapping_global

        # Darkest in A should map to darkest in B
        A = np.zeros((2, 1, 3), dtype=np.float32)
        A[0, 0] = [0, 0, 0]  # dark
        A[1, 0] = [255, 255, 255]  # bright
        B = np.zeros((2, 1, 3), dtype=np.float32)
        B[0, 0] = [10, 10, 10]  # dark
        B[1, 0] = [250, 250, 250]  # bright
        M = mapping_global(A, B)
        assert M[0] == 0  # dark->dark
        assert M[1] == 1  # bright->bright

    def test_multiscale_map_bijection(self):
        from pixelduet.mapping import multiscale_map

        np.random.seed(42)
        A = np.random.rand(20, 20, 3).astype(np.float32) * 255
        B = np.random.rand(20, 20, 3).astype(np.float32) * 255
        M, H, W = multiscale_map(
            A,
            B,
            levels=[20],
            tiles=[10],
            lambdas_spatial=[0.2],
            lambdas_parent=[0.0],
        )
        N = H * W
        assert M.shape == (N,)
        np.testing.assert_array_equal(np.sort(M), np.arange(N))

    def test_refine_preserves_bijection(self):
        from pixelduet.color import rgb_to_lab
        from pixelduet.mapping import mapping_global, refine_swaps

        A = np.random.rand(8, 8, 3).astype(np.float32) * 255
        B = np.random.rand(8, 8, 3).astype(np.float32) * 255
        M = mapping_global(A, B)
        A_lab = rgb_to_lab(A.reshape(-1, 3))
        B_lab = rgb_to_lab(B.reshape(-1, 3))
        M_refined = refine_swaps(M, A_lab, B_lab, 8, 8, iters=5, radius=1)
        np.testing.assert_array_equal(np.sort(M_refined), np.arange(64))


# ---------- Animation ----------


class TestAnimation:
    def test_ease_boundaries(self):
        from pixelduet.animation import ease_in_out_cos

        assert pytest.approx(ease_in_out_cos(np.array([0.0]))[0]) == 0.0
        assert pytest.approx(ease_in_out_cos(np.array([1.0]))[0]) == 1.0

    def test_ease_monotonic(self):
        from pixelduet.animation import ease_in_out_cos

        t = np.linspace(0, 1, 100)
        e = ease_in_out_cos(t)
        assert np.all(np.diff(e) >= -1e-10)

    def test_bezier_endpoints(self):
        from pixelduet.animation import bezier_cubic

        Sx, Sy = np.array([0.0]), np.array([0.0])
        Ex, Ey = np.array([10.0]), np.array([5.0])
        C1x, C1y = np.array([3.0]), np.array([1.0])
        C2x, C2y = np.array([7.0]), np.array([4.0])
        x0, y0 = bezier_cubic(Sx, Sy, C1x, C1y, C2x, C2y, Ex, Ey, np.array([0.0]))
        x1, y1 = bezier_cubic(Sx, Sy, C1x, C1y, C2x, C2y, Ex, Ey, np.array([1.0]))
        assert pytest.approx(x0[0]) == 0.0
        assert pytest.approx(y0[0]) == 0.0
        assert pytest.approx(x1[0]) == 10.0
        assert pytest.approx(y1[0]) == 5.0

    def test_compute_paths_shape(self):
        from pixelduet.animation import compute_paths

        paths = compute_paths(10, 10, np.arange(100, dtype=np.int64), arc=0.1)
        assert len(paths) == 8
        for p in paths:
            assert p.shape == (100,)

    def test_compute_stagger_range(self):
        from pixelduet.animation import compute_stagger

        s = compute_stagger(1000, stagger=0.5, seed=0)
        assert s.shape == (1000,)
        assert s.min() >= 0.0
        assert s.max() <= 0.5

    def test_frame_positions_hold(self):
        from pixelduet.animation import compute_paths, compute_stagger, frame_positions

        M = np.arange(25, dtype=np.int64)
        paths = compute_paths(5, 5, M)
        start = compute_stagger(25, stagger=0.3, seed=0)
        # Frame past motion should return None (hold frame)
        result = frame_positions(paths, start, frame=60, total_motion_frames=50)
        assert result is None

    def test_frame_positions_motion(self):
        from pixelduet.animation import compute_paths, compute_stagger, frame_positions

        M = np.arange(25, dtype=np.int64)
        paths = compute_paths(5, 5, M)
        start = compute_stagger(25, stagger=0.3, seed=0)
        result = frame_positions(paths, start, frame=0, total_motion_frames=50)
        assert result is not None
        x, _y = result
        assert x.shape == (25,)


# ---------- Public API ----------


class TestPublicAPI:
    def test_map_pixels(self):
        from pixelduet import map_pixels

        A = np.random.rand(15, 15, 3).astype(np.float32) * 255
        B = np.random.rand(15, 15, 3).astype(np.float32) * 255
        M, H, W = map_pixels(
            A, B, levels=[15], tiles=[8], lambdas_spatial=[0.2], lambdas_parent=[0.0]
        )
        assert H == 15
        assert W == 15
        np.testing.assert_array_equal(np.sort(M), np.arange(225))

    def test_auto_levels(self):
        from pixelduet import _auto_levels

        levels = _auto_levels(260)
        assert levels[-1] == 260
        assert len(levels) == 3
        assert levels[0] < levels[1] < levels[2]


# ---------- Chain ----------


class TestChain:
    def test_pixel_duet_chain_requires_two_images(self):
        from pixelduet import pixel_duet_chain

        with pytest.raises(ValueError, match="at least 2"):
            pixel_duet_chain(["only_one.png"])

    def test_chain_cumulative_mapping(self):
        """Verify that chaining A->B->C computes valid bijective mappings."""
        from pixelduet import map_pixels
        from pixelduet.utils import permute_image

        np.random.seed(99)
        A = np.random.rand(10, 10, 3).astype(np.float32) * 255
        B = np.random.rand(10, 10, 3).astype(np.float32) * 255
        C = np.random.rand(10, 10, 3).astype(np.float32) * 255

        # Segment 1: A -> B
        M1, H1, W1 = map_pixels(
            A, B, levels=[10], tiles=[5], lambdas_spatial=[0.2], lambdas_parent=[0.0]
        )
        N = H1 * W1
        np.testing.assert_array_equal(np.sort(M1), np.arange(N))

        # Segment 2: permuted(A,M1) -> C
        current = permute_image(A, M1).astype(np.float32)
        M2, _H2, _W2 = map_pixels(
            current, C, levels=[10], tiles=[5], lambdas_spatial=[0.2], lambdas_parent=[0.0]
        )
        np.testing.assert_array_equal(np.sort(M2), np.arange(N))

        # Cumulative mapping should also be a bijection
        cumulative = M1[M2]
        np.testing.assert_array_equal(np.sort(cumulative), np.arange(N))


# ---------- Batch ----------


class TestBatch:
    def test_find_image_pairs(self, tmp_path):
        from pixelduet.batch import find_image_pairs

        # Create 4 fake images
        for name in ["a.png", "b.png", "c.jpg", "d.jpeg"]:
            (tmp_path / name).write_bytes(b"fake")
        pairs = find_image_pairs(str(tmp_path))
        assert len(pairs) == 2
        assert all(len(p) == 2 for p in pairs)

    def test_find_image_pairs_odd_count(self, tmp_path):
        from pixelduet.batch import find_image_pairs

        for name in ["a.png", "b.png", "c.png"]:
            (tmp_path / name).write_bytes(b"fake")
        pairs = find_image_pairs(str(tmp_path))
        assert len(pairs) == 1  # only first pair, odd one left out

    def test_find_image_pairs_ignores_non_images(self, tmp_path):
        from pixelduet.batch import find_image_pairs

        (tmp_path / "a.png").write_bytes(b"fake")
        (tmp_path / "readme.txt").write_bytes(b"text")
        (tmp_path / "b.png").write_bytes(b"fake")
        pairs = find_image_pairs(str(tmp_path))
        assert len(pairs) == 1

    def test_generate_gallery_html(self, tmp_path):
        from pixelduet.batch import generate_gallery_html

        results = [str(tmp_path / "a_to_b.gif"), str(tmp_path / "c_to_d.mp4")]
        html_path = generate_gallery_html(results, output_dir=str(tmp_path))
        assert os.path.exists(html_path)
        with open(html_path) as f:
            content = f.read()
        assert "a_to_b.gif" in content
        assert "c_to_d.mp4" in content
        assert "<video" in content
        assert "<img" in content


# ---------- Web API ----------


class TestWebAPI:
    def test_submit_job(self, tmp_path):
        from pixelduet.web import create_app

        app = create_app(static_dir=str(tmp_path))  # no static files needed
        from starlette.testclient import TestClient

        client = TestClient(app)

        # Create tiny test images
        from PIL import Image as PILImage

        img_a = PILImage.new("RGB", (10, 10), color=(255, 0, 0))
        img_b = PILImage.new("RGB", (10, 10), color=(0, 0, 255))
        path_a = str(tmp_path / "a.png")
        path_b = str(tmp_path / "b.png")
        img_a.save(path_a)
        img_b.save(path_b)

        with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
            resp = client.post(
                "/api/duet",
                files={
                    "image_a": ("a.png", fa, "image/png"),
                    "image_b": ("b.png", fb, "image/png"),
                },
                data={"width": "10", "frames": "2", "hold": "1", "format": "gif"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data

    def test_job_status_not_found(self, tmp_path):
        from starlette.testclient import TestClient

        from pixelduet.web import create_app

        app = create_app(static_dir=str(tmp_path))
        client = TestClient(app)
        resp = client.get("/api/jobs/nonexistent/status")
        assert resp.status_code == 404

    def test_job_result_not_found(self, tmp_path):
        from starlette.testclient import TestClient

        from pixelduet.web import create_app

        app = create_app(static_dir=str(tmp_path))
        client = TestClient(app)
        resp = client.get("/api/jobs/nonexistent/result")
        assert resp.status_code == 404

    def test_invalid_format(self, tmp_path):
        from PIL import Image as PILImage
        from starlette.testclient import TestClient

        from pixelduet.web import create_app

        app = create_app(static_dir=str(tmp_path))
        client = TestClient(app)

        img = PILImage.new("RGB", (5, 5), color=(128, 128, 128))
        path = str(tmp_path / "test.png")
        img.save(path)

        with open(path, "rb") as f1, open(path, "rb") as f2:
            resp = client.post(
                "/api/duet",
                files={
                    "image_a": ("a.png", f1, "image/png"),
                    "image_b": ("b.png", f2, "image/png"),
                },
                data={"format": "bmp"},
            )
        assert resp.status_code == 400


# ---------- End-to-end render ----------


class TestRenderE2E:
    def test_render_gif_from_numpy(self, tmp_path):
        """Full pipeline: two numpy images -> GIF file on disk."""
        from pixelduet import map_pixels
        from pixelduet.render import render_gif

        np.random.seed(7)
        A = np.random.rand(8, 8, 3).astype(np.float32) * 255
        B = np.random.rand(8, 8, 3).astype(np.float32) * 255

        M, H, W = map_pixels(
            A, B, levels=[8], tiles=[4], lambdas_spatial=[0.2], lambdas_parent=[0.0]
        )
        output = str(tmp_path / "test.gif")
        result = render_gif(A, M, H, W, output, frames=3, hold=1, fps=10)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 100  # not empty

    def test_pixel_duet_end_to_end(self, tmp_path):
        """pixel_duet() with real image files on disk."""
        from PIL import Image as PILImage

        from pixelduet import pixel_duet

        # Create small test images
        img_a = PILImage.new("RGB", (16, 16), color=(200, 50, 50))
        img_b = PILImage.new("RGB", (16, 16), color=(50, 50, 200))
        path_a = str(tmp_path / "a.png")
        path_b = str(tmp_path / "b.png")
        img_a.save(path_a)
        img_b.save(path_b)

        output = str(tmp_path / "result.gif")
        result = pixel_duet(path_a, path_b, output_path=output, width=16, frames=3, hold=1)
        assert os.path.exists(result)
        assert os.path.getsize(result) > 100


# ---------- Web E2E ----------


class TestWebE2E:
    def test_full_job_lifecycle(self, tmp_path):
        """Submit job, poll to completion, download result."""
        import time

        from PIL import Image as PILImage
        from starlette.testclient import TestClient

        from pixelduet.web import create_app

        app = create_app(static_dir=str(tmp_path))
        client = TestClient(app)

        # Create tiny images
        img_a = PILImage.new("RGB", (8, 8), color=(255, 0, 0))
        img_b = PILImage.new("RGB", (8, 8), color=(0, 0, 255))
        path_a = str(tmp_path / "a.png")
        path_b = str(tmp_path / "b.png")
        img_a.save(path_a)
        img_b.save(path_b)

        # Submit
        with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
            resp = client.post(
                "/api/duet",
                files={
                    "image_a": ("a.png", fa, "image/png"),
                    "image_b": ("b.png", fb, "image/png"),
                },
                data={"width": "8", "frames": "2", "hold": "1", "format": "gif"},
            )
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        # Poll until done or timeout
        sd = None
        for _ in range(120):
            sr = client.get(f"/api/jobs/{job_id}/status")
            sd = sr.json()
            if sd["status"] in ("done", "failed"):
                break
            time.sleep(0.5)

        assert sd is not None, "Polling returned no result"
        assert sd["status"] == "done", f"Job failed: {sd.get('error')}"
        assert sd["progress"] == 100.0

        # Download result
        rr = client.get(f"/api/jobs/{job_id}/result")
        assert rr.status_code == 200
        assert len(rr.content) > 100
        assert rr.headers["content-type"] == "image/gif"


# ---------- CLI ----------


class TestCLI:
    def test_help(self):
        """CLI --help exits cleanly."""
        from pixelduet.cli import main

        try:
            main(["--help"])
        except SystemExit as e:
            assert e.code == 0

    def test_no_args_errors(self):
        """CLI with no args prints error."""
        from pixelduet.cli import main

        try:
            main([])
        except SystemExit as e:
            assert e.code != 0


# ---------- Job cleanup ----------


class TestJobCleanup:
    def test_cleanup_removes_expired_jobs(self):
        import time

        from pixelduet.web import Job, JobStatus, _cleanup_old_jobs, _jobs, _jobs_lock

        # Create a fake expired job
        job = Job("test_expired")
        job.status = JobStatus.done
        job.created_at = time.time() - 7200  # 2 hours ago

        with _jobs_lock:
            _jobs["test_expired"] = job

        _cleanup_old_jobs()

        with _jobs_lock:
            assert "test_expired" not in _jobs

    def test_cleanup_keeps_recent_jobs(self):
        import time

        from pixelduet.web import Job, JobStatus, _cleanup_old_jobs, _jobs, _jobs_lock

        job = Job("test_recent")
        job.status = JobStatus.done
        job.created_at = time.time()  # just now

        with _jobs_lock:
            _jobs["test_recent"] = job

        _cleanup_old_jobs()

        with _jobs_lock:
            assert "test_recent" in _jobs
            # Clean up after test
            del _jobs["test_recent"]


# ---------- Easing ----------


class TestEasing:
    """Tests for all easing functions and the get_easing dispatcher."""

    def test_all_easings_boundaries(self):
        """All easing functions map 0->0 and 1->1."""
        from pixelduet.animation import EASINGS

        for name, fn in EASINGS.items():
            t0 = np.array([0.0])
            t1 = np.array([1.0])
            v0 = fn(t0)
            v1 = fn(t1)
            assert pytest.approx(v0[0], abs=1e-6) == 0.0, f"{name} at t=0"
            assert pytest.approx(v1[0], abs=1e-6) == 1.0, f"{name} at t=1"

    def test_monotonic_easings(self):
        """Linear, cosine, and expo should be monotonically non-decreasing."""
        from pixelduet.animation import ease_expo, ease_in_out_cos, ease_linear

        t = np.linspace(0, 1, 200)
        for fn in [ease_linear, ease_in_out_cos, ease_expo]:
            e = fn(t)
            assert np.all(np.diff(e) >= -1e-10), f"{fn.__name__} is not monotonic"

    def test_get_easing_known(self):
        from pixelduet.animation import EASINGS, get_easing

        for name in EASINGS:
            fn = get_easing(name)
            assert callable(fn)

    def test_get_easing_default(self):
        from pixelduet.animation import ease_in_out_cos, get_easing

        fn = get_easing()
        assert fn is ease_in_out_cos

    def test_get_easing_unknown_raises(self):
        from pixelduet.animation import get_easing

        with pytest.raises(ValueError, match="Unknown easing"):
            get_easing("nonexistent")

    def test_bounce_shape(self):
        from pixelduet.animation import ease_bounce

        t = np.linspace(0, 1, 50)
        result = ease_bounce(t)
        assert result.shape == (50,)
        assert result[0] == pytest.approx(0.0, abs=1e-6)
        assert result[-1] == pytest.approx(1.0, abs=1e-6)

    def test_elastic_allows_overshoot(self):
        """Elastic easing may slightly overshoot 1.0 before settling."""
        from pixelduet.animation import ease_elastic

        t = np.linspace(0, 1, 100)
        result = ease_elastic(t)
        # Should still start at 0 and end at 1
        assert result[0] == pytest.approx(0.0, abs=1e-6)
        assert result[-1] == pytest.approx(1.0, abs=1e-6)

    def test_step_discrete(self):
        from pixelduet.animation import ease_step

        t = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
        result = ease_step(t, steps=4)
        # Steps should produce quantized values
        unique = np.unique(result)
        assert len(unique) <= 5  # at most steps+1 distinct values


# ---------- Presets endpoint ----------


class TestPresets:
    def test_get_presets(self, tmp_path):
        from starlette.testclient import TestClient

        from pixelduet.web import create_app

        app = create_app(static_dir=str(tmp_path))
        client = TestClient(app)

        resp = client.get("/api/presets")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) > 0
        # Each preset should have id, name, params
        for p in data:
            assert "id" in p
            assert "name" in p
            assert "params" in p
            assert "easing" in p["params"]


# ---------- URL input ----------


class TestURLInput:
    def test_submit_url_invalid(self, tmp_path):
        """POST /api/duet/url with invalid URLs should fail gracefully."""
        from starlette.testclient import TestClient

        from pixelduet.web import create_app

        app = create_app(static_dir=str(tmp_path))
        client = TestClient(app)

        resp = client.post(
            "/api/duet/url",
            data={
                "image_a_url": "http://invalid.test/nonexistent.png",
                "image_b_url": "http://invalid.test/nonexistent.png",
            },
        )
        assert resp.status_code == 400
        assert "error" in resp.json()


# ---------- Gallery ----------


class TestGallery:
    def test_gallery_empty(self, tmp_path):
        from starlette.testclient import TestClient

        from pixelduet.web import create_app

        app = create_app(static_dir=str(tmp_path))
        client = TestClient(app)

        resp = client.get("/api/gallery")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)

    def test_gallery_after_job(self, tmp_path):
        """Gallery should include completed jobs."""
        import time

        from PIL import Image as PILImage
        from starlette.testclient import TestClient

        from pixelduet.web import create_app

        app = create_app(static_dir=str(tmp_path))
        client = TestClient(app)

        img_a = PILImage.new("RGB", (8, 8), color=(255, 0, 0))
        img_b = PILImage.new("RGB", (8, 8), color=(0, 0, 255))
        path_a = str(tmp_path / "a.png")
        path_b = str(tmp_path / "b.png")
        img_a.save(path_a)
        img_b.save(path_b)

        with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
            resp = client.post(
                "/api/duet",
                files={
                    "image_a": ("a.png", fa, "image/png"),
                    "image_b": ("b.png", fb, "image/png"),
                },
                data={"width": "8", "frames": "2", "hold": "1", "format": "gif"},
            )
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        # Wait for completion
        sd = None
        for _ in range(120):
            sr = client.get(f"/api/jobs/{job_id}/status")
            sd = sr.json()
            if sd["status"] in ("done", "failed"):
                break
            time.sleep(0.5)

        assert sd is not None
        assert sd["status"] == "done"

        # Gallery should contain the job
        gr = client.get("/api/gallery")
        gallery = gr.json()
        job_ids = [j["job_id"] for j in gallery]
        assert job_id in job_ids


# ---------- Share ----------


class TestShare:
    def test_share_not_found(self, tmp_path):
        from starlette.testclient import TestClient

        from pixelduet.web import create_app

        app = create_app(static_dir=str(tmp_path))
        client = TestClient(app)

        resp = client.get("/api/jobs/nonexistent/share")
        assert resp.status_code == 404

    def test_share_completed_job(self, tmp_path):
        """Share endpoint returns URL and expires_in for completed job."""
        import time

        from PIL import Image as PILImage
        from starlette.testclient import TestClient

        from pixelduet.web import create_app

        app = create_app(static_dir=str(tmp_path))
        client = TestClient(app)

        img_a = PILImage.new("RGB", (8, 8), color=(255, 0, 0))
        img_b = PILImage.new("RGB", (8, 8), color=(0, 0, 255))
        path_a = str(tmp_path / "a.png")
        path_b = str(tmp_path / "b.png")
        img_a.save(path_a)
        img_b.save(path_b)

        with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
            resp = client.post(
                "/api/duet",
                files={
                    "image_a": ("a.png", fa, "image/png"),
                    "image_b": ("b.png", fb, "image/png"),
                },
                data={"width": "8", "frames": "2", "hold": "1", "format": "gif"},
            )
        job_id = resp.json()["job_id"]

        for _ in range(120):
            sr = client.get(f"/api/jobs/{job_id}/status")
            sd = sr.json()
            if sd["status"] in ("done", "failed"):
                break
            time.sleep(0.5)

        assert sd["status"] == "done"

        share = client.get(f"/api/jobs/{job_id}/share")
        assert share.status_code == 200
        data = share.json()
        assert "url" in data
        assert "expires_in" in data
        assert data["job_id"] == job_id
        assert data["expires_in"] > 0


# ---------- Easing parameter integration ----------


class TestEasingParam:
    def test_pixel_duet_with_bounce(self, tmp_path):
        """pixel_duet() with easing='bounce' runs without error."""
        from PIL import Image as PILImage

        from pixelduet import pixel_duet

        img_a = PILImage.new("RGB", (16, 16), color=(200, 50, 50))
        img_b = PILImage.new("RGB", (16, 16), color=(50, 50, 200))
        path_a = str(tmp_path / "a.png")
        path_b = str(tmp_path / "b.png")
        img_a.save(path_a)
        img_b.save(path_b)

        output = str(tmp_path / "bounce.gif")
        result = pixel_duet(
            path_a, path_b, output_path=output, width=16, frames=3, hold=1, easing="bounce"
        )
        assert os.path.exists(result)
        assert os.path.getsize(result) > 100

    def test_web_submit_with_easing(self, tmp_path):
        """Web API accepts easing parameter."""
        from PIL import Image as PILImage
        from starlette.testclient import TestClient

        from pixelduet.web import create_app

        app = create_app(static_dir=str(tmp_path))
        client = TestClient(app)

        img = PILImage.new("RGB", (8, 8), color=(128, 128, 128))
        path = str(tmp_path / "test.png")
        img.save(path)

        with open(path, "rb") as f1, open(path, "rb") as f2:
            resp = client.post(
                "/api/duet",
                files={
                    "image_a": ("a.png", f1, "image/png"),
                    "image_b": ("b.png", f2, "image/png"),
                },
                data={"width": "8", "frames": "2", "hold": "1", "format": "gif", "easing": "expo"},
            )
        assert resp.status_code == 200

    def test_web_submit_invalid_easing(self, tmp_path):
        """Web API rejects unknown easing."""
        from PIL import Image as PILImage
        from starlette.testclient import TestClient

        from pixelduet.web import create_app

        app = create_app(static_dir=str(tmp_path))
        client = TestClient(app)

        img = PILImage.new("RGB", (5, 5), color=(128, 128, 128))
        path = str(tmp_path / "test.png")
        img.save(path)

        with open(path, "rb") as f1, open(path, "rb") as f2:
            resp = client.post(
                "/api/duet",
                files={
                    "image_a": ("a.png", f1, "image/png"),
                    "image_b": ("b.png", f2, "image/png"),
                },
                data={"easing": "unknown"},
            )
        assert resp.status_code == 400
