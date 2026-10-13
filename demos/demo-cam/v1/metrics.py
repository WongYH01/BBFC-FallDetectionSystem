"""Measurement mode for v1 — per-frame metrics and an end-of-run summary.

A measurement run is deliberately NOT a recording. It writes no video, because
every figure collected while encoding a 2560x720 mp4 per frame measures the
encoder as much as the detector. What comes out is a JSON summary and a
per-frame CSV in metrics/.

Read METRICS.md before quoting any of these numbers — several of them measure
something narrower than their name suggests.
"""
import csv
import json
import math
import os
import platform
import threading
import time
import weakref
from collections import OrderedDict

METRICS_DIR = os.path.join(os.path.dirname(__file__), "metrics")

# Frames to discard from the summary. The first inference of a run pays for
# CUDA kernel compilation and cuDNN autotuning and can be an order of magnitude
# slower than steady state; averaging it in makes the run look worse than the
# system is.
METRICS_WARMUP_FRAMES = int(os.environ.get("METRICS_WARMUP_FRAMES", "30"))

# Matches Ultralytics' own Annotator.kpts default, so "visible" here means the
# same thing it means on screen.
KPT_CONF_THRESH = float(os.environ.get("KPT_CONF_THRESH", "0.25"))

CSV_FIELDS = [
    "frame", "t_rel_s", "latency_ms",
    "preprocess_ms", "inference_ms", "postprocess_ms",
    "n_people", "n_ids", "ids",
    "kpt_conf_mean", "kpt_conf_sum", "n_kpt_visible", "n_kpt_total",
    "warmup",
]


# ---------------------------------------------------------------------------
# Arrival tap
#
# Ultralytics' LoadStreams runs a daemon thread that pulls frames off the RTSP
# socket at camera rate and stores each as `self.imgs[i] = [im]`. With
# buffer=False the consumer then takes the newest and discards the rest, so
# frames dropped between inferences are never counted anywhere.
#
# Replacing `imgs` with this list subclass turns that assignment into a
# per-arrival callback: an exact count of what the camera delivered, and a
# timestamp for each frame so latency can be measured from arrival rather than
# from consumption.
# ---------------------------------------------------------------------------
_tap_lock = threading.Lock()
_tap: "_ArrivalTap | None" = None
_tap_ds = None          # the dataset object the tap is currently installed on
_tap_installed_at = 0.0  # when it went on; a run may start before the stream does


class _ArrivalTap(list):
    """Wraps LoadStreams.imgs. __setitem__ fires once per frame received."""

    def __init__(self, inner):
        super().__init__(inner)
        self.n = 0
        # id(ndarray) -> (arrival_time, weakref). Bounded: holding strong refs
        # to frames would leak ~2.7 MB each.
        self.stamps = OrderedDict()

    def __setitem__(self, i, value):
        super().__setitem__(i, value)
        if not value:
            return
        im = value[0]
        t = time.perf_counter()
        with _tap_lock:
            self.n += 1
            try:
                self.stamps[id(im)] = (t, weakref.ref(im))
            except TypeError:      # not weak-referenceable; skip the stamp
                return
            while len(self.stamps) > 64:
                self.stamps.popitem(last=False)


def install_tap(dataset) -> bool:
    """Install the tap on a LoadStreams instance. Idempotent per dataset.

    Call every frame — it is an identity check after the first time. The
    dataset is rebuilt whenever model.track() reconnects, and the tap does not
    survive that, so re-installing must be automatic rather than remembered.
    """
    global _tap, _tap_ds, _tap_installed_at
    if dataset is _tap_ds:
        return False
    imgs = getattr(dataset, "imgs", None)
    if imgs is None:
        return False
    # Swapping this while the reader thread runs can lose at most one frame to
    # the race. Irrelevant at these sample sizes.
    _tap = _ArrivalTap(imgs)
    dataset.imgs = _tap
    _tap_ds = dataset
    _tap_installed_at = time.perf_counter()
    return True


def arrival_count() -> int:
    with _tap_lock:
        return _tap.n if _tap is not None else 0


def arrival_of(im):
    """Arrival timestamp for this exact frame, or None if it cannot be matched.

    The weakref check is load-bearing, not defensive. With buffer=False the
    previous frame is freed almost immediately, so CPython readily reuses its
    address for the next one; keying on id() alone would silently attach a
    too-recent timestamp and make latency read BETTER than it is. Comparing the
    weakref back to the object proves it is the same frame, not a recycled
    address.
    """
    if _tap is None:
        return None
    with _tap_lock:
        ent = _tap.stamps.get(id(im))
    if ent is not None and ent[1]() is im:
        return ent[0]
    return None


# ---------------------------------------------------------------------------
# Per-run collection
# ---------------------------------------------------------------------------
class Run:
    """Accumulates one measurement run and writes its two files on finish()."""

    def __init__(self, ts: str, provenance: dict):
        self.ts = ts
        self.provenance = provenance
        self.rows = []
        self.t0 = time.perf_counter()
        self.arrivals0 = arrival_count()

    def add(self, result, t_done: float) -> None:
        """Score one frame. Called once per frame while the run is active."""
        i = len(self.rows)
        arrival = arrival_of(result.orig_img)
        speed = getattr(result, "speed", None) or {}

        boxes = getattr(result, "boxes", None)
        n_people = len(boxes) if boxes is not None else 0
        ids = []
        if boxes is not None and getattr(boxes, "id", None) is not None:
            ids = [int(v) for v in boxes.id.int().cpu().tolist()]

        # Visible keypoints only, per the chosen definition. n_kpt_total is
        # kept alongside so the confidence figure can be read honestly — see
        # visible_keypoint_rate in the summary.
        conf_sum = 0.0
        n_vis = 0
        n_total = 0
        kpts = getattr(result, "keypoints", None)
        if kpts is not None and getattr(kpts, "conf", None) is not None:
            c = kpts.conf.cpu().numpy()
            n_total = int(c.size)
            m = c[c > KPT_CONF_THRESH]
            n_vis = int(m.size)
            conf_sum = float(m.sum())

        self.rows.append({
            "frame": i,
            "t_rel_s": round((arrival if arrival is not None else t_done) - self.t0, 4),
            "latency_ms": (round((t_done - arrival) * 1000, 2)
                           if arrival is not None else ""),
            "preprocess_ms": _r(speed.get("preprocess")),
            "inference_ms": _r(speed.get("inference")),
            "postprocess_ms": _r(speed.get("postprocess")),
            "n_people": n_people,
            "n_ids": len(ids),
            "ids": ";".join(str(v) for v in ids),
            "kpt_conf_mean": round(conf_sum / n_vis, 4) if n_vis else "",
            "kpt_conf_sum": round(conf_sum, 4),
            "n_kpt_visible": n_vis,
            "n_kpt_total": n_total,
            "warmup": int(i < METRICS_WARMUP_FRAMES),
        })

    def finish(self) -> dict:
        """Write metrics_<ts>.json and metrics_<ts>_frames.csv; return names."""
        t_end = time.perf_counter()
        duration = t_end - self.t0
        arrivals = arrival_count() - self.arrivals0
        # The tap may have gone on AFTER this run started -- the stream can take
        # several seconds to deliver its first frame, and the run is allowed to
        # start before that. Divide the arrival count by the window it actually
        # covers, not by the whole run, or stream_fps reads low for that reason
        # alone. Deciding usability here rather than at __init__ is the point:
        # what matters is whether the tap was counting, not whether it happened
        # to be installed at the moment Record was pressed.
        covered = t_end - max(self.t0, _tap_installed_at)
        summary = self._summarise(duration, arrivals, covered)

        os.makedirs(METRICS_DIR, exist_ok=True)
        name_json = f"metrics_{self.ts}.json"
        name_csv = f"metrics_{self.ts}_frames.csv"

        with open(os.path.join(METRICS_DIR, name_csv), "w", newline="",
                  encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
            w.writeheader()
            w.writerows(self.rows)

        with open(os.path.join(METRICS_DIR, name_json), "w",
                  encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        return {"filename_metrics": name_json, "filename_frames": name_csv,
                "summary": summary}

    # -- summary ------------------------------------------------------------
    def _summarise(self, duration: float, arrivals: int, covered: float) -> dict:
        analysed = [r for r in self.rows if not r["warmup"]]
        base = {
            "run": self.ts,
            "duration_sec": round(duration, 2),
            "frames_total": len(self.rows),
            "frames_analysed": len(analysed),
            "warmup_frames_excluded": METRICS_WARMUP_FRAMES,
            "video_written": False,
            "provenance": self.provenance,
            "notes": {
                "latency": "arrival at this host -> annotated JPEG encoded. "
                           "Excludes camera sensor, Pi encode, network and "
                           "browser decode. See METRICS.md glass-to-glass.",
                "track_stability": "Proxies, not MOTA. True ID-switch counts "
                                   "need ground-truth annotation this run has "
                                   "none of.",
                "stream_fps": "Frames arriving at this host from MediaMTX. "
                              "Loss upstream of MediaMTX is invisible here.",
            },
        }
        if not analysed:
            # Fewer frames than the warm-up cut. Say so rather than dividing by
            # zero and emitting a confident-looking nonsense figure.
            base["insufficient_data"] = True
            return base

        base["insufficient_data"] = False
        span = max(a["t_rel_s"] for a in analysed) - min(a["t_rel_s"] for a in analysed)
        span = span or duration

        # -- rates ----------------------------------------------------------
        # (n-1), not n: n frames spanning `span` seconds have n-1 gaps between
        # them. Using n overstates the rate, which matters at small sample sizes.
        inference_fps = (len(analysed) - 1) / span if span > 0 and len(analysed) > 1 else 0.0

        # The tap depends on an Ultralytics internal. If it was never installed,
        # or counted nothing while frames were plainly being processed, the hook
        # is unavailable or has broken. Report NOTHING in that case — never a
        # zero, which reads as "the stream delivered no frames" and is the kind
        # of figure that ends up quoted in a report.
        tap_usable = _tap is not None and arrivals > 0 and covered > 0
        stream_fps = arrivals / covered if tap_usable else None
        base["rates"] = {
            "stream_fps": round(stream_fps, 2) if tap_usable else None,
            "stream_fps_source": "loader-tap" if tap_usable else "unavailable",
            "inference_fps": round(inference_fps, 2),
            "dropped_frame_pct": (round(max(0.0, 1 - inference_fps / stream_fps) * 100, 1)
                                  if tap_usable and stream_fps > 0 else None),
            "arrivals_counted": arrivals,
            "arrival_window_sec": round(covered, 2) if tap_usable else None,
            # Distinct from 1/inference_fps: the gap between them is everything
            # that is not the model (decode, plot, JPEG encode, MJPEG write).
            "model_inference_ms_mean": _mean([r["inference_ms"] for r in analysed]),
            "model_preprocess_ms_mean": _mean([r["preprocess_ms"] for r in analysed]),
            "model_postprocess_ms_mean": _mean([r["postprocess_ms"] for r in analysed]),
        }

        # -- latency --------------------------------------------------------
        lat = [r["latency_ms"] for r in analysed if r["latency_ms"] != ""]
        base["latency_ms"] = {
            "scope": "arrival -> annotated JPEG encoded",
            "mean": _mean(lat), "p50": _pct(lat, 50), "p95": _pct(lat, 95),
            "max": round(max(lat), 2) if lat else None,
            "samples": len(lat),
            # Frames whose arrival stamp could not be matched to the exact
            # ndarray. A large number here means the tap is unreliable.
            "unmatched": len(analysed) - len(lat),
        }

        # -- keypoint confidence --------------------------------------------
        vis = sum(r["n_kpt_visible"] for r in analysed)
        tot = sum(r["n_kpt_total"] for r in analysed)
        csum = sum(r["kpt_conf_sum"] for r in analysed)
        base["keypoint_confidence"] = {
            "threshold": KPT_CONF_THRESH,
            # Pooled over every visible keypoint, not a mean of per-frame
            # means, so busier frames weigh more.
            "mean_visible": round(csum / vis, 4) if vis else None,
            "visible_keypoints": vis,
            "total_keypoints": tot,
            # Not optional. Restricted to visible joints, a model that finds
            # two confident keypoints and misses fifteen still scores ~0.9;
            # this rate is what keeps that number honest.
            "visible_keypoint_rate": round(vis / tot, 4) if tot else None,
        }

        # -- presence continuity --------------------------------------------
        with_person = [r["n_people"] > 0 for r in analysed]
        gaps, run_len = [], 0
        for present in with_person:
            if present:
                if run_len:
                    gaps.append(run_len)
                run_len = 0
            else:
                run_len += 1
        if run_len:
            gaps.append(run_len)
        longest = max(gaps) if gaps else 0
        base["presence_continuity"] = {
            "frames_with_person_pct": round(100 * sum(with_person) / len(analysed), 1),
            "gap_count": len(gaps),
            "longest_gap_frames": longest,
            "longest_gap_sec": (round(longest / inference_fps, 2)
                                if inference_fps > 0 else None),
        }

        # -- track ID stability ---------------------------------------------
        detected = [r for r in analysed if r["n_people"] > 0]
        with_ids = [r for r in detected if r["n_ids"] > 0]
        seen = {}
        for r in analysed:
            for tid in (r["ids"].split(";") if r["ids"] else []):
                seen[tid] = seen.get(tid, 0) + 1
        max_conc = max((r["n_people"] for r in analysed), default=0)
        lifetimes = list(seen.values())
        base["track_stability"] = {
            "frames_with_detections": len(detected),
            "frames_with_ids_pct": (round(100 * len(with_ids) / len(detected), 1)
                                    if detected else None),
            "distinct_ids": len(seen),
            "max_concurrent_people": max_conc,
            # Crude but honest: many more IDs than people implies the tracker
            # kept losing and re-acquiring them.
            # None, not 0.0, when nothing carried an ID: a zero here reads as
            # perfect stability when it actually means no tracking happened.
            "fragmentation": (round(len(seen) / max_conc, 2)
                              if max_conc and seen else None),
            "mean_track_lifetime_frames": _mean(lifetimes),
            "short_tracks_under_5_frames": sum(1 for v in lifetimes if v < 5),
        }
        return base


# ---------------------------------------------------------------------------
def _r(v):
    return round(float(v), 3) if v is not None else ""


def _mean(vals):
    nums = [float(v) for v in vals if v != "" and v is not None]
    return round(sum(nums) / len(nums), 2) if nums else None


def _pct(vals, p):
    """Nearest-rank percentile: the smallest value at or above rank ceil(p/100*n).

    Deliberately not statistics.quantiles, which interpolates — with a handful of
    samples that reports a latency figure that was never actually measured.
    """
    nums = sorted(float(v) for v in vals if v != "" and v is not None)
    if not nums:
        return None
    k = max(1, math.ceil(p / 100 * len(nums)))
    return round(nums[min(k, len(nums)) - 1], 2)


def provenance_of(model, imgsz: int) -> dict:
    """Which machine and settings produced these figures.

    Not decoration. The CUDA WSL venv and the CPU Windows venv give figures that
    are not comparable, and the laptop drifts thermally across consecutive runs;
    a summary that does not say which one produced it cannot be used in a report.
    """
    try:
        device = str(next(model.model.parameters()).device)
    except Exception:
        device = "unknown"
    return {
        "model": os.path.basename(str(getattr(model, "model_name", "") or "")) or "unknown",
        "imgsz": imgsz,
        "device": device,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "kpt_conf_thresh": KPT_CONF_THRESH,
    }
