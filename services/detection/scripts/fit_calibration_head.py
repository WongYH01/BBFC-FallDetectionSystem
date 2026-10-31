"""Fit (and price) the deployment head: a logistic probe on frozen embeddings.

The transformer's pooled 256-d embedding already separates falls from ADLs on
the deployment camera -- what fails there is the *pretrained head*, which was
fitted on other corpora. This script fits a new head on the target room's own
clips and reports it honestly:

* `--protocol loso` (default): leave-one-subject-out. Five folds, each fitted on
  four subjects and scored on the fifth, that subject's falls included. This is
  the only number that predicts the next person; `picam` derives the subject
  ids from the capture timestamps.
* `--protocol fit-all`: one head on every clip, saved for deployment. Its picam
  score is in-sample and is printed as such -- a description of the fitted set,
  not a prediction.

The frozen backbone is never modified. The artefact is a ~1 KB head
(mean/scale/coef/intercept) plus the checkpoint it belongs to; scoring a window
costs one extra dot product after the forward pass the demo already does. The
head itself lives in `fallcore.calibrate`.

A head is also a function of the pose backbone whose keypoints it was fitted
on: `--scale` names that keypoint cache (`data/keypoints/<scale>/`), the head
records it (`ProbeHead.pose`), and the detector refuses a head whose pose
backbone is not the one it runs (`fallcore.fall_decision.DEPLOYED_POSE`). The
defaults are the deployed recipe: the deployed checkpoint, the deployed pose
backbone's keypoints (yolo26m INT8 at 384x640), segment labels.

    .venv-train/Scripts/python.exe scripts/fit_calibration_head.py
    # the shipped head
    .venv-train/Scripts/python.exe scripts/fit_calibration_head.py \
        --protocol fit-all --empty-negatives 10 \
        --out runs/checkpoints/probe_augnone_ms_coords_hn_s99_yolo26m-int8.npz
    # the retired yolo26n head
    ... --protocol fit-all --empty-negatives 10 --scale yolo26n --labels clip

Results are appended to `runs/metrics/calibration_head.csv`.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fallcore import calibrate, config as cfg, decide, picam, train as T
from fallcore.fall_decision import DEPLOYED_CHECKPOINT, DEPLOYED_POSE


#: Rows in half a second, at the row rate every window is cut at.
HALF_S = round(0.5 * calibrate.ROW_HZ)


def fall_row(scale: str, video_id: str) -> int:
    """Start row of the fastest half second of hip descent in a whole clip.

    The fall itself, as opposed to the standing before it or the lying after
    it. Undetected rows are bridged, as `calibrate.window_kinematics` does.
    """
    from fallcore.extract import load_keypoints

    kp = load_keypoints(scale, video_id)[::cfg.PREPROCESS.frame_stride]
    hips = kp[:, [cfg.LEFT_HIP, cfg.RIGHT_HIP], :]
    ok = (hips[:, :, 2] > 0.3).all(axis=1)
    idx = np.flatnonzero(ok)
    y = np.interp(np.arange(len(kp)), idx, hips[idx, :, 0].mean(axis=1))
    return int(np.argmax(y[HALF_S:] - y[:-HALF_S]))


def segment_labels(clips, scale: str, seq_len: int):
    """`(keep, label)` per window when a fall clip's label follows its fall.

    A fall clip is a few seconds of standing, about one second of falling, then
    lying still. Clip labels call every one of its windows Fall; here a window
    that ends before the fall is No-Fall (the same picture as walking), one that
    holds the whole fastest half second or starts after it is Fall, and one
    that cuts the fall in part is dropped. ADL windows stay No-Fall. On the
    picam sessions this is what lets a yolo26m head keep the falls a clip-label
    head misses in an unseen room (research/scripts/picam3_segments.py).
    """
    fall = clips.label.values == 1
    rows = {v: fall_row(scale, v) for v in clips.video_id[fall].unique()}
    e = clips.video_id.map(rows)
    end = clips.start + seq_len - 1
    pre = (end < e).values
    holds = ((clips.start <= e) & (e + HALF_S <= end)).values
    post = (clips.start > e + HALF_S).values
    label = np.where(fall & pre, 0, clips.label.values)
    keep = ~fall | pre | holds | post
    return keep, label


def nobody_windows(model, mcfg, tcfg, clips, meta, copies: int = 10,
                   scale: str = "yolo26n", load=None):
    """Synthetic "nobody visible" windows, to be fitted as No-Fall.

    The head otherwise inherits -- and amplifies -- the room's dropout artifact:
    in picam's own fall clips the pose model loses the subject for about half of
    a `fall forward` window against ~1% of an ADL window, so "sixty frames of
    zeros" is a *fall* cue in the fitting data. A live camera produces that input
    every time the room empties, and the head scored it 0.9999, i.e. a
    self-sustaining false alarm once the rolling mean latches.

    Three flavours, all unambiguously not a fall:

    * all-zero windows (nobody detected at all);
    * sparse-detection windows (two joints found, near the origin -- what a
      backlit or half-occluded frame leaves behind);
    * real ADL windows with the tail 25/50/75% zeroed, which is what a person
      leaving frame or walking behind furniture actually produces. Built from
      `clips`, so they carry this room's own pose statistics.

    `load(scale, video_id)` reads a clip's keypoints (default: the cache on
    disk). Fitted into the training folds only; the held-out subject never sees
    them. Measured effect on the LOSO score of the multi-scale backbone: AUC
    0.9923 -> 0.9909, rolling b4/b8 F1 unchanged at 0.967 (1 FP, 1 FN), while
    P(fall | all-zero window) falls 0.9999 -> 0.0077.
    """
    from fallcore.data import prepare_sequence, take_window

    if load is None:
        from fallcore.extract import load_keypoints as load

    T_, K = tcfg.seq_len, cfg.N_KEYPOINTS
    rng = np.random.default_rng(0)
    out = [np.zeros((T_, K, 3), dtype=np.float32)]

    sparse = np.zeros((T_, K, 3), dtype=np.float32)
    lo = 1 + int(0.8 * T_)
    sparse[lo:, [0, 5], 0] = rng.uniform(0.0, 0.08, (T_ - lo, 2))
    sparse[lo:, [0, 5], 1] = rng.uniform(0.0, 0.08, (T_ - lo, 2))
    sparse[lo:, [0, 5], 2] = 0.4
    out.append(sparse)

    adl = clips[clips.label == 0].video_id.drop_duplicates().head(6).tolist()
    for vid in adl:
        seq = prepare_sequence(load(scale, vid), None,
                               cfg.PREPROCESS.frame_stride)
        ref = take_window(seq, T_, start=0)
        for frac in (0.25, 0.50, 0.75):
            cut = int(T_ * (1 - frac))
            w = ref.copy()
            w[cut:] = 0.0
            out.append(w.astype(np.float32))

    windows = []
    for w in out:
        windows.extend([w] * copies)
    return np.stack(windows)


def embed_windows(model, mcfg, tcfg, windows, device: str = "cpu"):
    """Pooled embeddings for raw (n, T, 17, 3) windows, backbone frozen."""
    import torch

    from fallcore.data import build_features

    x = np.stack([build_features(w, mcfg.features) for w in windows])
    model.to(device).eval()
    with torch.no_grad():
        h = model.encoder(model.embed(torch.from_numpy(x).float().to(device)))
        return model._pool(h).cpu().numpy()


class NobodyNegatives:
    """A fold's nobody-visible windows, embedded, from that fold's own clips.

    Calling it with the clips a fold trains on returns `(embeddings,
    kinematics)` for the windows `nobody_windows` cuts from them (`kinematics`
    is None unless asked for). `embed(windows)` is the frozen backbone's pooled
    embedding. The windows depend only on which ADL clips they are cut from, so
    folds that share those clips share one build.
    """

    def __init__(self, tcfg, embed, copies: int = 10, scale: str = "yolo26n",
                 kinematics: bool = False, load=None):
        self.tcfg, self.embed, self.copies = tcfg, embed, copies
        self.scale, self.kinematics, self.load = scale, kinematics, load
        self._built = {}

    def __call__(self, clips):
        key = tuple(clips[clips.label == 0].video_id.drop_duplicates().head(6))
        if key not in self._built:
            windows = nobody_windows(None, None, self.tcfg, clips, None,
                                     copies=self.copies, scale=self.scale,
                                     load=self.load)
            # A window nobody is visible in has no measurable descent; zeros
            # are the honest value and are what `window_kinematics` returns.
            kin = (np.stack([calibrate.window_kinematics(w) for w in windows])
                   if self.kinematics else None)
            self._built[key] = (self.embed(windows), kin)
            print(f"nobody-visible windows: {len(windows)} labelled No-Fall "
                  f"({len(windows) // self.copies} flavours x {self.copies} "
                  f"copies, from {len(key)} ADL clips)", flush=True)
        return self._built[key]


def fold_fitting_set(train, clips, emb, y, keep, ext=None, negatives=None):
    """`(embeddings, labels, extras)` a head fitted on the `train` rows sees.

    `train` masks the rows of `clips` (one per window) the fold may learn from;
    `keep` drops the windows the label scheme leaves out. `negatives`, a
    `NobodyNegatives`, adds the synthetic No-Fall windows, cut from the
    training rows' clips only: a left-out subject's own ADL clips never reach
    the head that scores it.
    """
    real = train & keep
    e, lab = emb[real], y[real]
    x = ext[real] if ext is not None else None
    if negatives is not None:
        syn_emb, syn_ext = negatives(clips[train])
        e = np.concatenate([e, syn_emb])
        lab = np.concatenate([lab, np.zeros(len(syn_emb), dtype=lab.dtype)])
        if x is not None:
            x = np.concatenate([x, syn_ext])
    return e, lab, x


def report(name, prob, clips, clip_labels) -> list[dict]:
    """Window AUC plus clip F1 under the paper's peak rule and rolling rules."""
    rows = [{"head": name, "metric": "window_AUC",
             "value": float(roc_auc_score(clips.label.values, prob))}]
    w = clips.copy()
    w["p"] = prob
    for buf in (1, 4, 8):
        alarm = decide.rolling_alarm(w, score="p", buffer=buf, threshold=0.5)
        m = decide.rule_metrics(
            clip_labels.values, alarm.loc[clip_labels.index, "fired"].values)
        rows.append({"head": name, "metric": f"picam_rolling_b{buf}_F1",
                     "value": m["f1"], "FP": m["fp"], "FN": m["fn"],
                     "precision": m["precision"], "recall": m["recall"]})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(cfg.CKPT_DIR / DEPLOYED_CHECKPOINT))
    ap.add_argument("--protocol", choices=("loso", "fit-all", "curve"),
                    default="loso")
    ap.add_argument("--out", default=None,
                    help="npz path for the fitted head (fit-all only; default "
                         "probe_<ckpt>_<scale>.npz, which never overwrites a "
                         "shipped head)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite the metrics rows instead of appending")
    ap.add_argument("--empty-negatives", type=int, default=0, metavar="COPIES",
                    help="add synthetic nobody-visible windows (this many copies "
                         "of each flavour) to every training fold, labelled "
                         "No-Fall. 0 = off is what the shipped heads were fitted "
                         "with before this flag existed; 10 costs ~0.001 AUC on "
                         "the held-out subjects and takes P(fall | empty window) "
                         "from 0.9999 to 0.008")
    ap.add_argument("--extra-kinematics", action="store_true",
                    help="append the three scale-free descent measures "
                         "(fallcore.calibrate.KINEMATIC_NAMES) to every "
                         "embedding before fitting. Measured over three seeds "
                         "this is NOT an improvement -- a wash in the fitted "
                         "room and slightly worse in an unseen one; the "
                         "descent gate is what carries. Kept because it is "
                         "measured and reproducible")
    ap.add_argument("--scale", default=DEPLOYED_POSE,
                    help="keypoint cache the windows are read from "
                         "(data/keypoints/<scale>/): the pose backbone this "
                         "head will be paired with, recorded in the head")
    ap.add_argument("--labels", choices=("clip", "segment"), default="segment",
                    help="clip: every window of a fall clip is Fall (the yolo26n "
                         "head). segment: a fall clip's windows before the fall "
                         "are No-Fall and ones that cut it are dropped (the "
                         "yolo26m head; see segment_labels)")
    args = ap.parse_args()

    man = picam.scan()
    clips = picam.build_clips(man, scale=args.scale, seq_len=cfg.FINAL_TRAIN.seq_len,
                              stride=cfg.EVAL.sliding_stride)
    meta = man.set_index("video_id")
    subj = clips.video_id.map(meta.subject).values
    clip_labels = meta["label"]
    keep, y = np.ones(len(clips), dtype=bool), clips.label.values
    if args.labels == "segment":
        keep, y = segment_labels(clips, args.scale, cfg.FINAL_TRAIN.seq_len)

    model, mcfg, tcfg, *_ = T.load_checkpoint(args.ckpt, device="cpu")
    emb = calibrate.embeddings(model, mcfg, tcfg, clips, scale=args.scale)
    aspects = {v.video_id: float(v.width) / float(v.height)
               for _, v in man.iterrows()}
    ext = (calibrate.window_extras(tcfg, clips, scale=args.scale,
                                   aspects=aspects)
           if args.extra_kinematics else None)
    extra_names = calibrate.KINEMATIC_NAMES if args.extra_kinematics else ()
    print(f"{Path(args.ckpt).stem} on {args.scale}: {len(clips)} windows "
          f"({keep.sum()} fitted, {args.labels} labels), "
          f"{clips.video_id.nunique()} clips, embedding {emb.shape[1]}", flush=True)

    negatives = None
    if args.empty_negatives > 0:
        negatives = NobodyNegatives(
            tcfg, lambda w: embed_windows(model, mcfg, tcfg, w, device="cpu"),
            copies=args.empty_negatives, scale=args.scale,
            kinematics=args.extra_kinematics)

    def fit_fold(train_mask):
        """Fit a head on the masked real windows plus the synthetic negatives."""
        e, lab, x = fold_fitting_set(train_mask, clips, emb, y, keep, ext,
                                     negatives)
        return calibrate.fit(e, lab, backbone=args.ckpt, features=mcfg.features,
                             seq_len=tcfg.seq_len, extras=x,
                             extra_names=extra_names, pose=args.scale)

    rows = []
    if args.protocol == "curve":
        # How much commissioning data is enough: fit the head on k of the five
        # subjects, score the rest, over every choice of k. k = 4 is the LOSO
        # setup; k = 1 is a single staged session, which is what a field install
        # can realistically arrange.
        from itertools import combinations

        subs = sorted(set(subj))
        for k in (1, 2, 3, 4):
            aucs, f1s = [], []
            for combo in combinations(subs, k):
                tr = np.isin(subj, list(combo))
                te = ~tr
                head = fit_fold(tr)
                prob = np.zeros(len(clips))
                prob[te] = head.proba(emb[te], ext[te] if ext is not None else None)
                aucs.append(roc_auc_score(clips.label.values[te], prob[te]))
                w = clips[te].copy()
                w["p"] = prob[te]
                ids = clip_labels.index[clip_labels.index.isin(set(w.video_id))]
                alarm = decide.rolling_alarm(w, score="p", buffer=8, threshold=0.5)
                m = decide.rule_metrics(clip_labels.loc[ids].values,
                                        alarm.loc[ids, "fired"].values)
                f1s.append(m["f1"])
            print(f"  fit on {k} subject(s), {len(aucs)} folds: "
                  f"AUC {np.mean(aucs):.3f} +/- {np.std(aucs):.3f} | "
                  f"rolling-b8 F1 {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}",
                  flush=True)
            rows.append({"head": f"curve_k{k}", "metric": "heldout_AUC",
                         "value": float(np.mean(aucs)),
                         "std": float(np.std(aucs)), "folds": len(aucs)})
            rows.append({"head": f"curve_k{k}", "metric": "heldout_rolling_b8_F1",
                         "value": float(np.mean(f1s)),
                         "std": float(np.std(f1s)), "folds": len(aucs)})
    elif args.protocol == "loso":
        prob = np.zeros(len(clips))
        for s in sorted(set(subj)):
            tr, te = subj != s, subj == s
            head = fit_fold(tr)
            prob[te] = head.proba(emb[te], ext[te] if ext is not None else None)
            ids = meta.index[meta.subject == s]
            fired = (pd.DataFrame({"video_id": clips.video_id[te], "p": prob[te]})
                     .groupby("video_id")["p"].mean() >= 0.5)
            m = decide.rule_metrics(meta.loc[ids, "label"].values,
                                    fired.reindex(ids).fillna(False).values)
            print(f"  subj {s}: clip-mean F1 {m['f1']:.3f} "
                  f"(FP {m['fp']}, FN {m['fn']})", flush=True)
        name = f"loso_{Path(args.ckpt).stem}"
        rows += report(name, prob, clips, clip_labels)
        np.save(cfg.METRICS_DIR / f"{name}_window_probs.npy", prob)
    else:
        head = fit_fold(np.ones(len(clips), dtype=bool))
        out = Path(args.out or (cfg.CKPT_DIR /
                                f"probe_{Path(args.ckpt).stem}_{args.scale}.npz"))
        head.save(out)
        print(f"wrote {out} (in-sample numbers follow -- descriptive only)")
        rows += report(f"fitall_{Path(args.ckpt).stem}", head.proba(emb, ext),
                       clips, clip_labels)

    df = pd.DataFrame(rows)
    csv = cfg.METRICS_DIR / "calibration_head.csv"
    if csv.exists() and not args.force:
        df = pd.concat([pd.read_csv(csv), df], ignore_index=True)
    df.to_csv(csv, index=False)
    print(df.to_string(index=False))
    print(f"\nwrote {csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
