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
head itself lives in `fallcore.calibrate`, and the windows it is fitted on are
chosen by `fallcore.head_fitting`, which the weekly Refit shares.

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
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fallcore import calibrate, config as cfg, decide, picam, train as T
from fallcore.fall_decision import DEPLOYED_CHECKPOINT, DEPLOYED_POSE
from fallcore.head_fitting import (NobodyNegatives, embed_windows,
                                   fold_fitting_set, segment_labels)


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
    # fallcore.head_fitting reports what it adds through logging.
    logging.basicConfig(level=logging.INFO, format="%(message)s")

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
            tcfg.seq_len, lambda w: embed_windows(model, mcfg, w, device="cpu"),
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
