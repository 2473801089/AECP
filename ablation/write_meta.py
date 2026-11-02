#!/usr/bin/env python3
"""写单次运行的元数据 meta.json（消融规范 §4.1）。

只从运行目录里已存在的文件读取，不臆造字段；缺失项写 null 并记入 missing 列表。
"""
import argparse
import json
import os
import re
import sys


def read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return None


def stats_summary(path):
    """探索时间、终局覆盖率、航程取自 exploration_stats.csv 的最后一行。"""
    text = read_text(path)
    if not text:
        return None
    rows = [r for r in text.strip().splitlines() if r and not r.startswith("elapsed")]
    if not rows:
        return None
    last = rows[-1].split(",")
    try:
        return {
            "exploration_time_s": float(last[0]),
            "coverage_ratio": float(last[1]),
            "explored_volume_m3": float(last[2]),
            "traj_length_m": float(last[3]),
            "mean_speed_mps": float(last[4]),
        }
    except (IndexError, ValueError):
        return None


def known_voxels(path):
    """已知体素数：统计 1 Hz 采样里的覆盖率峰值，体素总数取自 launch 参数。"""
    text = read_text(path)
    if not text:
        return None
    rows = [r for r in text.strip().splitlines() if r and not r.startswith("elapsed")]
    if not rows:
        return None
    try:
        return max(float(r.split(",")[1]) for r in rows)
    except (IndexError, ValueError):
        return None


def git_info(path):
    text = read_text(path)
    if not text:
        return {}
    out = {}
    for line in text.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--termination", required=True)
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--begin", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--extra", default=None, help="额外键值，JSON 字符串")
    args = ap.parse_args()

    d = args.dir
    missing = []

    cfg_hash = read_text(os.path.join(d, "params.sha256"))
    cfg_hash = cfg_hash.strip() if cfg_hash else None
    if cfg_hash is None:
        missing.append("params.sha256")

    stats = stats_summary(os.path.join(d, "exploration_stats.csv"))
    if stats is None:
        missing.append("exploration_stats.csv")

    cov_peak = known_voxels(os.path.join(d, "exploration_stats.csv"))
    if cov_peak is None:
        missing.append("coverage_peak")

    git = git_info(os.path.join(d, "git_head.txt"))

    meta = {
        "config": args.config,
        "scene": args.scene,
        "run": int(args.run),
        "termination": args.termination,
        "duration_s": args.duration,
        "begin": args.begin,
        "end": args.end,
        "config_hash": cfg_hash,
        "git_commit": git.get("commit"),
        "git_branch": git.get("branch"),
        "stats": stats,
        "coverage_peak": cov_peak,
        "missing": missing,
    }
    if args.extra:
        try:
            meta["extra"] = json.loads(args.extra)
        except json.JSONDecodeError:
            meta["extra"] = args.extra

    with open(os.path.join(d, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
