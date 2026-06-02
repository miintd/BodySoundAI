import re
import os
import argparse
import pandas as pd


# ---------------------------------------------------------------------------
# Core parsing helpers
# ---------------------------------------------------------------------------

def _extract_auc_from_section(section: str) -> dict:
    """
    Extract {task_id: auc_float} from a test-evaluation section.
    """
    auc_dict = {}

    # Non-greedy per-task match; lookahead stops before the next Task header
    pattern = re.compile(
        r"Task\s+([ST]\d+)"                      # task id: S1, T3, etc.
        r"(?:(?!Task\s+[ST]\d+).)*?"             # any chars, not crossing next Task
        r"auc\s+(?:tensor\()?([\d.]+)\)?",        # auc value
        re.DOTALL
    )

    for m in pattern.finditer(section):
        task = m.group(1)
        try:
            auc_dict[task] = float(m.group(2))
        except ValueError:
            pass

    return auc_dict


def parse_auc_best_epoch(path: str) -> dict:
    """
    Read a log file and return {task_id: auc_float} at the BEST epoch.

    Strategy A (normal case):
      Find the last "New best model saved!" marker, then extract AUC from
      the test section(s) that immediately follow it (before the next epoch).

    Strategy B (fallback):
      If no marker exists, use the last test section in the file.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # ── Strategy A ──────────────────────────────────────────────────────────
    best_matches = list(re.finditer(
        r"New best model saved!\s*\(Loss:\s+([\d.e\-]+)\)", content
    ))

    if best_matches:
        last_best = best_matches[-1]
        after_best = content[last_best.end():]

        test_m = re.search(r"={10}test on", after_best)
        if test_m:
            # Boundary: next line that starts a new epoch header
            # Use \nEpoch: to avoid matching "epoch:" inside tqdm/iter lines
            next_epoch_m = re.search(r"\nEpoch:\s+\d+", after_best[test_m.end():])
            if next_epoch_m:
                test_end = test_m.end() + next_epoch_m.start()
            else:
                test_end = len(after_best)

            test_section = after_best[test_m.start():test_end]
            auc_dict = _extract_auc_from_section(test_section)
            if auc_dict:
                return auc_dict

    # ── Strategy B (fallback) ───────────────────────────────────────────────
    print(f"  [!] No best-epoch marker found — using last test section: {os.path.basename(path)}")
    segments = re.split(r"(?=={10}test on)", content)
    for segment in reversed(segments):
        if re.match(r"={10}test on", segment):
            auc_dict = _extract_auc_from_section(segment)
            if auc_dict:
                return auc_dict

    print(f"  [!] No AUC values found in: {os.path.basename(path)}")
    return {}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def merge_auc_logs(paths: list, names: list = None) -> pd.DataFrame:
    """
    Parse multiple log files → DataFrame.
      Rows    = experiment names
      Columns = task ids (S1…Sn, T1…Tn), sorted
      Values  = AUC at best epoch
    """
    if names is None:
        names = [os.path.splitext(os.path.basename(p))[0] for p in paths]

    assert len(paths) == len(names), "paths and names must have the same length"

    all_tasks: set = set()
    results: dict = {}

    for path, name in zip(paths, names):
        print(f"  Parsing  {name}")
        auc_dict = parse_auc_best_epoch(path)
        results[name] = auc_dict
        all_tasks.update(auc_dict.keys())

    # Sort: S-tasks first, then T-tasks, each numerically
    def task_key(t):
        return (0 if t.startswith("S") else 1, int(t[1:]))

    all_tasks_sorted = sorted(all_tasks, key=task_key)

    data = {task: [results[name].get(task, None) for name in names]
            for task in all_tasks_sorted}

    df = pd.DataFrame(data, index=names)
    df.index.name = "experiment"
    return df


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def find_log_files(folder: str, pattern: str = "") -> list:
    """Return sorted list of .txt files in `folder` whose name contains `pattern`."""
    if not os.path.isdir(folder):
        print(f"[!] Folder not found: {folder}")
        return []
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".txt") and pattern in f
    )


# ---------------------------------------------------------------------------
# Per-folder processing
# ---------------------------------------------------------------------------

def process_folder(folder: str, pattern: str, output_dir: str = None) -> pd.DataFrame:
    paths = find_log_files(folder, pattern)
    if not paths:
        print(f"  No matching .txt files in {folder}")
        return None

    print(f"\n{'='*60}")
    print(f"Folder : {folder}  |  pattern='{pattern}'  |  {len(paths)} file(s)")
    print(f"{'='*60}")

    df = merge_auc_logs(paths)

    out_dir = output_dir or folder
    os.makedirs(out_dir, exist_ok=True)
    tag = f"_{pattern}" if pattern else ""
    out_csv = os.path.join(out_dir, f"merged_auc_best_epoch{tag}.csv")
    df.to_csv(out_csv)

    print(f"\n✅ Saved → {out_csv}")
    print(df.to_string())
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract AUC at best epoch from training log files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    inp = parser.add_mutually_exclusive_group()
    inp.add_argument("--files",   nargs="+", metavar="FILE",
                     help="Explicit list of log file paths.")
    inp.add_argument("--folder",  metavar="DIR",
                     help="Single folder to search for .txt log files.")
    inp.add_argument("--folders", nargs="+", metavar="DIR",
                     help="Multiple folders (one CSV per folder).")

    parser.add_argument("--names",      nargs="+", metavar="NAME",
                        help="Custom row names (only with --files).")
    parser.add_argument("--pattern",    default="",
                        help="Filename filter when using --folder/--folders.")
    parser.add_argument("--output_dir", default=None, metavar="DIR",
                        help="Output directory for CSV(s). Default: source folder.")

    args = parser.parse_args()

    # ── Explicit file list ──────────────────────────────────────────────────
    if args.files:
        names = args.names or [os.path.splitext(os.path.basename(f))[0]
                                for f in args.files]
        print(f"\nProcessing {len(args.files)} file(s)…")
        df = merge_auc_logs(args.files, names)

        out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.files[0]))
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, "merged_auc_best_epoch.csv")
        df.to_csv(out_csv)
        print(f"\n✅ Saved → {out_csv}")
        print(df.to_string())

    # ── Single folder ───────────────────────────────────────────────────────
    elif args.folder:
        process_folder(args.folder, args.pattern, args.output_dir)

    # ── Multiple folders ────────────────────────────────────────────────────
    elif args.folders:
        all_dfs = []
        for folder in args.folders:
            df = process_folder(folder, args.pattern, args.output_dir)
            if df is not None:
                df.insert(0, "folder", os.path.basename(folder.rstrip("/\\")))
                all_dfs.append(df)

        if all_dfs:
            combined = pd.concat(all_dfs)
            out_dir = args.output_dir or "."
            os.makedirs(out_dir, exist_ok=True)
            combined_csv = os.path.join(out_dir, "merged_auc_all_folders.csv")
            combined.to_csv(combined_csv)
            print(f"\n✅ Combined CSV → {combined_csv}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()