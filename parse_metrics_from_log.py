"""
Extract AUC, F1, and Acc metrics at the BEST epoch from training log files.

Similar to parse_auc_from_log.py but extracts all three metrics:
  - AUC (Area Under Curve)
  - F1 (F1-score)
  - Acc (Accuracy)

Usage:
  python parse_metrics_from_log.py --folder logs_llm_fix_seed/rule_based
  python parse_metrics_from_log.py --folders logs_llm_fix_seed/* --output_dir results
  python parse_metrics_from_log.py --files file1.txt file2.txt file3.txt
"""

import re
import os
import argparse
import pandas as pd


# ---------------------------------------------------------------------------
# Core parsing helpers
# ---------------------------------------------------------------------------

def _extract_metrics_from_section(section: str) -> dict:
    """
    Extract {task_id: {auc, f1, acc}} from a test-evaluation section.
    
    Pattern: "auc: <float> | f1: <float> | acc: <float>"
    """
    metrics_dict = {}

    # Per-task match; lookahead stops before the next Task header
    pattern = re.compile(
        r"Task\s+([ST]\d+)"                                           # task id: S1, T3, etc.
        r"(?:(?!Task\s+[ST]\d+).)*?"                                  # any chars, not crossing next Task
        r"auc:\s+([\d.]+)\s*\|\s*f1:\s+([\d.]+)\s*\|\s*acc:\s+([\d.]+)",  # metrics
        re.DOTALL
    )

    for m in pattern.finditer(section):
        task = m.group(1)
        try:
            metrics_dict[task] = {
                'auc': float(m.group(2)),
                'f1':  float(m.group(3)),
                'acc': float(m.group(4))
            }
        except (ValueError, IndexError):
            pass

    return metrics_dict


def parse_metrics_best_epoch(path: str) -> dict:
    """
    Read a log file and return {task_id: {auc, f1, acc}} at the BEST epoch.

    Strategy A (normal case):
      Find the last "New best model saved!" marker, then extract metrics from
      the test section(s) that immediately follow it (before the next epoch).

    Strategy B (fallback):
      If no marker exists, extract metrics from ALL test sections (both seen and unseen).
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
            metrics_dict = _extract_metrics_from_section(test_section)
            if metrics_dict:
                return metrics_dict

    # ── Strategy B (fallback) ───────────────────────────────────────────────
    # Extract metrics from ALL test sections (both seen and unseen tasks)
    print(f"  [!] No best-epoch marker found — extracting from all test sections: {os.path.basename(path)}")
    metrics_dict = {}
    
    # Find all "test on" sections
    segments = re.split(r"(?=={10}test on)", content)
    for segment in segments:
        if re.match(r"={10}test on", segment):
            section_metrics = _extract_metrics_from_section(segment)
            metrics_dict.update(section_metrics)  # Merge all metrics found
    
    if metrics_dict:
        return metrics_dict

    print(f"  [!] No metrics values found in: {os.path.basename(path)}")
    return {}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def merge_metrics_logs(paths: list, names: list = None, metric: str = 'auc') -> pd.DataFrame:
    """
    Parse multiple log files → DataFrame.
      Rows    = experiment names
      Columns = task ids (S1…Sn, T1…Tn), sorted
      Values  = specified metric (auc/f1/acc) at best epoch
    """
    if names is None:
        names = [os.path.splitext(os.path.basename(p))[0] for p in paths]

    assert len(paths) == len(names), "paths and names must have the same length"
    assert metric in ['auc', 'f1', 'acc'], f"metric must be one of ['auc', 'f1', 'acc'], got {metric}"

    all_tasks: set = set()
    results: dict = {}

    for path, name in zip(paths, names):
        print(f"  Parsing  {name}")
        metrics_dict = parse_metrics_best_epoch(path)
        # Extract only the requested metric
        results[name] = {task: metrics[metric] for task, metrics in metrics_dict.items()}
        all_tasks.update(metrics_dict.keys())

    # Sort: S-tasks first, then T-tasks, each numerically
    def task_key(t):
        return (0 if t.startswith("S") else 1, int(t[1:]))

    all_tasks_sorted = sorted(all_tasks, key=task_key)

    data = {task: [results[name].get(task, None) for name in names]
            for task in all_tasks_sorted}

    df = pd.DataFrame(data, index=names)
    df.index.name = "experiment"
    return df


def merge_metrics_logs_consolidated(paths: list, names: list = None) -> pd.DataFrame:
    """
    Parse multiple log files → Single DataFrame with multi-level columns.
      Rows    = experiment names
      Columns = (Task, Metric) where Task is S1…Sn, T1…Tn and Metric is auc/f1/acc
      Values  = metric values at best epoch
      
    Creates a hierarchical column structure like:
      S1          S2          ...
      auc f1 acc  auc f1 acc  ...
    """
    if names is None:
        names = [os.path.splitext(os.path.basename(p))[0] for p in paths]

    assert len(paths) == len(names), "paths and names must have the same length"

    all_tasks: set = set()
    results: dict = {}

    for path, name in zip(paths, names):
        print(f"  Parsing  {name}")
        metrics_dict = parse_metrics_best_epoch(path)
        results[name] = metrics_dict
        all_tasks.update(metrics_dict.keys())

    # Sort: S-tasks first, then T-tasks, each numerically
    def task_key(t):
        return (0 if t.startswith("S") else 1, int(t[1:]))

    all_tasks_sorted = sorted(all_tasks, key=task_key)

    # Build data with multi-level columns: (Task, Metric)
    data_dict = {}
    for task in all_tasks_sorted:
        for metric in ['auc', 'f1', 'acc']:
            col_key = (task, metric)
            data_dict[col_key] = [results[name].get(task, {}).get(metric, None) for name in names]

    # Create DataFrame with MultiIndex columns
    df = pd.DataFrame(data_dict, index=names)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=['task', 'metric'])
    df.index.name = "experiment"
    
    return df



    """
    Parse multiple log files → dict of DataFrames (one per metric).
      Returns: {'auc': df_auc, 'f1': df_f1, 'acc': df_acc}
    """
    if names is None:
        names = [os.path.splitext(os.path.basename(p))[0] for p in paths]

    assert len(paths) == len(names), "paths and names must have the same length"

    all_tasks: set = set()
    results: dict = {'auc': {}, 'f1': {}, 'acc': {}}

    for path, name in zip(paths, names):
        print(f"  Parsing  {name}")
        metrics_dict = parse_metrics_best_epoch(path)
        
        for task, metrics in metrics_dict.items():
            if name not in results['auc']:
                results['auc'][name] = {}
                results['f1'][name] = {}
                results['acc'][name] = {}
            results['auc'][name][task] = metrics['auc']
            results['f1'][name][task] = metrics['f1']
            results['acc'][name][task] = metrics['acc']
        
        all_tasks.update(metrics_dict.keys())

    # Sort: S-tasks first, then T-tasks, each numerically
    def task_key(t):
        return (0 if t.startswith("S") else 1, int(t[1:]))

    all_tasks_sorted = sorted(all_tasks, key=task_key)

    dfs = {}
    for metric_name in ['auc', 'f1', 'acc']:
        data = {task: [results[metric_name].get(name, {}).get(task, None) for name in names]
                for task in all_tasks_sorted}
        dfs[metric_name] = pd.DataFrame(data, index=names)
        dfs[metric_name].index.name = "experiment"

    return dfs


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

def process_folder(folder: str, pattern: str, output_dir: str = None, consolidate: bool = False) -> dict:
    """
    Process all .txt files in a folder and save results.
    
    If consolidate=True: Saves one consolidated CSV (multi-level columns)
    Otherwise: Saves three separate CSVs (one per metric)
    """
    paths = find_log_files(folder, pattern)
    if not paths:
        print(f"  No matching .txt files in {folder}")
        return None

    print(f"\n{'='*60}")
    print(f"Folder : {folder}  |  pattern='{pattern}'  |  {len(paths)} file(s)")
    print(f"{'='*60}")

    out_dir = output_dir or folder
    os.makedirs(out_dir, exist_ok=True)
    tag = f"_{pattern}" if pattern else ""

    if consolidate:
        df = merge_metrics_logs_consolidated(paths)
        out_csv = os.path.join(out_dir, f"consolidated_metrics_best_epoch{tag}.csv")
        df.to_csv(out_csv)
        print(f"\n✅ Saved → {out_csv}")
        print(df.to_string())
        return {'consolidated': df}
    else:
        dfs = merge_metrics_logs_multi(paths)
        for metric_name, df in dfs.items():
            out_csv = os.path.join(out_dir, f"merged_{metric_name}_best_epoch{tag}.csv")
            df.to_csv(out_csv)
            print(f"\n✅ Saved → {out_csv}")
            print(df.to_string())
        return dfs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract AUC/F1/Acc metrics at best epoch from training log files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    inp = parser.add_mutually_exclusive_group()
    inp.add_argument("--files",   nargs="+", metavar="FILE",
                     help="Explicit list of log file paths.")
    inp.add_argument("--folder",  metavar="DIR",
                     help="Single folder to search for .txt log files.")
    inp.add_argument("--folders", nargs="+", metavar="DIR",
                     help="Multiple folders (one set of CSVs per folder).")

    parser.add_argument("--names",      nargs="+", metavar="NAME",
                        help="Custom row names (only with --files).")
    parser.add_argument("--pattern",    default="",
                        help="Filename filter when using --folder/--folders.")
    parser.add_argument("--output_dir", default=None, metavar="DIR",
                        help="Output directory for CSV(s). Default: source folder.")
    parser.add_argument("--metric", choices=['auc', 'f1', 'acc'], default='auc',
                        help="Single metric to extract (default: auc). Use --all-metrics for all three.")
    parser.add_argument("--all-metrics", action="store_true",
                        help="Extract all three metrics (auc, f1, acc) → three CSVs per folder.")
    parser.add_argument("--consolidated", action="store_true",
                        help="Create single consolidated CSV with multi-level columns (Task/Metric).")

    args = parser.parse_args()

    # ── Explicit file list ──────────────────────────────────────────────────
    if args.files:
        names = args.names or [os.path.splitext(os.path.basename(f))[0]
                                for f in args.files]
        print(f"\nProcessing {len(args.files)} file(s)…")

        out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.files[0]))
        os.makedirs(out_dir, exist_ok=True)

        if args.consolidated:
            df = merge_metrics_logs_consolidated(args.files, names)
            out_csv = os.path.join(out_dir, "consolidated_metrics_best_epoch.csv")
            df.to_csv(out_csv)
            print(f"\n✅ Saved → {out_csv}")
            print(df.to_string())
        elif args.all_metrics:
            dfs = merge_metrics_logs_multi(args.files, names)
            for metric_name, df in dfs.items():
                out_csv = os.path.join(out_dir, f"merged_{metric_name}_best_epoch.csv")
                df.to_csv(out_csv)
                print(f"\n✅ Saved → {out_csv}")
                print(df.to_string())
        else:
            df = merge_metrics_logs(args.files, names, metric=args.metric)
            out_csv = os.path.join(out_dir, f"merged_{args.metric}_best_epoch.csv")
            df.to_csv(out_csv)
            print(f"\n✅ Saved → {out_csv}")
            print(df.to_string())

    # ── Single folder ───────────────────────────────────────────────────────
    elif args.folder:
        process_folder(args.folder, args.pattern, args.output_dir, consolidate=args.consolidated)

    # ── Multiple folders ────────────────────────────────────────────────────
    elif args.folders:
        out_dir = args.output_dir or "."
        os.makedirs(out_dir, exist_ok=True)

        if args.consolidated:
            # Consolidate results from all folders into one CSV
            all_dfs_consolidated = []
            for folder in args.folders:
                result = process_folder(folder, args.pattern, args.output_dir, consolidate=True)
                if result and 'consolidated' in result:
                    df = result['consolidated']
                    df.insert(0, 'folder', os.path.basename(folder.rstrip("/\\")))
                    all_dfs_consolidated.append(df)
            
            if all_dfs_consolidated:
                combined = pd.concat(all_dfs_consolidated)
                combined_csv = os.path.join(out_dir, "consolidated_metrics_all_folders.csv")
                combined.to_csv(combined_csv)
                print(f"\n✅ Combined Consolidated CSV → {combined_csv}")
                print(combined.to_string())
        else:
            # Keep separate metric CSVs per folder
            all_dfs_list = []
            for folder in args.folders:
                dfs = process_folder(folder, args.pattern, args.output_dir, consolidate=False)
                if dfs is not None:
                    for metric_name, df in dfs.items():
                        df.insert(0, "folder", os.path.basename(folder.rstrip("/\\")))
                        all_dfs_list.append((metric_name, df))

            if all_dfs_list:
                # Group by metric and concatenate
                metric_dict = {}
                for metric_name, df in all_dfs_list:
                    if metric_name not in metric_dict:
                        metric_dict[metric_name] = []
                    metric_dict[metric_name].append(df)

                for metric_name, df_list in metric_dict.items():
                    combined = pd.concat(df_list)
                    combined_csv = os.path.join(out_dir, f"merged_{metric_name}_all_folders.csv")
                    combined.to_csv(combined_csv)
                    print(f"\n✅ Combined CSV → {combined_csv}")
                    print(combined.to_string())

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
