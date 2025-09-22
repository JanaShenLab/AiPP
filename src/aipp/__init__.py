# aipp/__init__.py

__version__ = "0.1.0"

from .dataload import load_dataset, _banner
from .parse   import parse_arguments, setup_thresholds, load_data
from .distill import (
    process_pdb_mappings,
    filter_conflicted_records,
    group_unambiguous_records,
    apply_filters,
    write_distilled_dataset,
    write_report as write_distilled_dataset_report,
)
from .resolve import resolve_unambiguous, resolve_ambiguous_records
from .report  import (
    compute_stats,
    generate_statistics,
    generate_ambiguity_report,
    write_report as write_statistics_report,
)

__all__ = [
    "load_dataset",
    # parse.py
    "parse_arguments", "setup_thresholds", "load_data",
    # distill.py
    "process_pdb_mappings", "filter_conflicted_records",
    "group_unambiguous_records", "apply_filters",
    "write_distilled_dataset", "write_distilled_dataset_report",
    # resolve.py
    "resolve_unambiguous", "resolve_ambiguous_records",
    # report.py
    "compute_stats", "generate_statistics",
    "generate_ambiguity_report", "write_statistics_report",
]

