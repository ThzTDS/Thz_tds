TEST ALIGNMENT SHIFTS — quick guide
===================================

What it does
------------
Checks the alignment of the traces in selected folders of each sample
(air = reference, point = sample measurements). For each folder it saves:
raw vs aligned traces, averages, Gaussian fits, t0 per trace, and
correction-factor plots. For each sample it also saves a summary CSV and
a summary plot (shift std, p2p, RMS per folder).
No plots open on screen. Everything is saved to disk.


How to run
----------
    cd /mnt
    python3 examples/diagnostics/test_alignment_shifts.py

    # or with another config:
    python3 examples/diagnostics/test_alignment_shifts.py config/other.yaml

Default config: config/all_samples_fp_batch.yaml


What to set in the YAML
-----------------------
1) Which samples  ->  "samples:" list (uncomment the ones you want)

2) Which folders  ->  bottom of the file:

    alignment_check:
      output: /mnt/Data/results
      folders:
        air:   [1, 7]        # air1, air7
        point: [1, 7]        # point1, point7

   Examples:
     air: [1, 4, 7]         -> only air1, air4, air7
     point: null            -> all point1..point10
     (delete the line)      -> skip that type completely
   A folder number that doesn't exist is skipped with a warning.

3) Alignment settings  ->  "alignment:" section (method, fit window, ...)
4) Number of traces drawn  ->  plot.n_traces_plot (null = all)


Where the results go
--------------------
Every run makes a NEW folder with a timestamp (nothing is overwritten):

    /mnt/Data/results/PMMA_N_3_2026-09-29_14-35-02/
        air1/  air7/  point1/  point7/      <- figures (.png) per folder
        summary.csv                         <- numbers for all folders
        summary_gaussian_median_integer.png <- comparison across folders


What to look at
---------------
- summary.csv / summary_*.png : compare shift std, t0 std, p2p, RMS
  across folders. Outliers = folders with a problem.
- raw_vs_*.png : the aligned traces should overlap tightly.
- gaussian_every_trace_*.png : t0 per trace. Look for jumps or drift.
