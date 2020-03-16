import pandas as pd, numpy as np, os, sys

# Baseline on AudioSet
baseline_audioset_results = pd.read_csv('baseline_audioset_results.csv')
tp_time = sum(baseline_audioset_results.tp_time)
fp_time = sum(baseline_audioset_results.fp_time)
tn_time = sum(baseline_audioset_results.tn_time)
fn_time = sum(baseline_audioset_results.fn_time)
accuracy = (tp_time + tn_time) / (tp_time + fp_time + tn_time + fn_time)
precision = tp_time / (tp_time + fp_time)
recall = tp_time / (tp_time + fn_time)
f1 = 2*(precision*recall)/(precision+recall)
print("Baseline results on Audioset:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"f1: {f1}")

print();print()

# Baseline on SWB Validation Set
baseline_swv_val_results = pd.read_csv('baseline_switchboard_val_results.csv')
tp_time = sum(baseline_swv_val_results.tp_time)
fp_time = sum(baseline_swv_val_results.fp_time)
tn_time = sum(baseline_swv_val_results.tn_time)
fn_time = sum(baseline_swv_val_results.fn_time)
accuracy = (tp_time + tn_time) / (tp_time + fp_time + tn_time + fn_time)
precision = tp_time / (tp_time + fp_time)
recall = tp_time / (tp_time + fn_time)
f1 = 2*(precision*recall)/(precision+recall)
print("Baseline results on SWB Validation Set:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"f1: {f1}")

print();print()

"""
# Baseline on SWB Test Set
baseline_swv_test_results = pd.read_csv('baseline_switchboard_test_results.csv')
tp_time = sum(baseline_swv_test_results.tp_time)
fp_time = sum(baseline_swv_test_results.fp_time)
tn_time = sum(baseline_swv_test_results.tn_time)
fn_time = sum(baseline_swv_test_results.fn_time)
accuracy = (tp_time + tn_time) / (tp_time + fp_time + tn_time + fn_time)
precision = tp_time / (tp_time + fp_time)
recall = tp_time / (tp_time + fn_time)
f1 = 2*(precision*recall)/(precision+recall)
print("Baseline results on SWB Test Set:")
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"f1: {f1}")
"""