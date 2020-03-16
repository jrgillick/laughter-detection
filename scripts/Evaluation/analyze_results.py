import pandas as pd, numpy as np, os, sys
from tqdm import tqdm

def resample(data, indices):
    new_data = []
    for i in indices:
        new_data.append(data[i])
    return new_data

# takes a list of tuples of precision, recall, f1, support
# returns the 95% confidence interval for each
def get_confidence_intervals(accs, precs, recs, f1s):
    accuracy_int = ("Accuracy: %s" % ' '.join(["%f" % (x) for x in np.percentile(accs, [2.5, 50, 97.5])]))
    precision_int = ("Precision: %s" % ' '.join(["%f" % (x) for x in np.percentile(precs, [2.5, 50, 97.5])]))
    recall_int = ("Recall: %s" % ' '.join(["%f" % (x) for x in np.percentile(recs, [2.5, 50, 97.5])]))
    f1_int = ("F1: %s" % ' '.join(["%f" % (x) for x in np.percentile(f1s, [2.5, 50, 97.5])]))

    return accuracy_int, precision_int, recall_int, f1_int

def get_metrics(tp_times,fp_times,tn_times,fn_times):
    tp_time = sum(tp_times)
    fp_time = sum(fp_times)
    tn_time = sum(tn_times)
    fn_time = sum(fn_times)
    accuracy = (tp_time + tn_time) / (tp_time + fp_time + tn_time + fn_time)
    precision = tp_time / (tp_time + fp_time)
    recall = tp_time / (tp_time + fn_time)
    f1 = 2*(precision*recall)/(precision+recall)
    return accuracy, precision, recall, f1

def bootstrap_metrics(tp_times,fp_times,tn_times,fn_times,n_samples=1000):
    accuracies = []; precisions = []; recalls = []; f1s = []
    for _ in tqdm(range(n_samples)):
        sample=np.random.choice(list(range(0,len(tp_times))),len(tp_times))
        sample_tp_times = resample(tp_times, sample)
        sample_fp_times = resample(fp_times, sample)
        sample_tn_times = resample(tn_times, sample)
        sample_fn_times = resample(fn_times, sample)
        metrics = get_metrics(sample_tp_times, sample_fp_times, sample_tn_times, sample_fn_times)
        accuracies.append(metrics[0]); precisions.append(metrics[1]); recalls.append(metrics[2]); f1s.append(metrics[3])
    
    intervals = get_confidence_intervals(accuracies, precisions, recalls, f1s)
    return intervals


print();print()
# Baseline on Audio Set
print("Baseline results on Audio Set...")
baseline_audioset_results = pd.read_csv('baseline_audioset_results.csv')

intervals = bootstrap_metrics(
    baseline_audioset_results.tp_time, baseline_audioset_results.fp_time,
    baseline_audioset_results.tn_time, baseline_audioset_results.fn_time,n_samples=1000)

for interval in intervals:
    print(interval)



print();print()
# Baseline on SWB Validation Set
print("Baseline results on SWB Validation Set...")
baseline_swv_val_results = pd.read_csv('baseline_switchboard_val_results.csv')

intervals = bootstrap_metrics(
    baseline_swv_val_results.tp_time, baseline_swv_val_results.fp_time,
    baseline_swv_val_results.tn_time, baseline_swv_val_results.fn_time,n_samples=1000)

for interval in intervals:
    print(interval)



print();print()
# Baseline on SWB Test Set
print("Baseline results on SWB Test Set...")
baseline_swv_test_results = pd.read_csv('baseline_switchboard_test_results.csv')

intervals = bootstrap_metrics(
    baseline_swv_test_results.tp_time, baseline_swv_test_results.fp_time,
    baseline_swv_test_results.tn_time, baseline_swv_test_results.fn_time,n_samples=1000)

for interval in intervals:
    print(interval)