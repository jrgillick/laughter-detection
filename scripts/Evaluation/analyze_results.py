import pandas as pd, numpy as np, os, sys, argparse
import eval_utils
from tqdm import tqdm

n_samples=100

parser = argparse.ArgumentParser()

######## OPTIONAL ARGS #########
# Include event
parser.add_argument('--include_event_based_results', type=str)
parser.add_argument('--include_results_without_distractors', type=str)

args = parser.parse_args()

if args.include_event_based_results is not None:
    # This assumes that laughter has a specific start and stop point and that an "event" is correct if you correctly
    # detect greater than a certain percentage of frames. Not used in the paper.
    include_event_based_results = True
else:
    include_event_based_results = False

if args.include_results_without_distractors is not None:
    # Results if you don't include extra audio "distractors" without laughter. Not used in the paper.
    include_results_without_distractors = True
else:
    include_results_without_distractors = False


def resample(data, indices):
    new_data = []
    for i in indices:
        new_data.append(data[i])
    return new_data

# takes a list of tuples of precision, recall, f1, support
# returns the 95% confidence interval for each
def get_confidence_intervals(accs, precs, recs, f1s):
    accuracy_int = ("Accuracy: %s" % ' '.join(["%f" % (x) for x in np.percentile(accs, [2.5, 50, 97.5])]) + " | Accuracy: " + str(np.percentile(accs,50)) + " +- " + str((np.percentile(accs, 97.5) - np.percentile(accs,2.5))/2))
    precision_int = ("Precision: %s" % ' '.join(["%f" % (x) for x in np.percentile(precs, [2.5, 50, 97.5])]) + " | Precision: " + str(np.percentile(precs,50)) + "  +- " + str((np.percentile(precs, 97.5) - np.percentile(precs,2.5))/2))
    recall_int = ("Recall: %s" % ' '.join(["%f" % (x) for x in np.percentile(recs, [2.5, 50, 97.5])]) + " | Recall: " + str(np.percentile(recs,50)) + "  +- " + str((np.percentile(recs, 97.5) - np.percentile(recs,2.5))/2))
    f1_int = ("F1: %s" % ' '.join(["%f" % (x) for x in np.percentile(f1s, [2.5, 50, 97.5])]) + " | F1: " + str(np.percentile(f1s,50)) + "  +- " + str((np.percentile(f1s, 97.5) - np.percentile(f1s,2.5))/2))

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

def get_event_metrics(df, cutoff_length=0.1, indices=None):
    if indices is None: # optionally accept list of subsampled indices
        indices = list(range(len(df)))
    tp_count = 0; fp_count = 0; tn_count = 0; fn_count = 0
    for index in indices:
        tp, fp, tn, fn = eval_utils.get_event_metrics_per_row(df, index, cutoff_length=cutoff_length)
        tp_count += tp; fp_count += fp; tn_count += tn; fn_count += fn
    accuracy = float(tp_count+tn_count) / (tp_count+fp_count+tn_count+fn_count)
    precision = float(tp_count) / (tp_count + fp_count)
    recall = float(tp_count) / (tp_count + fn_count)
    f1 = 2*(precision*recall)/(precision+recall)
    return accuracy, precision, recall, f1

def bootstrap_event_metrics(df,cutoff_length=0.1,n_samples=1000):
    accuracies = []; precisions = []; recalls = []; f1s = []
    for _ in tqdm(range(n_samples)):
        sample_indices=np.random.choice(list(range(0,len(df))),len(df))
        metrics = get_event_metrics(df, cutoff_length=cutoff_length, indices=sample_indices)
        accuracies.append(metrics[0]); precisions.append(metrics[1]); recalls.append(metrics[2]); f1s.append(metrics[3])
        
    intervals = get_confidence_intervals(accuracies, precisions, recalls, f1s)
    return intervals    
    
    
results_files_and_names = []    

results_files_and_names.append(['interannotator_agreement_results.csv', "Interannotator Agreement", 0])

results_files_and_names.append(['baseline_switchboard_test_results.csv', "Baseline on SWB Test Set", 203])
results_files_and_names.append(['baseline_audioset_results.csv', "Baseline on Audioset", 1000])

results_files_and_names.append(['resnet_base_switchboard_test_results.csv', "Resnet Base on SWB Test Set", 203])
results_files_and_names.append(['resnet_base_audioset_results.csv', "Resnet Base on Audioset", 1000])

results_files_and_names.append(['resnet_specaug_wavaug_switchboard_test_results.csv', "Resnet SpecAug+WaveAug on SWB Test Set", 203])
results_files_and_names.append(['resnet_specaug_wavaug_audioset_results.csv', "Resnet SpecAug+WaveAug on AudioSet", 1000])

results_files_and_names.append(['noisy_audioset_trained_resnet_specaug_wavaug_switchboard_test_results.csv', "Noisy Audioset-Trained Resnet SpecAug+WaveAug on SWB Test Set", 203])
results_files_and_names.append(['noisy_audioset_trained_resnet_specaug_wavaug_audioset_results.csv', "Noisy Audioset-Trained Resnet SpecAug+WaveAug on AudioSet", 1000])

results_files_and_names.append(['noisy_audioset_trained_resnet_base_switchboard_test_results.csv', "Noisy Audioset-Trained Resnet Base on SWB Test Set", 203])
results_files_and_names.append(['noisy_audioset_trained_resnet_base_audioset_results.csv', "Noisy Audioset-Trained Resnet Base on AudioSet", 1000])

results_files_and_names.append(['noisy_audioset_trained_baseline_switchboard_test_results.csv', "Noisy Audioset-Trained Baseline on SWB Test Set", 203])
results_files_and_names.append(['noisy_audioset_trained_baseline_audioset_results.csv', "Noisy Audioset-Trained Baseline on AudioSet", 1000])

#results_files_and_names.append(['consistency_resnet_audioset_results.csv', "Consistency Resnet SpecAug+WaveAug on AudioSet", 1000])

# Results Including Distractors
print("\n\n")
print("############################## RESULTS INCLUDING DISTRACTORS.....   ##############################")
print("\n\n")
for i in range(len(results_files_and_names)):
    f = results_files_and_names[i][0]
    desc = results_files_and_names[i][1]
    results_df = pd.read_csv(f)
    print();print()
    print(desc + "...")
    print("Per-Frame Results:")
    
    timing_confidence_intervals = bootstrap_metrics(
        results_df.tp_time, results_df.fp_time,
        results_df.tn_time, results_df.fn_time,n_samples=n_samples)
    for interval in timing_confidence_intervals:
        print(interval)
        
    if include_event_based_results:
        print("\nEvent-Based Results:")
        event_confidence_intervals = bootstrap_event_metrics(results_df, n_samples=n_samples, cutoff_length=0.1)
        for interval in event_confidence_intervals:
            print(interval)


if include_results_without_distractors:
    # Results Without Distractors
    print("############################## RESULTS WITHOUT DISTRACTORS.....   ##############################")
    for i in range(len(results_files_and_names)):
        f = results_files_and_names[i][0]
        desc = results_files_and_names[i][1]
        num_distractors = results_files_and_names[i][2]
        results_df = pd.read_csv(f)
        results_df = results_df[0:len(results_df)-num_distractors]
        print();print()
        print(desc + "...")
        print("\nTiming Results:")

        timing_confidence_intervals = bootstrap_metrics(
            results_df.tp_time, results_df.fp_time,
            results_df.tn_time, results_df.fn_time,n_samples=n_samples)
        for interval in timing_confidence_intervals:
            print(interval)

        if include_event_based_results:
            print("\nEvent Results:")
            event_confidence_intervals = bootstrap_event_metrics(results_df, n_samples=n_samples, cutoff_length=0.1)
            for interval in event_confidence_intervals:
                print(interval)
        


"""
print();print()
# Baseline on Audio Set
print("Baseline Timing results on Audio Set...")
baseline_audioset_results = pd.read_csv('baseline_audioset_results.csv')

intervals = bootstrap_metrics(
    baseline_audioset_results.tp_time, baseline_audioset_results.fp_time,
    baseline_audioset_results.tn_time, baseline_audioset_results.fn_time,n_samples=n_samples)

for interval in intervals:
    print(interval)
    
print("Baseline Event results on Audio Set...")
intervals = bootstrap_event_metrics(baseline_audioset_results, n_samples=n_samples)

for interval in intervals:
    print(interval)    


##### BEGIN BASELINE SWB   ######
baseline_swv_val_results = pd.read_csv('baseline_switchboard_val_results.csv')
print();print()
print("Baseline Timing results on SWB Validation Set...")

intervals = bootstrap_metrics(
    baseline_swv_val_results.tp_time, baseline_swv_val_results.fp_time,
    baseline_swv_val_results.tn_time, baseline_swv_val_results.fn_time,n_samples=n_samples)

for interval in intervals:
    print(interval)



baseline_swv_test_results = pd.read_csv('baseline_switchboard_test_results.csv')
print();print()
print("Baseline Timing results on SWB Test Set...")

intervals = bootstrap_metrics(
    baseline_swv_test_results.tp_time, baseline_swv_test_results.fp_time,
    baseline_swv_test_results.tn_time, baseline_swv_test_results.fn_time,n_samples=n_samples)

for interval in intervals:
    print(interval)
##### END BASELINE SWB   ######


##### BEGIN RESNET BASE ######
results = pd.read_csv('resnet_base_switchboard_val_results.csv')
print();print()
print("Resnet Base Timing results on SWB Validation Set...")

intervals = bootstrap_metrics(
    results.tp_time, results.fp_time,
    results.tn_time, results.fn_time,n_samples=n_samples)

for interval in intervals:
    print(interval)

results = pd.read_csv('resnet_base_switchboard_test_results.csv')
print();print()
print("Resnet Base Timing results on SWB Test Set...")

intervals = bootstrap_metrics(
    results.tp_time, results.fp_time,
    results.tn_time, results.fn_time,n_samples=n_samples)

for interval in intervals:
    print(interval)

    
results = pd.read_csv('resnet_base_audioset_results.csv')
print();print()
print("Resnet Base Timing results on AudioSet...")

intervals = bootstrap_metrics(
    results.tp_time, results.fp_time,
    results.tn_time, results.fn_time,n_samples=n_samples)

for interval in intervals:
    print(interval)

##### END RESNET BASE ######


##### BEGIN RESNET SPECAUG    ######
results = pd.read_csv('resnet_specaug_switchboard_val_results.csv')
print();print()
print("Resnet SpecAug Timing results on SWB Validation Set...")

intervals = bootstrap_metrics(
    results.tp_time, results.fp_time,
    results.tn_time, results.fn_time,n_samples=n_samples)

for interval in intervals:
    print(interval)

results = pd.read_csv('resnet_specaug_switchboard_test_results.csv')
print();print()
print("Resnet SpecAug Timing results on SWB Test Set...")

intervals = bootstrap_metrics(
    results.tp_time, results.fp_time,
    results.tn_time, results.fn_time,n_samples=n_samples)

for interval in intervals:
    print(interval)

    

results = pd.read_csv('resnet_specaug_audioset_results.csv')
print();print()
print("Resnet SpecAug Timing results on AudioSet...")

intervals = bootstrap_metrics(
    results.tp_time, results.fp_time,
    results.tn_time, results.fn_time,n_samples=n_samples)

for interval in intervals:
    print(interval)

##### END RESNET SPECAUG    ######

##### BEGIN RESNET SPECAUG+WAVE-AUG    ######
results = pd.read_csv('resnet_specaug_wavaug_switchboard_val_results.csv')
print();print()
print("Resnet SpecAug + Wave-Aug Timing results on SWB Validation Set...")

intervals = bootstrap_metrics(
    results.tp_time, results.fp_time,
    results.tn_time, results.fn_time,n_samples=n_samples)

for interval in intervals:
    print(interval)

results = pd.read_csv('resnet_specaug_wavaug_switchboard_test_results.csv')
print();print()
print("Resnet SpecAug + Wave-Aug Timing results on SWB Test Set...")

intervals = bootstrap_metrics(
    results.tp_time, results.fp_time,
    results.tn_time, results.fn_time,n_samples=n_samples)

for interval in intervals:
    print(interval)

   
results = pd.read_csv('resnet_specaug_wavaug_audioset_results.csv')
print();print()
print("Resnet SpecAug + Wave-Aug Timing results on AudioSet...")

intervals = bootstrap_metrics(
    results.tp_time, results.fp_time,
    results.tn_time, results.fn_time,n_samples=n_samples)

for interval in intervals:
    print(interval)

##### END RESNET SPECAUG + WAVE-AUG   ######

"""
"""

##### BEGIN CONSISTENCY RESNET ######
results = pd.read_csv('consistency_resnet_switchboard_val_results.csv')
print();print()
print("Consistency Resnet SpecAug results on SWB Validation Set...")

intervals = bootstrap_metrics(
    results.tp_time, results.fp_time,
    results.tn_time, results.fn_time,n_samples=n_samples)

for interval in intervals:
    print(interval)

results = pd.read_csv('consistency_resnet_switchboard_test_results.csv')
print();print()
print("Consistency Resnet SpecAug results on SWB Test Set...")

intervals = bootstrap_metrics(
    results.tp_time, results.fp_time,
    results.tn_time, results.fn_time,n_samples=n_samples)

for interval in intervals:
    print(interval)

    
results = pd.read_csv('consistency_resnet_audioset_results.csv')
print();print()
print("Consistency Resnet results on AudioSet...")

intervals = bootstrap_metrics(
    results.tp_time, results.fp_time,
    results.tn_time, results.fn_time,n_samples=n_samples)

for interval in intervals:
    print(interval)

##### END CONSISTENCY RESNET ######
"""
