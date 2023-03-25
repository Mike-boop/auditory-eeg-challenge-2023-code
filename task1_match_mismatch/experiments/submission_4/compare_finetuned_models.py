import glob
import json
import numpy as np
from scipy.stats import wilcoxon

evals1 = glob.glob('./task1_match_mismatch/experiments/results_submission_3/finetuned_env_models/*.json')
evals2 = glob.glob('./task1_match_mismatch/experiments/results_submission_4/finetuned_env_models/*.json')

subs = [f'sub-{i:03d}' for i in range(1, 72)]


acc1s = []
acc2s = []

for sub in subs:
    f1 = [x for x in evals1 if sub in x][0]
    f2 = [x for x in evals2 if sub in x][0]

    acc1 = json.load(open(f1, 'r'))[sub]['acc']
    acc2 = json.load(open(f2, 'r'))[sub]['acc']

    print(sub, acc1, acc2)

    #if acc1 == acc2: continue
    acc1s.append(acc1); acc2s.append(acc2)

print(np.mean(acc1s), np.mean(acc2s), 100*(np.mean(acc2s) - np.mean(acc1s)))
print(wilcoxon(acc1s, acc2s, alternative='less'))