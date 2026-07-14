import os, json, numpy as np
from scipy import stats
SP = 'C:/Users/yrdde/AppData/Local/Temp/claude/d--PhD-Experiments-classify-polyp/d5982003-bfbd-41f4-9d83-acfeabbb7c31/scratchpad'
PRE = os.path.join(SP, 'preds')
rng = np.random.default_rng(20260708)

def load(tag):
    d = np.load(os.path.join(PRE, tag + '.npz'))
    return (d['probs'], d['y_pred'], d['y_true'])

def mcnemar(y_true, pa, pb):
    ca = pa == y_true
    cb = pb == y_true
    b = int(np.sum(ca & ~cb))
    c = int(np.sum(~ca & cb))
    n = b + c
    p_exact = stats.binomtest(min(b, c), n, 0.5).pvalue if n > 0 else 1.0
    chi2 = (abs(b - c) - 1) ** 2 / n if n > 0 else 0.0
    p_chi = stats.chi2.sf(chi2, 1) if n > 0 else 1.0
    return dict(b=b, c=c, n_discordant=n, chi2_cc=round(chi2, 4), p_chi2_cc=p_chi, p_exact=p_exact, accA=float(ca.mean()), accB=float(cb.mean()))

def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2

def delong_var(y_true_bin, s1, s2):
    order = (-np.stack([s1, s2])).argsort()
    pos = y_true_bin == 1
    neg = ~pos
    scores = np.stack([s1, s2])
    m = pos.sum()
    n = neg.sum()
    k = 2
    tx = np.empty([k, m])
    ty = np.empty([k, n])
    tz = np.empty([k, m + n])
    aucs = np.empty(k)
    for r in range(k):
        pos_s = scores[r][pos]
        neg_s = scores[r][neg]
        tx[r] = compute_midrank(pos_s)
        ty[r] = compute_midrank(neg_s)
        tz[r] = compute_midrank(scores[r])
        aucs[r] = (tz[r][pos].sum() - m * (m + 1) / 2) / (m * n)
    v01 = (tz[:, pos] - tx) / n
    v10 = 1.0 - (tz[:, neg] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    S = sx / m + sy / n
    L = np.array([1, -1])
    var = L @ S @ L
    if var <= 0:
        z = 0.0
        p = 1.0
    else:
        z = (aucs[0] - aucs[1]) / np.sqrt(var)
        p = 2 * stats.norm.sf(abs(z))
    return (float(aucs[0]), float(aucs[1]), float(z), float(p))

def macro_auc(y_true, probs):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(np.eye(4)[y_true], probs, average='macro')

def boot_diff(y_true, probsA, probsB, metric, B=2000):
    n = len(y_true)
    diffs = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        diffs[i] = metric(y_true[idx], probsA[idx]) - metric(y_true[idx], probsB[idx])
    (lo, hi) = np.percentile(diffs, [2.5, 97.5])
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return dict(mean=float(diffs.mean()), lo=float(lo), hi=float(hi), p=float(min(p, 1.0)))

def acc_metric(yt, pr):
    return float((pr.argmax(1) == yt).mean())

def calibration(y_true, probs, nbins=10):
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, nbins + 1)
    ece = 0.0
    mce = 0.0
    rows = []
    N = len(y_true)
    for i in range(nbins):
        (lo, hi) = (bins[i], bins[i + 1])
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            rows.append((lo, hi, 0, None, None))
            continue
        acc = correct[m].mean()
        avg_conf = conf[m].mean()
        w = m.sum() / N
        gap = abs(acc - avg_conf)
        ece += w * gap
        mce = max(mce, gap)
        rows.append((round(lo, 2), round(hi, 2), int(m.sum()), round(float(acc), 4), round(float(avg_conf), 4)))
    brier = np.mean(np.sum((probs - np.eye(4)[y_true]) ** 2, axis=1))
    return dict(ECE=round(float(ece), 4), MCE=round(float(mce), 4), Brier=round(float(brier), 4), bins=rows)
OUT = {}
eff = [load(f'efficientnet_b1_full_run{r}_tta') for r in (1, 2, 3)]
mnv = [load(f'mobilenetv3_small_full_run{r}_tta') for r in (1, 2, 3)]
yt = eff[0][2]
assert np.array_equal(yt, mnv[0][2])
OUT['mcnemar_eff_vs_mnv3_per_run'] = [mcnemar(yt, eff[r][1], mnv[r][1]) for r in range(3)]
(_, base_p, _) = load('efficientnet_b1_baseline_run1_tta')
(_, cos_p, _) = load('efficientnet_b1_a2_cosine_run1_tta')
(_, mix_p, _) = load('efficientnet_b1_a3_mixup_run1_tta')
full_p = eff[0][1]
OUT['mcnemar_ablation'] = {'baseline_vs_full': mcnemar(yt, full_p, base_p), 'cosine_vs_full': mcnemar(yt, full_p, cos_p), 'cosine_vs_mixup': mcnemar(yt, cos_p, mix_p)}
classes = ['Gerd', 'Gerd_Normal', 'Polyp', 'Polyp_Normal']
dl = {}
for k in range(4):
    yb = (yt == k).astype(int)
    (a1, a2, z, p) = delong_var(yb, eff[0][0][:, k], mnv[0][0][:, k])
    dl[classes[k]] = dict(auc_eff=round(a1, 4), auc_mnv3=round(a2, 4), z=round(z, 3), p=round(p, 4))
OUT['delong_per_class_run1'] = dl
OUT['bootstrap_run1'] = {'macro_auc_diff_eff_minus_mnv3': boot_diff(yt, eff[0][0], mnv[0][0], macro_auc), 'accuracy_diff_eff_minus_mnv3': boot_diff(yt, eff[0][0], mnv[0][0], acc_metric)}
OUT['per_run'] = {'eff_acc': [round(float((eff[r][1] == yt).mean()), 4) for r in range(3)], 'mnv3_acc': [round(float((mnv[r][1] == yt).mean()), 4) for r in range(3)], 'eff_auc': [round(float(macro_auc(yt, eff[r][0])), 4) for r in range(3)], 'mnv3_auc': [round(float(macro_auc(yt, mnv[r][0])), 4) for r in range(3)]}
OUT['calibration_eff_run1'] = calibration(yt, eff[0][0])
OUT['calibration_mnv3_run1'] = calibration(yt, mnv[0][0])
OUT['ece_mean'] = {'eff': round(float(np.mean([calibration(yt, eff[r][0])['ECE'] for r in range(3)])), 4), 'mnv3': round(float(np.mean([calibration(yt, mnv[r][0])['ECE'] for r in range(3)])), 4)}
nd = np.load(os.path.join(SP, 'test_neardup_arrays.npz'), allow_pickle=True)
strict = (nd['cos'] >= 0.999) | (nd['ahash'] == 0) & (nd['dhash'] == 0) & (nd['phash'] == 0)
moderate = (nd['cos'] >= 0.99) | (nd['dhash'] == 0)

def acc_mask(p, mask_keep):
    return float((p[mask_keep] == yt[mask_keep]).mean())
keep_strict = ~strict
keep_mod = ~moderate
OUT['leakage_robustness'] = {'n_test': int(len(yt)), 'n_neardup_strict': int(strict.sum()), 'n_neardup_moderate': int(moderate.sum()), 'eff_acc_full_test_mean': round(float(np.mean([(eff[r][1] == yt).mean() for r in range(3)])), 4), 'eff_acc_clean_strict_mean': round(float(np.mean([acc_mask(eff[r][1], keep_strict) for r in range(3)])), 4), 'eff_acc_clean_moderate_mean': round(float(np.mean([acc_mask(eff[r][1], keep_mod) for r in range(3)])), 4), 'mnv3_acc_full_test_mean': round(float(np.mean([(mnv[r][1] == yt).mean() for r in range(3)])), 4), 'mnv3_acc_clean_strict_mean': round(float(np.mean([acc_mask(mnv[r][1], keep_strict) for r in range(3)])), 4)}
print(json.dumps(OUT, indent=2))
json.dump(OUT, open(os.path.join(SP, 'stats_results.json'), 'w'), indent=2)
print('\nSAVED stats_results.json')
