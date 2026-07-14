import os, sys, json
import numpy as np
from PIL import Image
from scipy.fftpack import dct
PROJ = 'd:/PhD/Experiments/classify_polyp'
os.chdir(PROJ)
SPLIT = 'data/split.json'

def load_gray(path, n):
    try:
        im = Image.open(path).convert('L').resize((n, n), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32)
    except Exception:
        return None

def ahash(path):
    g = load_gray(path, 8)
    if g is None:
        return None
    return (g > g.mean()).flatten()

def dhash(path):
    im = load_gray(path, None) if False else None
    try:
        g = np.asarray(Image.open(path).convert('L').resize((9, 8), Image.BILINEAR), np.float32)
    except Exception:
        return None
    return (g[:, 1:] > g[:, :-1]).flatten()

def phash(path):
    g = load_gray(path, 32)
    if g is None:
        return None
    d = dct(dct(g, axis=0, norm='ortho'), axis=1, norm='ortho')[:8, :8]
    med = np.median(d[1:].flatten())
    return (d > med).flatten()

def thumb(path):
    g = load_gray(path, 16)
    if g is None:
        return None
    v = g.flatten()
    v = v - v.mean()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def pack(bits):
    b = bits.astype(np.uint64)
    out = np.uint64(0)
    for i in range(64):
        out |= b[i] << np.uint64(i)
    return out

def hamming(a, b):
    x = int(a ^ b)
    return bin(x).count('1')
split = json.load(open(SPLIT))
sets = {k: split[k]['paths'] for k in ('train', 'val', 'test')}
print({k: len(v) for (k, v) in sets.items()}, flush=True)
feats = {}
for (k, paths) in sets.items():
    (ah, dh, ph, th, ok) = ([], [], [], [], [])
    for (i, p) in enumerate(paths):
        (a, d, h, t) = (ahash(p), dhash(p), phash(p), thumb(p))
        if a is None or d is None or h is None or (t is None):
            continue
        ah.append(pack(a))
        dh.append(pack(d))
        ph.append(pack(h))
        th.append(t)
        ok.append(p)
    feats[k] = dict(ahash=np.array(ah, dtype=np.uint64), dhash=np.array(dh, dtype=np.uint64), phash=np.array(ph, dtype=np.uint64), thumb=np.stack(th), paths=ok)
    print('hashed', k, len(ok), flush=True)

def min_hamming_to(query_hashes, ref_hashes):
    out = np.empty(len(query_hashes), dtype=np.int32)
    ref = ref_hashes
    for (i, q) in enumerate(query_hashes):
        x = np.bitwise_xor(ref, q)
        cnt = np.zeros(len(ref), dtype=np.int32)
        xx = x.copy()
        for _ in range(64):
            cnt += (xx & np.uint64(1)).astype(np.int32)
            xx >>= np.uint64(1)
        out[i] = cnt.min() if len(ref) else 64
    return out

def report(query, refs_key, ref_paths_pool, ref_feat_pool):
    res = {}
    for htype in ('ahash', 'dhash', 'phash'):
        d = min_hamming_to(feats[query][htype], ref_feat_pool[htype])
        res[htype] = d
    T = feats[query]['thumb']
    R = ref_feat_pool['thumb']
    sims = T @ R.T
    maxsim = sims.max(1)
    argmax = sims.argmax(1)
    res['cos'] = maxsim
    res['argmax'] = argmax
    return res

def pool(keys):
    return dict(ahash=np.concatenate([feats[k]['ahash'] for k in keys]), dhash=np.concatenate([feats[k]['dhash'] for k in keys]), phash=np.concatenate([feats[k]['phash'] for k in keys]), thumb=np.concatenate([feats[k]['thumb'] for k in keys]), paths=sum([feats[k]['paths'] for k in keys], []))
train_pool = pool(['train'])
trainval_pool = pool(['train', 'val'])
summary = {}
for (query, refpool, refname) in [('val', train_pool, 'train'), ('test', train_pool, 'train'), ('test', trainval_pool, 'train+val')]:
    r = report(query, refname, refpool['paths'], refpool)
    dmin = np.minimum(np.minimum(r['ahash'], r['dhash']), r['phash'])
    row = dict(n=len(dmin), exact_dHash0=int((r['dhash'] == 0).sum()), exact_all3_0=int(((r['ahash'] == 0) & (r['dhash'] == 0) & (r['phash'] == 0)).sum()), near_dHash_le5=int((r['dhash'] <= 5).sum()), near_pHash_le5=int((r['phash'] <= 5).sum()), near_combined_le5=int((dmin <= 5).sum()), near_combined_le10=int((dmin <= 10).sum()), cos_ge_0_999=int((r['cos'] >= 0.999).sum()), cos_ge_0_99=int((r['cos'] >= 0.99).sum()), cos_ge_0_95=int((r['cos'] >= 0.95).sum()), dHash_min=int(r['dhash'].min()), dHash_median=float(np.median(r['dhash'])), cos_max=float(r['cos'].max()), cos_median=float(np.median(r['cos'])))
    summary[f'{query}_vs_{refname}'] = row
    if query == 'test' and refname == 'train+val':
        np.savez(os.path.join('C:/Users/yrdde/AppData/Local/Temp/claude/d--PhD-Experiments-classify-polyp/d5982003-bfbd-41f4-9d83-acfeabbb7c31/scratchpad', 'test_neardup_arrays.npz'), ahash=r['ahash'], dhash=r['dhash'], phash=r['phash'], cos=r['cos'], test_paths=np.array(feats['test']['paths']))
    print(f"\n=== {query} vs {refname} (n={row['n']}) ===", flush=True)
    for (k, v) in row.items():
        print(f'  {k}: {v}', flush=True)
OUT = 'C:/Users/yrdde/AppData/Local/Temp/claude/d--PhD-Experiments-classify-polyp/d5982003-bfbd-41f4-9d83-acfeabbb7c31/scratchpad'
json.dump(summary, open(os.path.join(OUT, 'leakage_summary.json'), 'w'), indent=2)
print('\nSAVED leakage_summary.json', flush=True)
