"""Q-LORD on EyePACS (Kaggle Diabetic Retinopathy, resized). Script version of notebooks/qlord_eyepacs_main.ipynb.
Edit MODE / SMOKE_TEST in PART 0 ("main" = full model, "study" = baselines + ablations). Run on a Kaggle T4 GPU."""

# %% ============ PART 0: SETTINGS + HELPERS (always run) ============
import os, sys, time, json, math, glob, shutil, random, warnings, subprocess
from pathlib import Path
warnings.filterwarnings("ignore")
T0 = time.time()

# ------------------------------ USER SETTINGS ------------------------------
DATASET          = "eyepacs"                   # fixed: Diabetic Retinopathy (EyePACS 2015) resized dataset
MODE             = "main"                      # "main" or "study"  (see header)
RUN_PARTS        = ["prep", "train", "eval"]   # any subset; e.g. ["train"] on day 2
FOLDS_TO_RUN     = None                        # main mode: None = all folds; or e.g. [0] on day 1, [1, 2] on day 2
STUDY_FOLDS      = None                        # study mode: None = [0]
STUDY_VARIANTS   = ["qlord", "base_softmax", "base_mcdrop", "base_ordinal", "base_evidential",
                    "abl_no_graph", "abl_no_lesion_graph", "abl_no_ordinal", "abl_no_evidential",
                    "abl_no_quality", "abl_no_consistency", "abl_no_oversampling"]
SMOKE_TEST       = False                       # True = tiny end-to-end test (few minutes)
ALLOW_PARTIAL_EVAL = False                     # True = evaluate with only the folds finished so far
SESSION_BUDGET_H = 11.0                        # stop cleanly before Kaggle's 12 h limit

IMG_SIZE   = int(os.environ.get("QLORD_IMG", 512 if MODE == "main" else 384))
BACKBONE   = os.environ.get("QLORD_BACKBONE", "efficientnet_b3")
PRETRAINED = os.environ.get("QLORD_PRETRAINED", "1") == "1"
PRETRAINED_FILE = None      # optional: path to a .pth state_dict if Internet is OFF

SEED = 42
assert DATASET == "eyepacs", "this notebook is EyePACS-only"
EYE = True
N_FOLDS = 3
TEST_FRAC = 0.10   # held-out test split (test labels are not available, so we split the labelled set)
N_TOTAL = 35108    # labelled images (used only for the time estimate)
# EyePACS is large: one "epoch" = a random class-balanced draw of this many images
SAMPLES_PER_EPOCH = 6000 if MODE == "main" else 4000
VAL_MAX_PER_EPOCH = 3000    # per-epoch validation (model selection) uses at most this many images
BATCH = 16                  # train batch; CONS_N of them also get a 2nd view for consistency losses
CONS_N = 8
EPOCHS_STAGE1, EPOCHS_STAGE2, PATIENCE = (3, 25, 8) if MODE == "main" else (2, 12, 5)
PROTO_E1, PROTO_E2 = EPOCHS_STAGE1, EPOCHS_STAGE2   # stage 1 = frozen backbone, stage 2 = fine-tune all
MC_SAMPLES = 20             # MC-dropout baseline
LR_BACKBONE, LR_HEAD, WD = 3e-4, 1e-3, 1e-2
SAMPLER_POWER = 0.75        # 1.0 = fully balanced oversampling, 0 = none
QUALITY_BAD_FRAC = 0.10     # bottom 10% by image-statistics score are labelled "ungradable"
GRAD_CKPT = True            # saves GPU memory, about 20-25% slower
N_QUERIES, TOPM, D_MODEL = 8, 48, 256
W_ORD, W_EVID, W_Q, W_GRAPH, W_ATTN, W_ABST = 1.0, 1.0, 0.3, 0.2, 0.1, 0.2
KL_MAX, KL_ANNEAL_EPOCHS = 0.05, 10
TOPK_ENSEMBLE = 5
TARGET_COVERAGE = 0.90

if FOLDS_TO_RUN is None:
    FOLDS_TO_RUN = list(range(N_FOLDS))
if STUDY_FOLDS is None:
    STUDY_FOLDS = [0]
REAL_N_FOLDS = N_FOLDS
REAL_FOLDS = N_FOLDS if MODE == "main" else len(STUDY_FOLDS)
if SMOKE_TEST:
    N_FOLDS, FOLDS_TO_RUN, STUDY_FOLDS = 2, [0, 1], [0]
    EPOCHS_STAGE1, EPOCHS_STAGE2 = 1, 1
if MODE == "study":
    FOLDS_TO_RUN = list(STUDY_FOLDS)
VARIANTS_TO_RUN = ["qlord"] if MODE == "main" else list(STUDY_VARIANTS)
SMOKE_EST = {}

# ------------------------------ PATHS ------------------------------
ON_KAGGLE = Path("/kaggle/working").exists()
WORK = Path("/kaggle/working") if ON_KAGGLE else Path("./work")
TAG = ("smoke_" if SMOKE_TEST else "") + f"{DATASET}_{MODE}"
STATE = WORK / f"qlord_state_{TAG}"          # everything needed to resume lives here
RUNS = STATE / "runs"
TEMP = Path("/kaggle/temp") if Path("/kaggle/temp").exists() else WORK        # not saved as notebook output
CACHE = TEMP / f"cache_{IMG_SIZE}_{TAG}"
FIG = STATE / "figures"
for p in (WORK, STATE, RUNS, CACHE, FIG):
    p.mkdir(parents=True, exist_ok=True)

import numpy as np, pandas as pd, cv2
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
try:
    import timm
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm"], check=True)
    import timm
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEV.type == "cuda"
N_CLASSES = 5
STOP_REQUESTED = False
cv2.setNumThreads(1)


def elapsed_h():
    return (time.time() - T0) / 3600


def seed_everything(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def log(*a):
    print(f"[{elapsed_h():5.2f}h]", *a, flush=True)


def restore_previous_state():
    """If this is a new session and an earlier version's output is attached as Input, copy its state."""
    if (STATE / "split.csv").exists():
        return
    root = Path("/kaggle/input")
    if not root.exists():
        return
    cands = [p for p in root.rglob(f"qlord_state_{TAG}") if p.is_dir()]
    if not cands:
        return
    best = max(cands, key=lambda p: max([f.stat().st_mtime for f in p.rglob("*") if f.is_file()] or [0]))
    log("Restoring previous state from", best)
    shutil.copytree(best, STATE, dirs_exist_ok=True)


restore_previous_state()
seed_everything()
log("device:", DEV, "| torch", torch.__version__, "| timm", timm.__version__, "| image size", IMG_SIZE, "| backbone", BACKBONE)


def find_eyepacs():
    """Locate labels CSV + image folder of tanlikesmath/diabetic-retinopathy-resized (layout is auto-detected)."""
    env = os.environ.get("EYEPACS_DIR")
    root = Path(env) if env else Path("/kaggle/input")
    csvs = {p.name: p for p in root.rglob("trainLabels*.csv")}
    for csv_name, folder in (("trainLabels_cropped.csv", "resized_train_cropped"), ("trainLabels.csv", "resized_train")):
        if csv_name in csvs:
            dirs = [p for p in root.rglob(folder) if p.is_dir()]
            if dirs:
                inner = dirs[0] / folder           # these datasets usually nest a folder of the same name
                return csvs[csv_name], (inner if inner.is_dir() else dirs[0])
    raise FileNotFoundError("EyePACS data not found. Add 'tanlikesmath/diabetic-retinopathy-resized' as Input "
                            "(expects trainLabels*.csv and a resized_train* folder).")


def load_index():
    """Returns DataFrame[id_code, diagnosis, path] for EyePACS."""
    csv, img_dir = find_eyepacs()
    lab = pd.read_csv(csv)
    lab = lab.rename(columns={c: c.strip().lower() for c in lab.columns})
    df = pd.DataFrame(dict(id_code=lab["image"].astype(str), diagnosis=lab["level"].astype(int)))
    files = {Path(f).stem: f for f in os.listdir(img_dir) if f.lower().endswith((".jpeg", ".jpg", ".png"))}
    ok = df.id_code.isin(files)
    if (~ok).any():
        log(f"index: {int((~ok).sum())} labelled ids have no image file in {img_dir.name}; dropped")
    df = df[ok].reset_index(drop=True)
    df["path"] = [str(img_dir / files[i]) for i in df.id_code]
    log(f"index: {len(df)} EyePACS images from {img_dir}")
    return df


# %% ============ PART 1: PREPROCESSING + QUALITY PROXY + SPLITS ============
# Ben Graham preprocessing, cached as JPEG so training is fast.
# Neither dataset ships image-quality labels, so a proxy label is built from image statistics
# (sharpness, contrast, exposure, retina coverage). Replace with real labels if you have them.
from concurrent.futures import ThreadPoolExecutor


def crop_square(img, S):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    m = gray > 10
    if m.sum() > 0.01 * m.size:
        ys, xs = np.where(m.any(1))[0], np.where(m.any(0))[0]
        img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = img.shape[:2]
    side = max(h, w)
    img = cv2.copyMakeBorder(img, (side - h) // 2, side - h - (side - h) // 2,
                             (side - w) // 2, side - w - (side - w) // 2, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    interp = cv2.INTER_AREA if side > S else cv2.INTER_CUBIC
    return cv2.resize(img, (S, S), interpolation=interp)


def quality_feats(img):
    g = img[:, :, 1]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = (gray > 10).astype(np.uint8)
    me = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    if me.sum() < 100:
        me = mask
    sel = me > 0
    lap = cv2.Laplacian(g, cv2.CV_32F)
    vals = g[sel].astype(np.float32)
    return dict(sharp=float(lap[sel].var()), bright=float(vals.mean()),
                contrast=float(vals.std()), cover=float(mask.mean()))


def ben_graham(img):
    S = img.shape[0]
    out = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), S / 30), -4, 128)
    circ = np.zeros((S, S, 1), np.uint8)
    cv2.circle(circ, (S // 2, S // 2), int(S * 0.48), 1, -1)
    return (out * circ + 128 * (1 - circ)).astype(np.uint8)


def prep_one(args):
    id_code, path = args
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    img = crop_square(img, IMG_SIZE)
    f = quality_feats(img)
    cv2.imwrite(str(CACHE / f"{id_code}.jpg"), ben_graham(img), [cv2.IMWRITE_JPEG_QUALITY, 95])
    f["id_code"] = id_code
    return f


def run_jobs(jobs):
    feats = []
    with ThreadPoolExecutor(max_workers=max(2, os.cpu_count() or 2)) as ex:
        for i, f in enumerate(ex.map(prep_one, jobs)):
            feats.append(f)
            if (i + 1) % 2000 == 0:
                log(f"  preprocessed {i + 1}/{len(jobs)}")
    return pd.DataFrame(feats)


def run_prep():
    split_path = STATE / "split.csv"
    if split_path.exists():                       # keep the SAME split across days
        df = pd.read_csv(split_path, dtype={"id_code": str})
        missing = [i for i in df.id_code if not (CACHE / f"{i}.jpg").exists()]
        if not missing:
            log("prep: split + cache already present, skipping")
            return df
        paths = load_index().set_index("id_code").path
        log(f"prep: rebuilding cache for {len(missing)} images (split.csv kept unchanged)")
        run_jobs([(i, paths[i]) for i in missing])
        return df
    df = load_index()
    if SMOKE_TEST:
        df = df.groupby("diagnosis", group_keys=False).head(30).reset_index(drop=True)
    log(f"prep: {len(df)} images")
    fe = run_jobs(list(zip(df.id_code, df.path)))
    z = lambda s: (s - s.mean()) / (s.std() + 1e-6)
    score = z(np.log1p(fe.sharp)) + z(fe.contrast) - z(fe.bright).abs() + z(fe.cover)
    fe["q_score"] = score
    fe["q_label"] = (score > np.quantile(score, QUALITY_BAD_FRAC)).astype(float)   # 1 = gradable
    df = df.drop(columns="path").merge(fe, on="id_code")
    # group by patient so the left/right eyes of one patient never sit in different splits (EyePACS ids: "<patient>_left")
    groups = df.id_code.str.split("_").str[0]
    df["fold"], df["role"] = -1, "trainval"
    n_test = max(2, int(round(1 / TEST_FRAC)))
    _, te_idx = next(StratifiedGroupKFold(n_test, shuffle=True, random_state=SEED).split(df, df.diagnosis, groups))
    df.loc[df.index[te_idx], "role"] = "test"
    tv = df[df.role == "trainval"]
    sgkf = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
    for k, (_, vi) in enumerate(sgkf.split(tv, tv.diagnosis, groups[tv.index])):
        df.loc[tv.index[vi], "fold"] = k
    df.to_csv(split_path, index=False)
    log("prep done. class counts:", df.diagnosis.value_counts().sort_index().to_dict(),
        "| ungradable proxy:", int((df.q_label == 0).sum()), "| test images:", int((df.role == "test").sum()))
    return df


# %% ============ PART 2: DATA, MODEL, LOSSES, METRICS ============
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def corrupt(img):
    """Synthetic out-of-distribution proxy: heavy blur + noise + under-exposure."""
    rng = np.random.RandomState(0)
    img = cv2.GaussianBlur(img, (0, 0), 6)
    img = img.astype(np.float32) * 0.6 + rng.randn(*img.shape).astype(np.float32) * 20
    return np.clip(img, 0, 255).astype(np.uint8)


def augment(img):
    S = img.shape[0]
    ang, sc = np.random.uniform(0, 360), np.random.uniform(0.9, 1.1)
    M = cv2.getRotationMatrix2D((S / 2, S / 2), ang, sc)
    M[:, 2] += np.random.uniform(-0.04, 0.04, 2) * S
    img = cv2.warpAffine(img, M, (S, S), flags=cv2.INTER_LINEAR, borderValue=(128, 128, 128))
    if np.random.rand() < 0.5: img = img[:, ::-1]
    if np.random.rand() < 0.5: img = img[::-1]
    a, b = np.random.uniform(0.8, 1.2), np.random.uniform(-15, 15)
    img = np.clip((img.astype(np.float32) - 128) * a + 128 + b, 0, 255)
    if np.random.rand() < 0.2:
        img = cv2.GaussianBlur(img, (0, 0), np.random.uniform(0.5, 1.5))
    return np.ascontiguousarray(img.astype(np.uint8))


class DRDataset(Dataset):
    def __init__(self, df, train=False, ood=False):
        self.ids, self.y = df.id_code.values, df.diagnosis.values.astype(np.int64)
        self.q = df.q_label.values.astype(np.float32)
        self.train, self.ood = train, ood

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img = cv2.imread(str(CACHE / f"{self.ids[i]}.jpg"))
        if self.ood: img = corrupt(img)
        if self.train: img = augment(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(img).permute(2, 0, 1), self.y[i], self.q[i]


def make_loader(df, train=False, ood=False, bs=None, sampler=None):
    nw = min(4, os.cpu_count() or 1)
    return DataLoader(DRDataset(df, train, ood), batch_size=bs or (BATCH if train else 2 * BATCH),
                      sampler=sampler, shuffle=False, num_workers=nw, pin_memory=USE_AMP,
                      drop_last=train, persistent_workers=train and nw > 0)


def to_input(xb):
    x = xb.to(DEV, non_blocking=True).float() / 255.0
    x = (x - MEAN.to(DEV)) / STD.to(DEV)
    return x.contiguous(memory_format=torch.channels_last)


def dirichlet_unc(alpha):
    K = alpha.shape[-1]
    S = alpha.sum(-1, keepdim=True)
    p = alpha / S
    tot = -(p * torch.log(p.clamp_min(1e-8))).sum(-1)
    ale = (p * (torch.digamma(S + 1) - torch.digamma(alpha + 1))).sum(-1)
    return dict(vacuity=K / S.squeeze(-1), total=tot, aleatoric=ale, epistemic=(tot - ale).clamp_min(0))


def _v(spatial="graph", head="ord_evid", quality=True, consist=True, oversample=True):
    return dict(spatial=spatial, head=head, quality=quality, consist=consist, oversample=oversample)


# spatial: "graph" = lesion-query attention + patch graph | "lesion" = lesion-query attention only | "gap" = global pooling
# head   : "ord_evid" = ordinal + evidential (Q-LORD) | "evid" = flat Dirichlet | "ord" = ordinal only
#          "softmax" = plain cross-entropy | "mcdrop" = softmax + MC-dropout at inference
VARIANTS = {
    "qlord":               _v(),
    # ---- baselines (plain pooled features, no quality head, no consistency losses) ----
    "base_softmax":        _v("gap", "softmax", False, False),    # its fold ensemble = Deep Ensemble baseline
    "base_mcdrop":         _v("gap", "mcdrop", False, False),
    "base_ordinal":        _v("gap", "ord", False, False),
    "base_evidential":     _v("gap", "evid", False, False),
    # ---- ablations: remove ONE component from Q-LORD ----
    "abl_no_graph":        _v(spatial="lesion"),
    "abl_no_lesion_graph": _v(spatial="gap"),
    "abl_no_ordinal":      _v(head="evid"),
    "abl_no_evidential":   _v(head="ord"),
    "abl_no_quality":      _v(quality=False),
    "abl_no_consistency":  _v(consist=False),
    "abl_no_oversampling": _v(oversample=False),
}
EVID_HEADS = ("ord_evid", "evid")


def entropy_t(p):
    return -(p * torch.log(p.clamp_min(1e-8))).sum(-1)


class Net(nn.Module):
    """Configurable by a variant dict V. With the default V this is the full Q-LORD:
       backbone -> lesion-query attention -> patch graph (GCN) -> fused feature ->
       ordinal-evidential head + quality head + abstention gate."""

    def __init__(self, V, backbone=BACKBONE, pretrained=PRETRAINED, d=D_MODEL, n_queries=N_QUERIES,
                 heads=4, topm=TOPM, gcn_layers=2, sigma=0.25, tau=0.5):
        super().__init__()
        self.V = V
        try:
            self.backbone = timm.create_model(backbone, pretrained=pretrained and PRETRAINED_FILE is None,
                                              num_classes=0, global_pool="")
        except Exception as e:
            raise RuntimeError("Could not download pretrained weights. Turn Internet ON in the notebook "
                               "settings, or upload weights and set PRETRAINED_FILE.\n" + str(e))
        if PRETRAINED_FILE:
            self.backbone.load_state_dict(torch.load(PRETRAINED_FILE, map_location="cpu"), strict=False)
        if GRAD_CKPT and hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(True)
        C = self.backbone.num_features
        self.d, self.heads, self.topm, self.sigma, self.tau = d, heads, topm, sigma, tau
        sp, hd = V["spatial"], V["head"]
        if sp == "gap":
            self.fuse = nn.Sequential(nn.Linear(C, d), nn.GELU())
        else:
            self.proj = nn.Sequential(nn.Conv2d(C, d, 1, bias=False), nn.BatchNorm2d(d), nn.GELU())
            self.pos = nn.Parameter(torch.randn(1, d, 16, 16) * 0.02)
            self.ln_tok = nn.LayerNorm(d)
            self.queries = nn.Parameter(torch.randn(n_queries, d) * 0.02)   # learned lesion prototypes
            self.wq, self.wk, self.wv = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
            self.q_mlp = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
            if sp == "graph":
                self.gcn = nn.ModuleList([nn.Linear(d, d) for _ in range(gcn_layers)])
                self.gcn_ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(gcn_layers)])
                self.node_att = nn.Linear(d, 1)
            self.fuse = nn.Sequential(nn.Linear(3 * d, d), nn.GELU())
        self.drop = nn.Dropout(0.3)
        if V["quality"]:
            self.q_head = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))
        if hd in ("ord_evid", "ord"):
            self.ord_f = nn.Linear(d, 1)
            self.ord_raw = nn.Parameter(torch.tensor([-1.5, 0.5413, 0.5413, 0.5413]))   # ordered thresholds
        if hd in EVID_HEADS:
            self.ev_head = nn.Linear(d, N_CLASSES)
        if hd in ("softmax", "mcdrop"):
            self.cls = nn.Linear(d, N_CLASSES)
        self.gate = nn.Sequential(nn.Linear(5, 16), nn.GELU(), nn.Linear(16, 1))

    def head_params(self):
        return [p for n, p in self.named_parameters() if not n.startswith("backbone.")]

    def forward(self, x, backbone_grad=True):
        V, sp, hd = self.V, self.V["spatial"], self.V["head"]
        if backbone_grad:
            f = self.backbone.forward_features(x)
        else:
            with torch.no_grad():
                f = self.backbone.forward_features(x)
        B = f.shape[0]
        if sp != "gap":
            t = self.proj(f)
            _, d, H, W = t.shape
            t = t + F.interpolate(self.pos, size=(H, W), mode="bilinear", align_corners=False)
        attn, hw = None, None
        with torch.autocast(device_type=x.device.type, enabled=False):
            if sp == "gap":
                h = self.fuse(f.float().mean((2, 3)))
                gvec = F.normalize(h, dim=-1)
            else:
                d = self.d
                t = self.ln_tok(t.float().flatten(2).transpose(1, 2))                  # [B,N,d]
                N, nh = t.shape[1], self.heads
                dh = d // nh
                q = self.wq(self.queries).view(-1, nh, dh)
                k = self.wk(t).view(B, N, nh, dh)
                v = self.wv(t).view(B, N, nh, dh)
                att = (torch.einsum("khd,bnhd->bhkn", q, k) / math.sqrt(dh)).softmax(-1)
                lq = torch.einsum("bhkn,bnhd->bkhd", att, v).reshape(B, -1, d)
                lq = lq + self.q_mlp(lq)                                                # lesion tokens [B,K,d]
                attn, hw = att.mean(1), (H, W)                                          # [B,K,N]
                if sp == "graph":
                    s = attn.max(1).values
                    M = min(self.topm, N)
                    sv, idx = s.topk(M, dim=-1)
                    node = torch.gather(t, 1, idx.unsqueeze(-1).expand(-1, -1, d))
                    gate_w = sv / sv[:, :1].clamp_min(1e-8)
                    node = node * (0.5 + 0.5 * gate_w).unsqueeze(-1)
                    coords = torch.stack([(idx // W).float() / max(H - 1, 1), (idx % W).float() / max(W - 1, 1)], -1)
                    spatial = torch.exp(-torch.cdist(coords, coords) ** 2 / (2 * self.sigma ** 2))
                    nn_ = F.normalize(node, dim=-1)
                    adj = torch.exp(nn_ @ nn_.transpose(1, 2) / self.tau) * spatial
                    adj = adj / adj.sum(-1, keepdim=True)
                    hN = node
                    for lin, ln in zip(self.gcn, self.gcn_ln):
                        hN = ln(hN + F.gelu(lin(adj @ hN)))
                    wn = self.node_att(hN).softmax(1)
                    gpool = (wn * hN).sum(1)
                else:                                                                   # lesion attention only
                    gpool = lq.max(1).values
                gvec = F.normalize(gpool, dim=-1)
                h = self.fuse(torch.cat([gpool, lq.mean(1), t.mean(1)], -1))
            hf = self.drop(h)                                                           # active in train mode only
            # ------------------------------ heads ------------------------------
            if V["quality"]:
                q_logit = self.q_head(hf).squeeze(-1)
                qual = torch.sigmoid(q_logit)
            else:
                q_logit, qual = torch.zeros(B, device=h.device), torch.ones(B, device=h.device)
            alpha = p_ord = cum_z = logits = mc = None
            if hd in ("ord_evid", "ord"):
                cum_z = self.ord_f(hf) - torch.cumsum(torch.cat([self.ord_raw[:1], F.softplus(self.ord_raw[1:])]), 0)
                cum = torch.sigmoid(cum_z)                                              # P(y>k), monotone in k
                p_ord = torch.cat([1 - cum[:, :1], cum[:, :-1] - cum[:, 1:], cum[:, -1:]], 1).clamp_min(1e-6)
                p_ord = p_ord / p_ord.sum(1, keepdim=True)
            if hd in EVID_HEADS:
                alpha = F.softplus(self.ev_head(hf)) * (0.5 + 0.5 * qual).unsqueeze(-1) + 1   # quality-aware gating
                p_dir = alpha / alpha.sum(1, keepdim=True)
            if hd == "ord_evid":
                p = 0.5 * (p_ord + p_dir)
            elif hd == "evid":
                p = p_dir
            elif hd == "ord":
                p = p_ord
            else:
                logits = self.cls(hf)
                p = logits.softmax(-1)
                if hd == "mcdrop" and not self.training:
                    mc = torch.stack([self.cls(F.dropout(h, 0.3, training=True)).softmax(-1) for _ in range(MC_SAMPLES)])
                    p = mc.mean(0)
            # ----- abstention gate (inputs detached: it learns to predict errors, it does not steer the backbone) -----
            ud = dirichlet_unc(alpha) if alpha is not None else None
            ent = entropy_t(p) / math.log(N_CLASSES)
            top2 = p.topk(2, dim=1).values
            zeros = torch.zeros_like(ent)
            gf = torch.stack([ud["vacuity"] if ud else zeros, ent, 1 - qual, 1 - (top2[:, 0] - top2[:, 1]),
                              ud["epistemic"] / math.log(N_CLASSES) if ud else zeros], -1).detach()
            gate_logit = self.gate(gf).squeeze(-1)
            unc = None
            if not self.training:       # columns: native, total entropy, vacuity, aleatoric, epistemic
                tot = entropy_t(p)
                if ud is not None:
                    vac, ale, epi, nat = ud["vacuity"], ud["aleatoric"], ud["epistemic"], ud["vacuity"]
                elif mc is not None:
                    ale = torch.stack([entropy_t(m) for m in mc]).mean(0)
                    epi = (tot - ale).clamp_min(0)
                    vac, nat = zeros, epi
                else:
                    vac, ale, epi, nat = zeros, tot, zeros, tot
                unc = torch.stack([nat, tot, vac, ale, epi], -1)
        return dict(p=p, alpha=alpha, logits=logits, cum_logit=cum_z, qual=qual, q_logit=q_logit,
                    gate_logit=gate_logit, attn=attn, gvec=gvec, hw=hw, unc=unc)


def kl_dir_uniform(a):
    K = a.shape[1]
    S = a.sum(1, keepdim=True)
    return (torch.lgamma(S.squeeze(1)) - math.lgamma(K) - torch.lgamma(a).sum(1)
            + ((a - 1) * (torch.digamma(a) - torch.digamma(S))).sum(1))


def ordinal_evidential_loss(alpha, y, kl_w):
    """Sensoy-style evidential MSE on CUMULATIVE class probabilities (respects grade ordering)
       + annealed KL penalty on misleading evidence."""
    S = alpha.sum(1, keepdim=True)
    cp = (alpha / S).cumsum(1)[:, :-1]
    ct = (y.unsqueeze(1) <= torch.arange(N_CLASSES - 1, device=y.device).unsqueeze(0)).float()
    mse = ((ct - cp) ** 2 + cp * (1 - cp) / (S + 1)).sum(1)
    oh = F.one_hot(y, N_CLASSES).float()
    return (mse + kl_w * kl_dir_uniform(oh + (1 - oh) * alpha)).mean()


def flat_evidential_loss(alpha, y, kl_w):
    """Standard (non-ordinal) evidential loss on one-hot targets: used by the evidential baseline / ablation."""
    S = alpha.sum(1, keepdim=True)
    p = alpha / S
    oh = F.one_hot(y, N_CLASSES).float()
    mse = ((oh - p) ** 2 + p * (1 - p) / (S + 1)).sum(1)
    return (mse + kl_w * kl_dir_uniform(oh + (1 - oh) * alpha)).mean()


def jsd_attn(a1, a2):
    m = 0.5 * (a1 + a2)
    kl = lambda p, q: (p * (torch.log(p.clamp_min(1e-8)) - torch.log(q.clamp_min(1e-8)))).sum(-1)
    return (0.5 * kl(a1, m) + 0.5 * kl(a2, m)).mean()


def compute_losses(out, y, qlab, epoch, V):
    o = {k: (v.float() if torch.is_tensor(v) else v) for k, v in out.items()}
    B, hd = y.shape[0], V["head"]
    zero = torch.zeros((), device=y.device)
    kl_w = KL_MAX * min(1.0, epoch / KL_ANNEAL_EPOCHS)
    parts, total = {}, zero
    if hd in ("ord_evid", "ord"):
        ord_t = (y.unsqueeze(1) > torch.arange(N_CLASSES - 1, device=y.device).unsqueeze(0)).float()
        l = F.binary_cross_entropy_with_logits(o["cum_logit"][:B], ord_t)
        total = total + W_ORD * l; parts["ord"] = l.item()
    if hd == "ord_evid":
        l = ordinal_evidential_loss(o["alpha"][:B], y, kl_w); total = total + W_EVID * l; parts["evid"] = l.item()
    elif hd == "evid":
        l = flat_evidential_loss(o["alpha"][:B], y, kl_w); total = total + W_EVID * l; parts["evid"] = l.item()
    elif hd in ("softmax", "mcdrop"):
        l = F.cross_entropy(o["logits"][:B], y); total = total + l; parts["ce"] = l.item()
    if V["quality"]:
        l = F.binary_cross_entropy_with_logits(o["q_logit"][:B], qlab); total = total + W_Q * l; parts["qual"] = l.item()
    # abstention-calibration loss: confident errors are penalised more than safe referral
    pB = o["p"][:B]
    pred = pB.argmax(1)
    err = (pred - y).abs().clamp(max=2).float() / 2
    wgt = 1 + 2 * (pred != y).float() * pB.max(1).values.detach()
    l = (F.binary_cross_entropy_with_logits(o["gate_logit"][:B], err, reduction="none") * wgt).mean()
    total = total + W_ABST * l; parts["abst"] = l.item()
    if o["attn"] is not None and o["p"].shape[0] > B:       # second (flipped + jittered) view available
        n, (H, W) = CONS_N, o["hw"]
        a2 = o["attn"][B:B + n].view(n, -1, H, W).flip(-1).reshape(n, -1, H * W)   # un-flip the attention map
        l_a = jsd_attn(o["attn"][:n], a2)
        l_g = (1 - (o["gvec"][:n] * o["gvec"][B:B + n]).sum(-1)).mean()
        total = total + W_GRAPH * l_g + W_ATTN * l_a
        parts["graph"], parts["attn"] = l_g.item(), l_a.item()
    return total, parts


# ------------------------------ metrics ------------------------------
def ece_score(p, y, bins=15):
    conf, pred = p.max(1), p.argmax(1)
    e = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)


def metrics(p, y):
    pred = p.argmax(1)
    oh = np.eye(N_CLASSES)[y]
    return dict(acc=float((pred == y).mean()), qwk=float(cohen_kappa_score(y, pred, weights="quadratic")),
                macro_f1=float(f1_score(y, pred, average="macro")), ece=ece_score(p, y),
                brier=float(((p - oh) ** 2).sum(1).mean()),
                nll=float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-8, 1)).mean()))


@torch.no_grad()
def predict(model, loader, tta=True):
    """Returns p (class probabilities), unc [N,5] = native/total/vacuity/aleatoric/epistemic, q, risk (gate), y."""
    model.eval()
    res = {k: [] for k in ("p", "unc", "q", "risk", "y")}
    for xb, yb, _ in loader:
        x = to_input(xb)
        outs = []
        for v in ([x, x.flip(-1)] if tta else [x]):
            with torch.autocast(device_type=DEV.type, dtype=torch.float16, enabled=USE_AMP):
                outs.append(model(v))
        avg = lambda vals: torch.stack(vals).mean(0).cpu()
        res["p"].append(avg([o["p"].float() for o in outs]))
        res["unc"].append(avg([o["unc"].float() for o in outs]))
        res["q"].append(avg([o["qual"].float() for o in outs]))
        res["risk"].append(avg([torch.sigmoid(o["gate_logit"].float()) for o in outs]))
        res["y"].append(yb)
    return {k: torch.cat(v).numpy() for k, v in res.items()}


# %% ============ PART 3: TRAINING (resumable, variant by variant, fold by fold) ============
def atomic_save(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def lr_mults(epoch, it, spe):
    """returns (backbone_mult, head_mult)"""
    if epoch < EPOCHS_STAGE1:
        return 0.0, 1.0
    e2 = (epoch - EPOCHS_STAGE1) + it / spe
    cos = max(0.5 * (1 + math.cos(math.pi * e2 / EPOCHS_STAGE2)), 0.02)
    return min(1.0, 0.1 + 0.9 * e2) * cos, cos


def set_stage(model, stage):
    for p in model.backbone.parameters():
        p.requires_grad = stage == 2


def train_fold(vname, k, df):
    global STOP_REQUESTED
    V = VARIANTS[vname]
    fdir = RUNS / vname / f"fold{k}"
    fdir.mkdir(parents=True, exist_ok=True)
    if (fdir / "done.json").exists():
        log(f"{vname} fold {k}: already done"); return
    tv = df[df.role == "trainval"]
    tr, va = tv[tv.fold != k], tv[tv.fold == k]
    te = df[df.role == "test"]
    log(f"{vname} fold {k}: train {len(tr)} | val {len(va)} | test {len(te)} | {V}")

    power = SAMPLER_POWER if V["oversample"] else 0.0
    cnt = tr.diagnosis.value_counts().to_dict()
    wts = tr.diagnosis.map(lambda c: (1.0 / cnt[c]) ** power).values
    n_draw = len(tr) if SAMPLES_PER_EPOCH is None else min(len(tr), SAMPLES_PER_EPOCH)
    sampler = WeightedRandomSampler(torch.tensor(wts, dtype=torch.double), num_samples=n_draw, replacement=True)
    va_sub = va if len(va) <= VAL_MAX_PER_EPOCH else va.sample(VAL_MAX_PER_EPOCH, random_state=SEED)
    tl, vl = make_loader(tr, True, sampler=sampler), make_loader(va_sub)   # vl: per-epoch model selection only
    spe = len(tl)
    use_v2 = V["consist"] and V["spatial"] != "gap"

    model = Net(V).to(DEV).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW([
        dict(params=model.backbone.parameters(), lr=LR_BACKBONE),
        dict(params=model.head_params(), lr=LR_HEAD)], weight_decay=WD)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    start, best, best_ep, bad, hist = 0, -9.0, -1, 0, []
    if (fdir / "last.pt").exists():
        ck = torch.load(fdir / "last.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); scaler.load_state_dict(ck["scaler"])
        start, best, best_ep, bad, hist = ck["epoch"] + 1, ck["best"], ck["best_ep"], ck["bad"], ck["hist"]
        log(f"{vname} fold {k}: resumed at epoch {start}")

    total_epochs = EPOCHS_STAGE1 + EPOCHS_STAGE2
    ep_time, step_times = 0.0, {1: [], 2: []}
    for epoch in range(start, total_epochs):
        if elapsed_h() + 1.25 * ep_time / 3600 > SESSION_BUDGET_H:
            log("TIME GUARD: stopping cleanly. Save Version, then re-run on another day (state is saved).")
            STOP_REQUESTED = True
            return
        t_ep = time.time()
        stage = 1 if epoch < EPOCHS_STAGE1 else 2
        set_stage(model, stage)
        model.train()
        if stage == 1:
            model.backbone.eval()
        agg = {}
        for it, (xb, yb, qb) in enumerate(tl):
            t_s = time.time()
            bm, hm = lr_mults(epoch, it, spe)
            opt.param_groups[0]["lr"], opt.param_groups[1]["lr"] = LR_BACKBONE * bm, LR_HEAD * hm
            x, y, ql = to_input(xb), yb.to(DEV), qb.to(DEV)
            if use_v2:
                n = min(CONS_N, x.shape[0])
                x2 = x[:n].flip(-1) * (1 + 0.15 * (torch.rand(n, 1, 1, 1, device=DEV) * 2 - 1)) \
                     + 0.1 * (torch.rand(n, 1, 1, 1, device=DEV) * 2 - 1) + 0.02 * torch.randn_like(x[:n])
                inp = torch.cat([x, x2.contiguous(memory_format=torch.channels_last)])
            else:
                inp = x
            with torch.autocast(device_type=DEV.type, dtype=torch.float16, enabled=USE_AMP):
                out = model(inp, backbone_grad=(stage == 2))
            loss, parts = compute_losses(out, y, ql, epoch, V)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt); scaler.update()
            for kk, vv in parts.items():
                agg[kk] = agg.get(kk, 0) + vv / spe
            if DEV.type == "cuda": torch.cuda.synchronize()
            if it >= 1: step_times[stage].append(time.time() - t_s)
        va_pred = predict(model, vl, tta=False)
        m = metrics(va_pred["p"], va_pred["y"])
        ep_time = time.time() - t_ep
        hist.append(dict(epoch=epoch, stage=stage, **{f"loss_{a}": b for a, b in agg.items()},
                         **{f"val_{a}": b for a, b in m.items()}, sec=ep_time))
        log(f"{vname} f{k} ep {epoch + 1}/{total_epochs} (stage {stage}) | " + " ".join(f"{a}={b:.3f}" for a, b in agg.items())
            + f" | val QWK {m['qwk']:.4f} acc {m['acc']:.3f} ECE {m['ece']:.3f} | {ep_time:.0f}s")
        if m["qwk"] > best:
            best, best_ep, bad = m["qwk"], epoch, 0
            atomic_save(model.state_dict(), fdir / "best.pt")
        elif stage == 2:
            bad += 1
        atomic_save(dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), epoch=epoch,
                         best=best, best_ep=best_ep, bad=bad, hist=hist), fdir / "last.pt")
        if stage == 2 and bad >= PATIENCE:
            log(f"{vname} fold {k}: early stopping (best epoch {best_ep + 1}, QWK {best:.4f})"); break

    if SMOKE_TEST and step_times[1] and step_times[2]:
        n_real = N_TOTAL * (1 - TEST_FRAC) * (1 - 1 / REAL_N_FOLDS)
        full_steps = int(min(n_real, SAMPLES_PER_EPOCH or n_real) / BATCH)
        est = (PROTO_E1 * np.mean(step_times[1]) + PROTO_E2 * np.mean(step_times[2])) * full_steps / 3600
        SMOKE_EST[vname] = est
        log(f"TIME ESTIMATE (rough): {vname} ~{est:.2f} h per fold at full size (excludes per-epoch validation, final predictions and early stopping)")

    # ---- final predictions with the best checkpoint ----
    model.load_state_dict(torch.load(fdir / "best.pt", map_location=DEV))
    pv, pt = predict(model, make_loader(va)), predict(model, make_loader(te))
    po = predict(model, make_loader(te, ood=True))
    np.savez_compressed(fdir / "preds.npz", **{f"val_{a}": b for a, b in pv.items()},
                        **{f"test_{a}": b for a, b in pt.items()}, **{f"ood_{a}": b for a, b in po.items()},
                        val_ids=va.id_code.values, test_ids=te.id_code.values)
    json.dump(dict(best_epoch=best_ep + 1, best_val_qwk=best, history=hist), open(fdir / "done.json", "w"))
    (fdir / "last.pt").unlink(missing_ok=True)
    log(f"{vname} fold {k} finished. best val QWK {best:.4f} | test QWK {metrics(pt['p'], pt['y'])['qwk']:.4f}")
    del model, opt
    torch.cuda.empty_cache()


# %% ============ PART 4: ENSEMBLES, CALIBRATION, ABSTENTION, BASELINE + ABLATION COMPARISON ============
def temperature_fit(p, y):
    best_T, best_nll = 1.0, 1e9
    for T in np.linspace(0.5, 3.0, 51):
        q = p ** (1 / T); q /= q.sum(1, keepdims=True)
        nll = -np.log(np.clip(q[np.arange(len(y)), y], 1e-8, 1)).mean()
        if nll < best_nll: best_T, best_nll = T, nll
    return float(best_T)


def entropy(p):
    return -(p * np.log(np.clip(p, 1e-8, 1))).sum(1)


def selective_curve(p, y, risk, covs=np.linspace(0.3, 1.0, 15)):
    order = np.argsort(risk)
    rows = []
    for c in covs:
        keep = order[: max(int(round(c * len(y))), 5)]
        pr = p[keep].argmax(1)
        rows.append(dict(coverage=float(c), acc=float((pr == y[keep]).mean()),
                         qwk=float(cohen_kappa_score(y[keep], pr, weights="quadratic")),
                         risk=float(1 - (pr == y[keep]).mean())))
    return rows


def load_variant(name, folds):
    out = {}
    for k in folds:
        f = RUNS / name / f"fold{k}" / "preds.npz"
        if (RUNS / name / f"fold{k}" / "done.json").exists() and f.exists():
            out[k] = dict(np.load(f, allow_pickle=True))
    return out


def build_ensemble(V, P, members, T, split):
    """Uncertainty-weighted ensemble over fold models (uniform weights for non-evidential variants)."""
    pf = np.stack([(P[k][f"{split}_p"] ** (1 / T[k])) for k in members])
    pf = pf / pf.sum(-1, keepdims=True)                                         # [M,N,K]
    unc = np.stack([P[k][f"{split}_unc"] for k in members])                     # [M,N,5]
    if V["head"] in EVID_HEADS:
        w = np.exp(-unc[..., 2] / 0.5)
    else:
        w = np.ones(unc.shape[:2])
    w = w / w.sum(0, keepdims=True)
    pe = (w[..., None] * pf).sum(0)
    tot, ale = entropy(pe), (w * unc[..., 3]).sum(0)
    epi = np.clip(tot - ale, 0, None)
    vac = unc[..., 2].mean(0)
    native = vac if V["head"] in EVID_HEADS else (epi if V["head"] == "mcdrop" else tot)
    return dict(p=pe, native=native, total=tot, vacuity=vac, aleatoric=ale, epistemic=epi,
                q=np.stack([P[k][f"{split}_q"] for k in members]).mean(0),
                gate=np.stack([P[k][f"{split}_risk"] for k in members]).mean(0), conf=pe.max(1))


def eval_variant(name, P, df):
    V = VARIANTS[name]
    val_qwk = {k: metrics(P[k]["val_p"], P[k]["val_y"])["qwk"] for k in P}
    members = sorted(P, key=lambda k: -val_qwk[k])[:TOPK_ENSEMBLE]
    T = {k: temperature_fit(P[k]["val_p"], P[k]["val_y"]) for k in members}
    yt = P[members[0]]["test_y"]
    E, Eo = build_ensemble(V, P, members, T, "test"), build_ensemble(V, P, members, T, "ood")
    single = [metrics(P[k]["test_p"], yt) for k in members]
    res = dict(variant=name, config=V, members=[int(k) for k in members], temperatures=T,
               val_qwk_per_fold={int(k): v for k, v in val_qwk.items()},
               ensemble_test=metrics(E["p"], yt),
               single_fold_mean={a: float(np.mean([s[a] for s in single])) for a in single[0]},
               single_fold_std={a: float(np.std([s[a] for s in single])) for a in single[0]})
    risks = {"native": E["native"], "entropy": E["total"], "1-max_prob": 1 - E["conf"],
             "epistemic": E["epistemic"], "learned_gate": E["gate"]}
    if V["quality"]:
        risks["1-quality"] = 1 - E["q"]
    res["selective"] = {n: selective_curve(E["p"], yt, r) for n, r in risks.items()}
    res["aurc"] = {n: float(np.mean([r_["risk"] for r_ in c])) for n, c in res["selective"].items()}
    res["at90"] = {n: selective_curve(E["p"], yt, r, covs=[0.9])[0] for n, r in risks.items()}
    # deployment threshold chosen on out-of-fold validation data (no test leakage)
    oof_risk = np.concatenate([P[k]["val_risk"] for k in members])
    tau = float(np.quantile(oof_risk, TARGET_COVERAGE))
    keep = E["gate"] <= tau
    res["abstention"] = dict(threshold=tau, coverage_test=float(keep.mean()),
                             retained_metrics=metrics(E["p"][keep], yt[keep]) if keep.sum() > 10 else {},
                             referred_per_grade={int(g): float((~keep)[yt == g].mean()) for g in range(N_CLASSES) if (yt == g).any()})
    lab = np.r_[np.zeros(len(yt)), np.ones(len(yt))]
    res["ood_auroc_synthetic"] = {n: float(roc_auc_score(lab, np.r_[a, b])) for n, a, b in
                                  [("native", E["native"], Eo["native"]), ("entropy", E["total"], Eo["total"]),
                                   ("learned_gate", E["gate"], Eo["gate"])]}
    ok = E["p"].argmax(1) == yt
    mean_or_none = lambda a: float(a.mean()) if len(a) else None
    res["uncertainty_by_correctness"] = {n: dict(correct=mean_or_none(E[n][ok]), wrong=mean_or_none(E[n][~ok]))
                                         for n in ("aleatoric", "epistemic", "vacuity")}
    qte = df[df.role == "test"].set_index("id_code").loc[P[members[0]]["test_ids"]].q_label.values
    res["quality_proxy_auroc"] = float(roc_auc_score(qte, E["q"])) if V["quality"] and len(set(qte)) > 1 else None
    return res, E, yt, members


def bootstrap_qwk(preds, yt, ref, n_boot=500, seed=0):
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(yt), (n_boot, len(yt)))
    q = {n: np.array([cohen_kappa_score(yt[i], p[i], weights="quadratic") for i in idx]) for n, p in preds.items()}
    out = {}
    for n in preds:
        d = q[n] - q[ref] if ref in q else np.zeros(n_boot)
        out[n] = dict(qwk_ci=[float(np.percentile(q[n], 2.5)), float(np.percentile(q[n], 97.5))],
                      delta_vs_ref=float(np.mean(d)), delta_ci=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                      p_value=float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))) if n != ref else None)
    return out


def run_eval(df):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    expected = list(range(N_FOLDS)) if MODE == "main" else list(STUDY_FOLDS)
    (STATE / "results").mkdir(exist_ok=True)
    R, ENS, yt_all = {}, {}, None
    for name in VARIANTS_TO_RUN:
        P = load_variant(name, expected)
        if not P or (len(P) < len(expected) and not ALLOW_PARTIAL_EVAL):
            log(f"eval: skipping {name} (folds done: {sorted(P)}; needed {expected})"); continue
        res, E, yt, members = eval_variant(name, P, df)
        R[name], ENS[name] = res, (E, members, P)
        yt_all = yt
        json.dump(res, open(STATE / "results" / f"{name}.json", "w"), indent=2)
        log(f"{name}: ensemble test QWK {res['ensemble_test']['qwk']:.4f} acc {res['ensemble_test']['acc']:.4f} "
            f"ECE {res['ensemble_test']['ece']:.4f} | AURC native {res['aurc']['native']:.4f} gate {res['aurc']['learned_gate']:.4f}")
    if not R:
        log("eval: nothing to evaluate yet"); return

    # ---------------- comparison table (baselines + ablations) ----------------
    ref = "qlord" if "qlord" in R else next(iter(R))
    boot = bootstrap_qwk({n: ENS[n][0]["p"].argmax(1) for n in R}, yt_all, ref)
    rows = []
    for n, r in R.items():
        e, b = r["ensemble_test"], boot[n]
        rows.append(dict(variant=n, group="proposed" if n == "qlord" else n.split("_")[0], folds=len(r["members"]),
                         QWK=e["qwk"], QWK_CI_lo=b["qwk_ci"][0], QWK_CI_hi=b["qwk_ci"][1],
                         dQWK_vs_qlord=b["delta_vs_ref"], p_vs_qlord=b["p_value"], acc=e["acc"], macro_f1=e["macro_f1"],
                         ECE=e["ece"], Brier=e["brier"], NLL=e["nll"],
                         QWK_single_mean=r["single_fold_mean"]["qwk"], QWK_single_std=r["single_fold_std"]["qwk"],
                         AURC_native=r["aurc"]["native"], AURC_entropy=r["aurc"]["entropy"], AURC_gate=r["aurc"]["learned_gate"],
                         QWK_at90cov_native=r["at90"]["native"]["qwk"], QWK_at90cov_gate=r["at90"]["learned_gate"]["qwk"],
                         OOD_AUROC_native=r["ood_auroc_synthetic"]["native"]))
    tab = pd.DataFrame(rows)
    tab.to_csv(STATE / "comparison_table.csv", index=False)
    json.dump(boot, open(STATE / "bootstrap.json", "w"), indent=2)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print("\n=== COMPARISON (test set, fold-ensemble; reference =", ref, ") ===")
        print(tab[["variant", "group", "QWK", "QWK_CI_lo", "QWK_CI_hi", "dQWK_vs_qlord", "p_vs_qlord", "acc", "macro_f1", "ECE", "Brier"]].to_string(index=False))
        print("\n=== SELECTIVE PREDICTION / OOD ===")
        print(tab[["variant", "AURC_native", "AURC_entropy", "AURC_gate", "QWK_at90cov_native", "QWK_at90cov_gate", "OOD_AUROC_native"]].to_string(index=False))

    # ---------------- figures ----------------
    if len(R) > 1:
        fig, ax = plt.subplots(1, 2, figsize=(13, 0.45 * len(R) + 2.5))
        t2 = tab.iloc[::-1]
        ax[0].barh(t2.variant, t2.QWK, xerr=[t2.QWK - t2.QWK_CI_lo, t2.QWK_CI_hi - t2.QWK], color="#5b8def")
        ax[0].set(xlabel="QWK (95% bootstrap CI)", xlim=(max(0, t2.QWK_CI_lo.min() - 0.03), min(1, t2.QWK_CI_hi.max() + 0.02)))
        ax[1].barh(t2.variant, t2.AURC_native, color="#e08a3c"); ax[1].set(xlabel="AURC with native uncertainty (lower = better)")
        plt.tight_layout(); plt.savefig(FIG / "comparison_bars.png", dpi=130); plt.close()
        plt.figure(figsize=(7, 5))
        for n, r in R.items():
            c = r["selective"]["native"]
            plt.plot([x["coverage"] for x in c], [x["risk"] for x in c], label=n, lw=2.5 if n == ref else 1)
        plt.xlabel("coverage"); plt.ylabel("selective risk (1-acc)"); plt.legend(fontsize=6); plt.title("Selective risk, native uncertainty")
        plt.tight_layout(); plt.savefig(FIG / "comparison_selective_risk.png", dpi=130); plt.close()

    if "qlord" in R:
        E, members, P = ENS["qlord"]
        r = R["qlord"]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for n, c in r["selective"].items():
            ax[0].plot([x["coverage"] for x in c], [x["qwk"] for x in c], marker="o", ms=3, label=n)
            ax[1].plot([x["coverage"] for x in c], [x["risk"] for x in c], marker="o", ms=3, label=n)
        ax[0].set(xlabel="coverage (fraction kept)", ylabel="QWK on kept", title="Q-LORD retention curve"); ax[0].legend(fontsize=7)
        ax[1].set(xlabel="coverage", ylabel="selective risk (1-acc)", title="Selective risk (abstention-gate ablation = compare curves)")
        plt.tight_layout(); plt.savefig(FIG / "qlord_retention_curves.png", dpi=130); plt.close()
        conf, pred = E["p"].max(1), E["p"].argmax(1)
        bins = np.linspace(0, 1, 11); xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum(): xs.append(conf[m].mean()); ys.append((pred[m] == yt_all[m]).mean())
        plt.figure(figsize=(4.5, 4.5)); plt.plot([0, 1], [0, 1], "k--"); plt.plot(xs, ys, "o-")
        plt.xlabel("confidence"); plt.ylabel("accuracy"); plt.title(f"Reliability (ECE {r['ensemble_test']['ece']:.3f})")
        plt.tight_layout(); plt.savefig(FIG / "qlord_reliability.png", dpi=130); plt.close()
        cm = confusion_matrix(yt_all, pred, labels=range(N_CLASSES))
        plt.figure(figsize=(4.5, 4)); plt.imshow(cm, cmap="Blues")
        for i in range(N_CLASSES):
            for j in range(N_CLASSES): plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
        plt.xlabel("predicted"); plt.ylabel("true"); plt.title("Q-LORD confusion matrix (test)")
        plt.tight_layout(); plt.savefig(FIG / "qlord_confusion.png", dpi=130); plt.close()
        pd.DataFrame(dict(id_code=P[members[0]]["test_ids"], y=yt_all, pred=pred, conf=conf, quality=E["q"],
                          gate_risk=E["gate"], vacuity=E["vacuity"], aleatoric=E["aleatoric"], epistemic=E["epistemic"],
                          abstain=E["gate"] > r["abstention"]["threshold"])).to_csv(STATE / "qlord_test_predictions.csv", index=False)
        try:   # lesion-attention overlays from the best fold's model
            model = Net(VARIANTS["qlord"], pretrained=False).to(DEV).to(memory_format=torch.channels_last)
            model.load_state_dict(torch.load(RUNS / "qlord" / f"fold{members[0]}" / "best.pt", map_location=DEV)); model.eval()
            te = df[df.role == "test"].reset_index(drop=True)
            pick = pd.concat([te[te.diagnosis == g].head(2) for g in range(N_CLASSES)]).head(10)
            fig, axs = plt.subplots(2, len(pick), figsize=(2.2 * len(pick), 4.6))
            for j, rr in enumerate(pick.itertuples()):
                img = cv2.cvtColor(cv2.imread(str(CACHE / f"{rr.id_code}.jpg")), cv2.COLOR_BGR2RGB)
                with torch.no_grad(), torch.autocast(device_type=DEV.type, dtype=torch.float16, enabled=USE_AMP):
                    o = model(to_input(torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)))
                H, W = o["hw"]
                a = o["attn"][0].float().max(0).values.view(H, W).cpu().numpy()
                a = cv2.resize((a - a.min()) / (a.max() - a.min() + 1e-8), (img.shape[1], img.shape[0]))
                axs[0, j].imshow(img); axs[0, j].set_title(f"grade {rr.diagnosis}", fontsize=8)
                axs[1, j].imshow(img); axs[1, j].imshow(a, cmap="jet", alpha=0.45)
                for r_ in (0, 1): axs[r_, j].axis("off")
            plt.tight_layout(); plt.savefig(FIG / "qlord_lesion_attention.png", dpi=130); plt.close()
        except Exception as e:
            log("attention figure skipped:", e)
    log("eval finished. Everything is in", STATE)


# %% ============ RUN ============
df_split = run_prep() if "prep" in RUN_PARTS else pd.read_csv(STATE / "split.csv")
if "train" in RUN_PARTS:
    for vname in VARIANTS_TO_RUN:
        for k in FOLDS_TO_RUN:
            if STOP_REQUESTED: break
            train_fold(vname, k, df_split)
if "eval" in RUN_PARTS and not STOP_REQUESTED:
    run_eval(df_split)
if SMOKE_TEST and SMOKE_EST:
    tot = sum(SMOKE_EST.values()) * REAL_FOLDS
    log(f"SMOKE-TEST TOTAL ESTIMATE for a real {MODE} run: ~{tot:.1f} h over {len(SMOKE_EST)} variant(s) x {REAL_FOLDS} fold(s) "
        f"(rough; split across days if > {SESSION_BUDGET_H:.0f} h)")
log("STOPPED EARLY - resume on another day." if STOP_REQUESTED else "All requested parts finished.")

