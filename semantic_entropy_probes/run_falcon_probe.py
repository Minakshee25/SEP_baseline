"""
Standalone Stage-4 probe training for the falcon-7b smoke run (single dataset).

The notebook `train-latent-probe.ipynb` is wired for the 4-dataset paper
experiment (OOD cross-dataset tests + multi-panel plots) and its plotting/OOD
cells crash on a single dataset. This script reproduces ONLY the in-distribution
core, copying the probe functions VERBATIM from the notebook (cells 8, 12, 13,
14, 16, 19, 23, 26) so the baseline logic is unchanged:

    load_dataset -> best universal split -> binarize_entropy
    -> per-layer LogisticRegression -> AUROC

It trains SEPs (predict binarized SE) and Acc. Pr. (predict correctness) at both
TBG and SLT token positions, then saves the per-layer models.

NOTE: falcon is a pipeline-validation vehicle, not the SEP-paper baseline
(that is Llama-2-7b-chat). These AUROCs are a sanity check, not paper-comparable.
N must be large enough that test splits keep both classes (N=20 crashes; use 400).
"""
import os
import pickle
import warnings

import numpy as np
import scipy
import torch
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)

run_files = {
    'UNC_MEA': 'uncertainty_measures.pkl',
    'VAL_GEN': 'validation_generations.pkl',
}

# ---- config (this run) ----
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_name = 'falcon-7b'
ds_names = ['trivia-qa']
ds_paths = [os.path.join(
    REPO,
    'semantic_uncertainty/mn1025/uncertainty/wandb/run-20260620_184816-9ddn5y2k/files',
)]
SAVE_DIR = os.path.join(REPO, 'semantic_entropy_probes/models')


# ===================== verbatim from notebook cell 8 =====================
class Dataset:
    def __init__(self, values):
        self.tbg_dataset = values[0]
        self.slt_dataset = values[1]
        self.entropy = values[2]
        self.accuracies = values[3]


def load_dataset(path, n_sample=2000):
    os.chdir(path)
    with open(run_files['VAL_GEN'], 'rb') as f:
        generations = pickle.load(f)
    with open(run_files['UNC_MEA'], 'rb') as g:
        measures = pickle.load(g)
    entropy = torch.tensor(measures['uncertainty_measures']['cluster_assignment_entropy']).to(torch.float32)
    accuracies = torch.tensor([record['most_likely_answer']['accuracy'] for record in generations.values()])
    tbg_dataset = torch.stack([record['most_likely_answer']['emb_last_tok_before_gen']
                               for record in generations.values()]).squeeze(-2).transpose(0, 1).to(torch.float32)
    slt_dataset = torch.stack([record['most_likely_answer']['emb_tok_before_eos']
                               for record in generations.values()]).squeeze(-2).transpose(0, 1).to(torch.float32)
    return (tbg_dataset[:, :n_sample, :], slt_dataset[:, :n_sample, :], entropy[:n_sample], accuracies[:n_sample])


# ===================== verbatim from notebook cell 12 =====================
def create_Xs_and_ys(datasets, scores, val_test_splits=[0.2, 0.1], test_only=False, no_val=False):
    X = np.array(datasets)
    y = np.array(scores)
    if test_only:
        X_tests, y_tests = [], []
        for i in range(X.shape[0]):
            X_tests.append(X[i])
            y_tests.append(y)
        return (None, None, X_tests, None, None, y_tests)
    valid_size = val_test_splits[0]
    test_size = val_test_splits[1]
    X_trains, X_vals, X_tests, y_trains, y_vals, y_tests = [], [], [], [], [], []
    for i in range(X.shape[0]):
        X_train_val, X_test, y_train_val, y_test = train_test_split(X[i], y, test_size=test_size, random_state=42)
        X_tests.append(X_test)
        y_tests.append(y_test)
        if no_val:
            X_trains.append(X_train_val)
            y_trains.append(y_train_val)
            continue
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=valid_size, random_state=42)
        X_trains.append(X_train)
        y_trains.append(y_train)
        X_vals.append(X_val)
        y_vals.append(y_val)
    return X_trains, X_vals, X_tests, y_trains, y_vals, y_tests


# ===================== verbatim from notebook cell 13 =====================
def bootstrap_func(y_true, y_score, func):
    y_tuple = (y_true, y_score)
    metric_i = func(*y_tuple)
    metric_dict = {}
    metric_dict['mean'] = metric_i
    metric_dict['bootstrap'] = compatible_bootstrap(func, rng)(*y_tuple)
    return metric_dict


def bootstrap(function, rng, n_resamples=1000):
    def inner(data):
        bs = scipy.stats.bootstrap(
            (data, ), function, n_resamples=n_resamples, confidence_level=0.9,
            random_state=rng)
        return {
            'std_err': bs.standard_error,
            'low': bs.confidence_interval.low,
            'high': bs.confidence_interval.high
        }
    return inner


def auroc(y_true, y_score):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    del thresholds
    return metrics.auc(fpr, tpr)


def compatible_bootstrap(func, rng):
    def helper(y_true_y_score):
        y_true = np.array([i['y_true'] for i in y_true_y_score])
        y_score = np.array([i['y_score'] for i in y_true_y_score])
        out = func(y_true, y_score)
        return out

    def wrap_inputs(y_true, y_score):
        return [{'y_true': i, 'y_score': j} for i, j in zip(y_true, y_score)]

    def converted_func(y_true, y_score):
        y_true_y_score = wrap_inputs(y_true, y_score)
        return bootstrap(helper, rng=rng)(y_true_y_score)
    return converted_func


# ===================== verbatim from notebook cell 14 =====================
def sklearn_train_and_evaluate(model, X_train, y_train, X_valid, y_valid, silent=False):
    model.fit(X_train, y_train)
    train_probs = model.predict_proba(X_train)
    train_loss = log_loss(y_train, train_probs)
    valid_preds = model.predict(X_valid)
    valid_probs = model.predict_proba(X_valid)
    valid_loss = log_loss(y_valid, valid_probs)
    val_accuracy = np.mean((valid_preds == y_valid).astype(int))
    auroc_score = roc_auc_score(y_valid, valid_probs[:, 1])
    if not silent:
        print(f"Validation Accuracy: {val_accuracy:.4f}, AUROC: {auroc_score:.4f}")
        print(f"Training Loss: {train_loss:.4f}, Validation Loss: {valid_loss:.4f}")


def sklearn_evaluate_on_test(model, X_test, y_test, silent=False, bootstrap=True):
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)
    test_loss = log_loss(y_test, test_probs)
    test_accuracy = np.mean((test_preds == y_test).astype(int))
    if bootstrap:
        auroc_score = bootstrap_func(y_test, test_probs[:, 1], auroc)
        auroc_score_scalar = auroc_score['mean']
    else:
        auroc_score = auroc_score_scalar = roc_auc_score(y_test, test_probs[:, 1])
    if not silent:
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, AUROC: {auroc_score_scalar:.4f}")
    return test_loss, test_accuracy, auroc_score


def train_single_metric(D, token_type='tbg', metric='b_entropy'):
    var_name = token_type[0] + metric[0]
    X_trains, X_vals, X_tests, y_trains, y_vals, y_tests = create_Xs_and_ys(
        getattr(D, f'{token_type}_dataset'), getattr(D, metric)
    )
    accs = []
    aucs = []
    models = []
    for i, (X_train, X_val, X_test, y_train, y_val, y_test) in enumerate(
            zip(X_trains, X_vals, X_tests, y_trains, y_vals, y_tests)):
        print(f"Training on {D.name}-{token_type.upper()}-{metric.upper()} {i+1}/{len(X_trains)}")
        model = LogisticRegression()
        sklearn_train_and_evaluate(model, X_train, y_train, X_val, y_val)
        test_loss, test_acc, test_auc = sklearn_evaluate_on_test(model, X_test, y_test)
        accs.append(test_acc)
        aucs.append(test_auc)
        models.append(model)
    setattr(D, f'{var_name}_accs', accs)
    setattr(D, f'{var_name}_aucs', aucs)
    setattr(D, f'{var_name}_models', models)


auc = lambda aucs: [ac['mean'] for ac in aucs]


# ===================== verbatim from notebook cell 16 =====================
def best_split(entropy: torch.Tensor, label="Dx"):
    ents = entropy.numpy()
    splits = np.linspace(1e-10, ents.max(), 100)
    split_mses = []
    for split in splits:
        low_idxs, high_idxs = ents < split, ents >= split
        low_mean = np.mean(ents[low_idxs])
        high_mean = np.mean(ents[high_idxs])
        mse = np.sum((ents[low_idxs] - low_mean)**2) + np.sum((ents[high_idxs] - high_mean)**2)
        mse = np.sum(mse)
        split_mses.append(mse)
    split_mses = np.array(split_mses)
    return splits[np.argmin(split_mses)]


def binarize_entropy(entropy, thres=0.0):
    binary_entropy = torch.full_like(entropy, -1, dtype=torch.float)
    binary_entropy[entropy < thres] = 0
    binary_entropy[entropy > thres] = 1
    return binary_entropy


# ===================== driver =====================
def main():
    # cell 9: load
    Ds = []
    for path in ds_paths:
        Ds.append(Dataset(load_dataset(path)))
    for i, D in enumerate(Ds):
        D.name = ds_names[i]
        D.path = ds_paths[i]

    n_layers = Ds[0].slt_dataset.shape[0]
    print(f"\nLoaded {len(Ds)} dataset(s). N={Ds[0].entropy.shape[0]} samples, "
          f"{n_layers} layers, hidden_dim={Ds[0].slt_dataset.shape[-1]}\n")

    # cell 19: best universal split + binarize
    all_entropy = torch.cat([D.entropy for D in Ds], dim=0)
    split = best_split(all_entropy, "All datasets collective")
    print(f"Best universal SE split: {split:.4f}")
    for D in Ds:
        D.b_entropy = binarize_entropy(D.entropy, split)
        dummy = max(torch.mean(D.b_entropy).item(), 1 - torch.mean(D.b_entropy).item())
        print(f"Dummy accuracy for {D.name}: {dummy:.4f}")

    # cell 23: SEP (binarized SE) probes
    for D in Ds:
        train_single_metric(D, 'tbg', 'b_entropy')
        train_single_metric(D, 'slt', 'b_entropy')

    # cell 26: Accuracy probes
    for D in Ds:
        train_single_metric(D, 'tbg', 'accuracies')
        train_single_metric(D, 'slt', 'accuracies')

    # ---- per-layer AUROC summary + best layer ----
    print("\n" + "=" * 60)
    print(f"PER-LAYER TEST AUROC SUMMARY (falcon-7b, {ds_names[0]}, "
          f"N={Ds[0].entropy.shape[0]}) -- pipeline sanity check, not paper-comparable")
    print("=" * 60)
    for D in Ds:
        for var_name, desc in [('tb', 'SEP   TBG'), ('sb', 'SEP   SLT'),
                               ('ta', 'AccPr TBG'), ('sa', 'AccPr SLT')]:
            aucs = np.array(auc(getattr(D, f'{var_name}_aucs')))
            best_layer = int(np.argmax(aucs))
            print(f"  {desc} | mean AUROC={np.nanmean(aucs):.3f} | "
                  f"best layer {best_layer} AUROC={aucs[best_layer]:.3f}")

    # ---- save per-layer probes (regenerates the throwaway smoke pkl cleanly) ----
    os.makedirs(SAVE_DIR, exist_ok=True)
    out = []
    for D in Ds:
        out.append({
            'name': D.name,
            'tb_models': D.tb_models, 'sb_models': D.sb_models,
            'ta_models': D.ta_models, 'sa_models': D.sa_models,
            'tb_aucs': auc(D.tb_aucs), 'sb_aucs': auc(D.sb_aucs),
            'ta_aucs': auc(D.ta_aucs), 'sa_aucs': auc(D.sa_aucs),
            'se_split': float(split),
        })
    save_path = os.path.join(SAVE_DIR, f'{model_name}_smoke_inference.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(tuple(out), f)
    print(f"\nSaved per-layer probes -> {save_path}")


if __name__ == '__main__':
    main()
