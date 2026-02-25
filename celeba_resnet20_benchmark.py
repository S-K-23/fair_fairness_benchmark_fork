#!/usr/bin/env python
"""
CelebA + ResNet-20 Fairness Benchmark
Self-contained script for running fairness algorithms on CelebA with ResNet-20.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn import metrics
from pytorchcv.model_provider import get_model as ptcv_get_model

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# ============================================================================
# ResNet-20 Encoder
# ============================================================================

class Identity(nn.Module):
    def forward(self, x):
        return x

class ResNet20Encoder(nn.Module):
    """ResNet-20 encoder using pytorchcv CIFAR model."""

    def __init__(self, pretrained=False, n_hidden=64):
        super().__init__()
        # Load ResNet-20 CIFAR model
        self.resnet = ptcv_get_model("resnet20_cifar10", pretrained=pretrained)
        # Replace final layer to get features
        self.resnet.output = Identity()
        # Binary classification head
        self.fc = nn.Linear(n_hidden, 1)

    def forward(self, x):
        h = self.resnet(x)  # Hidden features
        out = self.fc(h)
        out = torch.sigmoid(out)
        return h, out

# ============================================================================
# Fairness Loss Functions
# ============================================================================

class DiffDP(nn.Module):
    """Demographic Parity loss."""
    def forward(self, y_pred, s):
        y_pred = y_pred.reshape(-1)
        s = s.reshape(-1)
        y0 = y_pred[s == 0]
        y1 = y_pred[s == 1]
        if len(y0) == 0 or len(y1) == 0:
            return torch.tensor(0.0, device=y_pred.device)
        return torch.abs(torch.mean(y0) - torch.mean(y1))

class DiffEOpp(nn.Module):
    """Equal Opportunity loss."""
    def forward(self, y_pred, s, y_gt):
        y_pred = y_pred.reshape(-1)
        s = s.reshape(-1)
        y_gt = y_gt.reshape(-1)

        y_pred = y_pred[y_gt == 1]
        s = s[y_gt == 1]

        y0 = y_pred[s == 0]
        y1 = y_pred[s == 1]
        if len(y0) == 0 or len(y1) == 0:
            return torch.tensor(0.0, device=y_pred.device)
        return torch.abs(torch.mean(y0) - torch.mean(y1))

class DiffEOdd(nn.Module):
    """Equalized Odds loss."""
    def forward(self, y_pred, s, y_gt):
        y_pred = y_pred.reshape(-1)
        s = s.reshape(-1)
        y_gt = y_gt.reshape(-1)

        # For y=1
        y_pred_y1 = y_pred[y_gt == 1]
        s_y1 = s[y_gt == 1]
        y0 = y_pred_y1[s_y1 == 0]
        y1 = y_pred_y1[s_y1 == 1]
        if len(y0) == 0 or len(y1) == 0:
            reg_loss_y1 = torch.tensor(0.0, device=y_pred.device)
        else:
            reg_loss_y1 = torch.abs(torch.mean(y0) - torch.mean(y1))

        # For y=0
        y_pred_y0 = y_pred[y_gt == 0]
        s_y0 = s[y_gt == 0]
        y0 = y_pred_y0[s_y0 == 0]
        y1 = y_pred_y0[s_y0 == 1]
        if len(y0) == 0 or len(y1) == 0:
            reg_loss_y0 = torch.tensor(0.0, device=y_pred.device)
        else:
            reg_loss_y0 = torch.abs(torch.mean(y0) - torch.mean(y1))

        return reg_loss_y1 + reg_loss_y0

def pairwise_distances(x):
    instances_norm = torch.sum(x**2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()

def GaussianKernelMatrix(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ / sigma)

class HSIC(nn.Module):
    """Hilbert-Schmidt Independence Criterion loss."""
    def __init__(self, s_x=1, s_y=1):
        super().__init__()
        self.s_x = s_x
        self.s_y = s_y

    def forward(self, x, y):
        device = x.device
        m = x.shape[0]
        K = GaussianKernelMatrix(x, self.s_x).to(device)
        L = GaussianKernelMatrix(y, self.s_y).to(device)
        H = (torch.eye(m) - 1.0 / m * torch.ones((m, m))).to(device)
        hsic = torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H)))) / ((m - 1) ** 2)
        return hsic

class PRLoss(nn.Module):
    """P-Rule / Mutual Information loss."""
    def forward(self, y_pred, s):
        device = y_pred.device
        output_f = y_pred[s == 0]
        output_m = y_pred[s == 1]

        if len(output_f) == 0 or len(output_m) == 0:
            return torch.tensor(0.0, device=device)

        N_female = torch.tensor(output_f.shape[0], dtype=torch.float, device=device)
        N_male = torch.tensor(output_m.shape[0], dtype=torch.float, device=device)
        Dxisi = torch.stack((N_male, N_female), axis=0)

        y_pred_female = torch.sum(output_f)
        y_pred_male = torch.sum(output_m)
        P_ys = torch.stack((y_pred_male, y_pred_female), axis=0) / Dxisi

        P = torch.cat((output_f, output_m), 0)
        P_y = torch.sum(P) / y_pred.shape[0]

        # Clamp to avoid log(0)
        eps = 1e-8
        P_ys = torch.clamp(P_ys, eps, 1 - eps)
        P_y = torch.clamp(P_y, eps, 1 - eps)

        P_s1y1 = torch.log(P_ys[1]) - torch.log(P_y)
        P_s1y0 = torch.log(1 - P_ys[1]) - torch.log(1 - P_y)
        P_s0y1 = torch.log(P_ys[0]) - torch.log(P_y)
        P_s0y0 = torch.log(1 - P_ys[0]) - torch.log(1 - P_y)

        PI_s1y1 = output_f * P_s1y1
        PI_s1y0 = (1 - output_f) * P_s1y0
        PI_s0y1 = output_m * P_s0y1
        PI_s0y0 = (1 - output_m) * P_s0y0

        PI = torch.sum(PI_s1y1) + torch.sum(PI_s1y0) + torch.sum(PI_s0y1) + torch.sum(PI_s0y0)
        return PI

# ============================================================================
# Metrics
# ============================================================================

def demographic_parity(y_pred, sensitive_attribute, threshold=0.5):
    y_z_1 = y_pred[sensitive_attribute == 1] > threshold if threshold else y_pred[sensitive_attribute == 1]
    y_z_0 = y_pred[sensitive_attribute == 0] > threshold if threshold else y_pred[sensitive_attribute == 0]
    if len(y_z_1) == 0 or len(y_z_0) == 0:
        return 0
    parity = abs(y_z_1.mean() - y_z_0.mean()) * 100
    return parity

def equal_opportunity(y_pred, y_gt, sensitive_attribute, threshold=0.5):
    y_pred = y_pred[y_gt == 1]
    sensitive_attribute = sensitive_attribute[y_gt == 1]
    y_z_1 = y_pred[sensitive_attribute == 1] > threshold if threshold else y_pred[sensitive_attribute == 1]
    y_z_0 = y_pred[sensitive_attribute == 0] > threshold if threshold else y_pred[sensitive_attribute == 0]
    if len(y_z_1) == 0 or len(y_z_0) == 0:
        return 0
    equality = abs(y_z_1.mean() - y_z_0.mean()) * 100
    return equality

def equalized_odds(y_pred, y_gt, sensitive_attribute, threshold=0.5):
    y_pred_all = y_pred.copy()
    sensitive_attribute_all = sensitive_attribute.copy()

    # TPR difference
    y_pred = y_pred_all[y_gt == 1]
    sensitive_attribute = sensitive_attribute_all[y_gt == 1]
    y_z_1 = y_pred[sensitive_attribute == 1] > threshold if threshold else y_pred[sensitive_attribute == 1]
    y_z_0 = y_pred[sensitive_attribute == 0] > threshold if threshold else y_pred[sensitive_attribute == 0]
    if len(y_z_1) == 0 or len(y_z_0) == 0:
        return 0
    equality = abs(y_z_1.mean() - y_z_0.mean())

    # FPR difference
    y_pred = y_pred_all[y_gt == 0]
    sensitive_attribute = sensitive_attribute_all[y_gt == 0]
    y_z_1 = y_pred[sensitive_attribute == 1] > threshold if threshold else y_pred[sensitive_attribute == 1]
    y_z_0 = y_pred[sensitive_attribute == 0] > threshold if threshold else y_pred[sensitive_attribute == 0]
    if len(y_z_1) == 0 or len(y_z_0) == 0:
        return 0
    equality += abs(y_z_1.mean() - y_z_0.mean())

    return equality * 100

def metric_evaluation(y_gt, y_pre, s, prefix=""):
    y_gt = y_gt.ravel()
    y_pre = y_pre.ravel()
    s = s.ravel()

    acc = metrics.accuracy_score(y_gt, y_pre > 0.5) * 100
    ap = metrics.average_precision_score(y_gt, y_pre) * 100
    try:
        auc = metrics.roc_auc_score(y_gt, y_pre) * 100
    except:
        auc = 0.0

    dp = demographic_parity(y_pre, s, threshold=0.5)
    eopp = equal_opportunity(y_pre, y_gt, s, threshold=0.5)
    eodd = equalized_odds(y_pre, y_gt, s, threshold=0.5)

    metric_name = ["acc", "ap", "auc", "dp", "eopp", "eodd"]
    metric_name = [prefix + "/" + x for x in metric_name]
    metric_val = [acc, ap, auc, dp, eopp, eodd]

    return dict(zip(metric_name, metric_val))

# ============================================================================
# Data Loading
# ============================================================================

def load_celeba_data(path="datasets/celeba/raw/"):
    df = pd.read_csv(os.path.join(path, "celeba.csv"), na_values="NA", index_col=None, sep=",", header=0)
    df['pixels'] = df['pixels'].apply(lambda x: np.array(x.split(), dtype="float32"))
    df['pixels'] = df['pixels'].apply(lambda x: x / 255)
    df['pixels'] = df['pixels'].apply(lambda x: np.reshape(x, (3, 48, 48)))

    X = df['pixels'].to_frame()
    df["Gender"] = df["Male"]
    attr = df[["Smiling", "Wavy_Hair", "Attractive", "Male", "Young"]]

    return X, attr

def InfiniteDataLoader(dataset, batch_size, shuffle=True, num_workers=0, pin_memory=False, drop_last=True):
    while True:
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                num_workers=num_workers, pin_memory=pin_memory, drop_last=drop_last)
        for data in data_loader:
            yield data

# ============================================================================
# Training
# ============================================================================

def train_step_erm(model, data, target, optimizer, scheduler, criterion, device):
    model.train()
    optimizer.zero_grad()
    _, output = model(data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item(), loss.item(), 0.0

def train_step_fairness(model, data, target, sensitive, optimizer, scheduler,
                        clf_criterion, fair_criterion, lam, device, method):
    model.train()
    optimizer.zero_grad()
    _, output = model(data)
    clf_loss = clf_criterion(output, target)

    if method == "diffdp":
        fair_loss = fair_criterion(output, sensitive)
    elif method == "diffeopp":
        fair_loss = fair_criterion(output, sensitive, target)
    elif method == "diffeodd":
        fair_loss = fair_criterion(output, sensitive, target)
    elif method == "hsic":
        fair_loss = fair_criterion(output, sensitive)
    elif method == "pr":
        fair_loss = fair_criterion(output, sensitive.reshape(-1))
    else:
        fair_loss = torch.tensor(0.0, device=device)

    loss = clf_loss + lam * fair_loss
    loss.backward()
    optimizer.step()
    scheduler.step()
    return loss.item(), clf_loss.item(), fair_loss.item()

def evaluate(model, data_loader, criterion, target_index, sensitive_index, device, prefix="test"):
    model.eval()
    target_hat_list = []
    target_list = []
    sensitive_list = []

    with torch.no_grad():
        for X, attr in data_loader:
            data = X.to(device)
            target = attr[:, target_index].to(device).unsqueeze(1).float()
            sensitive = attr[:, sensitive_index].to(device).unsqueeze(1).float()

            _, output = model(data)
            target_hat_list.append(output.cpu().numpy())
            target_list.append(target.cpu().numpy())
            sensitive_list.append(sensitive.cpu().numpy())

    target_hat_list = np.concatenate(target_hat_list, axis=0)
    target_list = np.concatenate(target_list, axis=0)
    sensitive_list = np.concatenate(sensitive_list, axis=0)

    return metric_evaluation(y_gt=target_list, y_pre=target_hat_list, s=sensitive_list, prefix=prefix)

# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CelebA + ResNet-20 Fairness Benchmark")
    parser.add_argument("--method", type=str, default="erm",
                       choices=["erm", "diffdp", "diffeopp", "diffeodd", "hsic", "pr"],
                       help="Fairness method")
    parser.add_argument("--lam", type=float, default=1.0, help="Fairness lambda")
    parser.add_argument("--target_attr", type=str, default="Smiling",
                       choices=["Smiling", "Wavy_Hair", "Attractive"],
                       help="Target attribute")
    parser.add_argument("--sensitive_attr", type=str, default="Male",
                       choices=["Male", "Young"],
                       help="Sensitive attribute")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--data_path", type=str, default="datasets/celeba/raw/")
    args = parser.parse_args()

    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    print("Loading CelebA dataset...")
    X, attr = load_celeba_data(path=args.data_path)
    X_np = np.stack(X["pixels"].to_list())
    attr_np = attr.to_numpy()

    # Attribute indices
    attr_names = ["Smiling", "Wavy_Hair", "Attractive", "Male", "Young"]
    target_index = attr_names.index(args.target_attr)
    sensitive_index = attr_names.index(args.sensitive_attr)

    print(f"Target: {args.target_attr} (index {target_index})")
    print(f"Sensitive: {args.sensitive_attr} (index {sensitive_index})")

    # Train/val/test split
    X_train, X_testval, attr_train, attr_testval = train_test_split(
        X_np, attr_np, test_size=0.4, stratify=attr_np[:, target_index], random_state=args.seed)
    X_val, X_test, attr_val, attr_test = train_test_split(
        X_testval, attr_testval, test_size=0.5, stratify=attr_testval[:, target_index], random_state=args.seed)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Create datasets
    X_train = torch.from_numpy(X_train).float()
    X_val = torch.from_numpy(X_val).float()
    X_test = torch.from_numpy(X_test).float()
    attr_train = torch.from_numpy(attr_train).float()
    attr_val = torch.from_numpy(attr_val).float()
    attr_test = torch.from_numpy(attr_test).float()

    train_dataset = TensorDataset(X_train, attr_train)
    val_dataset = TensorDataset(X_val, attr_val)
    test_dataset = TensorDataset(X_test, attr_test)

    train_loader = InfiniteDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    # Model
    print("Creating ResNet-20 model...")
    model = ResNet20Encoder(pretrained=False).to(device)

    # Loss and optimizer
    clf_criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

    # Fairness criterion
    fair_criterion = None
    if args.method == "diffdp":
        fair_criterion = DiffDP()
    elif args.method == "diffeopp":
        fair_criterion = DiffEOpp()
    elif args.method == "diffeodd":
        fair_criterion = DiffEOdd()
    elif args.method == "hsic":
        fair_criterion = HSIC()
    elif args.method == "pr":
        fair_criterion = PRLoss()

    # Training
    print(f"\nTraining with method: {args.method}, lambda: {args.lam}")
    print("=" * 70)
    print(f"{'Step':>6} | {'Loss':>8} | {'Val Acc':>8} | {'Val DP':>8} | {'Val EOpp':>8} | {'Test Acc':>8}")
    print("=" * 70)

    for step, (X, attr) in enumerate(train_loader):
        if step >= args.num_steps:
            break

        X = X.to(device)
        y = attr[:, target_index].to(device).unsqueeze(1)
        s = attr[:, sensitive_index].to(device).unsqueeze(1)

        if args.method == "erm":
            loss, clf_loss, fair_loss = train_step_erm(
                model, X, y, optimizer, scheduler, clf_criterion, device)
        else:
            loss, clf_loss, fair_loss = train_step_fairness(
                model, X, y, s, optimizer, scheduler,
                clf_criterion, fair_criterion, args.lam, device, args.method)

        if step % args.log_freq == 0 or step == args.num_steps - 1:
            val_metrics = evaluate(model, val_loader, clf_criterion,
                                  target_index, sensitive_index, device, prefix="val")
            test_metrics = evaluate(model, test_loader, clf_criterion,
                                   target_index, sensitive_index, device, prefix="test")

            print(f"{step:>6} | {loss:>8.4f} | {val_metrics['val/acc']:>8.2f} | "
                  f"{val_metrics['val/dp']:>8.2f} | {val_metrics['val/eopp']:>8.2f} | "
                  f"{test_metrics['test/acc']:>8.2f}")

    # Final evaluation
    print("=" * 70)
    print("\nFinal Results:")
    print("-" * 50)

    val_metrics = evaluate(model, val_loader, clf_criterion,
                          target_index, sensitive_index, device, prefix="val")
    test_metrics = evaluate(model, test_loader, clf_criterion,
                           target_index, sensitive_index, device, prefix="test")

    print(f"Validation:")
    print(f"  Accuracy: {val_metrics['val/acc']:.2f}%")
    print(f"  AUC: {val_metrics['val/auc']:.2f}%")
    print(f"  Demographic Parity: {val_metrics['val/dp']:.2f}%")
    print(f"  Equal Opportunity: {val_metrics['val/eopp']:.2f}%")
    print(f"  Equalized Odds: {val_metrics['val/eodd']:.2f}%")

    print(f"\nTest:")
    print(f"  Accuracy: {test_metrics['test/acc']:.2f}%")
    print(f"  AUC: {test_metrics['test/auc']:.2f}%")
    print(f"  Demographic Parity: {test_metrics['test/dp']:.2f}%")
    print(f"  Equal Opportunity: {test_metrics['test/eopp']:.2f}%")
    print(f"  Equalized Odds: {test_metrics['test/eodd']:.2f}%")

    return test_metrics

if __name__ == "__main__":
    main()
