# -*- coding: utf-8 -*-
"""训练脚本。

学习率调度严格采用论文 §5.3 式(3)：

    lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))

含义：前 warmup_steps 步线性增大，之后按步数的平方根反比衰减。
"""

import os
import sys
import json
import math
import time
import csv

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import Config
from model import TinyTransformer, count_parameters

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE)


def load_data(cfg):
    text = open(cfg.corpus_file, encoding="utf-8").read()
    v = json.load(open(cfg.vocab_file, encoding="utf-8"))
    itos = {int(k): c for k, c in v["itos"].items()}
    stoi = {c: i for i, c in itos.items()}

    # 丢弃词表外的字符（覆盖率 99.996%，影响可忽略）
    ids = [stoi[c] for c in text if c in stoi]
    data = torch.tensor(ids, dtype=torch.long)

    n = int(len(data) * 0.9)
    return data[:n], data[n:], itos, stoi


def get_batch(data, cfg, split="train"):
    """随机取一个批次。输入 x 与目标 y 相差一个位置（自回归的语言建模目标）。"""
    T = cfg.seq_len
    max_start = len(data) - T - 1
    idx = torch.randint(0, max_start, (cfg.batch_size,))
    x = torch.stack([data[i:i + T] for i in idx])
    y = torch.stack([data[i + 1:i + T + 1] for i in idx])
    return x.to(cfg.device), y.to(cfg.device)


def lr_lambda_factory(cfg):
    """论文式(3)。

    注意：LambdaLR 实际学习率 = optimizer 的 base_lr × lambda(step)。
    因此这里让 lambda 直接返回论文公式的值，并把 optimizer 的 base_lr 设为 1.0，
    这样实际学习率就等于论文公式本身，不会被额外缩放。
    """
    scale = cfg.lr_scale / math.sqrt(cfg.d_model)

    def f(step):
        step = max(step, 1)
        return scale * min(step ** -0.5, step * cfg.warmup_steps ** -1.5)

    return f


@torch.no_grad()
def evaluate(model, data, cfg, lossf):
    model.eval()
    losses = []
    for _ in range(cfg.eval_batches):
        x, y = get_batch(data, cfg)
        logits = model(x)
        loss = lossf(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    avg = sum(losses) / len(losses)
    return avg, math.exp(avg)


def main():
    cfg = Config().check()
    torch.manual_seed(cfg.seed)

    if not torch.cuda.is_available():
        print("警告: CUDA 不可用，将使用 CPU（会慢很多）")
        cfg.device = "cpu"

    print("=" * 74)
    print("训练配置")
    print("=" * 74)
    print("  设备        : %s (%s)" % (
        cfg.device,
        torch.cuda.get_device_name(0) if cfg.device == "cuda" else "CPU"))
    print("  d_model=%d  n_heads=%d  d_k=%d  d_ff=%d  n_layers=%d" % (
        cfg.d_model, cfg.n_heads, cfg.d_k, cfg.d_ff, cfg.n_layers))
    print("  序列长度=%d  批次=%d  训练步数=%d" % (
        cfg.seq_len, cfg.batch_size, cfg.steps))
    print("  学习率预热步数=%d（论文 §5.3 式(3)）" % cfg.warmup_steps)

    train, val, itos, stoi = load_data(cfg)
    print("  训练/验证字符数: %s / %s" % (format(len(train), ","), format(len(val), ",")))

    model = TinyTransformer(cfg.vocab_size, cfg.d_model, cfg.n_heads, cfg.d_ff,
                            cfg.n_layers, cfg.max_len, cfg.dropout).to(cfg.device)
    n_params = count_parameters(model)
    print("  模型参数量  : %s" % format(n_params, ","))
    print()

    # base_lr 设为 1.0：学习率完全由论文式(3) 的 LambdaLR 决定，不做额外缩放
    opt = torch.optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda_factory(cfg))
    lossf = nn.CrossEntropyLoss()

    os.makedirs("results", exist_ok=True)
    logf = open(cfg.log_file, "w", newline="", encoding="utf-8")
    writer = csv.writer(logf)
    writer.writerow(["step", "train_loss", "val_loss", "val_ppl", "lr", "elapsed_s"])

    print("%-8s %-12s %-12s %-10s %-12s %s" % (
        "步", "训练loss", "验证loss", "验证PPL", "学习率", "耗时"))
    print("-" * 74)

    model.train()
    t0 = time.time()
    running = []
    for step in range(1, cfg.steps + 1):
        x, y = get_batch(train, cfg)
        logits = model(x)
        loss = lossf(logits.view(-1, logits.size(-1)), y.view(-1))

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()

        running.append(loss.item())

        if step % cfg.eval_every == 0 or step == 1:
            vl, vppl = evaluate(model, val, cfg, lossf)
            tl = sum(running) / len(running)
            lr = sched.get_last_lr()[0]
            el = time.time() - t0
            print("%-8d %-12.4f %-12.4f %-10.2f %-12.2e %.1f" % (
                step, tl, vl, vppl, lr, el))
            writer.writerow([step, tl, vl, vppl, lr, el])
            logf.flush()
            running = []

    logf.close()
    torch.save({
        "model_state": model.state_dict(),
        "config": cfg.__dict__,
        "n_params": n_params,
    }, cfg.ckpt_file)
    print("-" * 74)
    print("训练完成，模型已保存: %s" % cfg.ckpt_file)
    print("日志: %s" % cfg.log_file)
    print("参数量: %s" % format(n_params, ","))


if __name__ == "__main__":
    main()
