"""Draw original, reproducible diagrams for the guided learning route."""
from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "guided"
FONT = Path("/System/Library/Fonts/PingFang.ttc")
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({"font.size": 12, "axes.unicode_minus": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white",
                     "text.color": "#183044", "svg.fonttype": "path"})
BLUE, INK, LIGHT, GRAY = "#245e9a", "#183044", "#e3effa", "#d4dce4"


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=160, bbox_inches="tight", pad_inches=.25)
    plt.close(fig)


def matrix(ax, values, highlight=(), fmt="g"):
    rows, cols = len(values), len(values[0])
    ax.set(xlim=(-.15, cols+.15), ylim=(-.15, rows+.15), aspect="equal")
    ax.axis("off")
    for r, line in enumerate(values):
        for c, value in enumerate(line):
            active = (r, c) in highlight
            ax.add_patch(Rectangle((c, rows-r-1), 1, 1,
                                  facecolor=LIGHT if active else "#f6f8fa",
                                  edgecolor=BLUE if active else GRAY, lw=1.6))
            ax.text(c+.5, rows-r-.5, format(value, fmt), ha="center", va="center",
                    fontsize=16, color=INK)


def convolution():
    values = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    kernel = [[1, 0], [0, 1]]
    fig, axes = plt.subplots(1, 4, figsize=(12, 4.1))
    names = ["左上", "右上", "左下", "右下"]
    for index, ax in enumerate(axes):
        r, c = divmod(index, 2)
        selected = {(r+i, c+j) for i in range(2) for j in range(2)}
        total = sum(values[r+i][c+j]*kernel[i][j] for i in range(2) for j in range(2))
        matrix(ax, values, selected)
        ax.set_title(f"{names[index]}窗口", fontsize=14, pad=12)
        ax.text(.5, -.12, f"输出 {total}", transform=ax.transAxes, ha="center",
                va="top", fontsize=14, weight="bold")
    fig.suptitle("移动的是窗口，共享的是同一组权重", fontsize=19, weight="bold", y=.99)
    fig.text(.5, .82, "权重 [[1, 0], [0, 1]] · 步长 1 · 无填充 · 偏置 0", ha="center")
    fig.text(.5, .035, "按位置排好输出：第一行 [6, 8]，第二行 [12, 14]", ha="center")
    fig.subplots_adjust(top=.74, bottom=.22, wspace=.3)
    save(fig, "convolution-windows")


def cnn():
    stages = [("输入图像", "3 × 32 × 32"), ("卷积 + ReLU", "16 × 32 × 32"),
              ("最大池化", "16 × 16 × 16"), ("卷积 + ReLU", "32 × 16 × 16"),
              ("最大池化", "32 × 8 × 8"), ("全局平均池化", "32 个特征"),
              ("全连接分类层", "10 个类别分数")]
    fig, ax = plt.subplots(figsize=(8.6, 9.2))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    fig.suptitle("一个完整 CNN：逐步追踪形状", fontsize=20, weight="bold", y=.98)
    ys = [.90-i*.13 for i in range(len(stages))]
    for index, ((label, shape), y) in enumerate(zip(stages, ys)):
        ax.add_patch(FancyBboxPatch((.12, y-.045), .76, .09,
                     boxstyle="round,pad=0.006", facecolor=LIGHT, edgecolor=GRAY, lw=1))
        ax.text(.16, y, label, ha="left", va="center", fontsize=14)
        ax.text(.84, y, shape, ha="right", va="center", fontsize=15, weight="bold")
        if index:
            ax.annotate("", xy=(.5, y+.052), xytext=(.5, ys[index-1]-.052),
                        arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 1.8})
    ax.text(.5, .02, "卷积：3×3，P=1，S=1    池化：2×2，S=2\n单张图像省略批次维度；ReLU 不改变形状。",
            ha="center", va="center", fontsize=12, linespacing=1.8)
    save(fig, "cnn-shapes")


def attention():
    s = 1 / math.sqrt(2)
    a = math.exp(s)/(math.exp(s)+1)
    scores = [[s, 0.], [0., s]]
    weights = [[a, 1-a], [1-a, a]]
    output = [[2*a, 4*(1-a)], [2*(1-a), 4*a]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
    for ax, title, vals in zip(axes,
                ["① 缩放后的匹配分数", "② 每行 Softmax", "③ 权重乘 V"],
                [scores, weights, output]):
        matrix(ax, vals, {(0, 0), (0, 1)}, ".4f")
        ax.set_title(title, fontsize=14, pad=13)
        ax.text(.5, -.12, "行 1：token 1    行 2：token 2", transform=ax.transAxes,
                ha="center", va="top", fontsize=11)
    fig.suptitle("两次查询，各自产生一个输出向量", fontsize=20, weight="bold", y=.98)
    fig.text(.5, .82, "Q = K = [[1, 0], [0, 1]]    V = [[2, 0], [0, 4]]    d_k = 2", ha="center")
    fig.text(.5, .055, "高亮第一行：约 0.6698 × [2, 0] + 0.3302 × [0, 4] = [1.3395, 1.3210]", ha="center")
    fig.subplots_adjust(top=.72, bottom=.22, wspace=.35)
    save(fig, "attention-two-tokens")


if __name__ == "__main__":
    convolution()
    cnn()
    attention()
    print(f"Saved 3 original diagrams (PNG + SVG): {OUT}")
