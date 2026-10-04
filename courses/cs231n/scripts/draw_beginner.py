"""Original beginner diagrams; no course slides or model output are reproduced."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets' / 'beginner'
OUT.mkdir(parents=True, exist_ok=True)
FONT = Path('/System/Library/Fonts/PingFang.ttc')
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({
    'font.size': 12, 'axes.unicode_minus': False,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
    'text.color': '#182e43', 'axes.titlecolor': '#182e43',
    'svg.fonttype': 'path',
})
INK = '#182e43'
BLUE = '#2563a6'
ORANGE = '#aa4e12'
GREEN = '#176a52'
GRAY = '#54697d'


def canvas(title, size=(11, 5.6)):
    fig, ax = plt.subplots(figsize=size)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    ax.set_title(title, fontsize=18, pad=24, weight='bold')
    return fig, ax


def box(ax, x, y, w, h, title, body='', color=BLUE, title_size=13):
    ax.add_patch(FancyBboxPatch((x-w/2, y-h/2), w, h,
                              boxstyle='round,pad=0.009',
                              fc=color+'0D', ec=color, lw=1.6))
    if body:
        ax.text(x, y+h*.18, title, ha='center', va='center',
                fontsize=title_size, weight='bold', color=color)
        ax.text(x, y-h*.20, body, ha='center', va='center',
                fontsize=11.5, linespacing=1.55)
    else:
        ax.text(x, y, title, ha='center', va='center',
                fontsize=title_size, color=INK, linespacing=1.5)


def arrow(ax, start, end, color=GRAY):
    ax.annotate('', xy=end, xytext=start,
                arrowprops={'arrowstyle': '->', 'color': color, 'lw': 1.8})


def save(fig, name):
    for ext in ('png', 'svg'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=170,
                    bbox_inches='tight', pad_inches=.25)
    plt.close(fig)


fig, ax = canvas('CS231n：先连通训练流程，再认识模型与应用', (11, 8))
stages = [
    (.88, '预备 + 第 1 章：输入与任务', '图片是数字数组；先明确需要输出什么', BLUE),
    (.69, '第 2–4 章：训练的共同基础', '预测 → 损失 → 梯度 → 参数更新', ORANGE),
    (.50, '第 5–10 章：模型与结构化输出', '空间局部、序列、注意力、位置与时间', BLUE),
    (.31, '第 11–14 章：计算资源、监督与生成', '多卡计算、自动构造目标、从分布采样', GREEN),
    (.12, '第 15–18 章：空间、语言、行动与人', '三维表示、图文关联、感知行动闭环、真实需求', BLUE),
]
for y, title, body, color in stages:
    box(ax, .5, y, .91, .135, title, body, color)
for upper, lower in zip(stages, stages[1:]):
    arrow(ax, (.5, upper[0]-.077), (.5, lower[0]+.077))
save(fig, 'course-roadmap')


fig, ax = canvas('广播：每行一个值 + 每列一个值 → 一张结果表', (10.4, 5.6))
ax.text(.12, .88, 'test_sq\n形状 (2, 1)', ha='center', va='center', color=BLUE)
ax.text(.60, .97, 'train_sq：形状 (3,) → 广播时按 (1, 3) 对齐',
        ha='center', va='center', fontsize=12, color=ORANGE)
left = [.39, .61, .83]
tops = [10, 50, 100]
rows = [(.59, 5), (.32, 25)]
for x, v in zip(left, tops):
    box(ax, x, .80, .16, .12, str(v), color=ORANGE, title_size=17)
for y, v in rows:
    box(ax, .12, y, .15, .15, str(v), color=BLUE, title_size=17)
    arrow(ax, (.21, y), (.275, y))
    for x, top in zip(left, tops):
        box(ax, x, y, .18, .18, f'{v} + {top}', f'= {v+top}', GREEN)
ax.text(.59, .10, '输出形状 (2, 3)：不混淆“哪一行”与“哪一列”', ha='center', fontsize=12)
save(fig, 'broadcast-grid')


fig, ax = canvas('两层 Sigmoid 网络：前向算数值，反向算导数', (13.8, 6.6))
positions = [.07, .30, .52, .74, .93]
for x, title, shape in zip(positions,
                           ['输入 X', 'A = XW₁', 'H = σ(A)', 'Ŷ = HW₂', '损失 L'],
                           ['64 × 1000', '64 × 100', '64 × 100', '64 × 10', '一个数']):
    box(ax, x, .74, .12 if x in (.07, .93) else .17, .19, title, shape, BLUE, 12.5)
for i in range(4):
    right = positions[i]+(.069 if i == 0 else .094)
    left_edge = positions[i+1]-(.069 if i == 3 else .094)
    arrow(ax, (right, .74), (left_edge, .74))
ax.text(.185, .94, 'W₁：1000 × 100', ha='center', fontsize=11, color=BLUE)
ax.text(.63, .94, 'W₂：100 × 10', ha='center', fontsize=11, color=BLUE)
ax.text(.07, .47, '反向', ha='center', fontsize=13, color=ORANGE, weight='bold')
for x, title, body in [(.30, 'G_A = G_H ⊙ H ⊙ (1−H)', '形状 64 × 100'),
                       (.57, 'G_H = G_Y W2.T', '形状 64 × 100'),
                       (.84, 'G_Y = 2(Ŷ−Y)', '形状 64 × 10')]:
    box(ax, x, .43, .235, .18, title, body, ORANGE, 11)
arrow(ax, (.93, .63), (.87, .535), ORANGE)
arrow(ax, (.714, .43), (.695, .43), ORANGE)
arrow(ax, (.444, .43), (.425, .43), ORANGE)
box(ax, .26, .16, .39, .16, '∂L/∂W₁ = X.T @ G_A', '1000 × 100，与 W₁ 同形状', GREEN, 12.5)
box(ax, .73, .16, .39, .16, '∂L/∂W₂ = H.T @ G_Y', '100 × 10，与 W₂ 同形状', GREEN, 12.5)
ax.text(.5, -.015, '梯度全部算好后：W ← W − 学习率 × 梯度', ha='center', fontsize=13, weight='bold')
save(fig, 'numpy-network')


fig, axs = plt.subplots(1, 2, figsize=(10.8, 5.5))
data = [[1, 10, 3], [5, 14, 9]]
for ax, title, mode in zip(axs, ['BatchNorm：对一个特征，跨样本统计',
                                'LayerNorm：对一个样本，跨特征统计'], ['bn', 'ln']):
    ax.set(xlim=(-1.4, 2.8), ylim=(-1.05, 2.2))
    ax.axis('off')
    ax.set_title(title, fontsize=13, pad=16, weight='bold')
    for i in range(2):
        for j in range(3):
            highlight = j == 1 if mode == 'bn' else i == 0
            color = BLUE if mode == 'bn' else ORANGE
            ax.add_patch(Rectangle((j-.43, 1-i-.35), .86, .70,
                                   fc=color+'20' if highlight else '#f0f3f6',
                                   ec=color if highlight else '#c2cbd3', lw=2 if highlight else 1))
            ax.text(j, 1-i, str(data[i][j]), ha='center', va='center', fontsize=20)
    for j in range(3):
        ax.text(j, 1.62, f'特征 {j+1}', ha='center', fontsize=11)
    for i in range(2):
        ax.text(-.70, 1-i, f'样本 {i+1}', ha='right', va='center', fontsize=11)
    ax.text(.8, -.68, '高亮列：[10, 14]' if mode == 'bn' else '高亮行：[1, 10, 3]',
            ha='center', color=BLUE if mode == 'bn' else ORANGE, fontsize=13)
fig.suptitle('先圈出参与同一次均值 / 方差计算的数', fontsize=18, weight='bold', y=1.00)
fig.text(.5, .035, '示例是 N × D 特征表；CNN 的 BN 通常对每个通道跨 N、H、W 统计。',
         ha='center', fontsize=11)
fig.subplots_adjust(top=.78, bottom=.18, wspace=.16)
save(fig, 'normalization-axes')


fig, ax = canvas('一次注意力读取：Q、K 决定比例，V 提供内容', (12, 5.6))
box(ax, .16, .74, .27, .20, '① 点积打分', 'q = 1，d_k = 1\nscores = [0, ln 2, ln 3]', BLUE)
box(ax, .50, .74, .27, .20, '② 取指数并归一化', '指数 [1, 2, 3]，总和 6\n权重 [1/6, 2/6, 3/6]', ORANGE)
box(ax, .84, .74, .27, .20, '③ 按权重读取 V', 'values = [10, 20, 40]\n每项乘自己的权重', GREEN)
arrow(ax, (.307, .74), (.352, .74))
arrow(ax, (.647, .74), (.692, .74))
arrow(ax, (.84, .625), (.84, .485), GREEN)
box(ax, .50, .34, .82, .25, '输出 = (1/6)×10 + (2/6)×20 + (3/6)×40',
    '= 170/6 ≈ 28.3333', GREEN, 14)
ax.text(.5, .065, '输出是 value 的加权和，不是最高分，也不是注意力权重本身。', ha='center', fontsize=12)
save(fig, 'attention-steps')

print(f'Saved 5 original diagrams as PNG + SVG in {OUT}')
