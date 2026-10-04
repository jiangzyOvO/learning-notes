"""Reproduce original educational figures for lecture 2 (no trained models)."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets' / 'chapter02'
OUT.mkdir(parents=True, exist_ok=True)
font = Path('/System/Library/Fonts/PingFang.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.unicode_minus': False,
                     'figure.facecolor': '#ffffff', 'axes.labelcolor': '#273449',
                     'text.color': '#273449', 'savefig.facecolor': '#ffffff'})
BLUE, ORANGE, PURPLE = '#2563a6', '#c26418', '#7b47a6'

def save(fig, name):
    fig.savefig(OUT / f'{name}.png', dpi=180, bbox_inches='tight', pad_inches=.25)
    fig.savefig(OUT / f'{name}.svg', bbox_inches='tight', pad_inches=.25)
    plt.close(fig)

# Same dataset in both panels; only K changes.
points = np.array([[.35,.05],[-.5,.2],[.1,.7],[-1.2,-.8],[1.,1.1]])
labels = np.array([0,1,1,0,1])
dist = np.linalg.norm(points, axis=1)
order = np.argsort(dist)
fig, axs = plt.subplots(1,2,figsize=(10.5,4.9), layout='constrained')
for ax,k,result in zip(axs,[1,3],['A 类：1 票','B 类：2 票；A 类：1 票']):
    for cls,color,marker in [(0,BLUE,'^'),(1,ORANGE,'o')]:
        mask=labels==cls
        ax.scatter(*points[mask].T,s=95,c=color,marker=marker,label=f'{"AB"[cls]} 类',zorder=3)
    ax.scatter(0,0,marker='*',s=240,c=PURPLE,label='待分类点 Q',zorder=5)
    for idx in order[:k]:
        ax.plot([0,points[idx,0]],[0,points[idx,1]],ls='--',lw=1.2,c='#8793a1',zorder=1)
    ax.add_patch(Circle((0,0),dist[order[k-1]],fill=False,ec='#a6afba',lw=1.2))
    for i,(x,y) in enumerate(points):
        ax.annotate(f'P{i+1}',(x,y),xytext=(7,-17),textcoords='offset points',fontsize=11)
    ax.annotate('Q',(0,0),xytext=(-18,-20),textcoords='offset points')
    ax.set(xlim=(-1.5,1.5),ylim=(-1.3,1.5),xlabel='特征 1',ylabel='特征 2')
    ax.set_aspect('equal');ax.grid(alpha=.15)
    ax.set_title(f'K = {k}  →  {result}',fontsize=13,pad=14)
axs[0].legend(loc='upper left',fontsize=10,frameon=False)
fig.suptitle('近邻投票：同一个查询点，K 不同，预测可能不同',fontsize=16)
save(fig,'knn-voting')

# Explicit shapes and numbers; no hidden preprocessing.
x=np.array([[2.],[1.]])
W=np.array([[1.,0.],[-1.,2.],[0.,-1.]])
b=np.array([[0.],[0.],[1.]])
s=W@x+b
fig,axs=plt.subplots(1,4,figsize=(10.8,4.3),layout='constrained',gridspec_kw={'width_ratios':[2,1,1,1]})
for ax,arr,title,foot in zip(axs,[W,x,b,s],['权重 W','输入 x','偏置 b','分数 s'],['3 类 × 2 个特征','2 × 1','3 × 1','3 × 1']):
    ax.imshow(arr,cmap='Blues',vmin=-1,vmax=2,aspect='equal')
    for (i,j),v in np.ndenumerate(arr):
        ax.text(j,i,f'{v:g}',ha='center',va='center',fontsize=22,color='white' if v>=1.7 else '#162c49')
    ax.set_xticks([]);ax.set_yticks([])
    if arr.shape[0]==3:ax.set_yticks([0,1,2],['A 类','B 类','C 类'])
    ax.set_title(title,pad=14);ax.set_xlabel(foot,labelpad=13)
    for spine in ax.spines.values():spine.set_visible(False)
fig.suptitle('s = W × x + b：每一行权重计算一个类别的分数',fontsize=16)
save(fig,'linear-scores')

shift=s.ravel()-s.max();p=np.exp(shift)/np.exp(shift).sum()
fig,axs=plt.subplots(1,2,figsize=(10.8,4.8),layout='constrained')
axs[0].bar(['A 类','B 类','C 类'],p,color=[BLUE,'#aebdcd','#aebdcd'],width=.55)
for i,v in enumerate(p):axs[0].text(i,v+.025,f'{v:.4f}',ha='center',fontsize=13)
axs[0].set(ylim=(0,1),ylabel='预测概率',title='分数 [2, 0, 0] 经过 Softmax')
axs[0].text(.5,.93,'概率和 = 1',transform=axs[0].transAxes,ha='center',fontsize=11)
xs=np.linspace(.01,1,400)
axs[1].plot(xs,-np.log(xs),c=BLUE,lw=2.2)
for prob,offset in [(.1,(15,9)),(.5,(-14,22)),(.9,(-70,23))]:
    loss=-np.log(prob);axs[1].scatter(prob,loss,c=ORANGE,s=38,zorder=3)
    axs[1].annotate(f'p={prob:.1f}\nL={loss:.3f}',(prob,loss),xytext=offset,textcoords='offset points',fontsize=10)
axs[1].set(xlim=(0,1.03),ylim=(0,4.8),xlabel='真实类别的预测概率 p',ylabel='损失 L',title='交叉熵：L = −ln(p)')
axs[1].grid(alpha=.15)
fig.suptitle('先得到概率，再用真实标签评价预测',fontsize=16)
save(fig,'softmax-loss')
print('Created 3 PNG figures and 3 SVG originals.')
print('Scores:',s.ravel().tolist(),'probabilities:',p.tolist(),'loss:',float(-np.log(p[0])))
