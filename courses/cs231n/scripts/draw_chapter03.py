"""Original, deterministic scalar illustrations; no fitted real-world models."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/chapter03'
OUT.mkdir(parents=True, exist_ok=True)
font = Path('/System/Library/Fonts/PingFang.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':12, 'axes.spines.top':False, 'axes.spines.right':False,
                     'axes.unicode_minus':False, 'figure.facecolor':'white', 'savefig.facecolor':'white'})
COLORS=['#2563a6','#c26418','#7b47a6','#43816d']
def save(fig,name):
    for ext in ['png','svg']:
        fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight',pad_inches=.2)
    plt.close(fig)

w=np.linspace(-.5,4,400)
fig,axs=plt.subplots(1,3,figsize=(12,4),layout='constrained')
for ax,lam,col in zip(axs,[0,2,10],COLORS):
    data=(w-3)**2; reg=lam*w*w/2; total=data+reg; best=6/(2+lam)
    ax.plot(w,data,'--',color='#7b8794',label='数据损失')
    ax.plot(w,reg,':',color='#c26418',label='正则化项')
    ax.plot(w,total,color=col,lw=2,label='总损失')
    ax.scatter(best,(best-3)**2+lam*best**2/2,color=col,zorder=3)
    ax.axvline(best,color=col,alpha=.3)
    ax.set(xlabel='参数 w',ylabel='损失',ylim=(-.5,22),title=f'λ = {lam}：最优 w = {best:g}')
    ax.grid(alpha=.15)
axs[0].legend(fontsize=10)
fig.suptitle('正则化改变优化目标：J(w) = (w − 3)² + λw²/2',fontsize=16)
save(fig,'regularization')

fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
for eta,col in zip([.01,.1,1.,1.1],COLORS):
    values=[0.]
    for _ in range(20):values.append(values[-1]-eta*2*(values[-1]-3))
    values=np.array(values)
    axs[0].plot(range(9),values[:9],marker='o',ms=3,color=col,label=f'η = {eta:g}')
    axs[1].semilogy((values-3)**2,color=col,label=f'η = {eta:g}')
axs[0].axhline(3,color='gray',ls='--',label='最优参数 w = 3')
axs[0].set(xlabel='更新次数',ylabel='参数 w',title='参数怎样移动（前 8 步）')
axs[1].set(xlabel='更新次数',ylabel='损失（对数刻度）',title='损失是否持续下降')
for ax in axs:ax.grid(alpha=.15);ax.legend(fontsize=10)
fig.suptitle('同一函数、同一起点：学习率改变训练轨迹',fontsize=16)
save(fig,'learning-rates')

u=np.linspace(0,1,501)
fig,axs=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
axs[0].plot(u,np.where(u<.5,1,np.where(u<.8,.1,.01)),label='阶梯衰减',color=COLORS[0])
axs[0].plot(u,1-u,label='线性衰减',color=COLORS[1])
axs[0].plot(u,.5*(1+np.cos(np.pi*u)),label='余弦衰减',color=COLORS[2])
warm=.1
schedule=np.where(u<warm,u/warm,.5*(1+np.cos(np.pi*(u-warm)/(1-warm))))
axs[1].plot(u,schedule,color=COLORS[2],lw=2)
axs[1].axvspan(0,warm,color=COLORS[1],alpha=.13,label='预热阶段（示例占 10%）')
axs[0].set_title('三种学习率衰减方式')
axs[1].set_title('线性预热 + 余弦衰减')
for ax in axs:
    ax.set(xlabel='训练进度',ylabel='学习率 / 峰值学习率',ylim=(-.03,1.08))
    ax.grid(alpha=.15);ax.legend(fontsize=10)
fig.suptitle('学习率调度：这些曲线是人为设定的计划，不是训练结果',fontsize=15)
save(fig,'schedules')
