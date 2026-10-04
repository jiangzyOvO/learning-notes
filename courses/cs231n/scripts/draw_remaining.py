"""Original illustrations for chapters 4–18. All plotted values are teaching examples."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'assets/remaining'; OUT.mkdir(parents=True,exist_ok=True)
font=Path('/System/Library/Fonts/PingFang.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':12,'axes.unicode_minus':False,'axes.spines.top':False,
 'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
BLUE='#2563a6'; ORANGE='#bc651d'; PURPLE='#784bad'; GREEN='#318579'; GRAY='#617082'
def save(fig,name):
    for ext in ['png','svg']: fig.savefig(OUT/f'{name}.{ext}',dpi=160,bbox_inches='tight',pad_inches=.22)
    plt.close(fig)
def diagram(title,figsize=(10.5,4.8)):
    fig,ax=plt.subplots(figsize=figsize);ax.set(xlim=(-.025,1.025),ylim=(0,1));ax.axis('off');ax.set_title(title,pad=20,fontsize=17)
    return fig,ax
def box(ax,x,y,text,w=.19,h=.17,color=BLUE):
    ax.add_patch(FancyBboxPatch((x-w/2,y-h/2),w,h,boxstyle='round,pad=0.012',ec=color,fc=color+'12',lw=1.6))
    ax.text(x,y,text,ha='center',va='center',fontsize=12,color='#24344a')
def arrow(ax,a,b,label=None,color=GRAY):
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':color,'lw':1.8})
    if label:
        vertical=abs(a[0]-b[0])<.03
        ax.text((a[0]+b[0])/2+(.075 if vertical else 0),(a[1]+b[1])/2+(0 if vertical else .055),label,ha='center',fontsize=11,color=color)

fig,ax=diagram('反向传播：沿依赖关系传回导数')
box(ax,.1,.75,'x = 1',w=.13);box(ax,.1,.3,'y = 2',w=.13)
box(ax,.4,.52,'q = x + y\nq = 3');box(ax,.72,.52,'f = q × z\nf = 12');box(ax,.72,.86,'z = 4',w=.13,h=.13)
arrow(ax,(.18,.73),(.3,.58));arrow(ax,(.18,.32),(.3,.46));arrow(ax,(.51,.52),(.61,.52));arrow(ax,(.72,.78),(.72,.62))
ax.text(.43,.17,'df/dq = 4 → df/dx = df/dy = 4',ha='center',color=ORANGE,fontsize=13)
ax.text(.78,.30,'df/dz = 3',ha='center',color=ORANGE,fontsize=13)
ax.text(.5,.02,'前向算数值；反向乘局部导数。这里只画计算，不做参数更新。',ha='center',fontsize=11)
save(fig,'04-backprop')

fig,axs=plt.subplots(1,3,figsize=(10,3.9),layout='constrained')
arrays=[np.arange(1,10).reshape(3,3),np.array([[1,0],[0,-1]]),np.full((2,2),-4)]
for ax,arr,title in zip(axs,arrays,['输入 3 × 3','卷积核 2 × 2','输出 2 × 2']):
    ax.imshow(arr,cmap='Blues' if arr.min()>=0 else 'RdBu',vmin=-4 if arr.min()<0 else 0,vmax=9 if arr.max()>1 else 4)
    for (i,j),v in np.ndenumerate(arr):ax.text(j,i,str(v),ha='center',va='center',fontsize=22,color='white' if (v < -2 or v >= 7) else 'black')
    ax.set_xticks([]);ax.set_yticks([]);ax.set_title(title,pad=12)
axs[0].add_patch(Rectangle((-.5,-.5),2,2,fill=False,ec=ORANGE,lw=3))
fig.suptitle('逐元素相乘再求和：1 × 1 + 2 × 0 + 4 × 0 − 5 × 1 = −4',fontsize=14)
save(fig,'05-convolution')

fig,ax=diagram('残差块：学习修正 F(x)，同时保留直接路径')
box(ax,.12,.42,'输入 x',w=.15);box(ax,.5,.42,'残差分支\nF(x)',w=.23);box(ax,.86,.42,'相加\ny = x + F(x)',w=.24)
arrow(ax,(.21,.42),(.37,.42));arrow(ax,(.63,.42),(.73,.42))
ax.plot([.24,.24,.86,.86],[.42,.8,.8,.53],c=GREEN,lw=2)
ax.annotate('',xy=(.86,.52),xytext=(.86,.7),arrowprops={'arrowstyle':'->','color':GREEN,'lw':2})
ax.text(.55,.84,'捷径：传递 x',ha='center',color=GREEN)
ax.text(.5,.10,'两条分支形状必须一致；否则需要投影或其他形状对齐。',ha='center')
save(fig,'06-residual')

fig,ax=diagram('RNN 展开：同一组参数，在不同时间重复使用')
for j,x in enumerate([.2,.5,.8],1):
    box(ax,x,.56,f'循环单元\nh{j}',w=.19)
    box(ax,x,.17,f'输入 x{j}',w=.17,h=.12)
    arrow(ax,(x,.24),(x,.46))
    arrow(ax,(x,.66),(x,.86),f'输出 s{j}')
arrow(ax,(.025,.56),(.095,.56));ax.text(.035,.64,'h0',ha='center')
arrow(ax,(.31,.56),(.39,.56));arrow(ax,(.61,.56),(.69,.56))
ax.text(.5,.98,'参数 W 共享，隐藏状态 h 随输入更新',ha='center',fontsize=12,color=PURPLE)
save(fig,'07-rnn')

logits=np.array([[2,1,0,-1],[1,2,1,0],[0,1,2,1],[-1,0,1,2]],float)
def softmax(a):
    b=np.exp(a-a.max(axis=-1,keepdims=True));return b/b.sum(axis=-1,keepdims=True)
full=softmax(logits);masked=logits.copy();masked[np.triu_indices(4,1)]=-np.inf;causal=softmax(masked)
fig,axs=plt.subplots(1,2,figsize=(9,4.4),layout='constrained')
for ax,arr,title in zip(axs,[full,causal],['所有位置可见','因果掩码：未来权重为 0']):
    ax.imshow(arr,cmap='Blues',vmin=0,vmax=1)
    for (i,j),v in np.ndenumerate(arr):ax.text(j,i,f'{v:.2f}',ha='center',va='center',color='white' if v>.6 else '#162c49')
    ax.set(xticks=range(4),yticks=range(4),xticklabels=range(1,5),yticklabels=range(1,5),xlabel='被读取的位置（key）',ylabel='查询位置（query）',title=title)
fig.suptitle('注意力权重：每一行的和为 1（原创构造示例）',fontsize=15)
save(fig,'08-attention')

fig,ax=plt.subplots(figsize=(6,4.8),layout='constrained')
ax.add_patch(Rectangle((0,0),2,2,ec=BLUE,fc=BLUE+'25',lw=2,label='A：面积 4'))
ax.add_patch(Rectangle((1,1),2,2,ec=ORANGE,fc=ORANGE+'25',lw=2,label='B：面积 4'))
ax.add_patch(Rectangle((1,1),1,1,fc=PURPLE+'60'))
ax.text(1.5,1.5,'交集\n面积 1',ha='center',va='center')
ax.set(xlim=(-.4,3.4),ylim=(-.4,3.4),xlabel='x',ylabel='y',title='IoU = 交集 / 并集 = 1 / (4 + 4 − 1) = 1/7');ax.set_aspect('equal');ax.legend(loc='upper left');ax.grid(alpha=.15)
save(fig,'09-iou')

fig,ax=plt.subplots(figsize=(10,3.5),layout='constrained')
for y,step,col,label in [(1,1,BLUE,'每隔 1 帧：首尾相隔 0.5 秒'),(0,3,ORANGE,'每隔 3 帧：首尾相隔 1.5 秒')]:
    t=np.arange(16)*step/30
    ax.hlines(y,t[0],t[-1],color=col,alpha=.4);ax.scatter(t,np.full(16,y),color=col,s=40,label=label)
ax.set(ylim=(-.5,1.5),yticks=[0,1],yticklabels=['稀疏采样','密集采样'],xlabel='原视频时间（秒）',title='同样取 16 帧，采样间隔决定时间覆盖（原视频 30 fps）')
ax.legend(loc='upper right',fontsize=10);ax.grid(axis='x',alpha=.2)
save(fig,'10-sampling')

fig,ax=diagram('数据并行：不同样本、相同模型、平均梯度')
box(ax,.21,.75,'GPU 1\n本地梯度 [1, 3]',w=.3,h=.20)
box(ax,.21,.25,'GPU 2\n本地梯度 [5, 7]',w=.3,h=.20)
box(ax,.66,.5,'汇总并平均\n全局梯度 [3, 5]',w=.32,h=.24,color=PURPLE)
arrow(ax,(.38,.73),(.48,.56));arrow(ax,(.38,.27),(.48,.44))
ax.text(.68,.16,'各卡得到同一梯度并同步更新',ha='center',fontsize=11)
ax.text(.5,.98,'前提：各卡样本数相同，损失均按本地样本取平均',ha='center',fontsize=11,color=GRAY)
save(fig,'11-data-parallel')

fig,ax=diagram('对比学习：用视图关系构造监督')
box(ax,.13,.52,'原始图片 x',w=.20)
box(ax,.46,.78,'增强视图 x₁',w=.22,color=BLUE);box(ax,.46,.27,'增强视图 x₂',w=.22,color=BLUE)
arrow(ax,(.24,.58),(.34,.75));arrow(ax,(.24,.46),(.34,.3))
box(ax,.83,.78,'表示 z₁',w=.17);box(ax,.83,.27,'表示 z₂',w=.17)
arrow(ax,(.58,.78),(.73,.78),'编码');arrow(ax,(.58,.27),(.73,.27),'编码')
ax.annotate('',xy=(.83,.4),xytext=(.83,.65),arrowprops={'arrowstyle':'<->','color':ORANGE,'lw':2})
ax.text(.94,.52,'正对\n拉近',ha='center',va='center',color=ORANGE)
ax.text(.5,.04,'其他图片常作负样本；同图增强也要保留与任务相关的语义。',ha='center',fontsize=11)
save(fig,'12-contrastive')

fig,ax=diagram('VAE：训练学习后验，生成从先验采样',figsize=(11,4.7))
for x,text in [(.08,'输入 x'),(.31,'编码器\nμ, log σ²'),(.56,'采样 z\nμ + σ × ε'),(.83,'解码器\np(x | z)')]:box(ax,x,.68,text,w=.14 if x==.08 else .2,h=.2)
arrow(ax,(.16,.68),(.20,.68));arrow(ax,(.42,.68),(.45,.68));arrow(ax,(.67,.68),(.72,.68))
box(ax,.56,.23,'生成：z ~ N(0, I)',w=.31,h=.16,color=GREEN)
arrow(ax,(.72,.25),(.83,.56),color=GREEN)
ax.text(.27,.30,'训练目标：\n重建解释 + 先验匹配',ha='center',va='center',fontsize=12,color=PURPLE)
ax.text(.5,.02,'生成新样本时可以不经过编码器。',ha='center',fontsize=11)
save(fig,'13-vae')

fig,ax=plt.subplots(figsize=(8,4.3),layout='constrained')
t=np.linspace(0,1,100);ax.plot(t,2-3*t,color=BLUE,lw=2)
steps=np.linspace(1,0,5);ax.scatter(steps,2-3*steps,color=ORANGE,zorder=3)
for a,b in zip(steps[:-1],steps[1:]):
    ax.annotate('',xy=(b,2-3*b),xytext=(a,2-3*a),arrowprops={'arrowstyle':'->','color':ORANGE,'lw':2})
ax.annotate('数据 x = 2',(0,2),xytext=(15,-2),textcoords='offset points',fontsize=12)
ax.annotate('噪声 z = −1',(1,-1),xytext=(-110,8),textcoords='offset points',fontsize=12)
ax.set(xlabel='t：0 为数据，1 为噪声',ylabel='标量 x_t',title='Rectified Flow：x_t = 2 − 3t，生成方向从右向左',xlim=(-.04,1.04),ylim=(-1.3,2.3));ax.grid(alpha=.18)
save(fig,'14-flow')

fig,axs=plt.subplots(1,4,figsize=(12,3.3),layout='constrained')
u=np.linspace(0,2*np.pi,17);circle=np.array([np.cos(u),np.sin(u)])
axs[0].scatter(*circle[:,:-1],s=25,color=BLUE)
axs[1].plot(*circle,'o-',ms=3,color=BLUE)
grid=np.linspace(-1.2,1.2,13);xx,yy=np.meshgrid(grid,grid)
axs[2].imshow((xx*xx+yy*yy<=1).astype(float),extent=(-1.3,1.3,-1.3,1.3),origin='lower',cmap='Blues')
axs[3].contour(xx,yy,np.sqrt(xx*xx+yy*yy)-1,levels=[0],colors=[PURPLE]);axs[3].text(0,0,'SDF = r − 1\n表面：SDF = 0',ha='center',va='center',fontsize=11)
for ax,title in zip(axs,['点集：无连接','网格：显式连接','占据网格：格子内外','隐式：查询函数']):
    ax.set(xlim=(-1.3,1.3),ylim=(-1.3,1.3),title=title,xticks=[],yticks=[]);ax.set_aspect('equal')
fig.suptitle('用二维圆类比三维表示：选择保存什么，决定怎样计算',fontsize=15)
save(fig,'15-representations')

fig,ax=plt.subplots(figsize=(6.4,4.5),layout='constrained')
arr=np.array([[.9,.2,.1],[.1,.85,.2],[.2,.1,.8]])
ax.imshow(arr,cmap='Blues',vmin=0,vmax=1)
for (i,j),v in np.ndenumerate(arr):ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=18,color='white' if v>.6 else BLUE)
ax.set(xticks=range(3),yticks=range(3),xticklabels=['文本 1','文本 2','文本 3'],yticklabels=['图像 1','图像 2','图像 3'],title='CLIP 图文匹配：正配对位于对角线\n相似度示意，不是概率或实验结果')
save(fig,'16-clip')

fig,ax=diagram('机器人学习：动作改变环境，也改变下一次观测')
box(ax,.17,.75,'环境状态',w=.23);box(ax,.76,.75,'传感器观测 o_t',w=.27)
box(ax,.76,.25,'策略 π\n生成动作 a_t',w=.27);box(ax,.17,.25,'执行与状态变化',w=.26)
arrow(ax,(.3,.75),(.61,.75),'感知');arrow(ax,(.76,.64),(.76,.37))
arrow(ax,(.61,.25),(.31,.25),'控制');arrow(ax,(.17,.37),(.17,.64),'反馈')
ax.text(.48,.5,'不确定性、时延\n与部分可观测性',ha='center',va='center',fontsize=11,color=PURPLE)
save(fig,'17-robot-loop')

fig,ax=diagram('以人为中心：用真实结果检验任务定义')
items=[(.16,.76,'人的需求\n谁需要什么帮助'),(.78,.76,'任务与数据\n目标、覆盖、隐私'),(.78,.24,'模型与交互\n能力、反馈、接管'),(.16,.24,'真实效果\n收益、错误与负担')]
for x,y,label in items:box(ax,x,y,label,w=.27,h=.22,color=BLUE if y>.5 else GREEN)
arrow(ax,(.31,.76),(.63,.76));arrow(ax,(.78,.63),(.78,.37));arrow(ax,(.63,.24),(.31,.24));arrow(ax,(.16,.37),(.16,.63))
ax.text(.48,.5,'原创复习框架\n并非课堂图复制',ha='center',va='center',fontsize=11,color=GRAY)
save(fig,'18-human-centered')
print('Generated 15 original figures in PNG and SVG.')
