"""Reproduce original scalar/array examples in lectures 4–18.
Run from the CS231n course directory: python3 experiments/remaining_examples.py
Uses only the Python standard library. No dataset or model training is performed.
"""
from math import exp, log, sqrt, tanh, isclose


def check(actual, expected, tol=1e-9):
    assert isclose(actual, expected, rel_tol=tol, abs_tol=tol), (actual, expected)


def softmax(values):
    shifted = [exp(v - max(values)) for v in values]
    return [v / sum(shifted) for v in shifted]


def network_loss(params):
    w, b, v, c = params
    a = 2 * w + b
    pred = v * max(0, a) + c
    return (pred - 1) ** 2 / 2


def iou(a, b):
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    return inter / (area_a + area_b - inter)


def main():
    # L4: analytical gradients compared against centered differences, away from the ReLU kink.
    p = [1., -1., 2., 0.]
    grad = [4., 2., 1., 1.]
    h = 1e-5
    for i, expected in enumerate(grad):
        plus, minus = p.copy(), p.copy()
        plus[i] += h; minus[i] -= h
        check((network_loss(plus) - network_loss(minus)) / (2 * h), expected)
    updates = {eta: network_loss([v - eta * g for v, g in zip(p, grad)]) for eta in [0.1, 0.01]}
    check(updates[0.1], .605); check(updates[0.01], .3049805)
    print('04 backprop gradient check passed; loss after lr=.1/.01:', updates)

    x = [[1,2,3],[4,5,6],[7,8,9]]
    k = [[1,0],[0,-1]]
    output = [[sum(x[i+a][j+b]*k[a][b] for a in range(2) for b in range(2)) for j in range(2)] for i in range(2)]
    assert output == [[-4,-4],[-4,-4]]
    params = 16 * (3*3*3+1)
    assert params == 448
    print('05 convolution:',output,'parameters:',params,'MACs:',32*32*16*3*3*3)

    vals=[1.,3.];mean=sum(vals)/2;variance=sum((v-mean)**2 for v in vals)/2
    bn=[2*(v-mean)/sqrt(variance)+1 for v in vals]
    assert bn==[-1.,3.]
    print('06 BN (epsilon omitted):',bn,'He std:',sqrt(2/100))

    state=0.;states=[]
    for value in [1,0,-1]:
        state=tanh(value+.5*state);states.append(state)
    cell=.9*2+.2*(-.5);hidden=.8*tanh(cell)
    print('07 RNN states:',[round(v,6) for v in states],'LSTM cell/hidden:',cell,hidden)

    weights=softmax([0,log(2),log(3)])
    attn=sum(w*v for w,v in zip(weights,[10,20,40]))
    check(sum(weights),1);check(attn,170/6)
    print('08 attention weights/output:',weights,attn)
    score=iou([0,0,2,2],[1,1,3,3]);check(score,1/7)
    print('09 IoU:',score)

    check(15*3/30,1.5)
    video_bytes=300*224*224*3;assert video_bytes==45158400
    print('10 video bytes:',video_bytes,'3D convolution parameters:',16*(3*3*3*3+1))

    gradients=[[1,3],[5,7]];mean_grad=[sum(col)/2 for col in zip(*gradients)]
    assert mean_grad==[3,5]
    print('11 averaged gradient:',mean_grad,'effective batch:',4*8*2,'MFU:',(1e15/20)/1e14)
    contrastive=-log(softmax([log(4),log(2),0])[0]);check(contrastive,-log(4/7))
    print('12 InfoNCE:',contrastive,'MAE visible patches:',196-int(.75*196))

    kl=.5*(1**2+2**2-1-log(2**2));check(kl,1.3068528194400546)
    print('13 AR probability/NLL:',.8*.5*.25,-log(.8*.5*.25),'KL:',kl)
    value=-1.;path=[value]
    for _ in range(4):value+=(-.25)*(-3);path.append(value)
    assert path==[-1.,-.25,.5,1.25,2.]
    print('14 flow reverse Euler path:',path)

    points=[0,2];target=[0,3]
    chamfer=sum(min((p-q)**2 for q in target) for p in points)/len(points)+sum(min((q-p)**2 for p in points) for q in target)/len(target)
    check(chamfer,1)
    ray_color=[.5,0,.25]
    print('15 Chamfer:',chamfer,'ray color:',ray_color,'remaining transmittance:',.25)
    assert max(range(3),key=[.8,.2,.1].__getitem__)==0
    print('16 CLIP teaching example predicted index: 0 (cat)')
    returns=sum(.9**t*r for t,r in enumerate([0,0,10]));check(returns,8.1)
    print('17 discounted return:',returns,'Bellman target:',1+.9*4)
    accuracy=(900*.99+100*.60)/1000;check(accuracy,.951)
    precision=90/(90+99);check(precision,10/21)
    print('18 pooled accuracy:',accuracy,'rare-event alarm precision:',precision)
    print('All numerical checks passed. These are teaching examples, not trained-model benchmarks.')


if __name__=='__main__':
    main()
