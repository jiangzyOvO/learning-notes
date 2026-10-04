"""Symmetric image/text matching on supplied vectors; no image or text encoder.
"""
import numpy as np


def softmax(x,axis):
    shifted=x-x.max(axis=axis,keepdims=True)
    exp=np.exp(shifted)
    return exp/exp.sum(axis=axis,keepdims=True)


def clip_loss_grad(image,text,temperature=.5):
    ni=np.linalg.norm(image,axis=1,keepdims=True)
    nt=np.linalg.norm(text,axis=1,keepdims=True)
    v,t=image/ni,text/nt
    scores=v@t.T/temperature
    row=softmax(scores,1)
    col=softmax(scores,0)
    B=len(image)
    loss=-.5*(np.log(np.diag(row)).mean()+np.log(np.diag(col)).mean())
    ds=(row+col-2*np.eye(B))/(2*B)
    dv,dt=ds@t/temperature,ds.T@v/temperature
    di=(dv-v*(dv*v).sum(axis=1,keepdims=True))/ni
    dtext=(dt-t*(dt*t).sum(axis=1,keepdims=True))/nt
    return float(loss),di,dtext,row,col


def numeric(fn,a):
    result=np.zeros_like(a)
    for idx in np.ndindex(a.shape):
        old=a[idx]
        a[idx]=old+1e-5
        plus=fn()
        a[idx]=old-1e-5
        minus=fn()
        a[idx]=old
        result[idx]=(plus-minus)/2e-5
    return result


def main():
    images=np.array([[1.,.1,0.],[.1,1.,.2],[.2,.1,1.]])
    texts=np.array([[.9,.2,.1],[.2,.8,.1],[0.,.2,1.]])
    loss,di,dt,row,col=clip_loss_grad(images,texts)
    fn=lambda:clip_loss_grad(images,texts)[0]
    ei=np.max(np.abs(numeric(fn,images)-di))
    et=np.max(np.abs(numeric(fn,texts)-dt))
    assert ei<1e-7 and et<1e-7
    np.testing.assert_allclose(row.sum(axis=1),1)
    np.testing.assert_allclose(col.sum(axis=0),1)
    # Supply candidate text embeddings, rather than pretend to encode actual text.
    query=np.array([1.,0.])
    candidates=np.array([[.9,.1],[.1,.9],[-1.,0.]])
    candidates/=np.linalg.norm(candidates,axis=1,keepdims=True)
    assert np.argmax(query@candidates.T)==0
    rng=np.random.default_rng(16)
    visual=rng.normal(size=(2,5,3))
    projector=rng.normal(size=(3,4))
    adapted=visual@projector
    assert adapted.shape==(2,5,4)
    print('Symmetric CLIP-style loss / image/text gradient errors:',loss,ei,et)
    print('Image-to-text probabilities:',row)
    print('Text-to-image probabilities (column-normalized):',col)
    print('Candidate similarity / visual projector output:',query@candidates.T,adapted.shape)
    print('Checks passed; no CLIP checkpoint, VLM language model, or SAM inference')


if __name__ == '__main__':
    main()
