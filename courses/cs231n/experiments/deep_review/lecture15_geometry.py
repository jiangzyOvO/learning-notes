"""Projection, PointNet aggregation, Chamfer, SDF and ray compositing checks.

Constructed coordinates/features; not a trained 3-D reconstruction model.
"""
import numpy as np


def chamfer(P,Q):
    distances = ((P[:,None]-Q[None])**2).sum(axis=-1)
    return float(distances.min(axis=1).mean()+distances.min(axis=0).mean())


def composite(density, distance, colors, background):
    alpha = -np.expm1(-density*distance)
    survival = 1-alpha
    transmission = np.concatenate([[1.],np.cumprod(survival)[:-1]])
    weights = transmission*alpha
    remainder = np.prod(survival)
    return weights@colors+remainder*background, weights, remainder


def main():
    points = np.array([[1.,2.,4.],[2.,4.,8.]])
    uv = 100*points[:,:2]/points[:,2:]
    np.testing.assert_allclose(uv,[[25,50],[25,50]])
    features = np.array([[1.,4.],[3.,2.],[0.,5.]])
    pooled = features.max(axis=0)
    np.testing.assert_allclose(pooled,[3,5])
    np.testing.assert_allclose(features[[2,0,1]].max(axis=0),pooled)
    P,Q = np.array([[0.],[2.]]), np.array([[0.],[3.]])
    np.testing.assert_allclose(chamfer(P,Q),1)
    np.testing.assert_allclose(chamfer(P[::-1],Q),1)
    positions = np.array([[0.,0.,0.],[1.,0.,0.],[2.,0.,0.]])
    sdf = np.linalg.norm(positions,axis=1)-1
    np.testing.assert_allclose(sdf,[-1,0,1])
    density = np.array([np.log(2),np.log(2)])
    colors = np.array([[1.,0.,0.],[0.,0.,1.]])
    color,weights,remainder = composite(density,np.ones(2),colors,np.zeros(3))
    np.testing.assert_allclose(color,[.5,0,.25])
    np.testing.assert_allclose(np.r_[weights,remainder],[.5,.25,.25])
    np.testing.assert_allclose(weights.sum()+remainder,1)
    reversed_color = composite(density,np.ones(2),colors[::-1],np.zeros(3))[0]
    assert not np.allclose(color,reversed_color)
    # Differentiate the red output wrt first density: exp(-density[0]) = .5.
    original = density[0]
    density[0] = original+1e-5
    plus = composite(density,np.ones(2),colors,np.zeros(3))[0][0]
    density[0] = original-1e-5
    minus = composite(density,np.ones(2),colors,np.zeros(3))[0][0]
    density[0] = original
    np.testing.assert_allclose((plus-minus)/2e-5,.5,atol=1e-8)
    print('Same 2-D projection from different depths:',uv)
    print('Point pooling / Chamfer / SDF:',pooled,chamfer(P,Q),sdf)
    print('Ray color/weights/background weight:',color,weights,remainder)
    print('Swapped front/back colors:',reversed_color)
    print('Voxel count 32^3/64^3:',32**3,64**3)
    print('Checks passed; no trained PointNet, NeRF or Gaussian splatting')


if __name__ == '__main__':
    main()
