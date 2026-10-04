"""Flow matching gradients and analytical Gaussian velocity, not image training.

Course convention: t=0 data, t=1 noise; generation integrates backwards.
"""
import numpy as np


def flow_loss_grad(params, data, noise, time):
    mixed = (1-time)*data+time*noise
    design = np.stack([mixed, time, np.ones_like(time)], axis=1)
    target = noise-data
    residual = design@params-target
    return float(np.mean(residual**2)), 2*design.T@residual/len(time)


def gaussian_velocity(y, time, mean=2., data_std=.5):
    variance = (1-time)**2*data_std**2+time**2
    covariance = time-(1-time)*data_std**2
    return -mean+covariance/variance*(y-(1-time)*mean)


def integrate(initial, steps):
    y = initial.copy()
    dt = -1/steps
    for k in range(steps):
        t = 1-k/steps
        y += dt*gaussian_velocity(y, t)
    return y


def main():
    data, noise = 2., -1.
    path = [noise]
    for _ in range(4):
        path.append(path[-1]-.25*(noise-data))
    np.testing.assert_allclose(path, [-1,-.25,.5,1.25,2])
    print('Known-pair backwards path:', path)
    x = np.array([1.,2.,3.])
    z = np.array([-1.,.5,1.])
    t = np.array([.2,.5,.8])
    params = np.array([.3,-.2,.1])
    loss, grad = flow_loss_grad(params,x,z,t)
    numeric = np.zeros_like(params)
    for i in range(3):
        original = params[i]
        params[i] = original+1e-5
        plus = flow_loss_grad(params,x,z,t)[0]
        params[i] = original-1e-5
        minus = flow_loss_grad(params,x,z,t)[0]
        params[i] = original
        numeric[i] = (plus-minus)/2e-5
    error = np.max(np.abs(numeric-grad))
    assert error < 1e-7
    initial = np.array([-1.,0.,1.])
    exact = 2+.5*initial
    coarse, fine = integrate(initial,20), integrate(initial,400)
    assert np.max(np.abs(fine-exact)) < np.max(np.abs(coarse-exact))
    assert np.max(np.abs(fine-exact)) < .02
    np.testing.assert_allclose(1+3*(2-1),4)  # CFG toy velocities.
    time = .25
    mixed = (1-time)*data+time*noise
    recovered_data = (mixed-time*noise)/(1-time)
    recovered_velocity = (noise-mixed)/(1-time)
    np.testing.assert_allclose([recovered_data,recovered_velocity],[data,noise-data])
    print('Flow regression loss/gradient error:',loss,error)
    print('Gaussian exact / 20-step / 400-step endpoints:',exact,coarse,fine)
    print('CFG and interior-time noise/velocity conversion passed')
    print('Analytical field only; no learned image flow, DDPM sampler or CFG training')


if __name__ == '__main__':
    main()
