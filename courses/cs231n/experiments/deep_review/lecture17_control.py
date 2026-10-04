"""Deterministic scalar control and learning-target demonstrations; no robot hardware.
"""
import itertools
import numpy as np


def mpc_action(state,target,gain,horizon=2):
    best_cost,best_sequence=np.inf,None
    for sequence in itertools.product([-.5,0.,.5],repeat=horizon):
        predicted=state
        cost=0.
        for action in sequence:
            predicted+=gain*action
            cost+=(predicted-target)**2+.01*action**2
        if cost<best_cost:
            best_cost,best_sequence=cost,sequence
    return best_sequence[0]


def main():
    rewards=np.array([0.,0.,10.])
    discounted=np.sum(.9**np.arange(3)*rewards)
    np.testing.assert_allclose(discounted,8.1)
    targets=np.array([1.,1.])+.9*(1-np.array([0,1]))*np.array([4.,4.])
    np.testing.assert_allclose(targets,[4.6,1])
    actions=np.array([-.5,.2,.5])
    changes=.7*actions
    estimated_gain=actions@changes/(actions@actions)
    np.testing.assert_allclose(estimated_gain,.7)
    # Open-loop actions planned under gain=1, actual plant gain=.7.
    open_endpoint=.7*(.5+.5)
    feedback_state=0.
    for _ in range(10):
        action=.5*(1-feedback_state)
        feedback_state+=.7*action
    assert abs(feedback_state-1)<abs(open_endpoint-1)
    state=0.
    trajectory=[state]
    for _ in range(8):
        action=mpc_action(state,1.,estimated_gain)
        state+=.7*action
        trajectory.append(state)
    assert abs(state-1)<.1
    expert_actions=np.array([-1.,1.])
    mean_action=expert_actions.mean()
    assert mean_action==0  # Neither observed mode; hypothetical obstacle example only.
    print('Discounted return / nonterminal and terminal Q targets:',discounted,targets)
    print('Fitted deterministic dynamics gain:',estimated_gain)
    print('Open-loop endpoint / feedback endpoint:',open_endpoint,feedback_state)
    print('Receding-horizon MPC trajectory:',trajectory)
    print('Conflicting expert modes / MSE optimal mean:',expert_actions,mean_action)
    print('Checks passed; no robot deployment, RL policy training or physical safety evaluation')


if __name__ == '__main__':
    main()
