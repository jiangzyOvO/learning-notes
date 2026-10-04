"""Constructed group and base-rate metrics, not an audit of a real system."""
import numpy as np


def main():
    counts=np.array([900,100])
    accuracies=np.array([.99,.60])
    overall=counts@accuracies/counts.sum()
    np.testing.assert_allclose(overall,.951)
    total,prevalence,recall,fpr=10_000,.01,.9,.01
    positive=total*prevalence
    negative=total-positive
    tp=positive*recall
    fp=negative*fpr
    precision=tp/(tp+fp)
    np.testing.assert_allclose([tp,fp],[90,99])
    np.testing.assert_allclose(precision,90/189)
    less_rare=.10
    changed_precision=less_rare*recall/(less_rare*recall+(1-less_rare)*fpr)
    assert changed_precision>precision
    # Independent stage success assumption, used only to illustrate accumulated errors.
    stage_success=np.array([.95,.95,.95])
    end_to_end=np.prod(stage_success)
    np.testing.assert_allclose(end_to_end,.857375)
    print('Group accuracy / sample-weighted accuracy:',accuracies,overall)
    print('Expected TP/FP / precision at 1% prevalence:',tp,fp,precision)
    print('Precision at 10% prevalence, same recall/FPR:',changed_precision)
    print('Three independent 95% stages, joint success:',end_to_end)
    print('Examples only; metrics do not define all fairness or privacy requirements')


if __name__ == '__main__':
    main()
