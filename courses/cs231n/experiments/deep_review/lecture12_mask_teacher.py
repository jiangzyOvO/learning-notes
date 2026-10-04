"""Masked MSE and a fixed-target student/teacher cross-entropy teaching demo.

No complete MAE/DINO networks, images, multi-crop training or downstream evaluation.
"""
import numpy as np


def masked_mse(prediction, target, masked):
    if prediction.shape != target.shape or masked.shape != target.shape[:-1] or not masked.any():
        raise ValueError('Expected aligned patch arrays and a nonempty masked set')
    errors = prediction[masked] - target[masked]
    grad = np.zeros_like(prediction)
    grad[masked] = 2 * errors / errors.size
    return float(np.mean(errors**2)), grad


def softmax(logits):
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def student_loss(student_logits, teacher_target, temperature=.5):
    scaled = student_logits / temperature
    shifted = scaled - scaled.max(axis=-1, keepdims=True)
    log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    loss = -(teacher_target * log_probs).sum(axis=-1).mean()
    probs = np.exp(log_probs)
    grad = (probs - teacher_target) / (len(student_logits) * temperature)
    return float(loss), grad


def numerical_gradient(fn, array, eps=1e-5):
    grad = np.zeros_like(array)
    for idx in np.ndindex(array.shape):
        original = array[idx]
        array[idx] = original + eps
        plus = fn()
        array[idx] = original - eps
        minus = fn()
        array[idx] = original
        grad[idx] = (plus - minus) / (2 * eps)
    return grad


def main():
    target = np.array([[[1.], [2.], [3.], [4.]]])
    prediction = np.array([[[1.], [0.], [0.], [4.]]])
    masked = np.array([[False, True, True, False]])
    loss, grad = masked_mse(prediction, target, masked)
    np.testing.assert_allclose(loss, 6.5)
    np.testing.assert_allclose(grad.ravel(), [0, -2, -3, 0])
    reconstruction_error = np.max(np.abs(numerical_gradient(
        lambda: masked_mse(prediction, target, masked)[0], prediction) - grad))
    assert reconstruction_error < 1e-7
    changed = prediction.copy()
    changed[~masked] = 100
    np.testing.assert_allclose(masked_mse(changed, target, masked)[0], loss)
    # Shuffle IDs, keep two visible patches, then restore original positions.
    shuffled_ids = np.array([2, 0, 3, 1])
    visible = target[:, shuffled_ids[:2]]
    shuffled_tokens = np.concatenate([visible, np.zeros((1,2,1))], axis=1)
    restored = shuffled_tokens[:, np.argsort(shuffled_ids)]
    np.testing.assert_allclose(restored.ravel(), [1,0,3,0])
    print('Masked MSE loss / gradient / error:', loss, grad.ravel(), reconstruction_error)
    print('Visible inputs / decoder restored input:', visible.ravel(), restored.ravel())
    rng = np.random.default_rng(12)
    teacher_logits = rng.normal(size=(2, 3))
    center = np.zeros((1, 3))
    teacher_temperature = .3
    teacher_target = softmax((teacher_logits-center)/teacher_temperature).copy()
    student_logits = rng.normal(size=(2, 3))
    student_temperature = .5
    before, student_grad = student_loss(student_logits, teacher_target, student_temperature)
    error = np.max(np.abs(numerical_gradient(
        lambda: student_loss(student_logits, teacher_target, student_temperature)[0],
        student_logits) - student_grad))
    assert error < 1e-7
    teacher_before = teacher_logits.copy()
    student_logits -= .1 * student_grad
    after = student_loss(student_logits, teacher_target, student_temperature)[0]
    assert after < before
    np.testing.assert_array_equal(teacher_before, teacher_logits)
    theta_teacher, theta_student = np.array([1.,2.]), np.array([3.,4.])
    theta_teacher = .9*theta_teacher + .1*theta_student
    np.testing.assert_allclose(theta_teacher, [1.2,2.2])
    new_center = .9*center + .1*teacher_logits.mean(axis=0, keepdims=True)
    print('Teacher target distribution:', teacher_target)
    print('Fixed-target student loss before/after:', before, after)
    print('Student raw-logit gradient error:', error)
    print('Teacher parameter EMA / center EMA:', theta_teacher, new_center)
    print('All checks passed; one-pair loss only, not complete DINO/MAE training')


if __name__ == '__main__':
    main()
