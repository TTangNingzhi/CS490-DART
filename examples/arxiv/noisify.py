import numpy as np
from numpy.testing import assert_array_almost_equal


def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :][0], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


# noisify_pairflip call the function "multiclass_noisify"
def noisify_pairflip(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the pair
    """
    P = np.eye(nb_classes)
    n = noise

    if n > 0.0:
        # 0 -> 1
        P[0, 0], P[0, 1] = 1. - n, n
        for i in range(1, nb_classes - 1):
            P[i, i], P[i, i + 1] = 1. - n, n
        P[nb_classes - 1, nb_classes - 1], P[nb_classes - 1, 0] = 1. - n, n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        y_train = y_train_noisy

    return y_train, actual_noise


def noisify_multiclass_symmetric(y_train, noise, random_state=None, nb_classes=10):
    """mistakes:
        flip in the symmetric way
    """
    P = np.ones((nb_classes, nb_classes))
    n = noise
    P = (n / (nb_classes - 1)) * P

    if n > 0.0:
        # 0 -> 1
        P[0, 0] = 1. - n
        for i in range(1, nb_classes - 1):
            P[i, i] = 1. - n
        P[nb_classes - 1, nb_classes - 1] = 1. - n

        y_train_noisy = multiclass_noisify(y_train, P=P,
                                           random_state=random_state)
        actual_noise = (y_train_noisy != y_train).mean()
        assert actual_noise > 0.0
        y_train = y_train_noisy

    return y_train, actual_noise


def noisify(nb_classes=10, train_labels=None, noise_type=None, noise_rate=0.0, random_state=0):
    assert noise_type in ['pairflip', 'symmetric']
    if noise_rate == 0.0:
        return train_labels, 0.0
    if noise_type == 'pairflip':
        train_noisy_labels, actual_noise_rate = noisify_pairflip(
            train_labels, noise_rate, random_state=random_state, nb_classes=nb_classes)
    else:
        train_noisy_labels, actual_noise_rate = noisify_multiclass_symmetric(
            train_labels, noise_rate, random_state=random_state, nb_classes=nb_classes)
    return train_noisy_labels, actual_noise_rate


if __name__ == '__main__':
    y = np.array([0, 1, 0, 1, 0, 1, 1, 1, 0, 0]).reshape(-1, 1)
    print(y)
    y_noisy, _ = noisify(nb_classes=2, train_labels=y, noise_type='pairflip', noise_rate=0.8, random_state=0)
    print(y_noisy)

    y = np.array([0, 1, 0, 1, 0, 1, 1, 1, 0, 0]).reshape(-1, 1)
    print(y)
    y_noisy, _ = noisify(nb_classes=2, train_labels=y, noise_type='symmetric', noise_rate=0.8, random_state=0)
    print(y_noisy)
