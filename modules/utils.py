import dezero.functions as F


def expand_2d(x):
    return F.expand_dims(F.expand_dims(x, 3), 4)
