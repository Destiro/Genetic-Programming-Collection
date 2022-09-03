import math
import operator


def div(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 1


def add(left, right):
    return left+right


def sub(left, right):
    return left-right


def mul(left, right):
    return left*right


def sin(left):
    return math.sin(left)


def cos(left):
    return math.cos(left)


def tan(left):
    return math.cos(left)


def abs(left):
    if left < 0:
        return left * -1
    return left


def neg(left):
    return operator.neg(left)


def pow(left):
    return left*left
