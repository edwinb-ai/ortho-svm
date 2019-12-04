from numba import njit
import json


def hermite_recurrence(x: float, n: int) -> float:

    if n == 0:
        return 1.0
    elif n == 1:
        return x
    else:
        return x * hermite_recurrence(x, n - 1) - (n - 1) * hermite_recurrence(x, n - 2)


@njit
def hermite_iterative(x: float, n: int) -> float:
    """Hermite polynomial computations using the three term recurrence
    relation.
    
    Arguments:
        x -- Value to evaluate the polynomial
        n -- Degree of the Hermite polynomial
    
    Returns:
        result -- Value of the Hermite polynomial, already evaluated.
    """
    first_value = 1.0
    second_value = x

    if n == 0:
        return first_value
    elif n == 1:
        return second_value
    else:
        result = 0.0

        for i in range(1, n):
            result = x * second_value - i * first_value
            first_value = second_value
            second_value = result

        return result


# Create the expressions dictionary
he_expressions = {}
with open("hermite_expressions.json", "r") as f:
    he_expressions = json.loads(f.read())


def hermite_expr(x: float, n: int) -> float:

    # Evaluate the polynomials while traversing through the dict
    result = 0.0
    for i, j in enumerate(he_expressions[f"{n}"]):
        result += j * x ** i

    return result

