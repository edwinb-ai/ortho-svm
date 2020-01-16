from functools import wraps
from typing import Callable


def decorate_a_decorator_with_parameters(f: Callable) -> Callable:
    """Creates a decorator that can handle parameters and no parameters to a decorator.
    In the case of parameters, these must be passed when calling the decorator
    like so:
    @wrap_with_parameters(*args, **kwargs)
    
    if no parameters are passed, then it's just
    @wrap_with_parameters
    
    Arguments:
        f {Callable} -- Any possible and valid callable.
    
    Returns:
        real_function {Callable} -- The function evaluated with the corresponding parameters.
    """

    @wraps(f)
    def new_decorator(*args, **kwargs) -> Callable:
        # If no arguments are passed, then it's a normal decorator
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return f(args[0])
        else:
            # Create a new function where the parameters are passed to the function
            def real_function(x):
                return f(x, *args, **kwargs)

            return real_function
