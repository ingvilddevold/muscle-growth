
def hill_function(x, K_A=1.0, n=4, max_response=2):
    """Hill function"""
    L_pow_n = x**n
    K_A_pow_n = K_A**n
    return max_response * L_pow_n / (K_A_pow_n + L_pow_n)


def hill_feedback(x, K_A=1.0, n=4, max_response=2):
    """Hill-type feedback function for protein synthesis rate"""
    return max_response - hill_function(x, K_A, n, max_response)


def linear_feedback(x, a=-2, b=3):
    """Linear feedback function for protein synthesis rate"""
    return a * x + b
