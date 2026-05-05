# AUP (Accuracy Under Parallelism) measure for parallel decoders
# See paper for detailed definition and motivation
import math

def weight_function(y: float, y_max: float, alpha: float = 3.0) -> float:
    """Quality-weighting function W(y) = min(exp(-alpha * (1 - y/y_max)), 1)"""
    return min(math.exp(-alpha * (1 - y / y_max)), 1.0)

def get_aup(rho: list[float], y: list[float], y_max: float, alpha: float = 3.0, y_min_offset: float = 5.0, is_print: bool = False) -> float:
    """
    Calculate the Accuracy Under Parallelism (AUP) of parallelism-accuracy pairs.
    
    Args:
        rho: list of parallelism values (TPF, tokens per forward)
        y: list of accuracy values in [0, 100] (percentage)
        y_max: maximum accuracy across all methods (for normalization)
        alpha: penalty factor for accuracy degradation (default: 3.0)
        y_min_offset: minimum accuracy threshold offset (default: 5.0, i.e., 5%)
    
    Returns:
        AUP score (scalar value)
    """
    assert len(rho) == len(y), "rho and y must have the same length"
    assert len(rho) > 0, "rho and y must not be empty"
    assert all(r > 0 for r in rho), "all rho must be positive"
    
    # Check if y values are in [0, 100] range
    if any(acc < 1.0 for acc in y):
        print("\033[91mWarning: Detected accuracy values < 1.0. Please check if accuracy should be in percentage (0-100) instead of (0-1).\033[0m")
    
    # Sort by rho
    sorted_pairs = sorted(zip(rho, y), key=lambda x: x[0])
    sorted_rho, sorted_y = zip(*sorted_pairs)
    sorted_rho, sorted_y = list(sorted_rho), list(sorted_y)
    
    # Filter by y_min threshold (y_1 - y_min_offset)
    y_1 = sorted_y[0]
    assert y_1 - sorted_y[-1] <= y_min_offset, f"Accuracy degradation is too large: minimum accuracy should be at least {y_min_offset:.2f} lower than the maximum accuracy. Max Acc: {y_1}, min Acc: {sorted_y[-1]}"
    y_min = y_1 - y_min_offset
    filtered_pairs = [(r, acc) for r, acc in zip(sorted_rho, sorted_y) if acc >= y_min]
    assert len(filtered_pairs) > 0, f"No valid pairs after filtering with y_min={y_min}"
    
    filtered_rho, filtered_y = zip(*filtered_pairs)
    filtered_rho, filtered_y = list(filtered_rho), list(filtered_y)
    
    # Calculate AUP: first term + trapezoidal sum
    aup = filtered_rho[0] * filtered_y[0]
    formula_parts = [f"{filtered_rho[0]:.2f} * {filtered_y[0]:.2f}"]
    
    for i in range(1, len(filtered_rho)):
        y_i = filtered_y[i]
        y_prev = filtered_y[i-1]
        w_i = weight_function(y_i, y_max, alpha)
        w_prev = weight_function(y_prev, y_max, alpha)
        term = 0.5 * (filtered_rho[i] - filtered_rho[i-1]) * (y_i * w_i + y_prev * w_prev)
        aup += term
        formula_parts.append(f"({filtered_rho[i]:.2f}-{filtered_rho[i-1]:.2f}) * ({y_i:.2f} * {w_i:.4f} + {y_prev:.2f} * {w_prev:.4f})")

    if is_print:
        formula = f"    AUP = " + " + ".join(formula_parts) + f" = {aup:.2f}"
        print(formula)
    
    return aup
