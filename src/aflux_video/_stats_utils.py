def get_sample_size(population_size: int) -> int:
    """Calculates the recommended sample size to obtain reliable proxies for Mean, Std Dev, and Min/Max.

    Returns the greater of the Mean, Std Dev, and Min/Max requirement, capped by the total population size.
    Finite Population Correction (FPC) is taken into account for Mean and Std Dev.

    Formulas & Assumptions:
    - Mean:
      - Assumption: 95% Confidence, Margin of Error = 0.1 * Population Sigma.
      - Base Requirement: ~385 samples.
      - Base Formula: n0 = (Z^2 * sigma^2) / E^2
      - FPC Formula: n = n0 / (1 + (n0 - 1) / N)

    - Std Dev:
      - Assumption: 95% Confidence, within 10% of true sigma.
      - Base Requirement: ~193 samples (converges faster than Mean, so not a constraint).
      - Base Formula: n0 = 0.5 * (Z / relative_error)^2
      - FPC Formula: n = n0 / (1 + n0 / N)

    - Min/Max:
      - Assumption: 95% Confidence of capturing at least one value in the top 1%.
      - Base Requirement: ~366 samples.
      - Formula: n = ln(1 - sqrt(C)) / ln(1 - p)
        - Asymptotic via small p = 0.01
        - sqrt(C) due to non-independent event between Min and Max

    Args:
        population_size: The total size of the population (N).

    Returns:
        The recommended sample size, capped at the population size.
    """

    BASE_MEAN_SAMPLE_SIZE = 385
    BASE_MIN_MAX_SAMPLE_SIZE = 366

    if population_size <= 0:
        return 0

    fpc_numerator = population_size * BASE_MEAN_SAMPLE_SIZE
    fpc_denominator = population_size + BASE_MEAN_SAMPLE_SIZE - 1
    fpc_mean_sample_size = (fpc_numerator - 1) // fpc_denominator + 1

    sample_size = max(fpc_mean_sample_size, BASE_MIN_MAX_SAMPLE_SIZE)
    sample_size = min(sample_size, population_size)
    return sample_size


def get_sample_indices(population_size: int) -> list[int]:
    sample_size = get_sample_size(population_size)
    if sample_size <= 1:
        return list(range(sample_size))
    if sample_size >= population_size:
        return list(range(population_size))
    step = (population_size - 1) / (sample_size - 1)
    return [round(i * step) for i in range(sample_size)]
