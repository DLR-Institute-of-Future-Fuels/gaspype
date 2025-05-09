from . import fluid_system


def syngas(*additional_gases: str) -> fluid_system:
    """
    This function returns a fluid system containing the following gases: H2, H2O, CO2,
    CO, CH4, C2H6, C2H4, C2H2 (acetylene), CH3OH, C2H5OH, C3H8.
    Any additional gases provided as arguments will also be included in the fluid system.

    Args:
        additional_gases: Any number of additional gases to include in the fluid system.

    Returns:
        fluid_system: A fluid system containing the specified gases.
    """
    return fluid_system(['H2', 'H2O', 'CO2', 'CO', 'CH4', 'C2H6',
                         'C2H4', 'C2H2', 'CH3OH', 'C2H5OH', 'C3H8'] + list(additional_gases))


def syngas_simple(*additional_gases: str) -> fluid_system:
    """
    This function returns a simplified fluid system containing the following gases: H2,
    H2O, CO2, CO, CH4.
    Any additional gases provided as arguments will also be included in the fluid system.

    Args:
        additional_gases: Any number of additional gases to include in the fluid system.

    Returns:
        fluid_system: A simplified fluid system containing the specified gases.
    """
    return fluid_system(['H2', 'H2O', 'CO2', 'CO', 'CH4'] + list(additional_gases))


def synthetic_air(*additional_gases: str) -> fluid_system:
    """
    This function returns a fluid system containing synthetic air (N2 and O2).
    Any additional gases provided as arguments will also be included in the fluid system.

    Args:
        additional_gases: Any number of additional gases to include in the fluid system.

    Returns:
        fluid_system: A fluid system containing synthetic air and any additional specified gases.
    """
    return fluid_system(['N2', 'O2'] + list(additional_gases))


def air(*additional_gases: str) -> fluid_system:
    """
    This function returns a fluid system containing atmospheric air (N2, O2, Ar and H2O).
    Any additional gases provided as arguments will also be included in the fluid system.

    Args:
        additional_gases: Any number of additional gases to include in the fluid system.

    Returns:
        fluid_system: A fluid system containing atmospheric air and any additional specified gases.
    """
    return fluid_system(['N2', 'O2', 'Ar', 'H2O'] + list(additional_gases))


def dry_air(*additional_gases: str) -> fluid_system:
    """
    This function returns a fluid system containing atmospheric air (N2, O2, and Ar).
    Any additional gases provided as arguments will also be included in the fluid system.

    Args:
        additional_gases: Any number of additional gases to include in the fluid system.

    Returns:
        fluid_system: A fluid system containing atmospheric air and any additional specified gases.
    """
    return fluid_system(['N2', 'O2', 'Ar'] + list(additional_gases))


def electrolysis(*additional_gases: str) -> fluid_system:
    """
    This function returns a fluid system containing the gases produced during
    electrolysis (H2, H2O, O2, N2). Any additional gases provided as arguments
    will also be included in the fluid system.

    Args:
        additional_gases: Any number of additional gases to include in the fluid system.

    Returns:
        fluid_system: A fluid system containing the gases produced during electrolysis and
        any additional specified gases.
    """
    return fluid_system(['H2', 'H2O', 'O2', 'N2'] + list(additional_gases))


def methanol_synthesis(*additional_gases: str) -> fluid_system:
    """
    This function returns a fluid system containing the gases involved in methanol synthesis
    (H2, H2O, O2, CO2, CO, CH4, CH3OH, C3H6O, C2H4O2). Any additional gases provided
    as arguments will also be included in the fluid system.

    Args:
        additional_gases: Any number of additional gases to include in the fluid system.

    Returns:
        fluid_system: A fluid system containing the gases involved in methanol
        synthesis and any additional specified gases.
    """
    return fluid_system(['H2', 'H2O', 'O2', 'CO2', 'CO',
                         'CH4', 'CH3OH', 'C3H6O', 'C2H4O2'] + list(additional_gases))
