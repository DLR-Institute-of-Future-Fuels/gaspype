class celsius_helper():
    def __mul__(self, other: float | int) -> float:
        return other + 273.15

    def __rmul__(self, other: float | int) -> float:
        return other + 273.15


C = celsius_helper()
