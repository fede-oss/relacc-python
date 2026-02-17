from decimal import Decimal, ROUND_HALF_UP


class MathUtil:
    """Math functions."""

    @staticmethod
    def roundTo(num, prec=None):
        if prec is None:
            prec = 3
        quant = Decimal("1").scaleb(-prec)
        return float(Decimal(str(num)).quantize(quant, rounding=ROUND_HALF_UP))

    @staticmethod
    def factorial(num):
        val = 1
        for n in range(2, num + 1):
            val *= n
        return val
