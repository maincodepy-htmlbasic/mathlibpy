# --- Additional math module wrappers ---
def fabs(x: int | float) -> float:
    """Returns the absolute value of x as a float."""
    return math.fabs(x)

def fmod(x: int | float, y: int | float) -> float:
    """Returns the remainder of x / y as a float."""
    return math.fmod(x, y)

def isfinite(x: int | float) -> bool:
    """Returns True if x is neither an infinity nor a NaN."""
    return math.isfinite(x)

def isnan(x: int | float) -> bool:
    """Returns True if x is NaN (not a number)."""
    return math.isnan(x)

def isinf(x: int | float) -> bool:
    """Returns True if x is positive or negative infinity."""
    return math.isinf(x)

def log10(x: int | float) -> float:
    """Returns the base-10 logarithm of x."""
    return math.log10(x)

def log(x: int | float, base: int | float = math.e) -> float:
    """Returns the logarithm of x to the given base (default is natural log)."""
    return math.log(x, base)

def remainder(x: int | float, y: int | float) -> float:
    """Returns the IEEE 754-style remainder of x with respect to y."""
    return math.remainder(x, y)

def ulp(x: int | float) -> float:
    """Returns the value of the least significant bit of x."""
    return math.ulp(x)

# --- Additional numpy module wrappers ---
def absolute(x):
    """Returns the absolute value element-wise."""
    return np.absolute(x)

def angle(x, deg: bool = False):
    """Returns the angle of the complex argument."""
    return np.angle(x, deg=deg)

def arccos(x):
    """Returns the element-wise arccosine of x."""
    return np.arccos(x)

def arccosh(x):
    """Returns the element-wise inverse hyperbolic cosine of x."""
    return np.arccosh(x)

def arcsin(x):
    """Returns the element-wise arcsine of x."""
    return np.arcsin(x)

def arcsinh(x):
    """Returns the element-wise inverse hyperbolic sine of x."""
    return np.arcsinh(x)

def arctan(x):
    """Returns the element-wise arctangent of x."""
    return np.arctan(x)

def arctan2(y, x):
    """Returns the element-wise arctangent of y/x."""
    return np.arctan2(y, x)

def arctanh(x):
    """Returns the element-wise inverse hyperbolic tangent of x."""
    return np.arctanh(x)

def ceil_np(x):
    """Returns the ceiling of each element in x."""
    return np.ceil(x)

def conj(x):
    """Returns the complex conjugate, element-wise."""
    return np.conj(x)

def conjugate(x):
    """Returns the complex conjugate, element-wise."""
    return np.conjugate(x)

def copysign_np(x1, x2):
    """Change the sign of x1 to that of x2, element-wise."""
    return np.copysign(x1, x2)

def deg2rad(x):
    """Converts angles from degrees to radians."""
    return np.deg2rad(x)

def degrees_np(x):
    """Converts angles from radians to degrees."""
    return np.degrees(x)

def exp_np(x):
    """Calculates the exponential of all elements in the input array."""
    return np.exp(x)

def exp2(x):
    """Calculates 2**x for all elements in the input array."""
    return np.exp2(x)

def expm1_np(x):
    """Calculates exp(x) - 1 for all elements in the input array."""
    return np.expm1(x)

def floor_np(x):
    """Returns the floor of each element in x."""
    return np.floor(x)

def hypot_np(x1, x2):
    """Returns sqrt(x1**2 + x2**2) element-wise."""
    return np.hypot(x1, x2)

def isfinite_np(x):
    """Tests element-wise for finiteness (not infinity or not Not a Number)."""
    return np.isfinite(x)

def isinf_np(x):
    """Tests element-wise for positive or negative infinity."""
    return np.isinf(x)

def isnan_np(x):
    """Tests element-wise for NaN and returns result as a boolean array."""
    return np.isnan(x)

def log_np(x):
    """Natural logarithm, element-wise."""
    return np.log(x)

def log10_np(x):
    """Base-10 logarithm, element-wise."""
    return np.log10(x)

def log1p_np(x):
    """Returns log(1 + x) element-wise."""
    return np.log1p(x)

def log2_np(x):
    """Base-2 logarithm, element-wise."""
    return np.log2(x)

def maximum(x1, x2):
    """Element-wise maximum of array elements."""
    return np.maximum(x1, x2)

def minimum(x1, x2):
    """Element-wise minimum of array elements."""
    return np.minimum(x1, x2)

def mod_np(x1, x2):
    """Returns element-wise remainder of division."""
    return np.mod(x1, x2)

def rad2deg(x):
    """Converts angles from radians to degrees."""
    return np.rad2deg(x)

def remainder_np(x1, x2):
    """Returns element-wise remainder of division as per IEEE 754."""
    return np.remainder(x1, x2)

def rint(x):
    """Rounds elements of the array to the nearest integer."""
    return np.rint(x)

def sign_np(x):
    """Returns an element-wise indication of the sign of a number."""
    return np.sign(x)

def sqrt_np(x):
    """Returns the non-negative square-root of an array, element-wise."""
    return np.sqrt(x)

def square(x):
    """Returns the element-wise square of the input."""
    return np.square(x)

def trunc_np(x):
    """Returns the truncated value of the input, element-wise."""
    return np.trunc(x)
"""

This module provides a comprehensive set of mathematical functions and utilities, including:

- Trigonometric functions (sin, cos, tan, cot, sec, csc) and their inverses (asin, acos, atan, acot, asec, acsc)
- Hyperbolic functions (sinh, cosh, tanh, coth, sech, csch) and their inverses (asinh, acosh, atanh, acoth, asech, acsch)
- Exponential and logarithmic functions (exp, expm1, log1p, log2, pow, sqrt, cbrt)
- Rounding and sign functions (floor, ceil, sign, trunc, copysign, modf, frexp, ldexp, nextafter)
- Angle conversions (radians, degrees)
- Distance and vector operations (hypot, dist, angle_between)
- Number theory utilities (gcd, lcm, is_prime, prime_factors, is_perfect_square, is_fibonacci, is_armstrong, is_palindrome, reverse_number, sum_of_digits, product_of_digits, count_digits, factorize)
- Combinatorial functions (factorial, combinations, permutations)
- Number property checks (is_deficient, is_odd, is_positive, is_negative, is_zero, is_integer, is_float, is_complex, is_real, is_rational, is_irrational, is_prime_power, is_perfect_number, is_abundant)
- Floating-point checks (is_nan, is_inf, is_finite)
- Fraction utilities (to_fraction, from_fraction, gcd_fraction, lcm_fraction)
- Geometry utilities (area_of_circle, circumference_of_circle, area_of_rectangle, perimeter_of_rectangle, area_of_triangle, pythagorean_theorem)
- Mathematical constants (pi, e, tau)

Dependencies:
    - numpy
    - math
    - fractions

This module is intended for educational and general-purpose mathematical computations.

num.py - Numeric and mathematical utility functions.

Provides trigonometric, logarithmic, combinatorial, number theory, and other mathematical functions.
Includes constants such as pi, e, and tau.
"""
import numpy as np
import numpy as np
import math
import fractions as frac 



def sin(x: int | float) -> float:
    """Returns the sine of x (in radians)."""
    return math.sin(x)
def cos(x: int | float) -> float:
    """Returns the cosine of x (in radians)."""
    return math.cos(x)
def tan(x: int | float) -> float:
    """Returns the tangent of x (in radians)."""
    return math.tan(x)
def cot(x: int | float) -> float:
    """Returns the cotangent of x (in radians)."""
    return 1 / math.tan(x)
def sec(x: int | float) -> float:
    """Returns the secant of x (in radians)."""
    return 1 / math.cos(x)
def csc(x: int | float) -> float:
    """Returns the cosecant of x (in radians)."""
    return 1 / math.sin(x)

def asin(x: int | float) -> float:
    """Returns the arcsine of x (in radians)."""
    return math.asin(x)
def acos(x: int | float) -> float:
    """Returns the arccosine of x (in radians)."""
    return math.acos(x)
def atan(x: int | float) -> float:
    """Returns the arctangent of x (in radians)."""
    return math.atan(x)
def acot(x: int | float) -> float:
    """Returns the arccotangent of x (in radians)."""
    return math.atan(1 / x)
def asec(x: int | float) -> float:
    """Returns the arcsecant of x (in radians)."""
    return math.acos(1 / x)
def acsc(x: int | float) -> float:
    """Returns the arccosecant of x (in radians)."""
    return math.asin(1 / x)

def sinh(x: int | float) -> float:
    """Returns the hyperbolic sine of x."""
    return np.sinh(x)
def cosh(x: int | float) -> float:
    """Returns the hyperbolic cosine of x."""
    return np.cosh(x)
def tanh(x: int | float) -> float:
    """Returns the hyperbolic tangent of x."""
    return np.tanh(x)
def coth(x: int | float) -> float:
    """Returns the hyperbolic cotangent of x."""
    return 1 / np.tanh(x)
def sech(x: int | float) -> float:
    """Returns the hyperbolic secant of x."""
    return 1 / np.cosh(x)
def csch(x: int | float) -> float:
    """Returns the hyperbolic cosecant of x."""
    return 1 / np.sinh(x)



def asinh(x: int | float) -> float:
    """Returns the inverse hyperbolic sine of x."""
    return np.arcsinh(x)
def acosh(x: int | float) -> float:
    """Returns the inverse hyperbolic cosine of x."""
    return np.arccosh(x)
def atanh(x: int | float) -> float:
    """Returns the inverse hyperbolic tangent of x."""
    return np.arctanh(x)
def acoth(x: int | float) -> float:
    """Returns the inverse hyperbolic cotangent of x."""
    return np.arctanh(1 / x)
def asech(x: int | float) -> float:
    """Returns the inverse hyperbolic secant of x."""
    return np.arccosh(1 / x)
def acsch(x: int | float) -> float:
    """Returns the inverse hyperbolic cosecant of x."""
    return np.arcsinh(1 / x)

def atan2(
        y: int | float,
        x: int | float
        ) -> float:
    """Returns the arctangent of y/x, handling the quadrant correctly."""
    return np.arctan2(y, x)

def exp(x: int | float) -> float:
    """Returns e raised to the power of x."""
    return math.exp(x)
def expm1(x: int | float) -> float:
    """Returns exp(x) - 1."""
    return math.expm1(x)
def log1p(x: int | float) -> float:
    """Returns log(1 + x)."""
    return math.log(x, 10)

def floor(x: int | float) -> int:
    """Returns the floor of x as an integer."""
    return math.floor(x)
def ceil(x: int | float) -> int:
    """Returns the ceiling of x as an integer."""
    return math.ceil(x)
def sign(x: int | float) -> int:
    """Returns 1 if x > 0, -1 if x < 0, 0 if x == 0."""
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
def radians(degrees: int | float) -> float:
    """Converts degrees to radians."""
    return math.radians(degrees)
def degrees(radians: int | float) -> float:
    """Converts radians to degrees."""
    return math.degrees(radians)
def hypot(
        x: int | float,
        y: int | float
        ) -> float:
    """Returns the Euclidean norm, sqrt(x*x + y*y)."""
    return math.hypot(x, y)
def isclose(
        a: int | float,
        b: int | float,
        rel_tol: float = 1e-09,
        abs_tol: float =0.0) -> bool:
    """Returns True if a and b are close in value."""
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
def copysign(
        x: int | float,
        y: int | float
        ) -> float:
    """Returns x with the sign of y."""
    return math.copysign(x, y)
def modf(x: int | float) -> tuple[float, float]:
    """Returns the fractional and integer parts of x."""
    return math.modf(x)
def trunc(
        x: int | float,
        y: int | float
        ) -> float:
    """Truncates x/y to an integer and multiplies by y."""
    return math.trunc(x / y) * y
def frexp(x: int | float) -> tuple[float, int]:
    """Returns the mantissa and exponent of x."""
    return math.frexp(x)
def ldexp(
        x: int | float,
        i: int | float
        ) -> float:
    """Returns x * (2**i)."""
    return math.ldexp(x, i)
def nextafter(
        x: int | float,
        y: int | float
        ) -> float:
    """Returns the next floating-point value after x towards y."""
    return math.nextafter(x, y)
def log2(x: int | float) -> float:
    """Returns the base-2 logarithm of x."""
    if x <= 0:
        raise ValueError("Logarithm base 2 is only defined for x > 0")
    return math.log(x, 2)
def pow(
        x: int | float,
        y: int | float
        ) -> float:
    """Returns x raised to the power of y for (x,y)"""
    return x ** y

def sqrt(x: int | float) -> float:
    """Returns the square root of x"""
    return math.sqrt(x)
def cbrt(x: int | float) -> float:
    """Returns the cube root of x"""
    return x ** (1/3)
def factorial(n: int) -> int:
    """Returns the factorial of x"""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    elif n != int(n):
        raise ValueError("Factorial is only defined for non-negative integers")
    return math.factorial(n)
def gcd(
        a: int | float,
        b: int | float
        ) -> int:
    """Returns the greatest common divisor (gcd) of a and b for (a,b)"""
    while b:
        a, b = b, a % b
    return abs(a)
def lcm(
        a: int | float,
        b: int | float
        ) -> int:
    """Returns the lowest common multiple (lcm) of a and b for (a,b)"""
    return abs(a * b) // gcd(a, b)
def is_prime(n: int) -> bool:
    """Returns True if prime, False if not."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True
def prime_factors(n: int) -> list[int]:
    """Returns the list of prime factors of n."""
    factors = []
    if n <= 1:
        return factors
    while n % 2 == 0:
        factors.append(2)
        n //= 2
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        while n % i == 0:
            factors.append(i)
            n //= i
    if n > 2:
        factors.append(n)
    return factors
def fibonacci(n: int) -> int:
    if n < 0:
        raise ValueError("Fibonacci is not defined for negative numbers")
    elif n == 0:
        return 0
    elif n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

def is_perfect_square(n: int) -> bool:
    if n < 0:
        return False
    root = int(math.sqrt(n))
    return root * root == n
def is_fibonacci(n: int) -> bool:
    if n < 0:
        return False
    a, b = 0, 1
    while b < n:
        a, b = b, a + b
    return b == n or a == n
def is_armstrong(n: int) -> bool:
    if n < 0:
        return False
    digits = str(n)
    power = len(digits)
    return sum(int(digit) ** power for digit in digits) == n
def is_palindrome(n: int) -> bool:
    if n < 0:
        return False
    s = str(n)
    return s == s[::-1]
def reverse_number(n: int) -> int:
    if n < 0:
        raise ValueError("Reverse is not defined for negative numbers")
    return int(str(n)[::-1])
def sum_of_digits(n: int) -> int:
    if n < 0:
        raise ValueError("Sum of digits is not defined for negative numbers")
    return sum(int(digit) for digit in str(n))
def product_of_digits(n: int) -> int:
    if n < 0:
        raise ValueError("Product of digits is not defined for negative numbers")
    product = 1
    for digit in str(n):
        product *= int(digit)
    return product
def count_digits(n: int) -> int:
    if n < 0:
        return len(str(-n))
    return len(str(n))

def floor(x: int | float) -> int:
    return math.floor(x)
def ceil(x):
    return math.ceil(x)
def sign(x):
    if x > 0:
        return 1
    elif x < 0:
        return -1
    else:
        return 0
def radians(degrees):
    return math.radians(degrees)
def degrees(radians):
    return math.degrees(radians)
def hypot(x, y):
    return math.hypot(x, y)
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
def copysign(x, y):
    return math.copysign(x, y)
def modf(x):
    return math.modf(x)
def trunc(x, y):
    return math.trunc(x / y) * y
def frexp(x):
    return math.frexp(x)
def ldexp(x, i):
    return math.ldexp(x, i)
def nextafter(x, y):
    return math.nextafter(x, y)


def combinations(n, k):
    """Returns the number of ways to choose k items from n without repetition and without order."""
    if k < 0 or k > n:
        raise ValueError("k must be in the range [0, n]")
    return math.comb(n, k)
def permutations(n, k):
    """Returns the number of ways to choose k items from n with order."""
    if k < 0 or k > n:
        raise ValueError("k must be in the range [0, n]")
    return math.perm(n, k)

def is_nan(x):
    """Returns True if x is NaN (not a number)."""
    return math.isnan(x)
def is_inf(x):
    """Returns True if x is infinite."""
    return math.isinf(x)
def is_finite(x):
    """Returns True if x is a finite number."""
    if math.isinf(x) or math.isnan(x):
        return False
    else:
        return True
def to_fraction(x):
    """Converts an integer or float to a Fraction object."""
    if isinstance(x, int):
        return frac.Fraction(x)
    elif isinstance(x, float):
        return frac.Fraction.from_float(x).limit_denominator()
    else:
        raise TypeError("Input must be an integer or a float")
def from_fraction(frac_obj):
    """Converts a Fraction object to a float."""
    if isinstance(frac_obj, frac.Fraction):
        return float(frac_obj)
    else:
        raise TypeError("Input must be a Fraction object")
def gcd_fraction(frac1, frac2):
    """Returns the GCD of the numerators of two Fraction objects."""
    if isinstance(frac1, frac.Fraction) and isinstance(frac2, frac.Fraction):
        return math.gcd(frac1.numerator, frac2.numerator)
    else:
        raise TypeError("Inputs must be Fraction objects")
def lcm_fraction(frac1, frac2):
    """Returns the LCM of the numerators of two Fraction objects."""
    if isinstance(frac1, frac.Fraction) and isinstance(frac2, frac.Fraction):
        return frac.lcm(frac1.numerator, frac2.numerator)
    else:
        raise TypeError("Inputs must be Fraction objects")
def dist(p1, p2):
    """Returns the Euclidean distance between two points p1 and p2."""
    if len(p1) != len(p2):
        raise ValueError("Points must have the same dimension")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
def angle_between(v1, v2):
    """Returns the angle in radians between two vectors v1 and v2."""
    if len(v1) != len(v2):
        raise ValueError("Vectors must have the same dimension")
    dot_product = sum(a * b for a, b in zip(v1, v2))
    magnitude_v1 = math.sqrt(sum(a ** 2 for a in v1))
    magnitude_v2 = math.sqrt(sum(b ** 2 for b in v2))
    if magnitude_v1 == 0 or magnitude_v2 == 0:
        raise ValueError("Cannot compute angle with zero-length vector")
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)
    return math.acos(cos_theta)
def area_of_circle(radius):
    if radius < 0:
        raise ValueError("Radius cannot be negative")
    return math.pi * radius ** 2
def circumference_of_circle(radius):
    if radius < 0:
        raise ValueError("Radius cannot be negative")
    return 2 * math.pi * radius
def area_of_rectangle(length, width):
    if length < 0 or width < 0:
        raise ValueError("Length and width cannot be negative")
    return length * width
def perimeter_of_rectangle(length, width):
    if length < 0 or width < 0:
        raise ValueError("Length and width cannot be negative")
    return 2 * (length + width)
def area_of_triangle(base, height):
    if base < 0 or height < 0:
        raise ValueError("Base and height cannot be negative")
    return 0.5 * base * height
def pythagorean_theorem(a, b):
    if a < 0 or b < 0:
        raise ValueError("Sides cannot be negative")
    return math.sqrt(a ** 2 + b ** 2)

def is_deficient(n):
    """Returns True if n is even."""
    return n % 2 == 0
def is_odd(n):
    """Returns True if n is odd."""
    return n % 2 != 0
def is_positive(n):
    """Returns True if n is positive."""
    return n > 0
def is_negative(n):
    """Returns True if n is negative."""
    return n < 0
def is_zero(n):
    """Returns True if n is zero."""
    return n == 0
def is_integer(n):
    """Returns True if n is an integer."""
    if isinstance(n, int):
        return True
    elif isinstance(n, float):
        return n.is_integer()
    else:
        raise TypeError("Input must be an integer or a float")
def is_float(n):
    """Returns True if n is a float."""
    if isinstance(n, float):
        return True
    elif isinstance(n, int):
        return False
    else:
        raise TypeError("Input must be an integer or a float")
def is_complex(n):
    """Returns True if n is a complex number."""
    if isinstance(n, complex):
        return True
    elif isinstance(n, (int, float)):
        return False
    else:
        raise TypeError("Input must be a complex number, integer, or float")
def is_real(n):
    """Returns True if n is a real number (int or float, or complex with zero imaginary part)."""
    if isinstance(n, (int, float)):
        return True
    elif isinstance(n, complex):
        return n.imag == 0
    else:
        raise TypeError("Input must be a real number or a complex number with zero imaginary part")
def is_rational(n):
    """Returns True if n is a rational number (int or Fraction)."""
    if isinstance(n, int):
        return True
    elif isinstance(n, float):
        return False
    elif isinstance(n, frac.Fraction):
        return n.denominator != 0
    elif isinstance(n, complex):
        return n.imag == 0 and n.real.is_rational()
    else:
        raise TypeError("Input must be an integer, float, or Fraction object")
def is_irrational(n):
    """Returns True if n is an irrational number (e.g., pi, e, or not rational)."""
    if n in [math.pi, math.e]:
        return True
    else:
        return not is_rational(n)
def is_prime_power(n):
    """Returns True if n is a power of a prime number."""
    if n < 1:
        raise ValueError("Input must be a positive integer")
    for i in range(2, int(math.sqrt(n)) + 1):
        power = 0
        while n % i == 0:
            n //= i
            power += 1
        if n == 1 and power > 0:
            return True
    return n == 1
def is_perfect_number(n):
    """Returns True if n is a perfect number (equal to the sum of its proper divisors)."""
    if n < 1:
        raise ValueError("Input must be a positive integer")
    divisors_sum = sum(i for i in range(1, n) if n % i == 0)
    return divisors_sum == n
def factorize(n):
    """Returns the list of prime factors of n."""
    if n < 1:
        raise ValueError("Input must be a positive integer")
    factors = []
    for i in range(2, int(math.sqrt(n)) + 1):
        while n % i == 0:
            factors.append(i)
            n //= i
    if n > 1:
        factors.append(n)
    return factors
def is_abundant(n):
    """Returns True if n is an abundant number (sum of divisors > n)."""
    if n < 1:
        raise ValueError("Input must be a positive integer")
    divisors_sum = sum(i for i in range(1, n) if n % i == 0)
    return divisors_sum > n
def is_deficient(n):
    """Returns True if n is a deficient number (sum of divisors < n)."""
    if n < 1:
        raise ValueError("Input must be a positive integer")
    divisors_sum = sum(i for i in range(1, n) if n % i == 0)
    return divisors_sum < n

def factorize(n):
    """Returns the list of factors or divisors of n."""
    if n < 1:
        raise ValueError("Input must be a positive integer")
    factors = []
    for i in range(1, n + 1):
        if n % i == 0:
            factors.append(i)
    return factors

num_pi = math.pi
num_e = math.e
num_tau = 2 * math.pi

if __name__ == "__main__":
    print(((math.nextafter(1.0, 3.0)-1)*10000000000000000 - 2.220446049250313) * 10000000000000000000) # returns the next float after 1.0 towards 2.0