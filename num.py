import numpy as np
import math
import fractions as frac 

def sin(x):
    return math.sin(x)
def cos(x):
    return math.cos(x)
def tan(x):
    return math.tan(x)
def cot(x):
    return 1 / math.tan(x)
def sec(x):
    return 1 / math.cos(x)
def csc(x):
    return 1 / math.sin(x)

def asin(x):
    return math.asin(x)
def acos(x):
    return math.acos(x)
def atan(x):
    return math.atan(x)
def acot(x):
    return math.atan(1 / x)
def asec(x):
    return math.acos(1 / x)
def acsc(x):
    return math.asin(1 / x)

def sinh(x):
    return np.sinh(x)
def cosh(x):
    return np.cosh(x)
def tanh(x):
    return np.tanh(x)
def coth(x):
    return 1 / np.tanh(x)
def sech(x):
    return 1 / np.cosh(x)
def csch(x):
    return 1 / np.sinh(x)

def asinh(x):
    return np.arcsinh(x)
def acosh(x):
    return np.arccosh(x)
def atanh(x):
    return np.arctanh(x)
def acoth(x):
    return np.arctanh(1 / x)
def asech(x):
    return np.arccosh(1 / x)
def acsch(x):
    return np.arcsinh(1 / x)

def atan2(y, x):
    return np.arctan2(y, x)

def exp(x):
    return math.e ** x
def ln(x):
    return math.log(x, math.e)
def log(x, base):
    return math.log(x, base)
def log10(x):
    return math.log(x, 10)
def log2(x):
    return math.log(x, 2)
def pow(x, y):
    return x ** y

def sqrt(x):
    return math.sqrt(x)
def cbrt(x):
    return x ** (1/3)
def factorial(n):
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    elif n != int(n):
        raise ValueError("Factorial is only defined for non-negative integers")
    return math.factorial(n)
def gcd(a, b):
    while b:
        a, b = b, a % b
    return abs(a)
def lcm(a, b):
    return abs(a * b) // gcd(a, b)
def is_prime(n):
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
def prime_factors(n):
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
def fibonacci(n):
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

def is_perfect_square(n):
    if n < 0:
        return False
    root = int(math.sqrt(n))
    return root * root == n
def is_fibonacci(n):
    if n < 0:
        return False
    a, b = 0, 1
    while b < n:
        a, b = b, a + b
    return b == n or a == n
def is_armstrong(n):
    if n < 0:
        return False
    digits = str(n)
    power = len(digits)
    return sum(int(digit) ** power for digit in digits) == n
def is_palindrome(n):
    if n < 0:
        return False
    s = str(n)
    return s == s[::-1]
def reverse_number(n):
    if n < 0:
        raise ValueError("Reverse is not defined for negative numbers")
    return int(str(n)[::-1])
def sum_of_digits(n):
    if n < 0:
        raise ValueError("Sum of digits is not defined for negative numbers")
    return sum(int(digit) for digit in str(n))
def product_of_digits(n):
    if n < 0:
        raise ValueError("Product of digits is not defined for negative numbers")
    product = 1
    for digit in str(n):
        product *= int(digit)
    return product
def count_digits(n):
    if n < 0:
        return len(str(-n))
    return len(str(n))

def floor(x):
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
    if k < 0 or k > n:
        raise ValueError("k must be in the range [0, n]")
    return math.comb(n, k)
def permutations(n, k):
    if k < 0 or k > n:
        raise ValueError("k must be in the range [0, n]")
    return math.perm(n, k)

def is_nan(x):
    return math.isnan(x)
def is_inf(x):
    return math.isinf(x) 
def is_finite(x):
    if math.isinf(x) or math.isnan(x):
        return False
    else:
        return True
def to_fraction(x):
    if isinstance(x, int):
        return frac.Fraction(x)
    elif isinstance(x, float):
        return frac.Fraction.from_float(x).limit_denominator()
    else:
        raise TypeError("Input must be an integer or a float")
def from_fraction(frac_obj):
    if isinstance(frac_obj, frac.Fraction):
        return float(frac_obj)
    else:
        raise TypeError("Input must be a Fraction object")
def gcd_fraction(frac1, frac2):
    if isinstance(frac1, frac.Fraction) and isinstance(frac2, frac.Fraction):
        return frac.gcd(frac1.numerator, frac2.numerator)
    else:
        raise TypeError("Inputs must be Fraction objects")
def lcm_fraction(frac1, frac2):
    if isinstance(frac1, frac.Fraction) and isinstance(frac2, frac.Fraction):
        return frac.lcm(frac1.numerator, frac2.numerator)
    else:
        raise TypeError("Inputs must be Fraction objects")
def dist(p1, p2):
    if len(p1) != len(p2):
        raise ValueError("Points must have the same dimension")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))
def angle_between(v1, v2):
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
def area_of_square(side):
    if side < 0:
        raise ValueError("Side cannot be negative")
    return side ** 2

def is_even(n):
    return n % 2 == 0
def is_odd(n):
    return n % 2 != 0
def is_positive(n):
    return n > 0
def is_negative(n):
    return n < 0
def is_zero(n):
    return n == 0
def is_integer(n):
    if isinstance(n, int):
        return True
    elif isinstance(n, float):
        return n.is_integer()
    else:
        raise TypeError("Input must be an integer or a float")
def is_float(n):
    if isinstance(n, float):
        return True
    elif isinstance(n, int):
        return False
    else:
        raise TypeError("Input must be an integer or a float")
def is_complex(n):
    if isinstance(n, complex):
        return True
    elif isinstance(n, (int, float)):
        return False
    else:
        raise TypeError("Input must be a complex number, integer, or float")
def is_real(n):
    if isinstance(n, (int, float)):
        return True
    elif isinstance(n, complex):
        return n.imag == 0
    else:
        raise TypeError("Input must be a real number or a complex number with zero imaginary part")
def is_rational(n):
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
    if n in [math.pi, math.e]:
        return True
    else:
        return not is_rational(n)
def is_prime_power(n):
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
    if n < 1:
        raise ValueError("Input must be a positive integer")
    divisors_sum = sum(i for i in range(1, n) if n % i == 0)
    return divisors_sum == n
def factorize(n):
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
    if n < 1:
        raise ValueError("Input must be a positive integer")
    divisors_sum = sum(i for i in range(1, n) if n % i == 0)
    return divisors_sum > n
def is_deficient(n):
    if n < 1:
        raise ValueError("Input must be a positive integer")
    divisors_sum = sum(i for i in range(1, n) if n % i == 0)
    return divisors_sum < n


pi = math.pi
e = math.e
tau = 2 * math.pi

if __name__ == "__main__":
    print(((math.nextafter(1.0, 3.0)-1)*10000000000000000 - 2.220446049250313) * 10000000000000000000) # returns the next float after 1.0 towards 2.0