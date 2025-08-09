import sympy as sy
import sympy.abc as syabc

# --- Additional sympy wrappers for bijection ---
def symbols(names, **kwargs):
    """Create symbolic variables (symbols) for symbolic computation."""
    return sy.symbols(names, **kwargs)

def Symbol(name, **kwargs):
    """Create a single symbolic variable (Symbol)."""
    return sy.Symbol(name, **kwargs)

def Eq(lhs, rhs=0):
    """Create a symbolic equation lhs = rhs."""
    return sy.Eq(lhs, rhs)

def diff(expr, *symbols, **kwargs):
    """Differentiate expr with respect to one or more symbols."""
    return sy.diff(expr, *symbols, **kwargs)

def integrate(expr, *symbols, **kwargs):
    """Integrate expr with respect to one or more symbols."""
    return sy.integrate(expr, *symbols, **kwargs)

def limit(expr, symbol, point, dir="+"):
    """Compute the limit of expr as symbol approaches point from the given direction."""
    return sy.limit(expr, symbol, point, dir=dir)

def expand_log(expr, **kwargs):
    """Expand logarithmic expressions."""
    return sy.expand_log(expr, **kwargs)

def expand_power_exp(expr, **kwargs):
    """Expand exponents in expressions."""
    return sy.expand_power_exp(expr, **kwargs)

def expand_power_base(expr, **kwargs):
    """Expand powers in the base of expressions."""
    return sy.expand_power_base(expr, **kwargs)

def expand_complex(expr, **kwargs):
    """Expand complex expressions."""
    return sy.expand_complex(expr, **kwargs)

def apart(expr, *args, **kwargs):
    """Partial fraction decomposition of rational functions."""
    return sy.apart(expr, *args, **kwargs)

def together(expr, *args, **kwargs):
    """Combine terms over a common denominator."""
    return sy.together(expr, *args, **kwargs)

def cancel(expr, *args, **kwargs):
    """Cancel common factors in a rational function."""
    return sy.cancel(expr, *args, **kwargs)

def collect(expr, syms, **kwargs):
    """Collect terms with respect to syms."""
    return sy.collect(expr, syms, **kwargs)

def factor_list(expr, **kwargs):
    """Return a list of irreducible factors of expr."""
    return sy.factor_list(expr, **kwargs)

def roots(expr, *symbols, **kwargs):
    """Find roots of a polynomial equation."""
    return sy.roots(expr, *symbols, **kwargs)

def solve_poly_system(eqs, *gens, **args):
    """Solve a system of polynomial equations."""
    return sy.solve_poly_system(eqs, *gens, **args)

def solve_linear_system(system, *symbols):
    """Solve a system of linear equations."""
    return sy.solve_linear_system(system, *symbols)

def expand_func(expr, **kwargs):
    """Expand special functions in expr."""
    return sy.expand_func(expr, **kwargs)

def simplify_logic(expr, form='dnf'):
    """Simplify a boolean expression to DNF or CNF."""
    return sy.simplify_logic(expr, form=form)

def apart_list(expr, *args, **kwargs):
    """Partial fraction decomposition, returning a list."""
    return sy.apart_list(expr, *args, **kwargs)

def together_list(expr, *args, **kwargs):
    """Combine terms over a common denominator, returning a list."""
    return sy.together_list(expr, *args, **kwargs)

def expand_multinomial(expr, **kwargs):
    """Expand multinomial expressions."""
    return sy.expand_multinomial(expr, **kwargs)

def expand_func(expr, **kwargs):
    """Expand special functions in expr."""
    return sy.expand_func(expr, **kwargs)

def apart(expr, *args, **kwargs):
    """Partial fraction decomposition of rational functions."""
    return sy.apart(expr, *args, **kwargs)

def together(expr, *args, **kwargs):
    """Combine terms over a common denominator."""
    return sy.together(expr, *args, **kwargs)

def cancel(expr, *args, **kwargs):
    """Cancel common factors in a rational function."""
    return sy.cancel(expr, *args, **kwargs)

def collect(expr, syms, **kwargs):
    """Collect terms with respect to syms."""
    return sy.collect(expr, syms, **kwargs)

def factor_list(expr, **kwargs):
    """Return a list of irreducible factors of expr."""
    return sy.factor_list(expr, **kwargs)

def roots(expr, *symbols, **kwargs):
    """Find roots of a polynomial equation."""
    return sy.roots(expr, *symbols, **kwargs)

def solve_poly_system(eqs, *gens, **args):
    """Solve a system of polynomial equations."""
    return sy.solve_poly_system(eqs, *gens, **args)

def solve_linear_system(system, *symbols):
    """Solve a system of linear equations."""
    return sy.solve_linear_system(system, *symbols)

def expand_func(expr, **kwargs):
    """Expand special functions in expr."""
    return sy.expand_func(expr, **kwargs)

def simplify_logic(expr, form='dnf'):
    """Simplify a boolean expression to DNF or CNF."""
    return sy.simplify_logic(expr, form=form)

a = syabc.a
b = syabc.b
c = syabc.c
d = syabc.d
e = syabc.e
f = syabc.f
g = syabc.g
h = syabc.h
i = syabc.i
j = syabc.j
k = syabc.k
l = syabc.l
m = syabc.m
n = syabc.n
o = syabc.o
p = syabc.p
q = syabc.q
r = syabc.r
s = syabc.s
t = syabc.t
u = syabc.u
v = syabc.v
w = syabc.w
x = syabc.x
y = syabc.y
z = syabc.z

alpha = syabc.alpha
beta = syabc.beta
gamma = syabc.gamma
delta = syabc.delta
epsilon = syabc.epsilon
zeta = syabc.zeta
eta = syabc.eta
theta = syabc.theta
iota = syabc.iota
kappa = syabc.kappa
lambda_ = sy.Symbol('lambda')
mu = syabc.mu
nu = syabc.nu
omicron = syabc.omicron
pi = syabc.pi
rho = syabc.rho
sigma = syabc.sigma
tau = syabc.tau
upsilon = syabc.upsilon
phi = syabc.phi
chi = syabc.chi
psi = syabc.psi
omega = syabc.omega

A = syabc.A
B = syabc.B
C = syabc.C
D = syabc.D
E = syabc.E
F = syabc.F
G = syabc.G
H = syabc.H
I = syabc.I
J = syabc.J
K = syabc.K
L = syabc.L
M = syabc.M
N = syabc.N
O = syabc.O
P = syabc.P
Q = syabc.Q
R = syabc.R
S = syabc.S
T = syabc.T
U = syabc.U
V = syabc.V
W = syabc.W
X = syabc.X
Y = syabc.Y
Z = syabc.Z

if __name__ == "__main__":
    # Example usage
    x, y = symbols('x y')
    expr = x**2 + y**2
    print("Expression:", expr)
    print("Differentiated with respect to x:", diff(expr, x))
    print("Integrated with respect to y:", integrate(expr, y))
    print("Limit as x approaches 0:", limit(expr, x, 0))
    print("Expanded logarithm:", expand_log(sy.log(x*y)))
    print("Roots of polynomial x^2 - 1:", roots(x**2 - 1))
    print("Symbolic variable a:", a)
    print("Symbolic variable beta:", beta)
    print("Equation x + y = 1:", Eq(x + y, 1))
    print("Factor list of x**2 - 1:", factor_list(x**2 - 1))
    print("Collect terms in x:", collect(expr, x))