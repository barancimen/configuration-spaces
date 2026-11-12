"""First-order formulas over the reals (SymPy) and their exact translation to z3."""
import z3
from sympy import (Add, And, Basic, Dummy, Equivalent, Implies, Mul, Not, Or, Poly, Pow,
                   Rational, Symbol, Tuple, expand, false, minimal_polynomial, true)
from sympy.core.relational import Eq, Ge, Gt, Le, Lt, Ne, Relational
from sympy.logic.boolalg import Boolean

from .spec import SpecError


class _Quantifier(Boolean, Basic):
    def __new__(cls, variables, formula):
        return Basic.__new__(cls, Tuple(*variables), formula)

    @property
    def variables(self):
        return self.args[0]

    @property
    def body(self):
        return self.args[1]

    @property
    def free_symbols(self):
        return self.body.free_symbols - set(self.variables)

    def _sympystr(self, printer):
        return f"{type(self).__name__}({printer._print(self.variables)}, {printer._print(self.body)})"

    def _mcode(self, printer):
        return f"{type(self).__name__}[{printer._print(self.variables)}, {printer._print(self.body)}]"


class ForAll(_Quantifier):
    pass


class Exists(_Quantifier):
    pass


def quantified(f):
    return f.has(ForAll) or f.has(Exists)


def map_relations(f, fn):
    """Rebuild a formula with fn applied to every comparison."""
    if isinstance(f, Relational):
        return fn(f)
    if isinstance(f, _Quantifier):
        return type(f)(f.variables, map_relations(f.body, fn))
    if isinstance(f, (And, Or, Not, Implies, Equivalent)):
        return f.func(*[map_relations(a, fn) for a in f.args])
    return f


class Z3:
    """Translate SymPy formulas to z3, exactly.

    Irrational constants must be algebraic: each becomes a fresh variable pinned down by its
    minimal polynomial and an isolating interval. Those conditions are collected in `defs`.
    """

    def __init__(self):
        self.vars, self.consts, self.defs = {}, {}, []

    def var(self, s):
        if s.name not in self.vars:
            self.vars[s.name] = z3.Real(s.name)
        return self.vars[s.name]

    def const(self, e):
        if e in self.consts:
            return self.consts[e]
        x = Dummy("x")
        try:
            P = Poly(minimal_polynomial(e, x), x)
        except Exception:
            raise SpecError(f"the constant {e} is not algebraic (or not recognised as such); "
                            "the prover only handles algebraic numbers")
        val = e.evalf(60)
        lo, hi = next((lo, hi) for (lo, hi), _ in P.intervals() if lo <= val <= hi)
        k = z3.Real(f"k{len(self.consts)}")
        self.defs += [z3.Sum([self._rat(c) * k**i for (i,), c in P.terms()]) == 0,
                      self._rat(lo) <= k, k <= self._rat(hi)]
        self.consts[e] = k
        return k

    @staticmethod
    def _rat(r):
        return z3.RealVal(f"{r.p}/{r.q}")

    def term(self, e):
        if e.is_Rational:
            return self._rat(e)
        if e.is_Float:                       # a decimal, read exactly as written
            return self._rat(Rational(str(e)))
        if e.is_number:
            return self.const(e)
        if isinstance(e, Symbol):
            return self.var(e)
        if isinstance(e, Add):
            return z3.Sum([self.term(a) for a in e.args])
        if isinstance(e, Mul):
            out = self.term(e.args[0])
            for a in e.args[1:]:
                out = out * self.term(a)
            return out
        if isinstance(e, Pow) and e.exp.is_Integer and e.exp > 0:
            b = self.term(e.base)
            out = b
            for _ in range(int(e.exp) - 1):
                out = out * b
            return out
        raise SpecError(f"cannot use {e} in a proof: only polynomials (division only by "
                        "nonzero constants) are supported")

    def formula(self, f):
        if f is true or f is True:
            return z3.BoolVal(True)
        if f is false or f is False:
            return z3.BoolVal(False)
        if isinstance(f, And):
            return z3.And([self.formula(a) for a in f.args])
        if isinstance(f, Or):
            return z3.Or([self.formula(a) for a in f.args])
        if isinstance(f, Not):
            return z3.Not(self.formula(f.args[0]))
        if isinstance(f, Implies):
            return z3.Implies(self.formula(f.args[0]), self.formula(f.args[1]))
        if isinstance(f, Equivalent):
            parts = [self.formula(a) for a in f.args]
            return z3.And([parts[0] == p for p in parts[1:]])
        if isinstance(f, _Quantifier):
            vs = [self.var(v) for v in f.variables]
            q = z3.ForAll if isinstance(f, ForAll) else z3.Exists
            return q(vs, self.formula(f.body))
        ops = {Eq: lambda a, b: a == b, Ne: lambda a, b: a != b, Ge: lambda a, b: a >= b,
               Gt: lambda a, b: a > b, Le: lambda a, b: a <= b, Lt: lambda a, b: a < b}
        for cls, op in ops.items():
            if isinstance(f, cls):
                return op(self.term(expand(f.lhs)), self.term(expand(f.rhs)))
        raise SpecError(f"cannot translate {f!r} to the prover")
