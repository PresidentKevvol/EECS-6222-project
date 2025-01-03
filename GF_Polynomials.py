import numpy as np

from polynomial import *
from gen_polys import *

'''
below are for generalized GF polynomials
'''

class AddTableManager:
    '''
    manager for Galois field addition tables
    the tables are stored in a dict, where keys are m
    and elements are addition tables of GF(2^m), in perspecitive of powers of alpha
    and each table is a dict where keys are 2-tuple
    where table[(i, j)] = i + j
    '''
    def __init__(self):
        self.add_tables = {}
        pass
    def gen_field_add_table(self, m, generator_polynomial):
        # prepare a map of the elements represented in powers of alpha
        elem_map_alpha = {}
        elem_map_alpha[-1] = p_0
        elem_map_alpha[0] = p_1
        for n in range(1, 2**m - 1):
            elem_map_alpha[n] = alpha.pow_mod(n, generator_polynomial)

        # generate the GF(2^m) lookup tables
        res = {}
        for i in range(-1, 2**m - 1):
            for j in range(-1, 2**m - 1):
                ii = elem_map_alpha[i]
                jj = elem_map_alpha[j]
                added = (ii + jj) % generator_polynomial
                # add element to lookup table
                for key in elem_map_alpha:
                    if added == elem_map_alpha[key]:
                        res[(i, j)] = key
                        break
        self.add_tables[m] = res

    def __getitem__(self, m):
        '''
        called when AddTableManager[m] is invoked
        gets the add table for GF(2^m) cached under here
        '''
        return self.add_tables[m]

'''
the table set object itself
'''
GF_add_tables = AddTableManager()
GF_add_tables.gen_field_add_table(4, generator_polynomials_list[4])
# GF_add_tables.gen_field_add_table(8, generator_polynomials_list[8])

class Element_GF2m:
    '''
    an element in GF(2^m)
    denoted as a power of alpha, alpha = X (GF(2) polynomial that is just X)
    ranges from -1 (0 polynomial of GF(2)), 0 (1 polynomial of GF(2)), 1 (alpha ** 1), 2 (alpha ** 2), ...
    '''
    def __init__(self, po, m):
        '''
        constructor
        creates an element of the Galois field of order 2^m
        the element is denoted as a power of alpha,
        where alpha = a(X) = X, the polynomial in GF(2) with just the X term
        aka identity function

        m is the log2 of the order of the field
        po is the element written as a power of alpha
        -1 means -inf, additive identity
        0 means alpha^0 = 1, multiplicative identity
        '''
        assert type(m) == int
        self.m = m
        assert type(po) == int and po >= -1 and po <= 2**m - 2
        self.po = po

    def __add__(self, other):
        '''
        add two element together using the lookup table
        '''
        i = self.po
        j = other.po
        return Element_GF2m(GF_add_tables[self.m][(i, j)], self.m)

    def __sub__(self, other):
        '''
        subtract is same as add in GF(2)
        '''
        return self.__add__(other)

    def __mul__(self, other):
        '''
        multiplication with an integer i is adding with itself i times
        
        multiplication with another element here is actually addition mod 2^m - 1
        unless either element is -1 (which is actually -inf, the 0 polynomial in GF(2)),
        in that case result is also -1
        '''
        # case 1: multiply by an integer
        # add with itself n times (this is repeated squaring)
        if type(other) == int:
            x = self
            if other == 0:
                # times integer zero = additive identity
                return self.add_identity()
            elif other == 1:
                return Element_GF2m(self.po, self.m)
            elif other % 2 == 0:
                return (x+x) * (other // 2)
            else:
                return x + ((x+x) * ((other - 1) // 2))

        # case 2: multiply by another element
        i = self.po
        j = other.po
        m = self.m
        # multiply by zero (additive identity)
        if i == -1 or j == -1:
            return self.add_identity()
        return Element_GF2m((i + j) % (2**m - 1), self.m)

    def __truediv__(self, other):
        '''
        multiplication here is actually subtraction mod 2^m - 1
        unless either element is -1 (which is actually -inf, the 0 polynomial in GF(2)),
        in that case result is also -1
        '''
        i = self.po
        j = other.po
        m = self.m
        assert j != -1, "cannot divide by 0 (additive identity)"
        if i == -1:
            return self.add_identity()
        return Element_GF2m((i - j) % (2**m - 1), self.m)

    def __pow__(self, n):
        '''
        raise to an integer power
        (type(n) == int)

        copied from:
        https://stackoverflow.com/questions/19375751/repetitive-squaring-python
        '''
        assert type(n) == int, "can only raise to a non negative integer power"
        assert n >= 0, "can only raise to a non negative integer power"
        x = self
        if n == 0:
            return self.mul_identity()
        elif n == 1:
            return x
        elif n % 2 == 0:
            return (x*x) ** (n // 2)
        else:
            return x * ((x*x) ** ((n - 1) // 2))
    
    def __eq__(self, other):
        '''
        compare two GF(2^m) element
        True only if they are same element in the same field
        '''
        i = self.po
        j = other.po
        return self.m == other.m and i == j

    def mul_identity(self):
        '''
        returns the multiplicative identity of this Galois field
        '''
        return Element_GF2m(0, self.m)
    def add_identity(self):
        '''
        returns the additive identity of this Galois field
        '''
        return Element_GF2m(-1, self.m)

class Polynomial_GF2m:
    '''
    a polynomial in galois field GF(2^m)
    with field generator polynomial X^4 + X + 1
    '''
    def __init__(self, coeff, m, degree=-1):
        '''
        constructor:
        coeff: List[int], list of integers in {-1, 0, 1, 2, ..., 2^m - 2}

        these are polynomials of m-th order polynomials in GF(2)
        meaning that, the coefficients (and variables) are elements
        of the field GF(2^m), which are each in turn represented as 
        m-th order polynomials in GF(2)
        '''
        self.m = m

        # convert back to python list for easier tinkering in code in this class
        if type(coeff) == np.ndarray:
            coeff = coeff.tolist()

        # skim higher order terms
        # -1 here means -inf and is the additive identity aka 0 polynomial in GF(2)
        # and 0 actually stands for the 1 polynomial
        j = 0
        for i in range(len(coeff)-1, -1, -1):
            if type(coeff[i]) == int:
                if coeff[i] != -1:
                    j = i+1
                    break
            elif type(coeff[i]) == Element_GF2m:
                if coeff[i].po != -1:
                    j = i+1
                    break
            else:
                raise TypeError(f"Only int or Gf(2^{self.m}) element are allowed")
        coeff = coeff[:j]
        for co in coeff:
            if type(co) == int:
                pass
            elif type(co) == Element_GF2m:
                assert self.m == co.m, "Elements that form the coefficients of the polynomial should be of the same field"
            else:
                raise TypeError(f"Only int or Gf(2^{self.m}) element are allowed")

        # keep the zero term in the zero polynomial
        if len(coeff) == 0:
            coeff = [-1,]
        # when degree not specified
        if degree == -1:
            degree = len(coeff)
        assert type(degree) == int, "non int degree of a polynomial is not allowed"

        # convert them all to gf(2^m) elements objects
        coeff_new = []
        for c in coeff:
            if type(c) == int:
                coeff_new.append(Element_GF2m(c, self.m))
            elif type(c) == Element_GF2m:
                coeff_new.append(c)
            else:
                raise TypeError(f"Only int or Gf(2^{self.m}) element are allowed") 
        
        self.degree = degree
        self.coeff = coeff_new

    def clone(self):
        '''
        make a clone of this polynomial
        '''
        return Polynomial_GF2m(self.coeff, self.m)
        
    def __add__(self, other):
        '''
        addition
        '''
        assert self.m == other.m, "Polynomials not in the same field cannot be added/subtracted"
        l = max(self.degree, other.degree)
        j = min(self.degree, other.degree)
        co = []
        for i in range(j):
            # gf(2^m) arithmetic
            r = (self.coeff[i] + other.coeff[i])
            co.append(r)
        # add the tail (higher order terms) too
        tail = self.coeff[j:] if self.degree > other.degree else other.coeff[j:]
        co += tail
        # print([c.po for c in co])
        return Polynomial_GF2m(co, self.m)

    def __sub__(self, other):
        '''
        subtract is same as add in GF(2), thus all of GF(2^m)
        '''
        return self.__add__( other)

    def __mul__(self, other):
        '''
        multiplication
        '''
        # case 1: multiply by a constant
        if type(other) == Element_GF2m:
            assert self.m == other.m, "Polynomial and constant not in the same field cannot be multiplied"
            co = [c * other for c in self.coeff]
            return Polynomial_GF2m(co, self.m)

        # case 2: multiply by another polynomial
        l = self.degree + other.degree - 1
        # reminder: -1 = alpha^(-inf) = 0 = the additive identity
        co = [Element_GF2m(-1, self.m),] * l

        # long multiplication in a 2 layer for loop
        for i in range(self.degree):
            for j in range(other.degree):
                co[i+j] = co[i+j] + (self.coeff[i] * other.coeff[j])

        return Polynomial_GF2m(co, self.m)

    def __mod__(self, other):
        '''
        polynomial moduolo
        self % other = the remainder polynomial when self / other
        '''
        l = self.degree
        j = other.degree
        # assert l > j
        assert self.m == other.m, "Polynomials not in the same field cannot be divided"
        assert other.coeff != [Element_GF2m(-1, self.m),], "cannot divide by additive identity"
        assert len(other.coeff) > 1, "polynomial modulo a constant does not work"
        
        if l < j:
            return self
        
        res = self.clone()
        # long division, but in base polynomial[ GF(2) ]
        # so we just subtract per each highest term
        while True:
            deg = res.degree
            # res = res - other * X_(deg - j)
            
            # multiply other by X**(deg - j) * (res_leading / other_leading)
            # like how in each step of long division
            # we multiply the divider by ..., 100, 10, 1
            # and then also the largest multiplier m such that
            # divider * m <= modified dividend at current step
            # new_coeff = [-1] * (deg - j) + other.coeff
            res_leading = res.coeff[-1]
            other_leading = other.coeff[-1]
            ratio = res_leading / other_leading
            other_m = other.pad(deg - j) * ratio

            # print("deg =", deg, "ratio =", ratio.po)

            # subtract step, again just like long division
            res = res - other_m
            
            # print(res.degree)
            if res.degree < j:
                break
        return res

    def __floordiv__(self, other):
        '''
        polynomial moduolo
        self % other = the remainder polynomial when self / other
        '''
        
        # case 1: divide by a constant
        if type(other) == Element_GF2m:
            assert self.m == other.m, "Polynomial and constant not in the same field cannot be multiplied"
            co = [c / other for c in self.coeff]
            return Polynomial_GF2m(co, self.m)

        # case 2: divide by another polynomial
        assert self.m == other.m, "Polynomials not in the same field cannot be divided"
        assert other.coeff != [Element_GF2m(-1, self.m),], "cannot divide by additive identity"

        l = self.degree
        j = other.degree
        if l < j:
            return self.add_identity()
        
        res = self.clone()
        res_coeffs = []
        # long division, but in base polynomial[ GF(2) ]
        # so we just subtract per each highest term
        while True:
            deg = res.degree
            # res = res - other * X_(deg - j)
            
            # multiply other by X**(deg - j) * (res_leading / other_leading)
            # like how in each step of long division
            # we multiply the divider by ..., 100, 10, 1
            # and then also the largest multiplier m such that
            # divider * m <= modified dividend at current step
            # new_coeff = [-1] * (deg - j) + other.coeff
            res_leading = res.coeff[-1]
            other_leading = other.coeff[-1]
            ratio = res_leading / other_leading
            other_m = other.pad(deg - j) * ratio

            # print("deg =", deg, "ratio =", ratio.po)

            # subtract step, again just like long division
            res = res - other_m

            # this is part of the result of the floor div
            # the ratio in each level forms a whole polynomial
            # that will be the ratio = self // other
            term_deg = (deg - j)
            res_coeffs.append((term_deg, ratio))
            
            # print(res.degree)
            if res.degree < j:
                break
        
        # degree of resulting polynomial is degree of highest term
        # start with zero polynomial
        new_res_coeff = [-1,] * (res_coeffs[0][0] + 1)
        # print(res_coeffs)
        # fill the terms back in
        for d, co in res_coeffs:
            new_res_coeff[d] = co
        
        return Polynomial_GF2m(new_res_coeff, self.m)

    def __pow__(self, n):
        '''
        raise to an integer power
        (type(other) == int)

        copied from:
        https://stackoverflow.com/questions/19375751/repetitive-squaring-python
        '''
        assert type(n) == int, "can only raise to a non negative integer power"
        assert n >= 0, "can only raise to a non negative integer power"
        x = self
        if n == 0:
            return self.mul_identity()
        elif n == 1:
            return x
        elif n % 2 == 0:
            return (x*x) ** (n // 2)
        else:
            return x * ((x*x) ** ((n - 1) // 2))

    def pow_mod(self, n, m):
        '''
        raise to an integer power modulo another GF(2) polynomial
        (type(other) == int)

        copied from:
        https://stackoverflow.com/questions/19375751/repetitive-squaring-python
        '''
        assert type(n) == int, "can only raise to a non negative integer power"
        assert n >= 0, "can only raise to a non negative integer power"
        x = self
        if n == 0:
            return self.mul_identity()
        elif n == 1:
            return x
        elif n % 2 == 0:
            return ((x*x) ** (n // 2)) % m
        else:
            return (x * ((x*x) ** ((n - 1) // 2))) % m

    def __eq__(self, other):
        '''
        if two polynomials are equal
        '''
        return self.m == other.m and self.coeff == other.coeff

    def __call__(self, x):
        '''
        evaluate this polynomial at x
        '''
        assert type(x) == Element_GF2m, "can only evaluate with an input that is an GF(2^m) element"
        assert x.m == self.m, "can only evaluate with an input that is an GF(2^m) element of the same field"

        # if the input to the function evaluation is
        # x = (additive identity), then all we need is to return the constant element
        if x == Element_GF2m(-1, self.m):
            return self.coeff[0]
        
        res = Element_GF2m(-1, self.m)    # result sum, start with 0 aka additive identity
        x_n = Element_GF2m(0, self.m)     # x to power of n, start with 1 aka multiplicative identity
        for i, co in enumerate(self.coeff):
            res = res + (co * x_n)
            x_n = x_n * x
        return res

    def pad(self, n):
        '''
        pad this polynomial by mulitplying by X^n
        so the new polynomial's coefficient list is
        [-1, -1, -1, ..., -1, coeff[0], coeff[1], ..., coeff[l-1]]
        '''
        return Polynomial_GF2m([-1,] * n + self.coeff, self.m)

    def __str__(self):
        return f"Polynomial in GF(2^{self.m}), coefficients: " + str(self.to_list())

    def to_list(self):
        return [n.po for n in self.coeff]

    def mul_identity(self):
        '''
        returns the multiplicative identity of this Galois field
        '''
        return Polynomial_GF2m([0,], self.m)
    def add_identity(self):
        '''
        returns the additive identity of this Galois field
        '''
        return Polynomial_GF2m([-1,], self.m)

    def formal_derivative(self):
        '''
        formal derivative of this polynomial
        basically like derivative of real number polynomials (same form)
        even when limits are not defined
        '''
        res_coeffs = []
        # loop through the coeffs and turn any
        # an*x^n -> n*an*x^(n-1)
        # thus skipping the constant element
        for i in range(1, len(self.coeff)):
            res_coeffs.append(self.coeff[i] * i)
        
        return Polynomial_GF2m(res_coeffs, self.m)