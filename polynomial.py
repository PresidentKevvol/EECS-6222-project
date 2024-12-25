class Polynomial_gf2:
    '''
    a polynomial in galois field GF(2)
    thus all arithemrtic operation are modulo 2
    1 + 1 = 0, 1 * 1 = 1
    '''
    def __init__(self, coeff, degree=-1):
        '''
        constructor initializes with a list as coefficents
        coefficients should be written in little endian (lest significant/lowest order term first)
        thus (1, 0, 0, 1, 1) = x^4 + x^3 + 1
        '''
        
        # skim higher order terms
        j = len(coeff)
        for i in range(len(coeff)-1, -1, -1):
            if coeff[i] != 0:
                j = i+1
                break
        coeff = coeff[:j]

        # keep the zero term in the zero polynomial
        if len(coeff) == 0 or sum(coeff) == 0:
            coeff = [0,]
        # when degree not specified
        if degree == -1:
            degree = len(coeff)
        
        self.degree = degree
        self.coeff = coeff

    def __add__(self, other):
        '''
        addition
        '''
        l = max(self.degree, other.degree)
        j = min(self.degree, other.degree)
        co = []
        for i in range(j):
            # modulo 2 arithmetic
            r = (self.coeff[i] + other.coeff[i]) % 2
            co.append(r)
        # add the tail (higher order terms) too
        tail = self.coeff[j:] if self.degree > other.degree else other.coeff[j:]
        co += tail
        return Polynomial_gf2(co)

    def __sub__(self, other):
        '''
        subtract is same as add in GF(2)
        '''
        return self.__add__( other)

    def __mul__(self, other):
        '''
        multiplication
        '''
        l = self.degree + other.degree - 1
        co = [0,] * l

        for i in range(self.degree):
            for j in range(other.degree):
                co[i+j] += self.coeff[i] * other.coeff[j]
        for k in range(len(co)):
            co[k] = co[k] % 2

        return Polynomial_gf2(co)

    def __mod__(self, other):
        '''
        polynomial moduolo
        self % other = the remainder polynomial when self / other
        '''
        l = self.degree
        j = other.degree
        # assert l > j
        assert other.coeff != [0,]
        if l < j:
            return self
        
        res = Polynomial_gf2(self.coeff)
        # long division, but in base polynomial[ GF(2) ]
        # so we just subtract per each highest term
        while True:
            deg = res.degree
            res = res - other * X_(deg - j)
            if res.degree < j:
                break
        return res

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
            return p_1
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
            return p_1
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
        return self.coeff == other.coeff
            
            


'''
define some single terms constants
'''
def X_(n):
    return Polynomial_gf2([0,] * n + [1,])

p_0 = Polynomial_gf2([0,])
p_1 = Polynomial_gf2([1,])
X = X_(1)
X2 = X_(2)
X3 = X_(3)
X4 = X_(4)
X5 = X_(5)
X6 = X_(6)

gf16_generator_polynomial = X4 + X + p_1

# we use root element alpha = X in GF(2) polynomials
alpha = X

def to_bin_array_4bit(n):
    bin_str = bin(n)[2:]
    bin_str = '0' * (4-len(bin_str)) + bin_str
    return [int(c) for c in bin_str.split()]

# 0, 1, alpha, alpha**2, alpha**3, ... MOD (X4 + X + 1)
# addition table in perspective of powers of alpha
# 0 = -inf, 1 = 0, alpha = 1, alpha**2 = 2, alpha**3 = 3, ...
elem_list_alpha = [p_0, p_1,] + [alpha.pow_mod(n, gf16_generator_polynomial) for n in range(1, 15)]
elem_map_alpha = {}
elem_map_alpha[-1] = p_0
elem_map_alpha[0] = p_1
for n in range(1, 15):
    elem_map_alpha[n] = alpha.pow_mod(n, gf16_generator_polynomial)

# generate the GF(16) lookup tables
gf16_add_table = {}
for i in range(-1, 15):
    for j in range(-1, 15):
        # ii = to_bin_array_4bit(i)[::-1]
        # jj = to_bin_array_4bit(j)[::-1]
        ii = elem_map_alpha[i]
        jj = elem_map_alpha[j]
        added = (ii + jj) % gf16_generator_polynomial
        # add element to lookup table
        for key in elem_map_alpha:
            if added == elem_map_alpha[key]:
                gf16_add_table[(i, j)] = key
                break

class Gf16_element:
    '''
    an element in GF(16)
    denoted as a power of alpha
    ranges from -1 (0 polynomial of GF(2)), 0 (1 polynomial of GF(2)), 1 (alpha ** 1), 2 (alpha ** 2), ...
    '''
    def __init__(self, po):
        assert type(po) == int and po >= -1 and po <= 14
        self.po = po

    def __add__(self, other):
        '''
        add two element together using the lookup table
        '''
        i = self.po
        j = other.po
        return Gf16_element(gf16_add_table[(i, j)])

    def __sub__(self, other):
        '''
        subtract is same as add in GF(2)
        '''
        return self.__add__(other)

    def __mul__(self, other):
        '''
        multiplication here is actually addition mod 15
        unless either element is -1 (which is actually -inf, the 0 polynomial in GF(2)),
        in that case result is also -1
        '''
        i = self.po
        j = other.po
        if i == -1 or j == -1:
            return Gf16_element(-1)
        return Gf16_element((i + j) % 15)

    def __truediv__(self, other):
        '''
        multiplication here is actually subtraction mod 15
        unless either element is -1 (which is actually -inf, the 0 polynomial in GF(2)),
        in that case result is also -1
        '''
        i = self.po
        j = other.po
        assert j != -1, "cannot divide by 0 (additive identity)"
        if i == -1:
            return Gf16_element(-1)
        return Gf16_element((i - j) % 15)

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
            return Gf16_element(0)
        elif n == 1:
            return x
        elif n % 2 == 0:
            return (x*x) ** (n // 2)
        else:
            return x * ((x*x) ** ((n - 1) // 2))
    
    def __eq__(self, other):
        i = self.po
        j = other.po
        return i == j

class Polynomial_gf16:
    '''
    a polynomial in galois field GF(2^4)
    with field generator polynomial X^4 + X + 1
    '''
    def __init__(self, coeff, degree=-1):
        '''
        constructor:
        coeff: List[int], list of integers in {0, 1, 2, .., 15}

        these are polynomials of 4th order polynomials in GF(2)
        meaning that, the coefficients (and variables) are elements
        of the field GF(16), which are each in turn represented as 
        4th order polynomials in GF(2)
        '''
        # self.generator_p = gf16_generator_polynomial

        # skim higher order terms
        # -1 here means -inf and is the additive identity aka 0 polynomial in GF(2)
        # and 0 actually stands for the 1 polynomial
        j = 0
        for i in range(len(coeff)-1, -1, -1):
            if type(coeff[i]) == int:
                if coeff[i] != -1:
                    # print("leading 0 found, i =", i)
                    j = i+1
                    break
            elif type(coeff[i]) == Gf16_element:
                if coeff[i].po != -1:
                    # print("leading 0 found, i =", i)
                    j = i+1
                    break
            else:
                raise TypeError("Only int or Gf16_element are allowed")
        coeff = coeff[:j]

        # keep the zero term in the zero polynomial
        if len(coeff) == 0:
            coeff = [-1,]
        # when degree not specified
        if degree == -1:
            degree = len(coeff)
        assert type(degree) == int, "non int degree of a polynomial is not allowed"

        # convert them all to gf16 elements objects
        coeff_new = []
        for c in coeff:
            if type(c) == int:
                coeff_new.append(Gf16_element(c))
            elif type(c) == Gf16_element:
                coeff_new.append(c)
            else:
                raise TypeError("Only int or Gf16_element are allowed") 
        
        self.degree = degree
        self.coeff = coeff_new

    def __add__(self, other):
        '''
        addition
        '''
        l = max(self.degree, other.degree)
        j = min(self.degree, other.degree)
        co = []
        for i in range(j):
            # gf16 arithmetic
            r = (self.coeff[i] + other.coeff[i])
            co.append(r)
        # add the tail (higher order terms) too
        tail = self.coeff[j:] if self.degree > other.degree else other.coeff[j:]
        co += tail
        return Polynomial_gf16(co)

    def __sub__(self, other):
        '''
        subtract is same as add in GF(2)
        '''
        return self.__add__( other)

    def __mul__(self, other):
        '''
        multiplication
        '''
        # case 1: multiply by a constant
        if type(other) == Gf16_element:
            co = [c * other for c in self.coeff]
            return Polynomial_gf16(co)

        # case 2: multiply by another polynomial
        l = self.degree + other.degree - 1
        # reminder: -1 = alpha^(-inf) = 0 = the additive identity
        co = [Gf16_element(-1),] * l

        # long multiplication in a 2 layer for loop
        for i in range(self.degree):
            for j in range(other.degree):
                co[i+j] = co[i+j] + (self.coeff[i] * other.coeff[j])
        # for k in range(len(co)):
        #     co[k] = co[k] % 2

        return Polynomial_gf16(co)

    def __mod__(self, other):
        # TODO: finish coding mod function for GF(16) polynomial
        '''
        polynomial moduolo
        self % other = the remainder polynomial when self / other
        '''
        l = self.degree
        j = other.degree
        # assert l > j
        assert other.coeff != [Gf16_element(-1),], "cannot divide by additive identity"
        assert len(other.coeff) > 1, "polynomial modulo a constant does not work"
        
        if l < j:
            return self
        
        res = Polynomial_gf16(self.coeff)
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
            new_coeff = [-1] * (deg - j) + other.coeff
            res_leading = res.coeff[-1]
            other_leading = other.coeff[-1]
            ratio = res_leading / other_leading
            # new_coeff = [if i != -1 else -1]
            other_m = Polynomial_gf16(new_coeff) * ratio

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
        l = self.degree
        j = other.degree
        # assert l > j
        assert other.coeff != [Gf16_element(-1),], "cannot divide by additive identity"
        assert len(other.coeff) > 1, "polynomial modulo a constant does not work"
        
        if l < j:
            return self
        
        res = Polynomial_gf16(self.coeff)
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
            new_coeff = [-1] * (deg - j) + other.coeff
            res_leading = res.coeff[-1]
            other_leading = other.coeff[-1]
            ratio = res_leading / other_leading
            # new_coeff = [if i != -1 else -1]
            other_m = Polynomial_gf16(new_coeff) * ratio

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
        
        return Polynomial_gf16(new_res_coeff)

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
            return Polynomial_gf16([0,])
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
            return p_1
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
        return self.coeff == other.coeff

    def __call__(self, x):
        '''
        evaluate this polynomial at x
        '''
        assert type(x) == Gf16_element, "can only evaluate with an input that is an GF(16) element"
        
        res = Gf16_element(-1)    # result sum, start with 0 aka additive identity
        x_n = Gf16_element(0)     # x to power of n, start with 1 aka multiplicative identity
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
        return Polynomial_gf16([-1,] * n + self.coeff)

    def __str__(self):
        return str(self.to_list())

    def to_list(self):
        return [n.po for n in self.coeff]
