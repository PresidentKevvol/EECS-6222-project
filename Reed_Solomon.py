import numpy as np
import random

from polynomial import *
from GF_Polynomials import *

'''
file for Reed-Solomon encoding scheme
'''

# utility functions

def generate_binary_string(n):
    '''
    Generate a random number/binary string with n bits
    should always return a str with length n of "1" and "0"
    '''
    number = random.getrandbits(n)
    # Convert the number to binary
    binary_string = format(number, '#0' + str(n+2) + 'b')[2:]
    return binary_string

def convert_to_chunks(bitstr, m, minus1=True):
    '''
    convert a bit string to chunks of m bits
    for RS encoding
    minus 1 might be needed as we are mapping each
    m bit chunk to [-1, 2^m-2] for the GF(2^m) representation

    result is a List[int] with values in [-1, 2^m -2]
    can be fed into constructor of Polynomial_GF2m to create polynomial 
    '''
    l = len(bitstr)
    # add padding at the end if not exactly multiple of m
    if l % m != 0:
        bs = bitstr + "0" * (m - l % m)
    else:
        bs = bitstr
    res = []
    for i in range(l // m):
        c = int(bs[i*m:(i+1)*m], 2)
        if minus1:
            c -= 1
        res.append(c)
    return res

def RS_encode_polynomial(t, mx, gx):
    '''
    Encode a message already converted to polynomial form
    using the systemic Reed-Solomon encoding method
    detailed in NASA memorandum

    returns the codeword polynomial
    '''
    mx_2t = mx.pad(2*t)
    cx = mx_2t + (mx_2t % gx)
    return cx


def codefn_list_to_bitstr(lis, m):
    '''
    convert code function to bitstr
    in our use case, elements in GF(2^m)
    are denoted as an integer in [-1, 2^m-2]
    and the codeword polynomial should be of order
    2^m - 1, if not, we pad in -1
    (which are actually the 0 polynomial, or the additive identity)

    this takes in a List[int]  with values in [-1, 2^m-2]
    so one might want to use Polynomial_GF2m.to_list() if putting in a polynomial
    '''
    if len(lis) < 2**m - 1:
        lis = lis + [-1,] * (2**m - 1 - len(lis))
    
    strs = [format(l+1, '#0' + str(m+2) + 'b')[2:] for l in lis]

    return ''.join(strs)

def ext_euclidean_algo(a, b):
    '''
    Extended Euclidean Algorithm
    for use in Reed Solomon decoding

    copied from:
    https://en.wikipedia.org/wiki/Extended_Euclidean_algorithm
    '''
    r_, r = a, b
    s_, s = a.mul_identity(), a.add_identity()
    t_, t = a.add_identity(), a.mul_identity()

    while r.degree >= t.degree:
        q = r_ // r
        r_, r = r, r_ - q * r
        s_, s = s, s_ - q * s
        t_, t = t, t_ - q * t

    return r, s, t

class ReedSolomon:
    '''
    Reprsesents a Reed-Solomon code system
    with encoding and decoding
    '''
    def __init__(self, m, t):
        '''
        constructor specifying the alphabet size, chunk size, and munimum distance of the scheme
        m is the alphabet size as a power of 2 (in number of bits), the alphabet is converted to elements in GF(2^m)
        t is the maximum distance of this coding scheme
        then,
        n = 2^m - 1 is the code word length
        k = n - 2t is the message word/chunk length
        both are in alphabets, thus,
        if m = 4, t = 3, each alphabet is a 4 bit chunk/word (from 0000 to 1111)
        and the minumum distance is error at 3 chunks
        codeword length is 15 of these 4 bit chunks/hex letters
        message  length is  9 of these 4 bit chunks/hex letters
        this corresponds to a (15, 9) Reed-Solomon code
        '''
        self.m = m
        self.n = 2**m - 1
        self.t = t
        self.k = self.n - 2 * t

        # generate the addition table if not existed
        # crucial for the algebraic opeartions of the field elements
        # and polynomials formed by them
        if m not in GF_add_tables.add_tables:
            GF_add_tables.gen_field_add_table(m, generator_polynomials_list[m])
        
        # root element alpha
        alph = Polynomial_GF2m([1,], self.m)
        # the X polynomial in GF(16)
        # the coefficients denote: 
        # a polynomial P(x) = X = alpha**0 * x + alpha**(-inf)
        X_gf2m = Polynomial_GF2m([-1, 0,], self.m)
        
        # create the generator polynomial for encoding
        gx = Polynomial_GF2m([0,], self.m) # mul. identity "1"
        for i in range(2*self.t):
            gx = gx * (X_gf2m + alph ** (i+1))
        self.gx = gx

        

    def bitstr_to_polynomial(self, bitstr):
        '''
        convert a bit string to a polynomial under this system 
        a polynomial in the field GF(2^m))
        '''
        return Polynomial_GF2m(convert_to_chunks(bitstr, self.m), self.m)
    
    def encode(self, message):
        '''
        encode a message using this R-S scheme
        
        message is a bit string of length m * k = m * (2^m-1 - 2t)
        returns a codeword, a bit string of length m * n = m * (2^m-1)
        '''
        mx = self.bitstr_to_polynomial(message)
        cx = RS_encode_polynomial(self.t, mx, self.gx)
        codeword = codefn_list_to_bitstr(cx.to_list(), self.m)
        return codeword

    def decode(self, codeword):
        '''
        decode a message using this R-S scheme
        
        codeword is a bit string of length m * n = m * (2^m-1)
        returns the decoded message, a bit string of length m * k = m * (2^m-1 - 2t) 
        '''
        # convert codeword back to a function
        code_fn = self.bitstr_to_polynomial(codeword)
        
        # compute syndrome
        synd_coeff = []
        for i in range(1, 2*self.t + 1):
            syn_i = code_fn(Element_GF2m(i, self.m))
            synd_coeff.append(syn_i)

        syndrome = Polynomial_GF2m(synd_coeff, self.m)
        
        # run extended Euclidean algorithm
        # a(x) = x^2t   (polynomial with just that term under this field)
        ax = Polynomial_GF2m([-1,] * (2*self.t) + [0], self.m)
        r_i, s_i, t_i = ext_euclidean_algo(ax, syndrome)

        # omega and lambda polynomial for subsequent steps
        field_zero = Element_GF2m(-1, self.m)
        omega = r_i // t_i(field_zero)
        lambd = t_i // t_i(field_zero)

        # compute locations of errors
        # Chien search
        # the locations are n-i for any lambda(alpha^i) == alpha^-1
        err_locations = []
        for i in range(self.n):
            el = Element_GF2m(i, self.m)
            func_eval = lambd(el)

            if func_eval == Element_GF2m(-1, self.m):
                err_locations.append(el)

        # finding the value of the errors (differences) using Forney's algorithm
        error_fixes = {}
        for e in err_locations:
            # if lambda(alpha^i) = alpha^(-1)
            # there is an error at position (n-i)
            err_pos = (self.n - e.po) % self.n
            err_diff = omega(e) / lambd.formal_derivative()(e)
            error_fixes[err_pos] = err_diff
        
        # finally, fix the errors by subtracting the back from the polynomial's coefficients
        decode_coeffs = code_fn.coeff
        # pad the list with additive identity if highest terms are all 0 aka alpha^(-inf)
        if len(decode_coeffs) < self.n:
            decode_coeffs = decode_coeffs + [Element_GF2m(-1, self.m),] * (self.n - len(decode_coeffs))
        # add is same as subtract
        for pos in error_fixes:
            decode_coeffs[pos] = decode_coeffs[pos] + error_fixes[pos]
        # convert back to bitstr
        err_corrected = codefn_list_to_bitstr([p.po for p in decode_coeffs], self.m)
        # the decoded message is the tail
        return err_corrected, err_corrected[self.m*2*self.t:]