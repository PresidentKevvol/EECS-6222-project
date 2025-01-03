import random
import numpy as np

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

def generate_binary_array(n):
    '''
    Generate a random binary array with n bits
    should always return a np array with length n of integers 1 and 0
    '''
    number = random.getrandbits(n)
    # Convert the number to binary
    binary_string = format(number, '#0' + str(n+2) + 'b')[2:]
    return np.array([int(i) for i in binary_string])

def binary_array_to_str(a):
    '''
    convert a binary array in numpy to bit string
    '''
    return ''.join([str(i) for i in a])

def binary_str_to_array(b):
    '''
    convert a bit string to binary array in numpy
    '''
    return np.array([int(i) for i in b])

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

    # different conversion methods for np arrays
    if type(bitstr) == np.ndarray:
        b2 = np.reshape(bitstr, newshape=(-1, m))
        pows = 2 ** (np.arange(m)[::-1])
        res = np.sum(b2 * pows, axis=1)
        if minus1:
            res = res - 1
        return res
    
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
    # add padding
    if len(lis) < 2**m - 1:
        lis = lis + [-1,] * (2**m - 1 - len(lis))
    
    strs = [format(l+1, '#0' + str(m+2) + 'b')[2:] for l in lis]

    return ''.join(strs)

def codefn_list_to_bit_array(lis, m):
    '''
    convert code function to bit asrray
    in our use case, elements in GF(2^m)
    are denoted as an integer in [-1, 2^m-2]
    and the codeword polynomial should be of order
    2^m - 1, if not, we pad in -1
    (which are actually the 0 polynomial, or the additive identity)

    this takes in a List[int]  with values in [-1, 2^m-2]
    so one might want to use Polynomial_GF2m.to_list() if putting in a polynomial
    '''
    # add padding
    if len(lis) < 2**m - 1:
        lis = lis + [-1,] * (2**m - 1 - len(lis))
    
    strs = [format(l+1, '#0' + str(m+2) + 'b')[2:] for l in lis]

    res = [[int(j) for j in i] for i in strs]
    
    return np.array(res).flatten()

def to_bits_array(x, n, dtype=int):
    '''
    convert an integer x to an n bit array
    '''
    return np.array([int(i) for i in format(x, '#0' + str(n+2) + 'b')[2:]], dtype=dtype)

def hamming_dist(a, b):
    '''
    hamming distance of two bit arrays
    '''
    return np.sum((a != b))