import numpy as np
from utils import *

'''
file for convolutional code
includes encoding
and decoding with Viterbi algorithm
'''

def convolute(data, kernel):
    '''
    single convcolution using primitive list
    '''
    l = len(data)
    return sum([data[j] * kernel[j] for j in range(l)]) % 2

def convolute_np(data, kernels):
    '''
    multiple convolution using np arrays
    Args:
        data (array_like): (l,) array, current frame of data
        kernels (array_like): (g, l) array, the kernels of convolution
    Returns:
        an array of size (g,), result of the convolution encoding
    '''
    return (kernels @ data) % 2

def convolutional_encode(data, kernels):
    '''
    data: list[int] stream of data as bits
    kernels: list[list[int]], the convolutional kernels in this code
    '''
    l = len(kernels[0])
    g = len(kernels)

    res = []

    # add a padding at the end
    data_padded = ([0,] * (l-1)) + data
    for i in range(len(data)):
        cur_window = data_padded[i:i+l]

        # for each kernel
        for k in kernels:
            # do the convolution and add to output
            res.append(sum([cur_window[j] * k[j] for j in range(l)]) % 2)
    return res

def convolutional_encode_np(data, kernels, dtype=int):
    '''
    Args:
        data (array_like): array(n), stream of data as bits
        kernels (array_like): array(g, l), the convolutional kernels in this code
    Returns:
        array(n*g), result of convolutional encoding
    '''
    n = data.shape[0]
    l = kernels.shape[1]
    g = kernels.shape[0]

    res = np.empty([n*g,], dtype=dtype)

    # add a padding of l-1 zeros at the beginning
    # just like the hardware encoder starts the intermediate states
    # with all 0 before the bits comes in
    data_padded = np.concatenate([np.zeros((l-1), dtype=dtype), data])
    for i in range(n):
        cur_window = data_padded[i:i+l]

        # for each kernel
        res[i*g:(i+1)*g] = convolute_np(cur_window, kernels)
    return res

def covolutional_decode(codewords, kernels):
    '''
    decode a convolutional code using Viterbi algorithm
    '''
    pass

def viterbi_transition_matrix(l):
    '''
    generate the transition matrix used in the Viterbi algorithm
    for use in decoding Convolution code
    a convolution code with kernel size n will have a 2^(n-1) by 2^(n-1) state transition matrix
    '''
    N = 2**l
    mat = np.ones((N, N)) * -1
    for i in range(N):
        ii = format(i, '#0' + str(l+2) + 'b')[2:]

        '''
        suppose we have a 4 bit convolution code
        its hidden state will be 3 bits
        and say hidden state in a stage is 101
        there can be 2 possibilities for the next bit - 1 or 0
        which will give a 4 bit input to the kernel as either 1010 or 1011
        and the next 3 bit hidden state will be 010 or 011
        '''
        j0 = int(ii[1:] + "0", 2)
        j1 = int(ii[1:] + "1", 2)

        mat[i, j0] = 0
        mat[i, j1] = 1
    return mat

def viterbi_transition_graph(kernels, dtype=int):
    '''
    generate the transition graph used in the Viterbi algorithm
    for use in decoding Convolution code
    '''
    # length of kernel
    l = kernels.shape[1]
    # number of kernels aka rate of code word bits per raw message bits
    g = kernels.shape[0]

    N = 2**(l-1)
    transition = {}
    # for every number in 0, 1, 2, ..., 2^(l-1) as (l-1) bit hidden states
    for i in range(N):
        # ii = np.array([int(i) for i in format(i, '#0' + str(l+1) + 'b')[2:]], dtype=dtype)
        ii = to_bits_array(i, l-1, dtype=dtype)

        '''
        suppose we have a 4 bit convolution code
        its hidden state will be 3 bits
        and say hidden state in a stage is 101
        there can be 2 possibilities for the next bit - 1 or 0
        which will give a 4 bit input to the kernel as either 1010 or 1011
        and the next 3 bit hidden state will be 010 or 011
        '''
        # 2 cases: is next bit a 0 or 1?
        j0 = np.concatenate([ii, [0,]])
        j1 = np.concatenate([ii, [1,]])

        # the emission (aka result of convolution) of a transition
        # is dependant on if the next bit is 1 or 0
        # transition[tuple(ii)] = {}
        # transition[tuple(ii)][0] = convolute_np(j0, kernels)
        # transition[tuple(ii)][1] = convolute_np(j1, kernels)
        transition[i] = {}
        transition[i][0] = convolute_np(j0, kernels)
        transition[i][1] = convolute_np(j1, kernels)
    return transition

def viterbi_decode_convolutional(codewords, g, l, trans):
    '''
    decode a code encoded by convolution code using Viterbi algorithm
    referenced from: https://en.wikipedia.org/wiki/Viterbi_algorithm
    '''
    # number of 'observations' aka chunks resulting from convolution
    t = len(codewords) // g
    # number of possible unique hidden states aka total number of 2^(l-1) bit numbers
    S = 2**(l-1)

    # total (cumulative) hamming distance of each hidden state at each location
    dist = np.empty((t, S), dtype=float)
    # initially, all states have infinite minimum culminative distance
    dist[:, :] = np.inf
    # keep track of each hidden state's previous hidden state at each chunk location
    prev = np.empty((t, S), dtype=int)

    # list for int to bit array, for easy access
    # (not need to convert number to array each cycle of the nested for loop)
    bit_conversions = []
    bit_conversions_p1 = []
    for i in range(S):
        as_bits = to_bits_array(i, l-1)
        bit_conversions.append(as_bits)
        bit_conversions_p1.append([])

    '''
    start with the fact that we know the hidden state must start with (0,0,0,...0)
    noted by the (l-1) zero padding or hardware encoder starts with all 0
    thus we only have 2 cases: start with hidden state 00...0 and its next aka the first bit is 0 or 1
    start by looking at the first chunk
    '''
    first_chunk = codewords[0:g]
    # case 00...0, next (first) bit 0
    dist[0, 0] = hamming_dist(first_chunk, trans[0][0])
    # case 00...0, next (first) bit 1
    dist[0, 1] = hamming_dist(first_chunk, trans[0][1])

    '''
    now we go through the trellis graph
    essentially performing a Dijkstra's
    recording the lowest culminative distance reaching each node
    '''
    for i in range(1, t):
        # current chunk, chunk of g bits output from encoding
        cur_chunk = codewords[(i)*g:(i+1)*g]
        # for each possible hidden state
        for si in range(S):
            # si_as_bits = bit_conversions[si]

            # 2 cases: 'next' bit, which is actually current bit is 0 or 1
            # when added to the hidden state, and then encoded with convolution, will
            # lead to a codeword chunk of trans[si][next_bit]
            # (si = current hidden state, some possible case of what last l-1 bits had been)
            for next_bit in [0, 1]:
                # hamming dist between: observed codeword's current chunk of bits
                # vs. the result of convolution if this trellis transition had been the case
                ham_dist = hamming_dist(cur_chunk, trans[si][next_bit])
                # add to cumulative hamming dist
                # current cumulative hamming dist after this transition
                cur_cuml_dist = dist[i-1, si] + ham_dist

                # the hidden state in the next chunk column
                # that this transition will transition to
                si_next = (si * 2) % (2**(l-1)) + next_bit

                # if lower cumulative distance, record and replace
                if cur_cuml_dist < dist[i, si_next]:
                    dist[i, si_next] = cur_cuml_dist
                    prev[i, si_next] = si

    res = np.empty((t), dtype=int)
    # from the last column, find out the hidden state with minimum cumulative dist
    min_dist_last_idx = np.argmin(dist[-1, :])
    # min_dist_last = dist[-1, min_dist_last_idx]
    # looping backwards for back trace
    # for each chunk, we recorded the minimum cumulative distance of each hidden state
    # and which previous hidden state feeds into this hidden state with min dist
    prev_idx = min_dist_last_idx
    # 'next' bit == least significant bit of 'current' hidden state
    res[-1] = prev_idx % 2
    for i in range(t-2, -1, -1):
        prev_idx = prev[i+1, prev_idx]
        res[i] = prev_idx % 2

    return res

class ConvCode:
    def __init__(self, kernels):
        '''
        create a convolutional code encoder and decoder object
        '''
        if type(kernels) != np.ndarray:
            self.kernels = np.array(kernels)
        else:
            self.kernels = kernels

        # length of kernel
        self.l = kernels.shape[1]
        # number of kernels aka rate of code word bits per raw message bits
        self.g = kernels.shape[0]

        # trandsition graph for use in Viterbi algorithm
        self.trans = viterbi_transition_graph(self.kernels)

    def encode(self, message):
        '''
        encode a message using this convolutional coding scheme
        '''
        return convolutional_encode_np(message, self.kernels)

    def decode(self, codeword):
        '''
        decode a message using this convolutional coding scheme
        '''
        return viterbi_decode_convolutional(codeword, self.g, self.l, self.trans)