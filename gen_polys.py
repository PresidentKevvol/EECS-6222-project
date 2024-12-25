from polynomial import *

'''
a list of generator polynomials
copied from
https://github.com/nceruchalu/reed_solomon/blob/master/primitives.h

the keys are m
the elements are generator polynomials that generates elements in GF(2^m)
'''
generator_polynomials_list = {}

generator_polynomials_list[1] = X_(1) + X_(0)
generator_polynomials_list[2] = X_(2) + X_(1) + X_(0)
generator_polynomials_list[3] = X_(3) + X_(1) + X_(0)
generator_polynomials_list[4] = X_(4) + X_(1) + X_(0)
generator_polynomials_list[5] = X_(5) + X_(1) + X_(0)
generator_polynomials_list[6] = X_(6) + X_(1) + X_(0)
generator_polynomials_list[7] = X_(7) + X_(1) + X_(0)
generator_polynomials_list[8] = X_(8) + X_(4) + X_(3) + X_(2) + X_(0)
generator_polynomials_list[9] = X_(9) + X_(4) + X_(0)
generator_polynomials_list[10] = X_(10) + X_(3) + X_(0)
