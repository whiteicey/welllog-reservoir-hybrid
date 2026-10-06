"""Gaussian convolution of the official triangular accuracy kernel."""
import math
import torch


def smoothed_error(signed_error,sigma=.25):
    def positive_part_mean(value):
        z=value/sigma
        return value*torch.special.ndtr(z)+sigma*torch.exp(-.5*z.square())/math.sqrt(2*math.pi)
    # E[max(0,1-|e+sigma*Z|)], with Z standard Gaussian.
    accuracy=positive_part_mean(signed_error+1)-2*positive_part_mean(signed_error)+positive_part_mean(signed_error-1)
    return 1-accuracy
