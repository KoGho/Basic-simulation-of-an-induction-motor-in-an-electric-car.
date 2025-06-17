import numpy as np
import numba as nb

@nb.njit(parallel=True, fastmath=True)
def start(ma, steps, t, f, ramp_s, maximum):
    ma_1 = 0#ma/4
    ma_2 = ma
    ma_t = np.zeros(steps)
    ramp_steps = int(ramp_s/t)
    ma_t[:ramp_steps] = np.linspace(ma_1, ma_2, ramp_steps)
    f_1 = 0#f/4
    f_2 = f
    f_t = np.zeros(steps)
    steps1 = int(5.2*steps/13)
    steps2 = int(11.4*steps/13)
    steps3 = int(8.2*steps/13)
    f_t[:ramp_steps] = np.linspace(f_1, f_2, int(ramp_steps))
    if steps == int(13/t):
        f_t[ramp_steps:steps2] = np.linspace(f_2, maximum, int(steps2-steps1))
        f_t[steps2:] = maximum
        ma_t[ramp_steps:steps3] = np.linspace(ma, 4*ma/np.pi, int(steps3-steps1))
        ma_t[steps3:] = 4*ma/np.pi
    elif steps == int(6/t) or steps == int(1/t):
        f_t[ramp_steps:] = f_2
        ma_t[ramp_steps:] = ma
    else:
        raise ValueError('An error occured.')
    return ma_t, f_t

@nb.njit(parallel=True, fastmath=True)
def during(ma, steps, t, f):
    ma_t = ma*np.ones(steps)
    f_t = f*np.ones(steps)
    return ma_t, f_t

@nb.njit(parallel=True, fastmath=True)
def stop(ma, steps, t, f, ramp_f):
    ma_1 = ma
    ma_2 = 0
    ma_t = np.zeros(steps)
    ramp_steps = int(ramp_f/t)
    ma_t[:steps-2*ramp_steps] = ma_1
    ma_t[steps-2*ramp_steps:steps-ramp_steps] = np.linspace(ma_1, ma_2, ramp_steps)
    ma_t[steps-ramp_steps:] = ma_2

    f_1 = f
    f_2 = 0
    f_t = np.zeros(steps)
    f_t[:steps-2*ramp_steps] = f_1
    f_t[steps-2*ramp_steps:steps-ramp_steps] = np.linspace(f_1, f_2, ramp_steps)
    f_t[steps-ramp_steps:] = f_2
    return ma_t, f_t
