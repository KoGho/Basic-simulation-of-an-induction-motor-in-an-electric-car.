import numpy as np
import matplotlib.pyplot as plt
from voltages import voltageSynchronous
from voltages import voltageInverter
from voltages import bipolarPwm
from voltages import unipolarPwm
from park import parkF
import numba as nb
from quickV import quickSynchronous
from quickV import quickInverter
from quickV import quickBipolarPwm
from quickV import quickUnipolarPwm
from park import inversePark

# Instructions.
def selectVoltage(ma_t, f_t, interval, Vd, steps, ma, mf, f):

    print('Select voltage type:')
    print('1 - Balanced sine Wave')
    print('2 - Square Wave')
    print('3 - Bipolar PWM')
    print('4 - Unipolar PWM')

    choice = input('Enter your choice (1/2/3/4): ')

    match choice:
        case '1':
            Va, Vb, Vc = voltageSynchronous(Vd, interval, ma_t, f_t)
        case '2':
            Va, Vb, Vc = voltageInverter(interval, steps, Vd, ma_t, f_t)
        case '3':
            Va, Vb, Vc = bipolarPwm(ma, mf, f, \
                interval, steps, Vd, ma_t, f_t)
        case '4':
            Va, Vb, Vc = unipolarPwm(ma, mf, f, \
                interval, steps, Vd, ma_t, f_t)
        case _:
            raise ValueError('Invalid choice.')
    if choice == '1':
        value = 1
    elif choice == '2':
        value = 2
    elif choice == '3':
        value = 3
    elif choice == '4':
        value = 4
    else:
        raise ValueError('Invalid choice')

    Vsd = np.zeros(steps)
    Vsq = np.zeros(steps)
    omega = 2*np.pi*f_t
    Theta = omega*interval

    for k in range(steps-1):
        [Vsd[k], Vsq[k]] = parkF(Va[k], Vb[k], Vc[k], Theta[k]-np.pi/2)

    v_s = Vsd + 1j*Vsq
    return v_s, value

# Compute voltage
@nb.njit(parallel=True, fastmath=True)
def computeVoltage(ma_t, f_t, interval, value, Vd, steps, ma, mf, f):

    match value:
        case 1:
            Va, Vb, Vc = voltageSynchronous(Vd, interval, ma_t, f_t)
        case 2:
            Va, Vb, Vc = voltageInverter(interval, steps, Vd, ma_t, f_t)
        case 3:
            Va, Vb, Vc = bipolarPwm(ma, mf, f, \
                interval, steps, Vd, ma_t, f_t)
        case 4:
            Va, Vb, Vc = unipolarPwm(ma, mf, f, \
                interval, steps, Vd, ma_t, f_t)
        case _:
            raise ValueError('Invalid choice.')

    Vsd = np.zeros(steps)
    Vsq = np.zeros(steps)
    omega = 2*np.pi*f_t
    Theta = omega*interval

    for k in range(steps-1):
        [Vsd[k], Vsq[k]] = parkF(Va[k], Vb[k], Vc[k], Theta[k]-np.pi/2)

    v_s = Vsd + 1j*Vsq
    return v_s

# Compute one voltage
@nb.njit(fastmath=True)
def computeQuickV(ma_t, f_t, interval, value, Vd, steps, ma, mf, f, k):

    match value:
        case 1:
            Va, Vb, Vc = quickSynchronous(Vd, interval, ma_t, f_t, k)
        case 2:
            Va, Vb, Vc = quickInverter(interval, steps, Vd, ma_t, f_t, k)
        case 3:
            Va, Vb, Vc = quickBipolarPwm(ma, mf, f, \
                interval, steps, Vd, ma_t, f_t, k)
        case 4:
            Va, Vb, Vc = quickUnipolarPwm(ma, mf, f, \
                interval, steps, Vd, ma_t, f_t, k)
        case _:
            raise ValueError('Invalid choice.')

    omega = 2*np.pi*f_t
    k = k+1
    Theta = omega*interval*k

    [Vsd, Vsq] = parkF(Va, Vb, Vc, Theta-np.pi/2)

    v_s = Vsd + 1j*Vsq
    return v_s

# Plots.
def plots(omega_m, Omega_m, psi_s, psi_r, i_s, i_r, theta, \
    theta_r, c_e, c_m, i_a, i_b, i_c, i_ar, i_br, i_cr, \
    psi_a, psi_b, psi_c, psi_ar, psi_br, psi_cr, time, \
    R_r, R_s, L_r, L_s, L_m, v_s, f_t):

    speed = Omega_m/(2*np.pi)*3.6*2.1/10
    plt.figure()
    plt.plot(time, speed, label= 'speed in km/h')
    plt.grid(True)
    plt.legend()

    distance = np.zeros(int(len(time)))
    for k in range(int(int(len(time) - 1))):
        distance[k+1] = distance[k] + speed[k]*1e-6/3.6

    plt.figure()
    plt.plot(time, distance, label= 'distance in m')
    plt.grid(True)
    plt.legend()

    power = np.real(v_s*i_s)
    energy = np.zeros(int(len(time)))
    for k in range(int(len(time) - 1)):
        energy[k+1] = energy[k] + power[k]*1e-6

    plt.figure()
    plt.plot(time, power, label= 'power')
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(time, energy, label='energy')
    plt.grid(True)
    plt.legend()
    plt.show()

    image = '2'
    while image != '1':
        print('Do you want to plot other images?')
        print('1 - No')
        print('2 - Stator currents in the time domain')
        print('3 - Rotor currents in the time domain')
        print('4 - Stator fluxes in the time domain')
        print('5 - Rotor fluxes in the time domain')
        print('6 - Evolution of stator currents in the park domain')
        print('7 - Evolution of rotor currents in the park domain')
        print('8 - Evolution of stator fluxes in the park domain')
        print('9 - Evolution of rotor fluxes in the park domain')
        print('10 - Stator currents in the park domain')
        print('11 - Rotor currents in the park domain')
        print('12 - Stator fluxes in the park domain')
        print('13 - Rotor fluxes in the park domain')
        print('14 - Evolution of the magnitude of currents in the park domain')
        print('15 - Evolution of the magnitude of fluxes in the park domain')
        print('16 - Evolution of mutual fluxes in the park domain')
        print('17 - Mutual fluxes in the park domain')
        print('18 - Electromagnetic torque and mechanical torque')
        print('19 - Slip')
        print('20 - Voltage in the time domain')
        print('21 - Evolution of voltage in the park domain')
        print('22 - Voltage in the park domain')
        print('23 - Evolution of the magnitude of the voltage in the park domain')
        print('24 - Frequency')
        print('25 - Rotational speed')
        #print('20 - Pseudo-Mechanical plots')
        image = input('Enter the code of the image you want to print:')
        if image == '1':
            break
        elif image == '2':
            #2
            plt.figure()
            plt.plot(time, i_a, label= 'i_a')
            plt.plot(time, i_b, label= 'i_b')
            plt.plot(time, i_c, label= 'i_c')
            plt.grid(True)
            plt.legend()
            plt.show()
            pass
        elif image == '3':
            #3
            plt.figure()
            plt.plot(time, i_ar, label = 'i_ar')
            plt.plot(time, i_br, label = 'i_br')
            plt.plot(time, i_cr, label = 'i_cr')
            plt.grid(True)
            plt.legend()
            plt.show()
            pass
        elif image == '4':
            #4
            plt.figure()
            plt.plot(time, psi_a, label = 'psi_as')
            plt.plot(time, psi_b, label = 'psi_bs')
            plt.plot(time, psi_c, label = 'psi_cs')
            plt.grid(True)
            plt.legend()
            plt.show()
            pass
        elif image == '5':
            #5
            plt.figure()
            plt.plot(time, psi_ar, label = 'psi_ar')
            plt.plot(time, psi_br, label = 'psi_br')
            plt.plot(time, psi_cr, label = 'psi_cr')
            plt.grid(True)
            plt.legend()
            plt.show()
            pass
        elif image == '6':
            # evolution over time of the direct and quadrature components of the
            # stator and rotor currents and fluxes.
            #6
            plt.figure()
            plt.plot(time, np.real(i_s), label = 'i_sd')
            plt.plot(time, np.imag(i_s), label = 'i_sq')
            plt.grid(True)
            plt.legend()
            plt.show()
            pass
        elif image == '7':
            #7
            plt.figure()
            plt.plot(time, np.real(i_r), label = 'i_rd')
            plt.plot(time, np.imag(i_r), label = 'i_rq')
            plt.grid(True)
            plt.legend()
            plt.show()
            pass
        elif image == '8':
            #8
            plt.figure()
            plt.plot(time, np.real(psi_s), label = 'psi_sd')
            plt.plot(time, np.imag(psi_s), label = 'psi_sq')
            plt.grid(True)
            plt.legend()
            plt.show()
            pass
        elif image == '9':
            #9
            plt.figure()
            plt.plot(time, np.real(psi_r), label = 'psi_rd')
            plt.plot(time, np.imag(psi_r), label = 'psi_rq')
            plt.grid(True)
            plt.legend()
            plt.show()
            pass
        elif image == '10':
            # figures about stator and rotor currents and fluxed in the dq plane
            #10
            plt.figure()
            plt.plot(np.real(i_s), np.imag(i_s), label = 'i_sd(i_sq)')
            plt.grid(True)
            plt.legend()
            plt.show()
            pass
        elif image == '11':
            #11
            plt.figure()
            plt.plot(np.real(i_r), np.imag(i_r), label = 'i_rd(i_rq)')
            plt.grid(True)
            plt.legend()
            plt.show()
            pass
        elif image == '12':
            #12
            plt.figure()
            plt.plot(np.real(psi_s), np.imag(psi_s), label = 'psi_sd(psi_sq)')
            plt.grid(True)
            plt.legend()
            plt.show()
            pass
        elif image == '13':
            #13
            plt.figure()
            plt.plot(np.real(psi_r), np.imag(psi_r), label = 'psi_rd(psi_rq)')
            plt.grid(True)
            plt.legend()
            plt.show()
            pass
        elif image == '14':
            # comparison of the absolute value of stator and rotor currents and fluxes
            #14
            plt.figure()
            plt.plot(time, abs(i_s), label = '|i_s|')
            plt.plot(time, abs(i_r), label = '|i_r|')
            plt.grid(True)
            plt.legend()
            plt.show()
            pass
        elif image == '15':
            # mutual flux
            psi_m = L_m*(i_s + i_r)
            #15
            plt.figure()
            plt.plot(time, abs(psi_s), label = '|psi_s|')
            plt.plot(time, abs(psi_r), label = '|psi_r|')
            plt.plot(time, abs(psi_m), label = '|psi_md|')
            plt.grid(True)
            plt.legend()
            plt.show()
            pass
        elif image == '16':
            psi_m = L_m*(i_s + i_r)
            #16
            plt.figure()
            plt.plot(time, np.real(psi_m), label = 'psi_md')
            plt.plot(time, np.imag(psi_m), label = 'psi_mq')
            plt.grid(True)
            plt.legend()
            plt.show()
            pass
        elif image == '17':
            psi_m = L_m*(i_s + i_r)
            #17
            plt.figure()
            plt.plot(np.real(psi_m), np.imag(psi_m), label = 'psi_mdq')
            plt.grid(True)
            plt.legend()
            plt.show()
            pass
        elif image == '18':
            # energetic figures
            #18
            plt.figure()
            plt.plot(time, c_e, label = 'c_e')
            plt.plot(time, c_m, label = 'c_m')
            plt.grid(True)
            plt.legend()
            plt.show()
            pass
        elif image == '19':
            numerator = 2*np.pi*f_t - 2*Omega_m
            denominator = 2*np.pi*f_t
            x = np.full_like(f_t, 0)  # Initialize with z values
            mask = (f_t > 1)  # Only calculate for f_t > 1
            x[mask] = numerator[mask] / denominator[mask]
            #19
            plt.figure()
            plt.plot(time, x*100, label = 'x')
            plt.grid(True)
            plt.legend()
            plt.show()
            pass
        elif image == '20':
            omega = 2*np.pi*f_t
            Theta = omega*time
            v_sd = np.real(v_s)
            v_sq = np.imag(v_s)
            v_a = v_b = v_c = np.zeros(int(len(time)))

            for k in range(int(len(time)-1)):
                [v_a[k], v_b[k], v_c[k]] = inversePark(v_sd[k], v_sq[k], Theta[k]-np.pi/2)
            plt.figure()
            plt.plot(time, v_a, label = 'v_a')
            plt.plot(time, v_b, label = 'v_b')
            plt.plot(time, v_c, label = 'v_c')
            plt.grid(True)
            plt.legend()
            plt.show()
            pass
        elif image == '21':
            plt.figure()
            plt.plot(time, np.real(v_s), label = 'v_sd')
            plt.plot(time, np.imag(v_s), label = 'v_sq')
            plt.grid(True)
            plt.legend()
            plt.show()
            pass
        elif image == '22':
            plt.figure()
            plt.plot(np.real(v_s), np.imag(v_s), label = 'v_sd(v_sq)')
            plt.grid(True)
            plt.legend()
            plt.show()
            pass
        elif image == '23':
            plt.figure()
            plt.plot(time, np.abs(v_s), label = '|v_s|')
            plt.grid(True)
            plt.legend()
            plt.show()
            pass
        elif image == '24':
            plt.figure()
            plt.plot(time, f_t, label = 'frequency')
            plt.grid(True)
            plt.legend()
            plt.show()
            pass
        elif image == '25':
            x = (2*np.pi*f_t - 2*(Omega_m))/(2*np.pi*f_t)
            #19
            plt.figure()
            plt.plot(time, 9.55*Omega_m, label = 'Omega_m')
            plt.grid(True)
            plt.legend()
            plt.show()
            pass
        else:
            raise ValueError('Invalid input.')
