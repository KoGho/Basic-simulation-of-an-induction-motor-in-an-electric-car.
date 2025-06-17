import numba as nb
import numpy as np
from park import inversePark
from utilities import computeQuickV

@nb.njit(parallel=True, fastmath=True)
def autoModel(steps, n, t, J, R_s, R_r, L_r, L_m, L_s, k_0, k_1, \
        psi_s0, psi_r0, omega_m0, c_e0, c_m0, i_s0, i_r0, i_a0, i_b0, i_c0, \
        psi_a0, psi_b0, psi_c0, theta0, i_ar0, i_br0, i_cr0, psi_ar0, psi_br0, \
        psi_cr0, theta_r0, v_r0, value, Vd, ma, f, mf, v_s0, f0, gear):
    # Initialize arrays
    f_t = f0*np.ones(steps)
    omega = 2*np.pi*f0*np.ones(steps)
    v_s = v_s0*np.ones(steps, dtype=np.complex128)
    psi_s = np.zeros(steps, dtype=np.complex128)
    psi_r = np.zeros(steps, dtype=np.complex128)
    omega_m = np.zeros(steps)
    Omega_m = np.zeros(steps)
    c_e = np.zeros(steps)
    c_m = np.zeros(steps)
    i_s = np.zeros(steps, dtype=np.complex128)
    i_r = np.zeros(steps, dtype=np.complex128)

    i_a = np.zeros(steps)
    i_b = np.zeros(steps)
    i_c = np.zeros(steps)
    psi_a = np.zeros(steps)
    psi_b = np.zeros(steps)
    psi_c = np.zeros(steps)
    theta = np.zeros(steps)
    i_ar = np.zeros(steps)
    i_br = np.zeros(steps)
    i_cr = np.zeros(steps)
    psi_ar = np.zeros(steps)
    psi_br = np.zeros(steps)
    psi_cr = np.zeros(steps)
    theta_r = np.zeros(steps)
    v_r = np.zeros(steps)

    # Assign initial conditions.
    psi_s[0] = psi_s0
    psi_r[0]  = psi_r0
    omega_m[0] = omega_m0
    Omega_m[0]  = omega_m0 / n
    c_e[0] = c_e0
    c_m[0] = c_m0
    i_s[0] = i_s0
    i_r[0]  = i_r0

    i_a[0]  = i_a0
    i_b[0]  = i_b0
    i_c[0]  = i_c0
    psi_a[0] = psi_a0
    psi_b[0] = psi_b0
    psi_c[0] = psi_c0
    theta[0] = theta0
    i_ar[0]  = i_ar0
    i_br[0]  = i_br0
    i_cr[0]  = i_cr0
    psi_ar[0]  = psi_ar0
    psi_br[0]  = psi_br0
    psi_cr[0]  = psi_cr0
    theta_r[0] = theta_r0
    v_r[0]  = v_r0

    # Controller parameters
    if value == 1:
        Kp = 0.05
        Ki = 2.5
        #Kp = 0.01
        #Ki = 1
    elif value == 2 or value == 3 or value == 4:
        Kp = 0.01   # proportional gain
        Ki = 1  # integral gain

    omega_ref = 2 * np.pi * f
    integral = 0.0
    f_base = f
    #V_base = np.sqrt(2)*600.0/np.sqrt(3)
    V_base = np.sqrt(2)*ma*Vd
    V_min = np.sqrt(2)*0.1*Vd
    V_max = np.sqrt(2)*Vd


    # Main loop
    for k in range(steps-1):
        # --- Speed control PI ---
        error = omega_ref - omega_m[k]
        integral += error * t
        integral = min(max(integral, -500), 500)
        freq_out = f_base + Kp * error + Ki * integral
        f_t[k+1] = min(max(freq_out, f-2*f/5), f+2*f/5)

        # --- Compute voltage using V/f ---
        omega[k+1] = 2 * np.pi * f_t[k+1]
        V_abs = V_base * f_t[k+1] / f_base
        V_mag = min(max(V_abs, V_min), V_max)
        #theta_e = theta[k+1]
        #v_s[k+1] = V_mag*np.exp(1j*theta_e)
        (v_s[k+1]) = computeQuickV(V_mag/(np.sqrt(2)*Vd), f_t[k+1], t, value, Vd, steps, ma, mf, f, k)

        omega_m[k+1] = omega_m[k] + t * ((n/J) * (c_e[k] - c_m[k])- 0.01 * omega_m[k])
        Omega_m[k+1] = omega_m[k+1] / n
        psi_s[k+1] = psi_s[k] + t * (v_s[k] - R_s * i_s[k] - 1j * omega[k] * \
            psi_s[k])
        psi_r[k+1] = psi_r[k] + t * (v_r[k] - R_r * i_r[k] - 1j * psi_r[k] * \
            (omega[k] - omega_m[k]))
        i_s[k+1] = (L_r * psi_s[k+1] - L_m * psi_r[k+1]) / (L_s * L_r - L_m**2)
        i_r[k+1] = (L_s * psi_r[k+1] - L_m * psi_s[k+1]) / (L_s * L_r - L_m**2)
        theta[k+1] = theta[k] + t * omega[k]
        theta_r[k+1] = theta_r[k] + t * (omega[k] - omega_m[k])
        c_e[k+1] = n * np.imag(i_s[k+1] * np.conj(psi_s[k+1]))
        c_tire = k_0 + k_1*Omega_m[k+1]/gear + k_1 * (Omega_m[k+1]/gear)**2
        c_m[k+1] = c_tire/gear

        (i_a[k], i_b[k], i_c[k]) = inversePark(np.real(i_s[k]), \
            (np.imag(i_s[k])), theta[k])
        (i_ar[k], i_br[k], i_cr[k]) = inversePark(np.real(i_r[k]), \
            (np.imag(i_r[k])), theta_r[k])
        (psi_a[k], psi_b[k], psi_c[k]) = inversePark(np.real(psi_s[k]), \
            (np.imag(psi_s[k])), theta[k])
        (psi_ar[k], psi_br[k], psi_cr[k]) = inversePark(np.real(psi_r[k]), \
            (np.imag(psi_r[k])), theta_r[k])

    return omega_m, Omega_m, psi_s, psi_r, i_s, i_r, theta, \
           theta_r, c_e, c_m, i_a, i_b, i_c, i_ar, i_br, i_cr, \
           psi_a, psi_b, psi_c, psi_ar, psi_br, psi_cr, v_r, v_s, f_t
