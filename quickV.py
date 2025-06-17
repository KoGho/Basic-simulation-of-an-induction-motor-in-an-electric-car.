import numpy as np
import numba as nb

@nb.njit(fastmath=True)
def quickSynchronous(Vd, time, ma_t, f_t, k):
    # Syncrhonous voltages.
    omega = 2*np.pi*f_t
    k = k+1
    Va = np.sqrt(2)*ma_t*Vd*np.sin(omega*time*k)
    Vb = np.sqrt(2)*ma_t*Vd*np.sin(omega*time*k+240/180*np.pi)
    Vc = np.sqrt(2)*ma_t*Vd*np.sin(omega*time*k+120/180*np.pi)
    return Va, Vb, Vc

  # Square wave voltages in a half bridge.
@nb.njit(fastmath=True)
def quickInverter(time, steps, Vd, ma_t, f_t, k):
    omega = 2*np.pi*f_t
    k = k+1
    v_ca = np.sqrt(2)*ma_t*np.sin(omega*time*k)
    v_cb = np.sqrt(2)*ma_t*np.sin(omega*time*k+240/180*np.pi)
    v_cc = np.sqrt(2)*ma_t*np.sin(omega*time*k+120/180*np.pi)

    v_s = 0

    # Phase a.
    if v_ca > v_s:
        transistor_a = 1
    elif v_ca <= v_s:
        transistor_a = 0

    if transistor_a == 1:
        v_a = Vd
    elif transistor_a == 0:
        v_a = -Vd

    # Phase b.
    if v_cb > v_s:
        transistor_b = 1
    elif v_cb <= v_s:
        transistor_b = 0

    if transistor_b == 1:
        v_b = Vd
    elif transistor_b == 0:
        v_b = -Vd

    # Phase c.
    if v_cc > v_s:
        transistor_c = 1
    elif v_cc <= v_s:
        transistor_c = 0

    if transistor_c == 1:
        v_c = Vd
    elif transistor_c == 0:
        v_c = -Vd

    Va = v_a
    Vb = v_b
    Vc = v_c
    return Va, Vb, Vc

  # Pwm voltages.
@nb.njit(fastmath=True)
def quickBipolarPwm(m_a, mf, f, time, steps, Vd, ma_t, f_t, k):
    omega_s = mf*f*2*np.pi
    ts = 2*np.pi/omega_s
    omega = f_t*2*np.pi

    v_ca = np.sqrt(2)*ma_t*np.sin(omega*time*k)
    v_cb = np.sqrt(2)*ma_t*np.sin(omega*time*k+240/180*np.pi)
    v_cc = np.sqrt(2)*ma_t*np.sin(omega*time*k+120/180*np.pi)

    if time*k >= k*ts and time*k < k*ts + ts/4:
        v_s = -4/ts*(time*k-k*ts)
    elif time*k >= k*ts + ts/4 and time*k < k*ts+3*ts/4:
        v_s = 4/ts*(time*k-k*ts)-2
    elif time*k >= k*ts + 3*ts/4 and time*k < k*ts + ts:
        v_s = -4/ts*(time*k-k*ts)+4


    # Phase a.
    if v_ca > v_s:
        transistor_a = 1
    elif v_ca <= v_s:
        transistor_a = 0

    if transistor_a == 1:
        v_a = Vd
    elif transistor_a == 0:
        v_a = -Vd

    # Phase b.
    if v_cb > v_s:
        transistor_b = 1
    elif v_cb <= v_s:
        transistor_b = 0

    if transistor_b == 1:
        v_b = Vd
    elif transistor_b == 0:
        v_b = -Vd

    # Phase c.
    if v_cc > v_s:
        transistor_c = 1
    elif v_cc <= v_s:
        transistor_c = 0

    if transistor_c == 1:
        v_c = Vd
    elif transistor_c == 0:
        v_c = -Vd

    Va = v_a
    Vb = v_b
    Vc = v_c
    return Va, Vb, Vc

  # Unipolar full bridge inverter pwm
@nb.njit(fastmath=True)
def quickUnipolarPwm(m_a, mf, f, time, steps, Vd, ma_t, f_t, k):
    omega_s = mf*f*2*np.pi
    ts = 2*np.pi/omega_s
    omega = 2*np.pi*f_t

    v_caa = np.sqrt(2)*ma_t*np.sin(omega*time*k)
    v_cba = np.sqrt(2)*ma_t*np.sin(omega*time*k+240/180*np.pi)
    v_cca = np.sqrt(2)*ma_t*np.sin(omega*time*k+120/180*np.pi)

    v_cab = -np.sqrt(2)*ma_t*np.sin(omega*time*k)
    v_cbb = -np.sqrt(2)*ma_t*np.sin(omega*time*k+240/180*np.pi)
    v_ccb = -np.sqrt(2)*ma_t*np.sin(omega*time*k+120/180*np.pi)


    if time*k >= k*ts and time*k < k*ts + ts/4:
        v_s = -4/ts*(time*k-k*ts)
    elif time*k >= k*ts + ts/4 and time*k < k*ts+3*ts/4:
        v_s = 4/ts*(time*k-k*ts)-2
    elif time*k >= k*ts + 3*ts/4 and time*k < k*ts + ts:
        v_s = -4/ts*(time*k-k*ts)+4


      # Phase a.
    if v_caa > v_s:
        transistor_aa = 1
    elif v_caa <= v_s:
        transistor_aa = 0

    if v_cab > v_s:
        transistor_ab = 1
    elif v_cab <= v_s:
        transistor_ab = 0

    if transistor_aa == 1 and transistor_ab == 0:
        v_a = Vd
    elif transistor_aa == 0 and transistor_ab == 1:
        v_a = -Vd
    elif transistor_aa == 1 and transistor_ab == 1:
        v_a = 0
    elif transistor_aa == 0 and transistor_ab == 0:
        v_a = 0

    # Phase b.
    if v_cba > v_s:
        transistor_ba = 1
    elif v_cba <= v_s:
        transistor_ba = 0

    if v_cbb > v_s:
        transistor_bb = 1
    elif v_cbb <= v_s:
        transistor_bb = 0

    if transistor_ba == 1 and transistor_bb == 0:
        v_b = Vd
    elif transistor_ba == 0 and transistor_bb == 1:
        v_b = -Vd
    elif transistor_ba == 1 and transistor_bb == 1:
        v_b = 0
    elif transistor_ba == 0 and transistor_bb == 0:
        v_b = 0

    # Phase c.
    if v_cca > v_s:
        transistor_ca = 1
    elif v_cca <= v_s:
        transistor_ca = 0

    if v_ccb > v_s:
        transistor_cb = 1
    elif v_ccb <= v_s:
        transistor_cb = 0

    if transistor_ca == 1 and transistor_cb == 0:
        v_c = Vd
    elif transistor_ca == 0 and transistor_cb == 1:
        v_c = -Vd
    elif transistor_ca == 1 and transistor_cb == 1:
        v_c = 0
    elif transistor_ca == 0 and transistor_cb == 0:
        v_c = 0

    Va = v_a
    Vb = v_b
    Vc = v_c
    return Va, Vb, Vc
