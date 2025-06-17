import numpy as np
import numba as nb

@nb.njit(fastmath=True)
def inversePark(i_sd, i_sq, theta):
      i_a = np.sqrt(2/3)*(i_sd * np.cos(theta) - i_sq * np.sin(theta))
      i_b = np.sqrt(2/3)*(i_sd * np.cos(theta - 2*np.pi/3) - i_sq * np.sin(theta - 2*np.pi/3))
      i_c = np.sqrt(2/3)*(i_sd * np.cos(theta + 2*np.pi/3) - i_sq * np.sin(theta + 2*np.pi/3))
      return i_a, i_b, i_c

@nb.njit(fastmath=True)
def parkF(v_a, v_b, v_c, theta):
    v_sd = np.sqrt(2/3)*(np.cos(theta)*v_a + np.cos(theta - (2*np.pi)/3)*v_b + np.cos(theta - (4*np.pi)/3)*v_c)
    v_sq = np.sqrt(2/3)*(-np.sin(theta)*v_a -np.sin(theta - (2*np.pi)/3)*v_b  -np.sin(theta - (4*np.pi)/3)*v_c)
    return v_sd, v_sq
