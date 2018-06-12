import csv
import sys

import numpy as np
import sympy as sp
import scipy as scp
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# from sympy import solve, Symbol
from pynverse import inversefunc

file = sys.argv[1]


t_begin = []
t_end = []

U_CD = []
U_DA = []
U_AC_B_plus = []
U_AC_B_minus = []

all_data = [t_begin, U_CD, U_DA, U_AC_B_plus, U_AC_B_minus, t_end]

with open(file, 'r') as f:
    reader = csv.reader(f, delimiter=";")
    headers = next(reader, None)
    for row in reader:
        for row_entry, arr in zip(row, all_data):
            arr.append(float(row_entry))
    f.close()

print(headers)

for data in all_data:
    assert len(data) == 41

reworked_file = "FP_hall_effect_reworked.csv"


def fifth_poly(x, a, b, c, d, e, f):#
    return f*x**5 + e*x**4 + a*x**3 + b*x**2 + c*x + d


temp = np.array(list(range(-200, 30, 10)) + [25, 30, 35, 40, 45] + list(range(50, 110, 10)))
mv_corresp = [-5.7, -5.51, -5.32, -5.12, -4.91, -4.69, -4.46, -4.21, -3.95, -3.68, -3.4, -3.11, -2.81, -2.5, -2.18,
              -1.85, -1.5, -1.14, -0.77, -0.39, 0.0, 0.4, 0.8, 1.0, 1.21, 1.42, 1.63, 1.84, 2.05, 2.48, 2.91, 3.35, 3.8,
              4.25]

popt, pcov = curve_fit(fifth_poly,temp,mv_corresp)

inver_fifth = inversefunc(fifth_poly, args=tuple(popt))

t_real_begin = inver_fifth(np.array(t_begin))
t_real_end = inver_fifth(np.array(t_end))


def correct(f,gamma):
    return np.cosh((gamma - 1) / (gamma + 1) * np.log(2) / f) - 0.5 * np.exp(np.log(2) / f)


# U_CD becomes now R_AB_CD since I_AB is 1 mv \pm 0.01
# U_DA becomes now R_BC_DA since I_BC is 1 mv \pm 0.01
dR_BD_AC = np.array(U_AC_B_plus) - np.array(U_AC_B_minus)

quotient = [scp.optimize.fsolve(correct, np.array([10]), args=(R_ab / R_bc), xtol=1e-10)[0]
            for R_ab, R_bc in zip(U_CD, U_DA)]

assert len(quotient) == 41
assert ((dR_BD_AC < 0).sum() == dR_BD_AC.size).astype(np.int)

csv.register_dialect("semicolon", delimiter=";")

new_headers = ["T_anf", "R_AB_CD", "R_BC_DA", "dR_BD_AC", "T_end", "f_kor"]

with open(reworked_file, "w") as f:
    writer = csv.writer(f, dialect="semicolon")
    writer.writerow(new_headers)
    rows = zip(t_real_begin, U_CD, U_DA, dR_BD_AC, t_real_end, quotient)
    writer.writerows(rows)

    f.close()

#
# temp_range = np.arange(-200,150,0.1)
# plt.plot(temp_range,fifth_poly(temp_range, *popt))
# plt.plot(t_real_begin, t_begin, '.r', label="T begin")
# plt.plot(t_real_end, t_end, '.y', label="T end")
# plt.plot(temp, mv_corresp,'.b')
# plt.legend(loc="best")
# plt.show()