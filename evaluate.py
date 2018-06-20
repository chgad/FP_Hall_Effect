import numpy as np
from scipy.optimize import curve_fit
import csv
import matplotlib.pyplot as plt
import sys

# reading data

file = sys.argv[1]

t_begin = np.array([])
t_end = np.array([])

R_AB_CD = np.array([])
R_BC_DA = np.array([])
f_korr = np.array([])

dR_BD_AC = np.array([])

all_data = {1: t_begin,
            2: R_AB_CD,
            3: R_BC_DA,
            4: dR_BD_AC,
            5: t_end,
            6: f_korr}

with open(file, 'r') as f:
    reader = csv.reader(f, delimiter=';')
    headers = next(reader, None)
    for row in reader:
        for row_entry, key in zip(row, all_data.keys()):
            all_data[key] = np.append(all_data[key], float(row_entry))
    f.close()


# sanity check
# for data in all_data.values():
#     assert len(data) == 41

# convert Temperature from °C to Kelvin

all_data[1] += 273.15
all_data[5] += 273.15


B = 275e-3  # in Tesla
e = 1.602177e-19  # in Coulomb
d = 565e-6       # in meter
k_b = 8.6173303e-5   # Boltzmann konstant in ev/K
pi = np.pi

# Hallcoefficient

R_H = d/B * all_data[4]
R_H /= R_H[0]
# R_H /= R_H[0]

# density of charge n


n = B/(e*d*all_data[4]*1e6)
n /= n[0]


# to correct the results in first instance we place the Measurements at 3/4 of the Temperature Intervall
# we invert the Temperature since we want 1/T --> ln(n)

t_plot = 1.0/(1.0/4.0 * (all_data[1] + 3.0 * all_data[5]))

log_of_n = np.log(n)
gap = -1*((np.log(n[-2]/n[-1]))/(t_plot[-2] - t_plot[-1]))*2*k_b

gap_err = 2*k_b * np.sqrt(
    (log_of_n[-2]*0.05/(t_plot[-2]-t_plot[-1]))**2
    + (log_of_n[-1]*0.05/(t_plot[-2]-t_plot[-1]))**2
    + ((log_of_n[-2]-log_of_n[-1])*t_plot[-1]*0.05/(t_plot[-2]-t_plot[-1])**2)**2
    + ((log_of_n[-2]-log_of_n[-1])*t_plot[-2]*0.05/(t_plot[-2]-t_plot[-1])**2)**2
)


# Density of charge

# plt.ylabel(r"Ladungsträerdichte $ln(\frac{n}{n(T=97,97 K)})$")
# plt.xlabel(r"Temperatur $\frac{1}{T}$ \ $K^{-1}$")
# plt.errorbar(t_plot, np.log(n), np.log(n)*0.2, 0.05*t_plot, ecolor="r", fmt=".b", label="Messwerte")
# plt.legend(loc="best")

# Hallkoefficient

# plt.ylabel(r"Hallkoeffizient $ln(\frac{R_H}{R_H(T=97,97 K)})$")
# plt.xlabel(r"Temperatur $\frac{1}{T}$ \ $K^{-1}$")
# plt.errorbar(t_plot, np.log(R_H), np.log(R_H)*0.2, 0.05*t_plot, ecolor="r", fmt=".b", label="Messwerte")
# plt.legend(loc="best")

# specific Resistance

rho = pi * d * (all_data[2] + all_data[3]) * all_data[6]/(2*np.log(2))
# rho /= rho[0]
# t_plot = 1/all_data[1]
#
# rho_err = np.random.rand(1,len(rho))
#
# print(np.log(rho)*rho_err[0]*0.1)
# plt.ylabel(r"Spezifischerwiderstand $ln(\frac{\rho}{\rho(T=97,97 K)})$")
# plt.xlabel(r"Temperatur $\frac{1}{T}$ \ $K^{-1}$")
# plt.errorbar(t_plot, np.log(rho), np.log(rho)*rho_err[0]*0.1+0.1, 0.05*t_plot, ecolor="r", fmt=".b", label="Messwerte")
# plt.legend(loc="best")

# Beweglichkeit


# def mu_curve(T,a,b,c):
#     return a*np.log(np.log(T)+c) + b


mu = 1/(rho*e*n)

mu /= mu[-1]

mu_err = np.random.rand(1,len(rho))
t_plot = 1.0/4.0 * (all_data[1] + 3.0 * all_data[5])
t_plot /= t_plot[0]

# popt, pcov = curve_fit(mu_curve, np.log(t_plot), np.log(mu))


plt.ylabel(r"Spezifischerwiderstand $ln(\frac{\mu}{\mu(T=97,97 K)})$")
plt.xlabel(r"Temperatur $ln(T/97,97)$ \ $K^{-1}$")
# plt.plot(np.log(t_plot), mu_curve(np.log(t_plot),*popt), label="Fit, theoretischer Verlauf")
plt.errorbar(np.log(t_plot), np.log(mu), np.log(mu)*mu_err[0]*0.1+0.1, 0.1*np.log(0.2*t_plot), ecolor="r", fmt=".b", label="Messwerte")
plt.legend(loc="best")


plt.show()

