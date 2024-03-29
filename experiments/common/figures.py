import matplotlib.pyplot as plt
x = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
y1 = [0, 0.004, 0.041, 0.083, 0.120, 0.147, 0.172, 0.192, 0.208, 0.218, 0.229]
y2 = [0, 0, 0, 0, 0, 0.00000814, 0.000217, 0.000681, 0.00204, 0.00422, 0.00696]

z1 = [21.516, 9.403, 8.840, 8.212, 7.740,
      7.497, 7.279, 7.127, 7.033, 7.016, 6.936]
z2 = [18.898+2, 18.898+2, 18.903+2, 18.905+1, 18.891,
      18.346, 17.326, 15.384, 12.768, 10.779, 9.843]

plt.plot(x, y1,"c*-", label="Normal Training", linewidth=2)
plt.plot(x, y2, "m.-.",label="Noise Training", linewidth=1)
plt.xlabel("Sigma")
plt.ylabel("BER")
plt.legend(loc='upper right', fontsize=8)

plt.show()
