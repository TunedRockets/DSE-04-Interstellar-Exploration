from Power.powerinsizeout import reactor, fuelcell, rtgsize


bigmass, fuelmass = fuelcell(2370.4, 199*24*3600)
print(bigmass, fuelmass)
print(bigmass+fuelmass)