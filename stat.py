import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf




def find_sequence(z):
    """Find the longest consecutive sequence in z.

    Given a vector, return the length of its longest run of values.

    Examples:
        find_sequence([0, 0, 1, 1, 1, 0])       # 3
        find_sequence([0, 1, 2, 2, 3, 3, 3], 3) # 3
    """
    z = np.asarray(z)
    if z.ndim > 2 or (z.ndim == 2 and min(z.shape) > 1):
        raise ValueError("z should be a vector")

    z = z.ravel()
    ##find locations of changes
    changes = np.flatnonzero(np.diff(z) != 0)
    ##find distance between changes, adding start and end values to detect
    ##runs of values at the start or end of the vector
    y = np.diff(np.r_[-1, changes, len(z) - 1]).max(initial=0)
    return int(y)

#z = stats.binom.rvs(n=1, p=0.5, size=N);
#print(find_sequence(z))
#en påhittad sekvens kommer troligen ha kortare sviter eftersom man felaktigt uppfattar dem som för osannolika.

#Uppgift 3.1 Längden av den längsta sviten från 200 slantsinglingar är L
#1 H0: Kasten är oberoende observationer av Bin(1,0.5), H1: Kasten ger L som är kortare än om de varit Bin(1,0.5)
#2 Fördelningen av 10000 oberoende L under H0 används för att beräkna p-värdet P(L<=L*|H0) för ett observerat L*. En signikansnivå på 0.05 väljs för det vänstersidiga testet.Om p<0.05 förkastas H0
n_sim = 10000
y = np.zeros(n_sim, dtype=int)
for i in range(n_sim):
    z = stats.binom.rvs(n=1, p=0.5, size=200)
    y[i] = find_sequence(z) 

ax = sns.histplot(x=y, discrete=True)
ax.set(xlabel="Längsta sekvens", ylabel="Antal")




fig, axes = plt.subplots(2, 1, sharex=True)
sns.histplot(x=y, discrete=True, stat="probability", ax=axes[0])
axes[0].set(xlabel="", ylabel="Sannolikhet")

sns.ecdfplot(x=y, ax=axes[1])
axes[1].set(xlabel="Längsta sekvens",
            ylabel="Fördelningsfunktion")

fig.tight_layout()
plt.show()
#3
#givet L*=5 så blir p värdet:
print(np.mean(y<=5))
#detta gav ett värde på 0.0361 vilket är lägre än valda signifikansnivå, därför förkastas H0 och sekvensen sägs påhittad 

#Uppgift 3.2
# Antal simuleringar
n_sim = 100
# Vilka sekvenser vi vill undersöka, antingen kan vi ha sekvenser i linjär eller
# exponentiel skala (logspace ger tal på formen 10^logspace)

#x = np.rint(np.linspace(20,10**14,200)).astype(int)
x = np.rint(np.logspace(0,5,100)).astype(int)
# logspace för att kunna vikta regressionen mot större x-värden
#skappa en pandas data.frame att spara resultat i
simulering = pd.DataFrame({"x": x,                 #Hur många gånger singlar vi slant
                           "y": np.zeros(x.size)}) #Hur lång är längsta sekvens

#enumerate gör att i går från 0 till len(x)-1 och length tar motsvarande värde i x, dvs x[i].
for i, length in enumerate(simulering.x):
  #spara varje simulering
  runs = np.zeros(n_sim, dtype=int)
  for j in range(n_sim):
    z = stats.binom.rvs(n=1, p=0.5, size=length)
    runs[j] = find_sequence(z)
  #beräkna medelvärden av simuleringarna för varje sekvens längd
  simulering.loc[i, 'y'] = runs.mean()

ax = sns.scatterplot(data=simulering, x="x", y="y")
ax.set(xlabel="Antal slantsinglingar", ylabel="Förväntad längsta sekvens")
#Undersök om x och/eller y borde vara i log-skala
#plt.xscale('log')
#plt.yscale('log')

#Längsta sviten y (L) ser ut att bero logaritmiskt av antal kast x. 
# Modellen antas därför ha formen L= a + b*ln(x)+ e, där a,b är konstanter och e det slumpmässiga felet som antas normalförldelat
#linjär regression
reg= smf.ols("y~np.log(x)", simulering).fit()
print(reg.summary())

plt.plot(simulering["x"],reg.predict(simulering),"red")
plt.xscale("log")
#plt.yscale('log')
plt.show()

print("skattningar")
print(reg.params)
print("osäkerheter")
print(reg.cov_params()) 
print("konfidensintervall")
print(reg.conf_int(0.05))