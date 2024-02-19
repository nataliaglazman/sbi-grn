import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_path = '/project/home23/ng319/Desktop/sbi-grn/smc-abc/smc_vary_pars/pars_final.out'
parameters = np.loadtxt(file_path)

plot_k = pd.DataFrame(data=parameters, columns=(r"$k_{1}$", r"$k_{2}$", r"$k_{3}$", r"$a_{1}$",r"$a_{2}$",r"$a_{3}$"))
 #Just create a dataframe with rows corresponding to params
print(plot_k)#Need to invert it for plotting
sns.pairplot(plot_k, kind="kde") #Just plot using a smoothing command
plt.show()