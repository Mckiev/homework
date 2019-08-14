import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

directory = "/Users/mckiev/Desktop/AI_study/Coding/homework/hw1/demo_compare/Hopper256"


bc = pd.read_csv('BC_vs_obs/Hopper-v2_64/results.csv')

daggrD = pd.read_csv('DAggr_Hopper-v2-64.csv')


plot_data = bc.append(daggrD, ignore_index=True, sort=False)
plot_data = plot_data[plot_data["Observations"] < 4001]

sns.set(context = 'notebook')
ax = sns.lineplot(x = 'Observations', y= 'Returns', hue="Algorithm", err_style="bars",
 				  markers=True, err_kws={'capsize':3},  ci='sd', data=plot_data)
plt.axhline(y=3776, color='g', linestyle='--', label= 'Average expert return')
plt.yticks(list(plt.yticks()[0]) + [3776])
plt.title('DAgger on Hopper-v2')
plt.legend(loc = 'upper right', bbox_to_anchor=(1,0.9))
plt.show()
