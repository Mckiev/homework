import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

"""
Using the plotter:

Call it from the command line, and supply it with logdirs to experiments.
Suppose you ran an experiment with name 'test', and you ran 'test' for 10 
random seeds. The runner code stored it in the directory structure

    data
    L test_EnvName_DateTime
      L  0
        L log.txt
        L params.json
      L  1
        L log.txt
        L params.json
       .
       .
       .
      L  9
        L log.txt
        L params.json

To plot learning curves from the experiment, averaged over all random
seeds, call

    python plot.py data/test_EnvName_DateTime --value AverageReturn

and voila. To see a different statistics, change what you put in for
the keyword --value. You can also enter /multiple/ values, and it will 
make all of them in order.


Suppose you ran two experiments: 'test1' and 'test2'. In 'test2' you tried
a different set of hyperparameters from 'test1', and now you would like 
to compare them -- see their learning curves side-by-side. Just call

    python plot.py data/test1 data/test2

and it will plot them both! They will be given titles in the legend according
to their exp_name parameters. If you want to use custom legend titles, use
the --legend flag and then provide a title for each logdir.

"""

def plot_data(data, x = 'Timestep', y="MeanEpisodeReward", hue = "Experiment", sci_label = False, savefig = None):
    if isinstance(data, list):
        data = pd.concat(data, ignore_index=True)

    sns.set(style="darkgrid")
    g = sns.lineplot(data=data, x=x, y=y, hue=hue)
    plt.legend(loc='best', title=None).set_draggable(True)
    if sci_label:
        xlabels = ['{:,.0f}'.format(x) + 'M' for x in g.get_xticks()/1e6]
        g.set_xticklabels(xlabels)
    sns.set(style="darkgrid")
    if savefig:
        os.makedirs('Figs', exist_ok=True)
        plt.savefig('Figs/'+savefig, dpi=600)
    plt.show()


def get_datasets(fpath, condition=None):

    datasets = pd.DataFrame()
    for root, dir, files in os.walk(fpath):
        if 'log.txt' in files:
            param_path = open(os.path.join(root,'params.json'))
            params = json.load(param_path)
            exp_name = params['exp_name']
            
            log_path = os.path.join(root,'log.txt')
            experiment_data = pd.read_csv(log_path, sep='\t')
     
            experiment_data.insert(
                len(experiment_data.columns),
                'Experiment',
                condition or exp_name
                )
            datasets = datasets.append(experiment_data)
    
    return datasets


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('logdir', nargs='*')
    parser.add_argument('--x', default='Timestep', help = 'x-axis parameter')
    parser.add_argument('--legend', nargs='*', help = 'custom legend for the curve. Defaults to experiment name')
    parser.add_argument('--value', default='MeanEpisodeReward', nargs='*', help ='y-axis parameter')
    parser.add_argument('--sci_label', action = 'store_true', help = 'Display x-axis in millions')
    parser.add_argument('--pic_name', help = "Filename for plot to be saved at 'Figs/' ")
    args = parser.parse_args()

    use_legend = False
    if args.legend is not None:
        assert len(args.legend) == len(args.logdir), \
            "Must give a legend title for each set of experiments."
        use_legend = True


    data = pd.DataFrame()
    if use_legend:
        for logdir, legend_title in zip(args.logdir, args.legend):
            data = data.append(get_datasets(logdir, legend_title))
    else:
        for logdir in args.logdir:
            data = data.append(get_datasets(logdir))


    if isinstance(args.value, list):
        values = args.value
    else:
        values = [args.value]
    for value in values:
        plot_data(data, x = args.x, y=value, sci_label = args.sci_label, savefig = args.pic_name)

if __name__ == "__main__":
    main()
