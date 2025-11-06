import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

sns.set_theme(context='paper', style='whitegrid', font_scale=2.2, font="serif", rc={
    "text.usetex": True
})

TYPE = ['AU', 'EU', 'TU']
DATA = ['bimodal', 'right_tail', 'left_tail', 'unimodal']
PRR = ["PRR (MAE)", "PRR (MCR)"]
GROUP = ['full', 'tail', 'head']

for group in GROUP:
    for prr in PRR:
        prrs_latex = pd.DataFrame()
        prrs_ranks = pd.DataFrame()
        for data in DATA:
            for type in TYPE:

                prrs = pd.read_csv(f'prrs_{data}.csv')

                # Rename uncertainty measures
                prrs = prrs.replace(f'Entropy', "ent")
                prrs = prrs.replace(f'Variance', "var")
                prrs = prrs.replace(f"Binary Entropy", "ord-ent")
                prrs = prrs.replace(f"Binary Variance", "ord-var")
                prrs = prrs.replace(f"Binary Entropy One vs. Rest", "bin-ent")
                prrs = prrs.replace(f"Binary Variance One vs. Rest", "bin-var")

                prrs = prrs[(prrs["Uncertainty Type"] == type) & (prrs["Group"] == group)]


                ax = sns.boxplot(data=prrs, showmeans=True,showfliers=False, x="Uncertainty Measure", y=prr, hue="Uncertainty Measure", meanprops={'marker':'^',
                       'markerfacecolor':'black',
                       'markeredgecolor':'black'
                                                                                                                                                   },
                                 order=["bin-ent", "bin-var", "ent","ord-ent","ord-var","var"])
                ax.set(xlabel=None)

                plt.savefig("./plots/" + type + "_" + data + "_" + group + "_" + prr + ".pdf", bbox_inches='tight',
                            dpi=300)

                plt.show()

                match data:
                    case "bimodal":
                        data_latex = 'D1'
                    case "right_tail":
                        data_latex = 'D2'
                    case "left_tail":
                        data_latex = 'D3'
                    case "unimodal":
                        data_latex = 'D4'

                prrs['Dataset'] = data_latex
                prrs_latex = pd.concat([prrs_latex, prrs])
                prrs_ranks = pd.concat([prrs_ranks, prrs])

        # Save as latex
        prrs_latex = prrs_latex.drop('Group', axis=1).groupby(
            ['Dataset', "Uncertainty Measure", "Uncertainty Type"]).agg(
            ['mean', 'std'])
        numeric_cols = prrs_latex.columns.get_level_values(0).unique()
        prrs_latex = prrs_latex.round(3)

        for col in numeric_cols:
            mean_col = (col, 'mean')
            std_col = (col, 'std')
            combined = (
                    prrs_latex[mean_col].astype(str) + " \\textpm " +
                    prrs_latex[std_col].astype(str)
            )
            # Store as a new column in the DataFrame, with a clear name
            prrs_latex[(col, 'mean_std')] = combined

        prrs_latex = prrs_latex.drop(columns=[(col, m) for col in numeric_cols for m in ['mean', 'std']])

        prrs_latex.columns = ['_'.join([str(c) for c in col if str(col)]) for col in prrs_latex.columns]

        prrs_latex.reset_index(drop=False, inplace=True)

        latex_df = prrs_latex.pivot(index=['Dataset', 'Uncertainty Type'], columns='Uncertainty Measure',
                                    values=prr + '_mean_std')

        # Export as a Booktabs LaTeX table
        latex_str = latex_df.to_latex(
            index=True,
            caption=group.title() + " - " + prr,
            label='tab:' + group + "-" + prr
        )

        # Save to file
        with open("./raw/" + group + "_" + prr + ".tex", 'w') as f:
            f.write(latex_str)

        # Save ranks as latex

        prrs_ranks = prrs_ranks[["Dataset", "Uncertainty Type", "Uncertainty Measure", prr]]

        prrs_ranks = prrs_ranks.groupby(["Dataset", "Uncertainty Type", "Uncertainty Measure"]).agg(['mean']).groupby(
            ["Dataset", "Uncertainty Type"]).rank(ascending=False)

        # Export as a Booktabs LaTeX table
        latex_str = prrs_ranks.to_latex(
            index=True,
            caption=group.title() + " - " + prr,
            label='tab:' + group + "-" + prr + "-ranks"
        )

        # Save to file
        with open("./ranks/" + group + "_" + prr + "_ranks.tex", 'w') as f:
            f.write(latex_str)
