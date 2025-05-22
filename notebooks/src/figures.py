# This python file contains the code to create the graphs that are used in the paper.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import statsmodels.api as sm
from linearmodels.panel import PanelOLS
from sklearn.preprocessing import StandardScaler

# Creative Destruction and Productivity Growth figure
def figure1():
    
    data = pd.read_csv("./data/productivity_and_dynamism.csv")
    
    # TFP growth plot
    aggregate_tfp_growth = data[["Year", "NAICS", "TFP_growth", "output_share"]].copy()
    
    # Calculate the sales weighted average of TFP growth
    weighted_growth = aggregate_tfp_growth.groupby("Year").apply(
        lambda x: np.average(x["TFP_growth"], weights=x["output_share"]),
        include_groups=False
    ).reset_index()
    
    aggregate_tfp_growth = weighted_growth.rename(columns={0: "TFP_growth"})
    
    # Calculate the 5 year moving average of TFP growth
    aggregate_tfp_growth["5year_TFP_growth"] = aggregate_tfp_growth["TFP_growth"].rolling(5).mean()
    
    aggregate_tfp_growth['TFP_growth'] = aggregate_tfp_growth['TFP_growth'] * 100
    aggregate_tfp_growth['5year_TFP_growth'] = aggregate_tfp_growth['5year_TFP_growth'] * 100
    
    aggregate_tfp_growth = aggregate_tfp_growth[aggregate_tfp_growth["Year"] < 2022]
    aggregate_tfp_growth = aggregate_tfp_growth[aggregate_tfp_growth["Year"] > 1991]
    
    
    # Job reallocation rate
    reallocation_rate = data[["Year", "NAICS", "job_reallocation_rate", "output_share"]].copy()
    
    weighted_reallocation_rate = reallocation_rate.groupby("Year").apply(
        lambda x: np.average(x["job_reallocation_rate"], weights=x["output_share"]),
        include_groups=False
    ).reset_index()
    
    reallocation_rate = weighted_reallocation_rate.rename(columns={0: "job_reallocation_rate"})
    
    reallocation_rate["5year_reallocation_rate"] = reallocation_rate["job_reallocation_rate"].rolling(5).mean()
    
    reallocation_rate['job_reallocation_rate'] = reallocation_rate['job_reallocation_rate'] * 100
    reallocation_rate['5year_reallocation_rate'] = reallocation_rate['5year_reallocation_rate'] * 100
    
    reallocation_rate = reallocation_rate[reallocation_rate["Year"] < 2022]
    reallocation_rate = reallocation_rate[reallocation_rate["Year"] > 1991]
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Creative Destruction Plot
    ax1.plot(reallocation_rate['Year'], reallocation_rate['job_reallocation_rate'],
            label='Job Reallocation Rate',
            color='#111fA8', linewidth=2)

    ax1.plot(reallocation_rate['Year'], reallocation_rate['5year_reallocation_rate'],
            label='5 Year Moving Average',
            color='#1396f0', linewidth=2, zorder=-10,
            alpha=0.8, linestyle='--')

    ax1.set_xlim(1992, 2021)
    ax1.grid(visible=True, which='both', axis='y', color='gray', linewidth=0.5)

    # format the y ticks to have a percentage sign
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.0f}%'.format(x)))

    ax1.legend()

    ax1.set_xlabel('Year')
    ax1.set_ylabel('Job Reallocation Rate')
    ax1.set_title('Declining Business Dynamism in the US')
    
    # TFP Growth Plot
    ax2.plot(aggregate_tfp_growth['Year'], aggregate_tfp_growth['TFP_growth'],
            label='TFP Growth Rate',
            color='#111fA8', linewidth=2)

    ax2.plot(aggregate_tfp_growth['Year'], aggregate_tfp_growth['5year_TFP_growth'],
            label='5 Year Moving Average',
            color='#1396f0', linewidth=2, zorder=-10,
            alpha=0.8, linestyle='--')

    ax2.set_xlim(1992, 2021)
    ax2.grid(visible=True, which='both', axis='y', color='gray', linewidth=0.5)

    # format the y ticks to have a percentage sign
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.1f}%'.format(x)))

    ax2.legend()

    ax2.set_xlabel('Year')
    ax2.set_ylabel('TFP Growth Rate')
    ax2.set_title('Steady Productivity Growth in the US')
    
    return fig
    
figure1().savefig("./figs/fig1.png", dpi=300, bbox_inches='tight')

# Creative Destruction and Productivity Growth Regression Table
def create_regression_table():

    data = pd.read_csv('./data/productivity_and_dynamism.csv')
    
    df = data.copy()
    df.dropna(inplace=True)
    
    # Standardize 'output'
    scaler = StandardScaler()
    df['output'] = scaler.fit_transform(df[['output']])
    
    # Set index after transformation
    df = df.set_index(['NAICS', 'Year'])
    
    # Create exogenous vars and model
    exog_vars = ['firm_death_rate', 'firm_birth_rate', 'job_reallocation_rate', 'output']
    exog = df[exog_vars]
    exog = sm.add_constant(exog)
    dependent = df['TFP_growth']
    
    model = PanelOLS(dependent, exog, entity_effects=True, time_effects=True)
    results = model.fit()

    table = results.summary.tables[1].as_csv()
    
    with open('./figs/table1.csv', 'w') as f:
        f.write(table)
        f.close()
        
        
create_regression_table()


# Markup and Productivity Dispersion
def figure2():
    
    data = pd.read_csv("./data/markup_and_productivity.csv")
    
    # Markup dispersion
    markup_df = data[(data['year'] > 1991) & data['year'] < 2022].copy()

    years = []
    p50_values = []
    p80_values = []
    p95_values = []
    weighted_avgs = []

    # Group by year and calculate percentiles and sales-weighted average markup
    for year, group_data in markup_df.groupby('year'):
        years.append(year)

        # Calculate percentiles of markup
        p50 = np.percentile(group_data['markup'], 50)
        p80 = np.percentile(group_data['markup'], 80)
        p95 = np.percentile(group_data['markup'], 95)

        p50_values.append(p50)
        p80_values.append(p80)
        p95_values.append(p95)

        # Calculate sales-weighted average markup
        total_sales = group_data['sale'].sum()
        weighted_markup = sum(group_data['markup'] * group_data['sale']) / total_sales
        weighted_avgs.append(weighted_markup)
        
    markup_df = pd.DataFrame({
        'year': years,
        'p50': p50_values,
        'p80': p80_values,
        'p95': p95_values,
        'weighted_avg': weighted_avgs
    })
    
    # Productivity dispersion
    productivity_df = data[(data['year'] > 1991) & (data['year'] < 2022)]

    years = []
    p50_values = []
    p80_values = []
    p95_values = []
    weighted_avgs = []

    # Group by year and calculate percentiles and sales-weighted average markup
    for year, group_data in productivity_df.groupby('year'):
        years.append(year)

        # Calculate percentiles of markup
        p50 = np.percentile(group_data['a'], 50)
        p80 = np.percentile(group_data['a'], 80)
        p95 = np.percentile(group_data['a'], 95)

        p50_values.append(p50)
        p80_values.append(p80)
        p95_values.append(p95)

        # Calculate sales-weighted average markup
        total_sales = group_data['sale'].sum()
        weighted_markup = sum(group_data['a'] * group_data['sale']) / total_sales
        weighted_avgs.append(weighted_markup)
        
    productivity_df = pd.DataFrame({
        'year': years,
        'p50': p50_values,
        'p80': p80_values,
        'p95': p95_values,
        'weighted_avg': weighted_avgs
    })
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Markup Dispersion Plot
    ax1.plot(markup_df['year'], markup_df['p50'],
            label='P50',
            color='#111fA8', linewidth=2,
            linestyle='-.')
    
    ax1.plot(markup_df['year'], markup_df['p80'],
            label='P80',
            color='#111fA8', linewidth=2,
            linestyle=':')
    
    ax1.plot(markup_df['year'], markup_df['p95'],
            label='P95',
            color='#111fA8', linewidth=2,
            linestyle='--')

    ax1.plot(markup_df['year'], markup_df['weighted_avg'],
            label='Average',
            color='#1396f0', linewidth=2, zorder=-10)

    ax1.set_xlim(1992, 2021)
    ax1.grid(visible=True, which='both', axis='y', color='gray', linewidth=0.5)

    # format the y ticks to have a percentage sign
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.1f}'.format(x)))

    ax1.legend()

    ax1.set_xlabel('Year')
    ax1.set_ylabel('Markup')
    ax1.set_title('Markup Dispersion')
    
    # Productivity Dispersion Plot
    ax2.plot(productivity_df['year'], productivity_df['p50'],
            label='P50',
            color='#111fA8', linewidth=2,
            linestyle='-.')
    
    ax2.plot(productivity_df['year'], productivity_df['p80'],
            label='P80',
            color='#111fA8', linewidth=2,
            linestyle=':')
    
    ax2.plot(productivity_df['year'], productivity_df['p95'],
            label='P95',
            color='#111fA8', linewidth=2,
            linestyle='--')

    ax2.plot(productivity_df['year'], productivity_df['weighted_avg'],
            label='Average',
            color='#1396f0', linewidth=2, zorder=-10)

    ax2.set_xlim(1992, 2021)
    ax2.grid(visible=True, which='both', axis='y', color='gray', linewidth=0.5)

    # format the y ticks to have a percentage sign
    ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:.1f}'.format(x)))

    ax2.legend()

    ax2.set_xlabel('Year')
    ax2.set_ylabel('TFP Estimate')
    ax2.set_title('Productivity Dispersion')
    
    return fig

figure2().savefig("./figs/fig2.png", dpi=300, bbox_inches='tight')

def figure3():
    
    data = pd.read_csv('./data/real_rnd_patents.csv')
    
    
    df = data[(data['year'] > 1991) & (data['year'] < 2022)].copy()
    
    df['patents'] = df['patents'] / 1000
    
    fig, ax = plt.subplots(figsize=(6,4))
    
    ax.plot(df['year'], df['patents'],
            label='US Originated Patents',
            color='#111fA8', linewidth=2)

    ax1 = ax.twinx()

    ax1.plot(df['year'], df['rnd'],
            label='R&D Spending',
            color='#1396f0', linewidth=2)

    ax.set_xlim(1992, 2021)
    ax.grid(visible=True, which='both', axis='y', color='gray', linewidth=0.5)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax1.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    ax.set_xlabel('Year')
    ax.set_ylabel('US Originated Patents (Thousands)')
    ax1.set_ylabel('Real R&D Spending (2017 $USDmn)')
    ax.set_title('Growing Innovation')
    
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '${:,.0f}m'.format(x)))
    
    return fig
    
figure3().savefig("./figs/fig3.png", dpi=300, bbox_inches='tight')