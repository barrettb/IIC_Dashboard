import streamlit as st
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
import requests
import matplotlib.pyplot as plt
import plotly.express as px

def portfolio_weightings(subset):
    weights = [0.016666666666666666, 0.00164788, 0.000178035, 0.033145842, 0.024828715, 0.016666666666666666, 0.026928044, 0.044189697, 0.055261441, 0.000517012, 0.000594662, 0.016666666666666666, 0.016666666666666666, 0.00013656, 0.016666666666666666, 0.016666666666666666, 0.016666666666666666, 0.016666666666666666, 0.016666666666666666, 0.089653034, 0.087917666, 0.000561825, 0.003104912, 0.011592297, 9.0385e-05, 0.016666666666666666, 0.016666666666666666, 0.016666666666666666, 0.000541409, 0.000345167, 0.000311743, 0.000237095, 0.079347957, 0.035077629, 0.085020162, 0.00772739, 0.013748099, 0.000631485, 0.000814736, 0.000620857, 0.000327856, 0.015386971, 0.000952365, 0.000876036, 0.000951154, 0.032092231, 0.010361308, 0.082507035, 2.8038e-05, 0.000972577, 0.050754194, 1.8499e-05]
    latest_prices = subset.head(1)
    portfolio_weightings = {}
    i = -1
    for column in subset.columns:
        i= i+1
        portfolio_weightings[column] = weights[i]
    subset = subset.reset_index()
    portfolio_returns = []
    capital = 20000 

    shares = []
    row = subset.iloc[1, 1:]  # Assuming subset is a DataFrame with stock prices, and you want to exclude the first column from the iteration
    
    for stock, weight in portfolio_weightings.items():
        price = row[stock]
        shares_to_buy = (capital * weight) / price
        shares.append((stock, shares_to_buy))
    
    portfolio_value = []  # To store the portfolio value for each day

    for index, row in subset.iterrows():
        date = row['Date']
        portfolio_total = 0.0

        for stock, num_shares in shares:
            if stock in row:
                stock_price = row[stock]
                stock_value = stock_price * num_shares
                portfolio_total += stock_value
        
        portfolio_value.append((date, portfolio_total))
    
    gains = []
    previous_value = portfolio_value[0][1]
    
    for date, value in portfolio_value:
        daily_gain = (value - previous_value) / previous_value * 100
        gains.append((date, daily_gain))
        previous_value = value
    
    cumulative_returns = [ret for date, ret in gains]
    dates = [date for date, gain in gains]
    daily_gains = [gain for date, gain in gains]
    values = [value for date, value in portfolio_value]

    return values, daily_gains, dates, cumulative_returns

def tab1(values, daily_gains, dates, cumulative_returns):
    tickers = subset.columns.tolist()
    sectors = []

    for ticker in tickers:
        stock = yf.Ticker(ticker)
        sector = stock.info.get("sector")
        sectors.append(sector)

    sector_counts = pd.Series(sectors).value_counts()
    sector_labels = sector_counts.index.tolist()
    sector_sizes = sector_counts.values.tolist()

    last_cumulative_return = cumulative_returns[-2]
    last_daily_dollars = values[-2] - values[-3]
    format_daily_dollars = "${:.2f}".format(last_daily_dollars)
    total_dollars = values[-2] - values[0]
    format_total_dollars = "${:.2f}".format(total_dollars)
    total_dollars2 = total_dollars + 20000
    format_total_dollars2 = "${:.2f}".format(total_dollars2)

    formatted_cumulative_return = "{:.1f}%".format(last_cumulative_return)
    formatted_total_return = "{:.1f}%".format(sum(cumulative_returns[0:-2]))

    st.header('Climate Resilience Fund Performance')
    st.markdown('### Metrics')
    col1, col2, col3 = st.columns(3)
    col1.metric("Day's Gain", formatted_cumulative_return, format_daily_dollars)
    col2.metric("Total Gain", formatted_total_return, format_total_dollars)
    col3.metric("Total Value", format_total_dollars2)

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=dates, y=values, mode='lines', name='Portfolio Value'))

    spy_data = yf.download("SPY", start=min(dates), end=max(dates))
    spy_dates = spy_data.index
    spy_values = spy_data["Adj Close"].values

    # Calculate the number of shares of SPY based on the portfolio value
    spy_price = spy_data["Adj Close"].iloc[-1]
    spy_shares = 20000 / spy_price
    spy_portfolio_value = (spy_shares * spy_values) + 2733

    fig1.add_trace(go.Scatter(x=spy_dates, y=spy_portfolio_value, mode='lines', name='SPY Portfolio'))

    fig1.update_layout(title='Portfolio Value Over Time', xaxis_title='Date', yaxis_title='Value')
    fig2 = px.pie(
        values=sector_sizes,
        names=sector_labels,
        title='Sector Breakdown of Portfolio',
    )
    fig2.update_traces(textposition='inside', textinfo='percent+label')
    fig2.update_layout(showlegend=False)

    col4, col5 = st.columns([2, 1])
    col4.plotly_chart(fig1, use_container_width=True)
    col5.plotly_chart(fig2, use_container_width=True)




def tab3():
    return None


def tab4():
    return None


def tab5():
    return None


def tab6():
    return None



def run():
    st.set_page_config(layout='wide', initial_sidebar_state='expanded')
    with open('style.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    portfolio_value, daily_gains, dates, cumulative_returns = portfolio_weightings(subset)
    tab1(*portfolio_weightings(subset))

def read_data():
    
        start_date = datetime.datetime(2021, 1, 21)
        end_date = datetime.datetime.now()

        tickers = [
        'BLK', 'ADBE', 'POOL', 'SONY', 'CRM', 'NVDA', 'MSFT', 'MRK', 'TSLA', 'SIEGY', 'ALB', 'MS', 'TGT', 'HD', 'NEE', 'ICLN', 'LIT', 'INTC', 'HAS', 'SEDG', 'WM', 'JLL', 'CBRE', 'WY', 'JBGS', 'PLD', 'AY', 'CI', 'AVT',
        'FLEX', 'EQIX', 'AVB', 'PEAK', 'SMPL', 'PSMT', 'LIN', 'APD', 'ABB', 'JCI', 'XYL', 'CAKE', 'NKE', 'ED', 'XEL', 'NGG', 'FSLR', 'FERG', 'DLR', 'ABG',
        'TMO', 'ECL', 'ASML']
        data = yf.download(tickers, start=start_date, end=end_date)
        spy_ticker = ['SPY']
        
        subset = data.xs('Adj Close', axis=1, level=0)
        
        return subset
subset = read_data()    

if __name__ == "__main__":
    run()
