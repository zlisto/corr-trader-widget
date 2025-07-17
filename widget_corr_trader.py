# widget_app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from corr_trade_simulator import CorrTraderSimulator

# UI
st.title("Correlation Trading Simulator")

# Date selection
start_date = st.date_input("Start Date", value=pd.to_datetime('2025-04-10').date(), min_value=pd.to_datetime('2025-04-10').date())
stop_date = '2025-07-10'

# Set up simulator
sim = CorrTraderSimulator(str(start_date), stop_date)
pairs = sim.df_returns.columns

# Pair selection in a row
col1, col2 = st.columns(2)
with col1:
    default_pair = "AEVOUSD"
    if default_pair in pairs.unique():
        default_index = list(pairs.unique()).index(default_pair)
    else:
        default_index = 0  # fallback
    pair1 = st.selectbox("Signal Pair", pairs.unique(), index=default_index)

with col2:
    default_traded_pair = "PEPEUSD"
    if default_traded_pair in pairs.unique():
        default_traded_index = list(pairs.unique()).index(default_traded_pair)
    else:
        default_traded_index = 0  # fallback
    pair2 = st.selectbox("Traded Pair", pairs.unique(), index=default_traded_index)

# Trading parameters in a row
col3, col4, col5 = st.columns(3)
with col3:
    tstat_enter = st.number_input("t-stat Enter", value=0.0)
with col4:
    window_tstat = st.number_input("Window t-stat", value=12)
with col5:
    trade_fee = st.number_input("Trade Fee", value=0.0013, format="%.4f")
print(f"Running correlation trading simulation from {start_date} to {stop_date}")

if st.button("Simulate & Plot"):
    sim.simulate_trading(pair1, pair2, int(window_tstat), float(tstat_enter), float(trade_fee))

    # Interactive Plot with Plotly
    fig = go.Figure()
    
    # Add traces for each line
    fig.add_trace(go.Scatter(
        x=sim.df.index,
        y=sim.df.cum_pair2_signal,
        mode='lines',
        name=f'{pair2} Signal',
        line=dict(color='orange'),
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      'Date: %{x}<br>' +
                      'Value: %{y:.4f}<br>' +
                      't-stat: %{customdata:.4f}<br>' +
                      '<extra></extra>',
        customdata=sim.df.tstat_pair1
    ))
    
    fig.add_trace(go.Scatter(
        x=sim.df.index,
        y=sim.df.cum_pair2,
        mode='lines',
        name=f'{pair2} HODL',
        line=dict(color='blue'),
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      'Date: %{x}<br>' +
                      'Value: %{y:.4f}<br>' +
                      '<extra></extra>'
    ))
    
    fig.add_trace(go.Scatter(
        x=sim.df.index,
        y=sim.df.cum_pair1,
        mode='lines',
        name=pair1,
        line=dict(color='white'),
        visible="legendonly",
        hovertemplate='<b>%{fullData.name}</b><br>' +
                      'Date: %{x}<br>' +
                      'Value: %{y:.4f}<br>' +
                      '<extra></extra>'
    ))
    
    # Add scatter points for signal points
    signal_mask = sim.df.signal == 1
    fig.add_trace(go.Scatter(
        x=sim.df.index[signal_mask],
        y=sim.df.cum_pair2_signal[signal_mask],
        mode='markers',
        name='Signal Points',
        marker=dict(color='red', size=6),
        hovertemplate='<b>Signal Point</b><br>' +
                      'Date: %{x}<br>' +
                      'Value: %{y:.4f}<br>' +
                      't-stat: %{customdata:.4f}<br>' +
                      '<extra></extra>',
        customdata=sim.df.tstat_pair1[signal_mask]
    ))
    signal_mask = sim.df.signal == 0
    fig.add_trace(go.Scatter(
        x=sim.df.index[signal_mask],
        y=sim.df.cum_pair2_signal[signal_mask],
        mode='markers',
        name='Non Signal Points',
        marker=dict(color='cyan', size=6),
        hovertemplate='<b>Signal Point</b><br>' +
                      'Date: %{x}<br>' +
                      'Value: %{y:.4f}<br>' +
                      't-stat: %{customdata:.4f}<br>' +
                      '<extra></extra>',
        customdata=sim.df.tstat_pair1[signal_mask]
    ))
    
    # Update layout
    fig.update_layout(
        title=f"Trading Simulation: Signal = {pair1}, Trade = {pair2}",
        xaxis_title="Date",
        yaxis_title="Cumulative Returns",
        hovermode='closest',
        showlegend=True,
        width=1000,
        height=500
    )
    
    # Update axes
    fig.update_xaxes(
        gridcolor='lightgray',
        gridwidth=0.5
    )
    fig.update_yaxes(
        gridcolor='lightgray',
        gridwidth=0.5
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Trading Summary below the plot
    df_trades_pair = sim.df_trades
    final_drawdown = 1 - sim.df.cum_pair2_signal.iloc[-1] / sim.df.cum_pair2_signal.max()
    sharpe_signal = sim.sharpe_ratio(sim.df.returns_pair2_signal)
    sharpe_hodl = sim.sharpe_ratio(sim.df.returns_pair2)
    annualized_return_signal = sim.df.returns_pair2_signal.mean() * 365*24
    annualized_return_hodl = sim.df.returns_pair2.mean() * 365*24

    st.write(f"### Trading Summary for {pair2} (signal: {pair1})")
    st.write(f"{len(df_trades_pair)} buy/sell trades executed")
    st.write(f"Total return: {df_trades_pair['returns'].sum() * 100:.2f}%")
    st.write(f"Average trade return: {df_trades_pair['returns'].mean() * 100:.2f}%")
    st.write(f"Average trade duration: {df_trades_pair['age_hours'].mean():.2f} hours")
    st.write(f"Sharpe ratio (signal/HODL): {sharpe_signal:.2f}/{sharpe_hodl:.2f}")
    st.write(f"Annualized return (signal/HODL): {annualized_return_signal*100:.2f}%/{annualized_return_hodl*100:.2f}%")
    st.write(f"Final drawdown: {final_drawdown * 100:.2f}%")
