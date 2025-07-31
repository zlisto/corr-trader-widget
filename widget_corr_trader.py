# widget_app.py
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from corr_trade_simulator import CorrTraderSimulator

# Set page config for wider layout
st.set_page_config(
    page_title="Correlation Trading Simulator",
    page_icon="💖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# BlackPink theme CSS
st.markdown("""
<style>
    /* BlackPink Theme */
    .main {
        background: linear-gradient(135deg, #000000 0%, #1a1a1a 50%, #000000 100%);
        color: #ffffff;
    }
    
    /* Bigger fonts */
    h1 {
        font-size: 3rem !important;
        color: #ff69b4 !important;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        margin-bottom: 2rem;
    }
    
    h3 {
        font-size: 1.8rem !important;
        color: #ff1493 !important;
        margin-top: 2rem;
    }
    
    /* Widget styling */
    .stSelectbox, .stNumberInput, .stDateInput {
        font-size: 1.2rem !important;
    }
    
    .stButton > button {
        background: linear-gradient(45deg, #ff69b4, #ff1493) !important;
        color: white !important;
        font-size: 1.3rem !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.8rem 2rem !important;
        box-shadow: 0 4px 15px rgba(255, 105, 180, 0.3) !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(45deg, #ff1493, #ff69b4) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(255, 105, 180, 0.4) !important;
    }
    
    /* Text styling */
    p, div {
        font-size: 1.1rem !important;
        color: #ffffff !important;
    }
    
    /* Plot container */
    .element-container {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 15px !important;
        padding: 1rem !important;
        margin: 1rem 0 !important;
    }
</style>
""", unsafe_allow_html=True)

# UI
st.title("💖 Correlation Trading Simulator 💖")

# Date selection
start_date = st.date_input("Start Date", value=pd.to_datetime('2025-04-10').date(), min_value=pd.to_datetime('2025-04-10').date())
stop_date = '2025-07-30'

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
        width=1400,
        height=600,
        plot_bgcolor='rgba(0,0,0,0.8)',
        paper_bgcolor='rgba(0,0,0,0.8)',
        font=dict(color='white', size=14),
        title_font=dict(size=20, color='#ff69b4')
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
    
    st.plotly_chart(fig, use_container_width=False)

    # Trading Summary below the plot
    df_trades_pair = sim.df_trades
    final_drawdown = 1 - sim.df.cum_pair2_signal.iloc[-1] / sim.df.cum_pair2_signal.max()
    sharpe_signal = sim.sharpe_ratio(sim.df.returns_pair2_signal)
    sharpe_hodl = sim.sharpe_ratio(sim.df.returns_pair2)
    annualized_return_signal = sim.df.returns_pair2_signal.mean() * 365*24
    annualized_return_hodl = sim.df.returns_pair2.mean() * 365*24
    total_return_signal = (sim.df.cum_pair2_signal.iloc[-1]) * 100
    total_return_hodl = (sim.df.cum_pair2.iloc[-1] ) * 100

    st.write(f"### 📊 Trading Summary: {pair2} (Signal: {pair1})")
    
    # Create comparison table
    summary_data = {
        'Metric': [
            'Total Return (%)',
            'Annualized Return (%)', 
            'Sharpe Ratio',
            'Final Drawdown (%)',
            'Trades Executed',
            'Avg Trade Return (%)',
            'Avg Trade Duration (hrs)'
        ],
        'Signal Strategy': [
            f"{total_return_signal:.2f}%",
            f"{annualized_return_signal*100:.2f}%",
            f"{sharpe_signal:.2f}",
            f"{final_drawdown * 100:.2f}%",
            f"{len(df_trades_pair)}",
            f"{df_trades_pair['returns'].mean() * 100:.2f}%",
            f"{df_trades_pair['age_hours'].mean():.2f}"
        ],
        'HODL Strategy': [
            f"{total_return_hodl:.2f}%",
            f"{annualized_return_hodl*100:.2f}%",
            f"{sharpe_hodl:.2f}",
            "N/A",
            "0",
            "N/A",
            "N/A"
        ]
    }
    
    df_summary = pd.DataFrame(summary_data)
    st.table(df_summary)
