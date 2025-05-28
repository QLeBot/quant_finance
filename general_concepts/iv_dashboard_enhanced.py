import numpy as np
import plotly.graph_objects as go
from scipy.stats import norm
from scipy.optimize import brentq
from scipy.interpolate import griddata
import yfinance as yf
from datetime import datetime
import streamlit as st
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform

st.set_page_config(
    page_title="Implied Volatility Surface",
    layout="wide"
)

st.title("Implied Volatility Surface")

def black_scholes(S, K, T, r, sigma, q=0, option_type='call'):
    """
    Calculate Black-Scholes option price
    S: current stock price
    K: strike price
    T: time to maturity (in years)
    r: risk-free rate
    sigma: volatility
    option_type: 'call' or 'put'
    """
    # Input validation
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return np.nan
        
    try:
        d1 = (np.log(S/K) + (r + ((sigma**2)/2)) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)
        
        return price
    except (ValueError, ZeroDivisionError):
        return np.nan
    
def implied_volatility(price, S, K, T, r, q=0,option_type='call'):
    """
    Calculate implied volatility using Brent's method
    """
    if T <= 0 or price <= 0:
        return np.nan
    
    def objective(sigma):
        return black_scholes(S, K, T, r, sigma, q, option_type) - price
    
    try:
        return brentq(objective,  1e-6, 5)
    except (ValueError, RuntimeError):
        return np.nan

def check_convexity_violations(strikes, times, ivs, spot_price):
    """
    Check for convexity violations in the implied volatility surface.
    Returns a DataFrame with potential arbitrage opportunities.
    """
    # Create a DataFrame with the data
    df = pd.DataFrame({
        'strike': strikes,
        'time': times,
        'iv': ivs,
        'moneyness': strikes / spot_price
    })
    
    # Sort by time and strike
    df = df.sort_values(['time', 'strike'])
    
    # Initialize list to store violations
    violations = []
    
    # For each time slice, check for convexity violations
    for t in df['time'].unique():
        time_slice = df[df['time'] == t].sort_values('strike')
        
        if len(time_slice) < 3:
            continue
            
        # Calculate second differences
        strikes = time_slice['strike'].values
        ivs = time_slice['iv'].values
        
        # Calculate second differences
        second_diff = np.diff(np.diff(ivs))
        
        # Find where second differences are negative (violating convexity)
        violation_indices = np.where(second_diff < 0)[0]
        
        for idx in violation_indices:
            violations.append({
                'time': t,
                'strike1': strikes[idx],
                'strike2': strikes[idx + 1],
                'strike3': strikes[idx + 2],
                'iv1': ivs[idx],
                'iv2': ivs[idx + 1],
                'iv3': ivs[idx + 2],
                'violation_magnitude': abs(second_diff[idx])
            })
    
    return pd.DataFrame(violations)

def calculate_surface_quality(strikes, times, ivs):
    """
    Calculate a quality metric for the volatility surface based on convexity.
    Returns a score between 0 and 1, where 1 is perfect convexity.
    """
    # Create points for convex hull
    points = np.column_stack((times, strikes, ivs))
    
    try:
        # Calculate convex hull
        hull = ConvexHull(points)
        
        # Calculate the volume of the convex hull
        hull_volume = hull.volume
        
        # Calculate the volume of the actual surface
        # We'll use a simple approximation based on the points
        actual_volume = np.sum(ivs) * (np.max(times) - np.min(times)) * (np.max(strikes) - np.min(strikes)) / len(ivs)
        
        # Calculate quality score
        quality_score = min(actual_volume / hull_volume, 1.0)
        
        return quality_score
    except:
        return 0.0

# --- SIDEBAR LAYOUT ---
st.sidebar.title("Model Parameters")
st.sidebar.write("Adjust the parameters for the Black-Scholes model.")
r = st.sidebar.number_input("Risk-Free Rate (e.g., 0.015 for 1.5%)", value=0.015, format="%.4f")
q = st.sidebar.number_input("Dividend Yield (e.g., 0.013 for 1.3%)", value=0.013, format="%.4f")

st.sidebar.subheader("Visualization Parameters")
y_axis_label = st.sidebar.selectbox("Select Y-axis:", ["Strike Price", "Moneyness"])

st.sidebar.subheader("Ticker Symbol")
ticker_symbol = st.sidebar.text_input("Enter Ticker Symbol", value="SPY", max_chars=10).upper()
option_type = st.sidebar.selectbox("Select Option Type", ["Call", "Put"])

st.sidebar.subheader("Strike Price Filter Parameters")
min_strike_pct = st.sidebar.number_input("Minimum Strike Price (% of Spot Price)", min_value=50.0, max_value=199.0, value=80.0, step=1.0, format="%.1f")
max_strike_pct = st.sidebar.number_input("Maximum Strike Price (% of Spot Price)", min_value=51.0, max_value=200.0, value=120.0, step=1.0, format="%.1f")

if min_strike_pct > max_strike_pct:
    st.sidebar.error("Minimum strike price must be less than maximum strike price")
    st.stop()

ticker = yf.Ticker(ticker_symbol)
date_now = pd.Timestamp('today').normalize()

try:
    expirations = ticker.options
except Exception as e:
    st.error(f"Error fetching options for {ticker_symbol}: {str(e)}")
    st.stop()

expirations = [pd.Timestamp(exp) for exp in expirations if pd.Timestamp(exp) > date_now + pd.Timedelta(days=7)]
    
if not expirations:
    st.error(f"No expiration dates found for {ticker_symbol}")
else:
    option_data = []

    for expiration in expirations:
        try:
            options = ticker.option_chain(expiration.strftime('%Y-%m-%d'))
            calls = options.calls
            puts = options.puts 
        except Exception as e:
            st.warning(f"Error fetching options for {ticker_symbol} on {expiration.strftime('%Y-%m-%d')}: {str(e)}")
            continue
        
        calls = calls[(calls['bid'] > 0) & (calls['ask'] > 0)]
        puts = puts[(puts['bid'] > 0) & (puts['ask'] > 0)]
        
        if option_type == "Call":
            options = calls
        else:
            options = puts
        
        for index, row in options.iterrows():
            strike = row['strike']
            bid = row['bid']
            ask = row['ask']
            mid_price = (bid + ask) / 2

            option_data.append({
                'expiration': expiration,
                'strike': strike,
                'bid': bid,
                'ask': ask,
                'mid': mid_price
            })
    
    if not option_data:
        st.error(f"No valid options found for {ticker_symbol}")
    else:
        options_df = pd.DataFrame(option_data)
        
        try:
            spot_history = ticker.history(period='5d')
            if spot_history.empty:
                st.error(f"Could not fetch recent price data for {ticker_symbol}")
                st.stop()
            else:
                spot_price = spot_history['Close'].iloc[-1]
        except Exception as e:
            st.error(f"Error fetching price data for {ticker_symbol}: {str(e)}")
            st.stop()

        options_df['daysToExpiration'] = (options_df['expiration'] - date_now).dt.days
        options_df['timeToExpiration'] = options_df['daysToExpiration'] / 365
        
        options_df = options_df[
            (options_df['strike'] >= spot_price * (min_strike_pct / 100)) &
            (options_df['strike'] <= spot_price * (max_strike_pct / 100))
        ]

        options_df.reset_index(drop=True, inplace=True)
        
        with st.spinner("Calculating Implied Volatility Surface..."):
            options_df['implied_volatility'] = options_df.apply(
                lambda row: implied_volatility(
                    price = row['mid'], 
                    S = spot_price, 
                    K = row['strike'], 
                    T = row['timeToExpiration'], 
                    r = r, 
                    q = q, 
                    option_type = option_type
                ), axis=1
            )

        options_df.dropna(subset=['implied_volatility'], inplace=True)
        
        if len(options_df) == 0:
            st.error("No valid options data available after processing. Try adjusting the strike price filter parameters.")
            st.stop()

        options_df['implied_volatility'] = options_df['implied_volatility'] * 100

        options_df.sort_values('strike', inplace=True)

        options_df['moneyness'] = options_df['strike'] / spot_price

        if y_axis_label == "Strike Price":
            Y = options_df['strike'].values
            y_label = "Strike Price ($)"
        else:
            Y = options_df['moneyness'].values
            y_label = "Moneyness (Strike/Spot)"

        X = options_df['timeToExpiration'].values
        Z = options_df['implied_volatility'].values

        ti = np.linspace(X.min(), X.max(), 50)
        ki = np.linspace(Y.min(), Y.max(), 50)
        T,K = np.meshgrid(ti, ki)

        Z_mesh = griddata(
            (X, Y), Z, (T, K), method='linear'
        )

        Z_mesh = np.ma.array(Z_mesh, mask=np.isnan(Z_mesh))

        fig = go.Figure(data=[go.Surface(
            z=Z_mesh, 
            x=T, 
            y=K, 
            colorscale='Viridis',
            colorbar_title='Implied Volatility (%)'
        )])
        fig.update_layout(
            title=f'Implied Volatility Surface for {ticker_symbol} Options', 
            scene=dict(
                xaxis_title='Time to Expiration (years)', 
                yaxis_title=y_label, 
                zaxis_title='Implied Volatility (%)'
            ),
            autosize=False,
            width=1200,
            height=800,
            margin=dict(l=50, r=50, b=65, t=90)
        )
        st.plotly_chart(fig)

        # After calculating implied volatilities and before creating the surface plot
        with st.spinner("Analyzing surface convexity..."):
            # Check for convexity violations
            violations_df = check_convexity_violations(
                options_df['strike'].values,
                options_df['timeToExpiration'].values,
                options_df['implied_volatility'].values,
                spot_price
            )
            
            # Calculate surface quality
            quality_score = calculate_surface_quality(
                options_df['strike'].values,
                options_df['timeToExpiration'].values,
                options_df['implied_volatility'].values
            )
            
            # Display surface quality
            st.metric("Surface Quality Score", f"{quality_score:.2%}")
            
            # Display violations if any
            if not violations_df.empty:
                st.warning(f"Found {len(violations_df)} potential arbitrage opportunities!")
                st.dataframe(violations_df)
                
                # Create a scatter plot of violations
                fig_violations = go.Figure()
                
                # Add the surface
                fig_violations.add_trace(go.Surface(
                    z=Z_mesh,
                    x=T,
                    y=K,
                    colorscale='Viridis',
                    opacity=0.7,
                    showscale=False
                ))
                
                # Add violation points
                fig_violations.add_trace(go.Scatter3d(
                    x=violations_df['time'],
                    y=violations_df['strike2'],
                    z=violations_df['iv2'],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color='red',
                        symbol='x'
                    ),
                    name='Convexity Violations'
                ))
                
                fig_violations.update_layout(
                    title='Implied Volatility Surface with Convexity Violations',
                    scene=dict(
                        xaxis_title='Time to Expiration (years)',
                        yaxis_title=y_label,
                        zaxis_title='Implied Volatility (%)'
                    ),
                    autosize=False,
                    width=1200,
                    height=800,
                    margin=dict(l=50, r=50, b=65, t=90)
                )
                
                st.plotly_chart(fig_violations)






