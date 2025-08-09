import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
import time
import gc
import os
import sys
from datetime import datetime, timedelta
import torch
from typing import Optional, Dict, Any, Tuple

warnings.filterwarnings('ignore')

# Streamlit Cloud optimizations
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Memory and CPU optimizations for cloud deployment
if torch.cuda.is_available():
    torch.cuda.empty_cache()
torch.set_num_threads(min(4, os.cpu_count() or 1))

# Add error handling for import issues on cloud
def safe_import(module_name, error_msg):
    try:
        return __import__(module_name)
    except ImportError as e:
        st.error(f"❌ {error_msg}: {str(e)}")
        st.info("This may be due to package installation issues. Please check requirements.txt")
        return None

st.set_page_config(
    page_title="⚡ Fast AI Stock Predictor - Chronos & Moirai",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/ai-stock-predictor',
        'Report a bug': 'https://github.com/yourusername/ai-stock-predictor/issues',
        'About': "AI Stock Predictor using Chronos & Moirai models"
    }
)

# Global model cache with cloud-specific settings
@st.cache_resource(show_spinner=False, max_entries=2, ttl=3600)  # 1 hour TTL for cloud
def get_model_cache():
    return {}

MODEL_CACHE = get_model_cache()

# Initialize session state
for key, default in [
    ('current_page', 'home'),
    ('analysis_data', None),
    ('last_model_type', None),
    ('model_ready', False)
]:
    if key not in st.session_state:
        st.session_state[key] = default

def display_disclaimers():
    """Compact disclaimer section"""
    with st.expander("⚠️ IMPORTANT DISCLAIMERS - Click to Read", expanded=False):
        st.markdown("""
        **🚨 NOT FINANCIAL ADVICE** | **⚡ Educational Use Only** | **🎯 AI Predictions May Be Wrong**
        
        - Stock markets are unpredictable. AI models can fail.
        - Past performance ≠ future results. You can lose money.
        - Always consult financial professionals before investing.
        - Only invest what you can afford to lose.
        """)

class FastAIStockAnalyzer:
    """Optimized AI Stock Analyzer with faster model loading"""
    
    def __init__(self):
        self.context_length = 32  # Reduced for faster processing
        self.prediction_length = 7
        self.device = "cpu"  # Force CPU for stability
        
    @st.cache_data(ttl=900, show_spinner=False)  # Reduced TTL for fresh data
    def fetch_stock_data(_self, symbol: str, period: str = "6mo") -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """Optimized data fetching with shorter period for speed"""
        try:
            ticker = yf.Ticker(symbol)
            # Use shorter period and specific fields for faster download
            data = ticker.history(period=period, interval="1d", 
                                actions=False, auto_adjust=True, 
                                back_adjust=False, repair=False)
            
            if data.empty:
                return None, None
                
            # Get minimal info for speed
            try:
                info = {
                    'longName': ticker.info.get('longName', symbol),
                    'sector': ticker.info.get('sector', 'Unknown'),
                    'marketCap': ticker.info.get('marketCap', 0)
                }
            except:
                info = {'longName': symbol, 'sector': 'Unknown', 'marketCap': 0}
            
            return data, info
            
        except Exception as e:
            st.error(f"Data fetch error: {str(e)[:100]}...")
            return None, None
    
    def load_chronos_tiny(self) -> Tuple[Optional[Any], str]:
        """Load tiny Chronos model for fastest performance - Cloud optimized"""
        model_key = "chronos_tiny"
        
        # Check if model is already loaded
        if model_key in MODEL_CACHE:
            return MODEL_CACHE[model_key], "chronos"
        
        try:
            st.info("🔄 Loading Amazon Chronos Tiny (Cloud Optimized)...")
            
            # Cloud-safe import
            try:
                from chronos import ChronosPipeline
            except ImportError as e:
                st.error("📦 Chronos not available. Please ensure chronos-forecasting is in requirements.txt")
                st.code("pip install chronos-forecasting")
                return None, None
            
            # Load with cloud-specific optimizations
            try:
                pipeline = ChronosPipeline.from_pretrained(
                    "amazon/chronos-t5-tiny",
                    device_map="auto" if torch.cuda.is_available() else "cpu",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                    trust_remote_code=True  # Required for some models
                )
                
                # Cache the model
                MODEL_CACHE[model_key] = pipeline
                
                st.success("✅ Chronos Tiny loaded successfully!")
                return pipeline, "chronos"
                
            except Exception as load_error:
                st.error(f"Model loading failed: {str(load_error)}")
                # Try fallback without safetensors
                try:
                    pipeline = ChronosPipeline.from_pretrained(
                        "amazon/chronos-t5-tiny",
                        device_map="cpu",
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True,
                        use_safetensors=False
                    )
                    MODEL_CACHE[model_key] = pipeline
                    st.success("✅ Chronos loaded with fallback settings!")
                    return pipeline, "chronos"
                except Exception as e:
                    st.error(f"Fallback loading also failed: {str(e)}")
                    return None, None
                
        except Exception as e:
            st.error(f"❌ Chronos loading failed: {str(e)[:200]}...")
            return None, None
    
    def load_moirai_small(self) -> Tuple[Optional[Any], str]:
        """Load Moirai small model with cloud optimizations"""
        model_key = "moirai_small"
        
        # Check cache first
        if model_key in MODEL_CACHE:
            return MODEL_CACHE[model_key], "moirai"
        
        try:
            st.info("🔄 Loading Salesforce Moirai Small (Cloud Optimized)...")
            
            # Cloud-safe import
            try:
                from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
            except ImportError as e:
                st.error("📦 Moirai not available. Please ensure uni2ts is in requirements.txt")
                st.code("pip install uni2ts")
                return None, None
            
            # Load with memory optimizations for cloud
            try:
                module = MoiraiModule.from_pretrained(
                    "Salesforce/moirai-1.0-R-small",
                    low_cpu_mem_usage=True,
                    device_map="auto" if torch.cuda.is_available() else "cpu",
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True
                )
                
                model = MoiraiForecast(
                    module=module,
                    prediction_length=self.prediction_length,
                    context_length=self.context_length,
                    patch_size="auto",
                    num_samples=15,  # Reduced for cloud performance
                    target_dim=1,
                    feat_dynamic_real_dim=0,
                    past_feat_dynamic_real_dim=0
                )
                
                # Cache the model
                MODEL_CACHE[model_key] = model
                
                st.success("✅ Moirai Small loaded successfully!")
                return model, "moirai"
                
            except Exception as load_error:
                st.error(f"Moirai loading failed: {str(load_error)}")
                return None, None
            
        except Exception as e:
            st.error(f"❌ Moirai loading failed: {str(e)[:200]}...")
            return None, None
    
    def predict_chronos_fast(self, pipeline: Any, data: np.ndarray) -> Optional[Dict]:
        """Fast Chronos prediction with reduced samples"""
        try:
            # Use only recent data for speed
            context_data = data[-self.context_length:]
            context = torch.tensor(context_data, dtype=torch.float32).unsqueeze(0)
            
            # Reduced samples for faster prediction
            with torch.no_grad():
                forecast = pipeline.predict(
                    context=context,
                    prediction_length=self.prediction_length,
                    num_samples=10,  # Reduced from 30
                    temperature=1.0,
                    top_k=50,
                    top_p=1.0
                )
            
            # Quick statistics calculation
            forecast_array = forecast[0].numpy()
            predictions = {
                'mean': np.median(forecast_array, axis=0),  # Use median for robustness
                'q10': np.quantile(forecast_array, 0.1, axis=0),
                'q90': np.quantile(forecast_array, 0.9, axis=0),
                'std': np.std(forecast_array, axis=0)
            }
            
            return predictions
            
        except Exception as e:
            st.error(f"Chronos prediction error: {str(e)[:100]}...")
            return None
    
    def predict_moirai_fast(self, model: Any, data: np.ndarray) -> Optional[Dict]:
        """Fast Moirai prediction"""
        try:
            from gluonts.dataset.common import ListDataset
            from gluonts.dataset.util import to_pandas
            
            # Prepare minimal dataset
            dataset = ListDataset([{
                "item_id": "stock",
                "start": "2023-01-01",  # Fixed start date for speed
                "target": data[-self.context_length:].tolist()
            }], freq='D')
            
            # Create predictor with optimizations
            predictor = model.create_predictor(
                batch_size=1,
                num_parallel_samples=10  # Reduced for speed
            )
            
            # Generate forecast
            forecasts = list(predictor.predict(dataset))
            forecast = forecasts[0]
            
            predictions = {
                'mean': forecast.mean,
                'q10': forecast.quantile(0.1),
                'q90': forecast.quantile(0.9),
                'std': np.std(forecast.samples, axis=0)
            }
            
            return predictions
            
        except Exception as e:
            st.error(f"Moirai prediction error: {str(e)[:100]}...")
            return None
    
    def calculate_investment_scenarios(self, current_price: float, predictions: Dict, amounts: list) -> Dict:
        """Fast calculation of investment scenarios"""
        results = {}
        mean_pred = predictions['mean']
        
        for amount in amounts:
            shares = amount / current_price
            scenarios = {}
            
            for day, pred_price in enumerate(mean_pred, 1):
                profit_loss = (pred_price - current_price) * shares
                profit_loss_pct = ((pred_price - current_price) / current_price) * 100
                
                scenarios[f'Day_{day}'] = {
                    'predicted_price': pred_price,
                    'profit_loss': profit_loss,
                    'profit_loss_pct': profit_loss_pct,
                    'portfolio_value': amount + profit_loss,
                    'is_profit': profit_loss > 0
                }
            
            results[f'${amount:,}'] = scenarios
        
        return results

def show_home_page():
    """Streamlined home page"""
    st.title("⚡ Fast AI Stock Predictor")
    st.markdown("**🤖 Powered by Amazon Chronos & Salesforce Moirai**")
    
    display_disclaimers()
    
    analyzer = FastAIStockAnalyzer()
    
    # Sidebar configuration
    st.sidebar.header("🎯 Quick Setup")
    
    # Popular stocks with simpler display
    stocks = {
        'AAPL': '🍎 Apple', 'GOOGL': '🔍 Google', 'MSFT': '💻 Microsoft',
        'TSLA': '🚗 Tesla', 'AMZN': '📦 Amazon', 'META': '📱 Meta',
        'NFLX': '🎬 Netflix', 'NVDA': '🔥 NVIDIA'
    }
    
    selected_stock = st.sidebar.selectbox(
        "Select Stock", 
        list(stocks.keys()),
        format_func=lambda x: stocks[x]
    )
    
    # Custom stock option
    use_custom = st.sidebar.checkbox("Use Custom Symbol")
    if use_custom:
        custom_stock = st.sidebar.text_input("Symbol", "AAPL").upper()
        selected_stock = custom_stock
    
    # Simplified AI model selection
    model_choice = st.sidebar.radio(
        "AI Model",
        ["🚀 Chronos (Fast)", "🎯 Moirai (Accurate)"],
        help="Chronos: Faster loading, good predictions\nMoirai: Slower loading, higher accuracy"
    )
    
    model_type = "chronos" if "Chronos" in model_choice else "moirai"
    
    # Investment amounts
    st.sidebar.markdown("💰 **Investment Amounts:**")
    amounts = []
    for amount, default in [(1000, True), (5000, False), (10000, False)]:
        if st.sidebar.checkbox(f"${amount:,}", default):
            amounts.append(amount)
    
    # Custom amount
    if st.sidebar.checkbox("Custom Amount"):
        custom_amt = st.sidebar.number_input("Amount ($)", 500, 50000, 2500, 500)
        amounts.append(custom_amt)
    
    if not amounts:
        amounts = [1000]
    
    # Analyze button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Analyze Stock Now", type="primary", use_container_width=True):
            st.session_state.analysis_data = {
                'stock_symbol': selected_stock,
                'model_type': model_type,
                'investment_amounts': amounts,
                'analyzer': analyzer
            }
            st.session_state.current_page = 'results'
            st.rerun()

def show_results_page():
    """Optimized results page"""
    # Navigation
    col1, col2 = st.columns([1, 4])
    with col1:
        if st.button("← Home", type="secondary"):
            st.session_state.current_page = 'home'
            st.rerun()
    
    # Get analysis data
    data = st.session_state.analysis_data
    if not data:
        st.error("No analysis data found.")
        return
    
    symbol = data['stock_symbol']
    model_type = data['model_type']
    amounts = data['investment_amounts']
    analyzer = data['analyzer']
    
    st.title(f"🎯 {symbol} Analysis Results")
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Step 1: Fetch data
    status_text.text("📊 Fetching stock data...")
    progress_bar.progress(20)
    
    stock_data, stock_info = analyzer.fetch_stock_data(symbol)
    
    if stock_data is None or len(stock_data) < 50:
        st.error("❌ Insufficient data for analysis")
        return
    
    current_price = stock_data['Close'].iloc[-1]
    company_name = stock_info.get('longName', symbol) if stock_info else symbol
    
    progress_bar.progress(40)
    
    # Display basic info
    st.markdown(f"## 🏢 {company_name} ({symbol})")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    with col2:
        day_change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]
        day_change_pct = (day_change / stock_data['Close'].iloc[-2]) * 100
        st.metric("1-Day Change", f"${day_change:.2f}", f"{day_change_pct:+.2f}%")
    with col3:
        week_change = ((current_price - stock_data['Close'].iloc[-7]) / stock_data['Close'].iloc[-7]) * 100 if len(stock_data) > 7 else 0
        st.metric("1-Week Change", f"{week_change:+.2f}%")
    
    # Step 2: Load AI model
    status_text.text(f"🤖 Loading {model_type.title()} AI model...")
    progress_bar.progress(60)
    
    model = None
    model_name = ""
    
    if model_type == "chronos":
        model, loaded_type = analyzer.load_chronos_tiny()
        model_name = "Amazon Chronos Tiny"
    else:
        model, loaded_type = analyzer.load_moirai_small()
        model_name = "Salesforce Moirai Small"
    
    if model is None:
        st.error("❌ Failed to load AI model")
        return
    
    # Step 3: Generate predictions
    status_text.text("🔮 Generating AI predictions...")
    progress_bar.progress(80)
    
    if model_type == "chronos":
        predictions = analyzer.predict_chronos_fast(model, stock_data['Close'].values)
    else:
        predictions = analyzer.predict_moirai_fast(model, stock_data['Close'].values)
    
    if predictions is None:
        st.error("❌ Prediction failed")
        return
    
    progress_bar.progress(100)
    status_text.text("✅ Analysis complete!")
    
    # Clear progress indicators
    time.sleep(1)
    progress_bar.empty()
    status_text.empty()
    
    # Results
    st.markdown("---")
    st.markdown("## 🎯 **AI PREDICTIONS**")
    
    mean_pred = predictions['mean']
    final_pred = mean_pred[-1]
    week_change = ((final_pred - current_price) / current_price) * 100
    
    # Decision logic
    if week_change > 5:
        decision = "🟢 STRONG BUY"
        color = "green"
        explanation = "AI expects significant gains!"
    elif week_change > 2:
        decision = "🟢 BUY"
        color = "green"  
        explanation = "AI expects moderate gains"
    elif week_change < -5:
        decision = "🔴 STRONG SELL"
        color = "red"
        explanation = "AI expects significant losses"
    elif week_change < -2:
        decision = "🔴 SELL"
        color = "red"
        explanation = "AI expects losses"
    else:
        decision = "⚪ HOLD"
        color = "gray"
        explanation = "AI expects stable prices"
    
    # Display recommendation
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {color}22, {color}11); 
                border: 2px solid {color}; padding: 20px; border-radius: 15px; text-align: center;">
        <h2 style="color: {color}; margin: 0;">🤖 {decision}</h2>
        <p style="font-size: 18px;"><strong>{explanation}</strong></p>
        <p style="color: #666;">Powered by {model_name}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("7-Day Prediction", f"${final_pred:.2f}", f"{week_change:+.2f}%")
    with col2:
        confidence = min(100, max(50, 70 + abs(week_change) * 1.5))
        st.metric("AI Confidence", f"{confidence:.0f}%")
    with col3:
        volatility = np.std(predictions.get('std', [0])) * 100
        st.metric("Volatility", f"{volatility:.1f}%")
    
    # Investment scenarios
    st.markdown("---")
    st.markdown("## 💰 **Investment Scenarios**")
    
    scenarios = analyzer.calculate_investment_scenarios(current_price, predictions, amounts)
    
    tabs = st.tabs([f"${amt:,}" for amt in amounts])
    for tab, amount in zip(tabs, amounts):
        with tab:
            amount_key = f"${amount:,}"
            day_7 = scenarios[amount_key]['Day_7']
            profit = day_7['profit_loss']
            profit_pct = day_7['profit_loss_pct']
            
            color = "green" if profit > 0 else "red"
            
            st.markdown(f"""
            <div style="background-color: {color}15; border: 2px solid {color}; 
                        padding: 15px; border-radius: 10px;">
                <h4 style="color: {color};">Investment: ${amount:,}</h4>
                <p><strong>Shares:</strong> {amount/current_price:.2f}</p>
                <p><strong>7-Day Value:</strong> ${day_7['portfolio_value']:,.2f}</p>
                <p><strong>Profit/Loss:</strong> <span style="color: {color}; font-weight: bold;">
                   ${profit:+,.2f} ({profit_pct:+.2f}%)</span></p>
            </div>
            """, unsafe_allow_html=True)
    
    # Chart
    st.markdown("---")
    st.markdown("## 📈 **Price Chart & Predictions**")
    
    fig = go.Figure()
    
    # Historical data (last 30 days)
    recent = stock_data.tail(30)
    fig.add_trace(go.Scatter(
        x=recent.index, y=recent['Close'],
        mode='lines', name='Historical',
        line=dict(color='blue', width=2)
    ))
    
    # Predictions
    future_dates = pd.date_range(
        start=stock_data.index[-1] + pd.Timedelta(days=1),
        periods=7, freq='D'
    )
    
    fig.add_trace(go.Scatter(
        x=future_dates, y=mean_pred,
        mode='lines+markers', name='AI Prediction',
        line=dict(color='red', width=3),
        marker=dict(size=8)
    ))
    
    # Confidence bands
    if 'q10' in predictions and 'q90' in predictions:
        fig.add_trace(go.Scatter(
            x=future_dates.tolist() + future_dates[::-1].tolist(),
            y=predictions['q90'].tolist() + predictions['q10'][::-1].tolist(),
            fill='toself', fillcolor='rgba(255,0,0,0.1)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Range', showlegend=True
        ))
    
    fig.update_layout(
        title=f"{symbol} - AI Forecast",
        xaxis_title="Date", yaxis_title="Price ($)",
        height=400, showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # New analysis button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 New Analysis", type="primary", use_container_width=True):
            # Clear cache periodically to free memory
            if len(MODEL_CACHE) > 2:
                MODEL_CACHE.clear()
                gc.collect()  # Force garbage collection
            
            st.session_state.current_page = 'home'
            st.session_state.analysis_data = None
            st.rerun()

def main():
    """Main application router with cloud deployment optimizations"""
    
    # Add deployment info
    if st.sidebar.button("ℹ️ About This App"):
        st.sidebar.info("""
        **AI Stock Predictor**
        
        🤖 Models: Amazon Chronos & Salesforce Moirai
        ☁️ Deployed on Streamlit Cloud
        ⚠️ Educational use only - not financial advice
        
        Made with ❤️ using Streamlit
        """)
    
    # Memory optimization for cloud
    try:
        # Clear excessive cache periodically
        if len(MODEL_CACHE) > 2:
            # Keep only the most recent model
            keys_to_remove = list(MODEL_CACHE.keys())[:-1]
            for key in keys_to_remove:
                if key in MODEL_CACHE:
                    del MODEL_CACHE[key]
            gc.collect()
    except Exception:
        pass  # Silent fail for cloud robustness
    
    # Route pages
    try:
        if st.session_state.current_page == 'home':
            show_home_page()
        elif st.session_state.current_page == 'results':
            show_results_page()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please refresh the page and try again.")
        if st.button("🔄 Reset App"):
            # Clear all session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

if __name__ == "__main__":
    main()
