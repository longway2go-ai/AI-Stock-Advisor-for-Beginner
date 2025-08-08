import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
import time
from datetime import datetime, timedelta
import torch

warnings.filterwarnings('ignore')

# --- Lazy Loading Models ---
@st.cache_resource
def load_chronos_model():
    from chronos import Autoformer
    return Autoformer.from_pretrained("amazon/chronos-t5-small")

@st.cache_resource
def load_moirai_model():
    from uni2ts import Uni2TSForecaster
    return Uni2TSForecaster.from_pretrained("salesforce/moirai-tsnet-large")

st.set_page_config(
    page_title="📈 Smart Stock Advisor - AI-Powered Investment Guide",
    page_icon="💰",
    layout="wide"
)

# Initialize session state for page navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None

def display_header_and_disclaimers():
    st.title("💰 Smart Stock Advisor - Your AI Investment Guide")
    st.markdown("### 🤖 Powered by Amazon Chronos & Salesforce Moirai AI Models - Pretrained Models")
    
    # Important disclaimers at the top
    st.markdown("""
    <div style="background-color: #ff6b6b; color: white; padding: 15px; border-radius: 10px; margin: 20px 0;">
        <h3>⚠️ IMPORTANT DISCLAIMER - READ BEFORE USING</h3>
        <ul>
            <li><strong>NOT FINANCIAL ADVICE:</strong> This tool is for educational purposes only. Do NOT make investment decisions solely based on AI predictions.</li>
            <li><strong>PAST PERFORMANCE ≠ FUTURE RESULTS:</strong> Stock markets are unpredictable. AI models can be wrong.</li>
            <li><strong>RISK WARNING:</strong> You can lose money investing in stocks. Only invest what you can afford to lose.</li>
            <li><strong>CONSULT PROFESSIONALS:</strong> Always speak with a certified financial advisor before making investment decisions.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
    <div style="background-color: #212529; color: white; padding: 15px; border-radius: 5px; margin: 20px 0;">
        <h4>⚖️ Legal Disclaimer</h4>
        <p><small>
        This tool provides educational information only and should not be considered as financial advice. 
        The predictions are based on AI models and historical data, which may not reflect future performance. 
        Stock investments involve risk of loss. Past performance does not guarantee future results. 
        Always consult with a qualified financial advisor before making investment decisions. 
        The creators of this tool are not responsible for any financial losses incurred from using this information.
        </small></p>
    </div>
    """, unsafe_allow_html=True)

class BeginnerFriendlyStockAnalyzer:
    def __init__(self):
        self.model = None
        self.model_type = None
        self.context_length = 60
        self.prediction_length = 7
        
    @st.cache_data(ttl=1800)
    def fetch_stock_data(_self, symbol, period="1y"):
        """Fast data fetching with caching"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            info = ticker.info
            return data, info
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None, None
    
    def load_chronos_model(self, model_size="small"):
        """Load Amazon Chronos model"""
        try:
            st.info(f"🔄 Loading Amazon Chronos ({model_size}) - The world's most reliable time series AI...")
            
            from chronos import ChronosPipeline
            
            model_names = {
                "tiny": "amazon/chronos-t5-tiny",
                "small": "amazon/chronos-t5-small",
                "base": "amazon/chronos-t5-base"
            }
            
            model_name = model_names.get(model_size, "amazon/chronos-t5-small")
            
            pipeline = ChronosPipeline.from_pretrained(
                model_name,
                device_map="cpu",
                torch_dtype=torch.float32,
            )
            
            st.success(f"✅ Amazon Chronos {model_size} loaded successfully!")
            return pipeline, "chronos"
            
        except ImportError:
            st.warning("Amazon Chronos not installed. Run: pip install chronos-forecasting")
            return None, None
        except Exception as e:
            st.warning(f"Chronos loading failed: {e}")
            return None, None
    
    def load_moirai_model(self, model_size="small"):
        """Load Salesforce Moirai Universal model"""
        try:
            st.info(f"🔄 Loading Salesforce Moirai ({model_size}) - Universal AI trained on 27 billion observations...")
            
            # Import Moirai components
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
            
            model_names = {
                "small": "Salesforce/moirai-1.0-R-small",
                "base": "Salesforce/moirai-1.0-R-base", 
                "large": "Salesforce/moirai-1.0-R-large"
            }
            
            model_name = model_names.get(model_size, "Salesforce/moirai-1.0-R-small")
            
            model = MoiraiForecast(
                module=MoiraiModule.from_pretrained(model_name),
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                patch_size="auto",
                num_samples=50,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0
            )
            
            st.success(f"✅ Salesforce Moirai {model_size} loaded successfully!")
            return model, "moirai"
            
        except ImportError:
            st.warning("Moirai not installed. Run: pip install uni2ts")
            return None, None
        except Exception as e:
            st.warning(f"Moirai loading failed: {e}")
            return None, None
    
    def predict_with_chronos(self, pipeline, data):
        """Generate predictions using Amazon Chronos"""
        try:
            context = torch.tensor(data[-self.context_length:], dtype=torch.float32)
            
            forecast = pipeline.predict(
                context=context,
                prediction_length=self.prediction_length,
                num_samples=30,
            )
            
            predictions = {
                'mean': forecast[0].median(dim=0).values.numpy(),
                'q10': forecast[0].quantile(0.1, dim=0).numpy(),
                'q25': forecast[0].quantile(0.25, dim=0).numpy(),
                'q75': forecast[0].quantile(0.75, dim=0).numpy(),
                'q90': forecast[0].quantile(0.9, dim=0).numpy()
            }
            
            return predictions
            
        except Exception as e:
            st.error(f"Chronos prediction failed: {e}")
            return None
    
    def predict_with_moirai(self, model, data):
        """Generate predictions using Salesforce Moirai"""
        try:
            # Convert to GluonTS format
            from gluonts.dataset.common import ListDataset
            
            dataset = ListDataset([{
                "item_id": "stock",
                "start": "2020-01-01",
                "target": data.tolist()
            }], freq='D')
            
            predictor = model.create_predictor(batch_size=1)
            forecasts = list(predictor.predict(dataset))
            forecast = forecasts[0]
            
            predictions = {
                'mean': forecast.mean,
                'q10': forecast.quantile(0.1),
                'q25': forecast.quantile(0.25),
                'q75': forecast.quantile(0.75),
                'q90': forecast.quantile(0.9)
            }
            
            return predictions
            
        except Exception as e:
            st.error(f"Moirai prediction failed: {e}")
            return None
    
    def calculate_profit_loss_scenarios(self, current_price, predictions, investment_amounts):
        """Calculate detailed profit/loss scenarios for beginners"""
        results = {}
        
        for amount in investment_amounts:
            shares = amount / current_price
            scenarios = {}
            
            # Daily predictions
            for day, pred_price in enumerate(predictions['mean'], 1):
                profit_loss = (pred_price - current_price) * shares
                profit_loss_pct = ((pred_price - current_price) / current_price) * 100
                new_value = amount + profit_loss
                
                scenarios[f'Day_{day}'] = {
                    'predicted_price': pred_price,
                    'profit_loss_amount': profit_loss,
                    'profit_loss_pct': profit_loss_pct,
                    'new_portfolio_value': new_value,
                    'is_profit': profit_loss > 0
                }
            
            results[f'${amount:,}'] = scenarios
        
        return results

def show_home_page():
    """Display the main configuration page"""
    # Display header and disclaimers first
    display_header_and_disclaimers()
    
    analyzer = BeginnerFriendlyStockAnalyzer()
    
    # Sidebar configuration
    st.sidebar.header("💰 Investment Advisor Settings")
    
    # Popular stocks for beginners
    popular_stocks = {
        'AAPL': '🍎 Apple (iPhone maker)',
        'GOOGL': '🔍 Google (Search engine)',
        'MSFT': '💻 Microsoft (Windows, Xbox)',
        'TSLA': '🚗 Tesla (Electric cars)',
        'AMZN': '📦 Amazon (Online shopping)',
        'META': '📱 Meta (Facebook, Instagram)', 
        'NFLX': '🎬 Netflix (Streaming)',
        'NVDA': '🔥 NVIDIA (AI chips)'
    }
    
    selected_stock_display = st.sidebar.selectbox(
        "🏢 Select a Company to Analyze", 
        list(popular_stocks.keys()),
        format_func=lambda x: popular_stocks[x]
    )
    stock_symbol = selected_stock_display
    
    # Custom stock option
    if st.sidebar.checkbox("📝 Enter Custom Stock Symbol"):
        custom_stock = st.sidebar.text_input("Stock Symbol (e.g., AAPL)", "AAPL")
        stock_symbol = custom_stock.upper()
    
    # AI Model selection with explanations
    model_options = {
        "🚀 Amazon Chronos (Recommended)": ("chronos", "small"),
        "🎯 Salesforce Moirai Universal": ("moirai", "small")
    }
    
    selected_model = st.sidebar.selectbox("🤖 Choose AI Model", list(model_options.keys()))
    model_type, model_size = model_options[selected_model]
    
    # Investment amount scenarios
    st.sidebar.markdown("💵 **Investment Scenarios to Analyze:**")
    investment_amounts = []
    
    if st.sidebar.checkbox("💰 $1,000 (Small investment)", True):
        investment_amounts.append(1000)
    if st.sidebar.checkbox("💰 $5,000 (Medium investment)"):
        investment_amounts.append(5000)
    if st.sidebar.checkbox("💰 $10,000 (Large investment)"):
        investment_amounts.append(10000)
    if st.sidebar.checkbox("💰 Custom amount"):
        custom_amount = st.sidebar.number_input("Enter amount ($)", min_value=100, max_value=100000, value=2500, step=500)
        investment_amounts.append(custom_amount)
    
    if not investment_amounts:
        investment_amounts = [1000]  # Default
    
    # Center the analyze button and make it more prominent
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔮 Analyze Stock & Predict Profits", type="primary", use_container_width=True):
            # Store all analysis parameters in session state
            st.session_state.analysis_data = {
                'stock_symbol': stock_symbol,
                'model_type': model_type,
                'model_size': model_size,
                'investment_amounts': investment_amounts,
                'analyzer': analyzer
            }
            # Navigate to results page
            st.session_state.current_page = 'results'
            st.rerun()
    
    # Footer with educational content on home page
    st.markdown("---")

def show_results_page():
    """Display the analysis results page"""
    # Back button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Back to Home", type="secondary"):
            st.session_state.current_page = 'home'
            st.rerun()
    
    # Get analysis parameters from session state
    analysis_data = st.session_state.analysis_data
    if analysis_data is None:
        st.error("No analysis data found. Please go back and run analysis.")
        return
    
    stock_symbol = analysis_data['stock_symbol']
    model_type = analysis_data['model_type']
    model_size = analysis_data['model_size']
    investment_amounts = analysis_data['investment_amounts']
    analyzer = analysis_data['analyzer']
    
    st.title(f"🎯 Analysis Results for {stock_symbol}")
    st.markdown("---")
    
    start_time = time.time()
    
    # Fetch stock data
    with st.spinner(f"📊 Getting {stock_symbol} data from Yahoo Finance..."):
        stock_data, stock_info = analyzer.fetch_stock_data(stock_symbol, period="1y")
    
    if stock_data is not None and len(stock_data) > 60:
        current_price = stock_data['Close'].iloc[-1]
        company_name = stock_info.get('longName', stock_symbol) if stock_info else stock_symbol
        
        # Display company information
        st.markdown(f"## 🏢 **{company_name} ({stock_symbol})**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("💰 Current Price", f"${current_price:.2f}")
        
        with col2:
            day_change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]
            day_change_pct = (day_change / stock_data['Close'].iloc[-2]) * 100
            st.metric("📈 1-Day Change", f"${day_change:.2f}", f"{day_change_pct:+.2f}%")
        
        with col3:
            week_change = ((current_price - stock_data['Close'].iloc[-6]) / stock_data['Close'].iloc[-6]) * 100 if len(stock_data) > 6 else 0
            st.metric("📅 1-Week Change", f"{week_change:+.2f}%")
        
        with col4:
            month_change = ((current_price - stock_data['Close'].iloc[-21]) / stock_data['Close'].iloc[-21]) * 100 if len(stock_data) > 21 else 0
            st.metric("📆 1-Month Change", f"{month_change:+.2f}%")
        
        # Load AI model and make predictions
        model_loaded = False
        predictions = None
        used_model = ""
        
        if model_type == "chronos":
            model, loaded_type = analyzer.load_chronos_model(model_size)
            if model is not None:
                with st.spinner("🤖 Amazon's AI is analyzing the stock..."):
                    predictions = analyzer.predict_with_chronos(model, stock_data['Close'].values)
                if predictions is not None:
                    model_loaded = True
                    used_model = load_chronos_model()
        
        elif model_type == "moirai":
            model, loaded_type = analyzer.load_moirai_model(model_size)
            if model is not None:
                with st.spinner("🤖 Salesforce's AI is analyzing the stock..."):
                    predictions = analyzer.predict_with_moirai(model, stock_data['Close'].values)
                if predictions is not None:
                    model_loaded = True
                    used_model = load_moirai_model()
        
        # Fallback to Chronos if primary failed
        if not model_loaded:
            st.warning("Primary AI model failed, trying Amazon Chronos backup...")
            model, loaded_type = analyzer.load_chronos_model("tiny")
            if model is not None:
                predictions = analyzer.predict_with_chronos(model, stock_data['Close'].values)
                if predictions is not None:
                    model_loaded = True
                    used_model = "Amazon Chronos (Backup)"
        
        total_time = time.time() - start_time
        
        # Display results if successful
        if model_loaded and predictions is not None:
            st.markdown("---")
            st.markdown(f"## 🎯 **AI PREDICTION RESULTS** ✨")
            
            # AI Decision
            mean_pred = predictions['mean']
            next_day_pred = mean_pred[0]
            week_end_pred = mean_pred[-1]
            
            next_day_change = ((next_day_pred - current_price) / current_price) * 100
            week_change_pred = ((week_end_pred - current_price) / current_price) * 100
            
            # Simple decision logic for beginners
            if week_change_pred > 5:
                decision = "🟢 STRONG BUY"
                decision_color = "green"
                decision_explanation = "AI expects significant price increase! Good time to invest."
            elif week_change_pred > 2:
                decision = "🟢 BUY"
                decision_color = "green"
                decision_explanation = "AI expects moderate price increase. Consider investing."
            elif week_change_pred < -5:
                decision = "🔴 STRONG SELL"
                decision_color = "red"
                decision_explanation = "AI expects significant price drop! Risky to invest now."
            elif week_change_pred < -2:
                decision = "🔴 SELL"
                decision_color = "red"
                decision_explanation = "AI expects price to drop. Consider waiting."
            else:
                decision = "⚪ HOLD"
                decision_color = "gray"
                decision_explanation = "AI expects stable prices. Neither buy nor sell strongly recommended."
            
            # Display decision prominently
            st.markdown(f"""
            <div style="background-color: {decision_color}20; border: 3px solid {decision_color}; padding: 20px; border-radius: 15px; text-align: center;">
                <h2 style="color: {decision_color}; margin: 0;">🤖 AI RECOMMENDATION: {decision}</h2>
                <p style="font-size: 18px; margin: 10px 0;"><strong>{decision_explanation}</strong></p>
                <p style="color: #666;">Prediction powered by {used_model}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Quick summary metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🔮 Tomorrow's Prediction", f"${next_day_pred:.2f}", f"{next_day_change:+.2f}%")
            
            with col2:
                st.metric("📅 Week-End Prediction", f"${week_end_pred:.2f}", f"{week_change_pred:+.2f}%")
            
            with col3:
                confidence_score = max(0, min(100, 70 + abs(week_change_pred) * 2))
                st.metric("🎯 AI Confidence", f"{confidence_score:.0f}%")
            
            # Profit/Loss Analysis
            st.markdown("---")
            st.markdown("## 💰 **PROFIT & LOSS FORECAST** - Your Money Scenarios")
            
            profit_loss_results = analyzer.calculate_profit_loss_scenarios(
                current_price, predictions, investment_amounts
            )
            
            # Create tabs for different investment amounts
            tabs = st.tabs([f"💵 Invest {amount}" for amount in [f"${amt:,}" for amt in investment_amounts]])
            
            for i, (tab, amount) in enumerate(zip(tabs, investment_amounts)):
                with tab:
                    amount_key = f"${amount:,}"
                    scenarios = profit_loss_results[amount_key]
                    
                    # Summary for this investment amount
                    day_7_scenario = scenarios['Day_7']
                    profit_loss_7day = day_7_scenario['profit_loss_amount']
                    profit_loss_pct_7day = day_7_scenario['profit_loss_pct']
                    
                    # Color coding
                    color = "green" if profit_loss_7day > 0 else "red" if profit_loss_7day < -50 else "orange"
                    
                    st.markdown(f"""
                    <div style="background-color: {color}20; border: 2px solid {color}; padding: 15px; border-radius: 10px; margin: 10px 0;">
                        <h3 style="color: {color}; margin-top: 0;">💰 Investment: ${amount:,}</h3>
                        <p><strong>Shares you'd own:</strong> {amount/current_price:.2f} shares</p>
                        <p><strong>Week-end value:</strong> ${day_7_scenario['new_portfolio_value']:,.2f}</p>
                        <p><strong>Profit/Loss after 7 days:</strong> <span style="color: {color}; font-weight: bold;">${profit_loss_7day:+,.2f} ({profit_loss_pct_7day:+.2f}%)</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Daily breakdown table
                    st.subheader("📅 Day-by-Day Forecast (Status : PROFIT / LOSS)")
                    
                    daily_data = []
                    for day in range(1, 8):
                        day_key = f'Day_{day}'
                        scenario = scenarios[day_key]
                        daily_data.append({
                            'Day': f"Day {day}",
                            'Predicted Price': f"${scenario['predicted_price']:.2f}",
                            'Your Portfolio Value': f"${scenario['new_portfolio_value']:,.2f}",
                            'Profit/Loss': f"${scenario['profit_loss_amount']:+,.2f}",
                            'Percentage': f"{scenario['profit_loss_pct']:+.2f}%",
                            'Status': "📈 Profit" if scenario['is_profit'] else "📉 Loss"
                        })
                    
                    df = pd.DataFrame(daily_data)
                    st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Visualization
            st.markdown("---")
            st.markdown("## 📈 **PRICE PREDICTION CHART**")
            
            fig = go.Figure()
            
            # Historical prices (last 30 days)
            recent_data = stock_data.tail(30)
            fig.add_trace(go.Scatter(
                x=recent_data.index,
                y=recent_data['Close'],
                mode='lines',
                name='📊 Recent Prices',
                line=dict(color='blue', width=2)
            ))
            
            # Future predictions
            future_dates = pd.date_range(
                start=stock_data.index[-1] + pd.Timedelta(days=1),
                periods=7,
                freq='D'
            )
            
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=mean_pred,
                mode='lines+markers',
                name=f'🤖 {used_model} Predictions',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ))
            
            # Confidence bands
            if 'q10' in predictions and 'q90' in predictions:
                fig.add_trace(go.Scatter(
                    x=future_dates.tolist() + future_dates[::-1].tolist(),
                    y=predictions['q90'].tolist() + predictions['q10'][::-1].tolist(),
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.15)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='🎯 AI Confidence Range',
                    showlegend=True
                ))
            
            fig.update_layout(
                title=f"📈 {company_name} ({stock_symbol}) - AI Price Forecast",
                xaxis_title="📅 Date",
                yaxis_title="💰 Price (USD)",
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk Assessment for Beginners
            st.markdown("---")
            st.markdown("## ⚖️ **RISK ASSESSMENT** - Important for Beginners")
            
            volatility = stock_data['Close'].pct_change().std() * np.sqrt(252) * 100
            
            if volatility > 40:
                risk_level = "🔴 HIGH RISK"
                risk_color = "red"
                risk_explanation = "This stock moves up and down a lot. You could make or lose money quickly!"
            elif volatility > 25:
                risk_level = "🟡 MEDIUM RISK"
                risk_color = "orange"
                risk_explanation = "This stock has moderate ups and downs. Suitable for some risk tolerance."
            else:
                risk_level = "🟢 LOW RISK"
                risk_color = "green"
                risk_explanation = "This stock is relatively stable. Less chance of big gains or losses."
            
            st.markdown(f"""
            <div style="background-color: {risk_color}20; border: 2px solid {risk_color}; padding: 15px; border-radius: 10px;">
                <h3 style="color: {risk_color}; margin-top: 0;">⚖️ Risk Level: {risk_level}</h3>
                <p><strong>Annual Volatility:</strong> {volatility:.1f}%</p>
                <p><strong>What this means:</strong> {risk_explanation}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Final summary and recommendations
            st.markdown("---")
            st.success(f"""
            ## ✅ **ANALYSIS COMPLETE!**
            
            **🤖 AI Model Used:** {used_model}  
            **⏱️ Analysis Time:** {total_time:.1f} seconds  
            **🎯 Recommendation:** {decision.replace('*', '').strip()}  
            **📊 7-Day Prediction:** ${week_end_pred:.2f} ({week_change_pred:+.2f}%)  
            **⚖️ Risk Level:** {risk_level.replace('🔴 ', '').replace('🟡 ', '').replace('🟢 ', '')}
            
            **Remember:** ⚠️ This is AI prediction, not guaranteed! Always do your own research and consult financial advisors ⚠️
            """)
            
            # New analysis button
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🔄 Analyze Another Stock", type="primary", use_container_width=True):
                    st.session_state.current_page = 'home'
                    st.session_state.analysis_data = None
                    st.rerun()
        
        else:
            st.error("❌ AI model failed to load. Please try again or contact support.")
            if st.button("🔄 Try Again"):
                st.session_state.current_page = 'home'
                st.rerun()
    
    else:
        st.error("❌ Could not fetch stock data. Please check the symbol and try again.")
        if st.button("🔄 Try Again"):
            st.session_state.current_page = 'home'
            st.rerun()

def main():
    # Page router
    if st.session_state.current_page == 'home':
        show_home_page()
    elif st.session_state.current_page == 'results':
        show_results_page()

if __name__ == "__main__":
    main()


