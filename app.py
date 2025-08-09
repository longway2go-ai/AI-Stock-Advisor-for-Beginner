import gradio as gr
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import warnings
import time
import gc
import os
import torch
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple

warnings.filterwarnings('ignore')

# Environment optimizations for Hugging Face Spaces
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

if torch.cuda.is_available():
    torch.cuda.empty_cache()
torch.set_num_threads(min(4, os.cpu_count() or 1))

class FastAIStockAnalyzer:
    """Optimized AI Stock Analyzer for Gradio with robust error handling"""
    
    def __init__(self):
        self.context_length = 32
        self.prediction_length = 7
        self.device = "cpu"
        self.model_cache = {}
        
    def fetch_stock_data(self, symbol: str, period: str = "6mo") -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        """Fetch stock data with error handling"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval="1d", 
                                actions=False, auto_adjust=True, 
                                back_adjust=False, repair=False)
            
            if data.empty:
                return None, None
                
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
            return None, None
    
    def load_chronos_tiny(self) -> Tuple[Optional[Any], str]:
        """Load Chronos model with caching and fallback"""
        model_key = "chronos_tiny"
        
        if model_key in self.model_cache:
            return self.model_cache[model_key], "chronos"
        
        try:
            from chronos import ChronosPipeline
            
            # Try primary loading method
            pipeline = ChronosPipeline.from_pretrained(
                "amazon/chronos-t5-tiny",
                device_map="cpu",
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            self.model_cache[model_key] = pipeline
            return pipeline, "chronos"
            
        except ImportError:
            # Chronos not available
            return None, None
        except Exception as e:
            # Try fallback loading method
            try:
                pipeline = ChronosPipeline.from_pretrained(
                    "amazon/chronos-t5-tiny",
                    device_map="auto",
                    torch_dtype=torch.float32
                )
                self.model_cache[model_key] = pipeline
                return pipeline, "chronos"
            except:
                return None, None
    
    def load_moirai_small(self) -> Tuple[Optional[Any], str]:
        """Load Moirai model with updated method and fallbacks"""
        model_key = "moirai_small"
        
        if model_key in self.model_cache:
            return self.model_cache[model_key], "moirai"
        
        try:
            from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
            
            # Method 1: Try the standard approach
            try:
                module = MoiraiModule.from_pretrained(
                    "Salesforce/moirai-1.0-R-small",
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True
                )
                
                model = MoiraiForecast(
                    module=module,
                    prediction_length=self.prediction_length,
                    context_length=self.context_length,
                    patch_size="auto",
                    num_samples=10,
                    target_dim=1,
                    feat_dynamic_real_dim=0,
                    past_feat_dynamic_real_dim=0
                )
                
                self.model_cache[model_key] = model
                return model, "moirai"
                
            except Exception as e1:
                # Method 2: Try newer version
                try:
                    module = MoiraiModule.from_pretrained(
                        "Salesforce/moirai-1.1-R-small",
                        device_map="cpu",
                        torch_dtype=torch.float32,
                        trust_remote_code=True
                    )
                    
                    model = MoiraiForecast(
                        module=module,
                        prediction_length=self.prediction_length,
                        context_length=self.context_length,
                        patch_size="auto",
                        num_samples=5,  # Reduced for stability
                        target_dim=1,
                        feat_dynamic_real_dim=0,
                        past_feat_dynamic_real_dim=0
                    )
                    
                    self.model_cache[model_key] = model
                    return model, "moirai"
                    
                except Exception as e2:
                    # Method 3: Minimal configuration
                    try:
                        module = MoiraiModule.from_pretrained("Salesforce/moirai-1.0-R-small")
                        model = MoiraiForecast(
                            module=module,
                            prediction_length=7,
                            context_length=32,
                            patch_size="auto",
                            num_samples=5,
                            target_dim=1
                        )
                        
                        self.model_cache[model_key] = model
                        return model, "moirai"
                        
                    except Exception as e3:
                        return None, None
                        
        except ImportError:
            # uni2ts not available
            return None, None
        except Exception as e:
            return None, None
    
    def predict_chronos_fast(self, pipeline: Any, data: np.ndarray) -> Optional[Dict]:
        """Fast Chronos prediction with error handling"""
        try:
            context_data = data[-self.context_length:]
            context = torch.tensor(context_data, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                forecast = pipeline.predict(
                    context=context,
                    prediction_length=self.prediction_length,
                    num_samples=10,
                    temperature=1.0,
                    top_k=50,
                    top_p=1.0
                )
            
            forecast_array = forecast[0].numpy()
            predictions = {
                'mean': np.median(forecast_array, axis=0),
                'q10': np.quantile(forecast_array, 0.1, axis=0),
                'q90': np.quantile(forecast_array, 0.9, axis=0),
                'std': np.std(forecast_array, axis=0)
            }
            
            return predictions
            
        except Exception as e:
            return None
    
    def predict_moirai_fast(self, model: Any, data: np.ndarray) -> Optional[Dict]:
        """Fast Moirai prediction with enhanced error handling"""
        try:
            from gluonts.dataset.common import ListDataset
            
            # Prepare dataset with minimal configuration
            dataset = ListDataset([{
                "item_id": "stock",
                "start": "2023-01-01",
                "target": data[-self.context_length:].tolist()
            }], freq='D')
            
            # Create predictor with safe parameters
            predictor = model.create_predictor(
                batch_size=1,
                num_parallel_samples=5  # Further reduced for stability
            )
            
            # Generate forecast
            forecasts = list(predictor.predict(dataset))
            
            if not forecasts:
                return None
                
            forecast = forecasts[0]
            
            predictions = {
                'mean': forecast.mean,
                'q10': forecast.quantile(0.1),
                'q90': forecast.quantile(0.9),
                'std': np.std(forecast.samples, axis=0) if hasattr(forecast, 'samples') else np.zeros(7)
            }
            
            return predictions
            
        except Exception as e:
            return None
    
    def generate_simple_prediction(self, data: np.ndarray) -> Dict:
        """Fallback prediction method using simple statistical models"""
        try:
            # Simple moving average with trend
            recent_data = data[-30:]  # Last 30 days
            short_ma = np.mean(recent_data[-7:])   # 7-day average
            long_ma = np.mean(recent_data[-21:])   # 21-day average
            
            # Calculate trend
            trend = (short_ma - long_ma) / long_ma if long_ma != 0 else 0
            
            # Generate predictions
            current_price = data[-1]
            predictions = []
            
            for i in range(7):
                # Simple trend projection with some noise
                predicted_price = current_price * (1 + trend * (i + 1) * 0.1)
                predictions.append(predicted_price)
            
            predictions = np.array(predictions)
            
            return {
                'mean': predictions,
                'q10': predictions * 0.95,  # 5% lower
                'q90': predictions * 1.05,  # 5% higher
                'std': np.full(7, np.std(recent_data) * 0.5)
            }
            
        except Exception:
            # Ultimate fallback - flat prediction
            current_price = data[-1]
            return {
                'mean': np.full(7, current_price),
                'q10': np.full(7, current_price * 0.98),
                'q90': np.full(7, current_price * 1.02),
                'std': np.full(7, 0.01)
            }

# Initialize analyzer globally for caching
analyzer = FastAIStockAnalyzer()

def analyze_stock(stock_symbol, model_choice, investment_amount, progress=gr.Progress()):
    """Main analysis function with comprehensive error handling and fallbacks"""
    
    try:
        progress(0.1, desc="Fetching stock data...")
        
        # Validate input
        if not stock_symbol or stock_symbol.strip() == "":
            return (
                "❌ Error: Please enter a valid stock symbol.",
                None,
                "❌ Invalid Input",
                "N/A",
                "N/A"
            )
        
        # Fetch data
        stock_data, stock_info = analyzer.fetch_stock_data(stock_symbol.upper())
        
        if stock_data is None or len(stock_data) < 50:
            return (
                f"❌ Error: Insufficient data for {stock_symbol.upper()}. Please check the stock symbol or try a different one.",
                None,
                "❌ Data Error",
                "N/A",
                "N/A"
            )
        
        current_price = stock_data['Close'].iloc[-1]
        company_name = stock_info.get('longName', stock_symbol) if stock_info else stock_symbol
        
        progress(0.3, desc="Loading AI model...")
        
        # Determine model type and load
        model_type = "chronos" if "Chronos" in model_choice else "moirai"
        model = None
        model_name = ""
        prediction_method = None
        
        if model_type == "chronos":
            model, loaded_type = analyzer.load_chronos_tiny()
            model_name = "Amazon Chronos Tiny"
            prediction_method = "chronos"
        else:
            model, loaded_type = analyzer.load_moirai_small()
            model_name = "Salesforce Moirai Small"
            prediction_method = "moirai"
            
            # Fallback to Chronos if Moirai fails
            if model is None:
                progress(0.4, desc="Moirai unavailable, switching to Chronos...")
                model, loaded_type = analyzer.load_chronos_tiny()
                model_name = "Amazon Chronos Tiny (Fallback)"
                prediction_method = "chronos"
        
        # If both models fail, use simple prediction
        if model is None:
            progress(0.5, desc="Using statistical fallback method...")
            model_name = "Statistical Trend Model (Fallback)"
            prediction_method = "simple"
        
        progress(0.6, desc="Generating predictions...")
        
        # Generate predictions based on available method
        predictions = None
        
        if prediction_method == "chronos" and model is not None:
            predictions = analyzer.predict_chronos_fast(model, stock_data['Close'].values)
        elif prediction_method == "moirai" and model is not None:
            predictions = analyzer.predict_moirai_fast(model, stock_data['Close'].values)
        
        # Use simple prediction if AI models fail
        if predictions is None:
            predictions = analyzer.generate_simple_prediction(stock_data['Close'].values)
            model_name = "Statistical Trend Model (AI Models Unavailable)"
        
        progress(0.8, desc="Calculating investment scenarios...")
        
        # Analysis results
        mean_pred = predictions['mean']
        final_pred = mean_pred[-1]
        week_change = ((final_pred - current_price) / current_price) * 100
        
        # Decision logic
        if week_change > 5:
            decision = "🟢 STRONG BUY"
            explanation = "Model expects significant gains!"
        elif week_change > 2:
            decision = "🟢 BUY"
            explanation = "Model expects moderate gains"
        elif week_change < -5:
            decision = "🔴 STRONG SELL"
            explanation = "Model expects significant losses"
        elif week_change < -2:
            decision = "🔴 SELL"
            explanation = "Model expects losses"
        else:
            decision = "⚪ HOLD"
            explanation = "Model expects stable prices"
        
        # Create analysis text
        analysis_text = f"""
# 🎯 {company_name} ({stock_symbol.upper()}) Analysis
## 🤖 RECOMMENDATION: {decision}
**{explanation}**
*Powered by {model_name}*
## 📊 Key Metrics
- **Current Price**: ${current_price:.2f}
- **7-Day Prediction**: ${final_pred:.2f} ({week_change:+.2f}%)
- **Confidence Level**: {min(100, max(50, 70 + abs(week_change) * 1.5)):.0f}%
- **Analysis Method**: {model_name}
## 💰 Investment Scenario (${investment_amount:,.0f})
- **Shares**: {investment_amount/current_price:.2f}
- **Current Value**: ${investment_amount:,.2f}
- **Predicted Value**: ${investment_amount + ((final_pred - current_price) * (investment_amount/current_price)):,.2f}
- **Profit/Loss**: ${((final_pred - current_price) * (investment_amount/current_price)):+,.2f} ({week_change:+.2f}%)
## ⚠️ Important Disclaimers
- **This is for educational purposes only**
- **Not financial advice - consult professionals**
- **AI predictions can be wrong - invest responsibly**
- **Past performance ≠ future results**
"""
        
        progress(0.9, desc="Creating visualizations...")
        
        # Create chart
        fig = go.Figure()
        
        # Historical data (last 30 days)
        recent = stock_data.tail(30)
        fig.add_trace(go.Scatter(
            x=recent.index, 
            y=recent['Close'],
            mode='lines', 
            name='Historical Price',
            line=dict(color='blue', width=2)
        ))
        
        # Predictions
        future_dates = pd.date_range(
            start=stock_data.index[-1] + pd.Timedelta(days=1),
            periods=7, 
            freq='D'
        )
        
        fig.add_trace(go.Scatter(
            x=future_dates, 
            y=mean_pred,
            mode='lines+markers', 
            name='Prediction',
            line=dict(color='red', width=3),
            marker=dict(size=8)
        ))
        
        # Confidence bands
        if 'q10' in predictions and 'q90' in predictions:
            fig.add_trace(go.Scatter(
                x=future_dates.tolist() + future_dates[::-1].tolist(),
                y=predictions['q90'].tolist() + predictions['q10'][::-1].tolist(),
                fill='toself', 
                fillcolor='rgba(255,0,0,0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Confidence Range', 
                showlegend=True
            ))
        
        fig.update_layout(
            title=f"{stock_symbol.upper()} - Stock Forecast ({model_name})",
            xaxis_title="Date", 
            yaxis_title="Price ($)",
            height=500, 
            showlegend=True,
            template="plotly_white"
        )
        
        progress(1.0, desc="Analysis complete!")
        
        # Create summary metrics
        try:
            day_change = stock_data['Close'].iloc[-1] - stock_data['Close'].iloc[-2]
            day_change_pct = (day_change / stock_data['Close'].iloc[-2]) * 100
        except:
            day_change_pct = 0
        
        current_metrics = f"${current_price:.2f} ({day_change_pct:+.2f}%)"
        prediction_metrics = f"${final_pred:.2f} ({week_change:+.2f}%)"
        
        return (
            analysis_text,
            fig,
            decision,
            current_metrics,
            prediction_metrics
        )
        
    except Exception as e:
        # Ultimate error handler
        error_msg = f"""
# ❌ Analysis Error
**Something went wrong during the analysis:**
- **Error**: {str(e)[:200]}...
- **Stock**: {stock_symbol}
- **Model**: {model_choice}
## 🔧 Try these solutions:
1. **Check stock symbol** - Make sure it's valid (e.g., AAPL, GOOGL)
2. **Try different model** - Switch between Chronos and Moirai
3. **Refresh and try again** - Temporary server issues
4. **Use popular stocks** - AAPL, MSFT, GOOGL work best
## 📞 Still having issues?
This may be due to Hugging Face Spaces resource limitations or model availability.
"""
        
        return (
            error_msg,
            None,
            "❌ Error",
            "N/A",
            "N/A"
        )

# Create Gradio interface with enhanced UI and improved disclaimer
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="⚡ Fast AI Stock Predictor",
    css="""
    footer {visibility: hidden}
    .gradio-container {max-width: 1200px; margin: auto;}
    .main-header {text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 20px;}
    .disclaimer {background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 8px; margin: 10px 0;}
    """
) as demo:
    
    gr.HTML("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5em;">⚡ AI Stock Predictor</h1>
        <p style="margin: 10px 0 0 0; font-size: 1.2em;"><strong>🤖 Powered by Amazon Chronos & Salesforce Moirai</strong></p>
        <p style="margin: 5px 0 0 0; opacity: 0.9;">Advanced AI models for stock price forecasting</p>
    </div>
    """)
    
        # Enhanced Disclaimer Section with Fully Visible Headings
    gr.HTML("""
    <div style="max-width: 900px; margin: 20px auto; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;">
        
        <!-- Main Disclaimer Box -->
        <div style="background: #ffffff; 
                    border: 5px solid #ff6b6b; 
                    border-radius: 15px; 
                    padding: 30px; 
                    margin-bottom: 30px; 
                    box-shadow: 0 10px 30px rgba(255, 107, 107, 0.3);">
            
            <!-- FIXED DISCLAIMER HEADING -->
            <div style="background: #ff6b6b; 
                        color: #ffffff; 
                        text-align: center; 
                        margin: -30px -30px 25px -30px; 
                        padding: 25px; 
                        border-radius: 10px 10px 0 0;">
                <h2 style="color: #ffffff; 
                          margin: 0; 
                          font-size: 2.2em; 
                          font-weight: bold; 
                          text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                    ⚠️ IMPORTANT DISCLAIMER ⚠️
                </h2>
            </div>
            
            <div style="background: #ffffff; 
                        border-radius: 12px; 
                        padding: 25px; 
                        color: #000000; 
                        font-size: 16px; 
                        line-height: 1.8;">
                
                <!-- Educational Purpose Header -->
                <div style="background: #f8f9fa; 
                            border: 3px solid #dee2e6; 
                            border-radius: 10px; 
                            padding: 20px; 
                            text-align: center; 
                            margin-bottom: 25px;">
                    <h3 style="margin: 0; 
                              font-weight: bold; 
                              font-size: 1.4em; 
                              color: #000000;">
                        📚 FOR EDUCATIONAL PURPOSES ONLY 📚
                    </h3>
                </div>
                
                <!-- Disclaimer Points -->
                <div style="display: grid; gap: 15px;">
                    <div style="background: #ffffff; 
                                border: 3px solid #3498db; 
                                border-radius: 10px; 
                                padding: 20px; 
                                box-shadow: 0 3px 10px rgba(0,0,0,0.1);">
                        <h4 style="color: #000000; margin: 0 0 8px 0; font-size: 1.1em;">
                            🚫 NOT FINANCIAL ADVICE
                        </h4>
                        <p style="color: #000000; margin: 0; font-size: 15px;">
                            This AI tool is for learning and research only
                        </p>
                    </div>
                    
                    <div style="background: #ffffff; 
                                border: 3px solid #e74c3c; 
                                border-radius: 10px; 
                                padding: 20px; 
                                box-shadow: 0 3px 10px rgba(0,0,0,0.1);">
                        <h4 style="color: #000000; margin: 0 0 8px 0; font-size: 1.1em;">
                            ⚡ AI CAN BE WRONG
                        </h4>
                        <p style="color: #000000; margin: 0; font-size: 15px;">
                            Predictions may be inaccurate or completely wrong
                        </p>
                    </div>
                    
                    <div style="background: #ffffff; 
                                border: 3px solid #f39c12; 
                                border-radius: 10px; 
                                padding: 20px; 
                                box-shadow: 0 3px 10px rgba(0,0,0,0.1);">
                        <h4 style="color: #000000; margin: 0 0 8px 0; font-size: 1.1em;">
                            💼 CONSULT PROFESSIONALS
                        </h4>
                        <p style="color: #000000; margin: 0; font-size: 15px;">
                            Always seek qualified financial advisors
                        </p>
                    </div>
                    
                    <div style="background: #ffffff; 
                                border: 3px solid #9b59b6; 
                                border-radius: 10px; 
                                padding: 20px; 
                                box-shadow: 0 3px 10px rgba(0,0,0,0.1);">
                        <h4 style="color: #000000; margin: 0 0 8px 0; font-size: 1.1em;">
                            💰 INVEST RESPONSIBLY
                        </h4>
                        <p style="color: #000000; margin: 0; font-size: 15px;">
                            Only invest what you can afford to lose
                        </p>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Trading Decision Guide for Beginners -->
        <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 25px; margin: 30px 0;">
            
            <!-- BUY Signal -->
            <div style="background: #27ae60; 
                        color: #ffffff; 
                        padding: 25px; 
                        border-radius: 15px; 
                        text-align: center; 
                        box-shadow: 0 8px 25px rgba(46, 204, 113, 0.4);
                        border: 4px solid #229954;">
                <div style="font-size: 3em; margin-bottom: 15px;">📈</div>
                <h3 style="margin: 0 0 15px 0; 
                          font-size: 1.3em; 
                          font-weight: bold; 
                          color: #ffffff;
                          text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
                    GREEN = BUY SIGNAL
                </h3>
                <p style="margin: 0; 
                         font-size: 14px; 
                         color: #ffffff; 
                         line-height: 1.4;">
                    AI predicts price increase<br>
                    <strong style="color: #ffffff;">⚠️ Still risky - do research!</strong>
                </p>
            </div>
            
            <!-- HOLD Signal -->
            <div style="background: #f39c12; 
                        color: #ffffff; 
                        padding: 25px; 
                        border-radius: 15px; 
                        text-align: center; 
                        box-shadow: 0 8px 25px rgba(243, 156, 18, 0.4);
                        border: 4px solid #e67e22;">
                <div style="font-size: 3em; margin-bottom: 15px;">⚪</div>
                <h3 style="margin: 0 0 15px 0; 
                          font-size: 1.3em; 
                          font-weight: bold; 
                          color: #ffffff;
                          text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
                    YELLOW = HOLD
                </h3>
                <p style="margin: 0; 
                         font-size: 14px; 
                         color: #ffffff; 
                         line-height: 1.4;">
                    AI expects stable prices<br>
                    <strong style="color: #ffffff;">Wait and watch strategy</strong>
                </p>
            </div>
            
            <!-- SELL Signal -->
            <div style="background: #e74c3c; 
                        color: #ffffff; 
                        padding: 25px; 
                        border-radius: 15px; 
                        text-align: center; 
                        box-shadow: 0 8px 25px rgba(231, 76, 60, 0.4);
                        border: 4px solid #c0392b;">
                <div style="font-size: 3em; margin-bottom: 15px;">📉</div>
                <h3 style="margin: 0 0 15px 0; 
                          font-size: 1.3em; 
                          font-weight: bold; 
                          color: #ffffff;
                          text-shadow: 1px 1px 2px rgba(0,0,0,0.3);">
                    RED = SELL SIGNAL
                </h3>
                <p style="margin: 0; 
                         font-size: 14px; 
                         color: #ffffff; 
                         line-height: 1.4;">
                    AI predicts price decrease<br>
                    <strong style="color: #ffffff;">Consider reducing exposure</strong>
                </p>
            </div>
        </div>
        
        <!-- Beginner Tips - COMPLETELY FIXED VERSION -->
        <div style="background: #ffffff; 
                    border: 5px solid #3498db; 
                    border-radius: 15px; 
                    padding: 30px; 
                    margin-top: 30px;
                    box-shadow: 0 10px 30px rgba(52, 152, 219, 0.3);">
            
            <!-- FIXED BEGINNER TIPS HEADING -->
            <div style="background: #3498db; 
                        color: #ffffff; 
                        text-align: center; 
                        margin: -30px -30px 25px -30px; 
                        padding: 25px; 
                        border-radius: 10px 10px 0 0;">
                <h2 style="color: #ffffff; 
                          margin: 0; 
                          font-size: 2em; 
                          font-weight: bold;
                          text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">
                    💡 BEGINNER TIPS 💡
                </h2>
            </div>
            
            <div style="display: grid; 
                        grid-template-columns: 1fr 1fr; 
                        gap: 25px;">
                <div style="background: #ffffff; 
                            padding: 25px; 
                            border-radius: 12px; 
                            box-shadow: 0 6px 20px rgba(0,0,0,0.15); 
                            border: 4px solid #27ae60;">
                    <h4 style="color: #000000; 
                              font-size: 1.2em; 
                              margin: 0 0 12px 0;
                              font-weight: bold;">🎯 Start Small</h4>
                    <p style="color: #000000; 
                             font-size: 15px; 
                             margin: 0;
                             line-height: 1.5;">Begin with small amounts you can afford to lose completely</p>
                </div>
                <div style="background: #ffffff; 
                            padding: 25px; 
                            border-radius: 12px; 
                            box-shadow: 0 6px 20px rgba(0,0,0,0.15); 
                            border: 4px solid #e74c3c;">
                    <h4 style="color: #000000; 
                              font-size: 1.2em; 
                              margin: 0 0 12px 0;
                              font-weight: bold;">📚 Keep Learning</h4>
                    <p style="color: #000000; 
                             font-size: 15px; 
                             margin: 0;
                             line-height: 1.5;">Study finance basics before making real investments</p>
                </div>
                <div style="background: #ffffff; 
                            padding: 25px; 
                            border-radius: 12px; 
                            box-shadow: 0 6px 20px rgba(0,0,0,0.15); 
                            border: 4px solid #f39c12;">
                    <h4 style="color: #000000; 
                              font-size: 1.2em; 
                              margin: 0 0 12px 0;
                              font-weight: bold;">🏦 Use Real Platforms</h4>
                    <p style="color: #000000; 
                             font-size: 15px; 
                             margin: 0;
                             line-height: 1.5;">Practice with paper trading before using real money</p>
                </div>
                <div style="background: #ffffff; 
                            padding: 25px; 
                            border-radius: 12px; 
                            box-shadow: 0 6px 20px rgba(0,0,0,0.15); 
                            border: 4px solid #9b59b6;">
                    <h4 style="color: #000000; 
                              font-size: 1.2em; 
                              margin: 0 0 12px 0;
                              font-weight: bold;">⏰ Think Long-term</h4>
                    <p style="color: #000000; 
                             font-size: 15px; 
                             margin: 0;
                             line-height: 1.5;">Don't panic on daily market fluctuations</p>
                </div>
            </div>
        </div>
        
        <!-- Final Warning -->
        <div style="background: #2c3e50; 
                    color: #ffffff; 
                    padding: 25px; 
                    border-radius: 12px; 
                    text-align: center; 
                    margin-top: 30px; 
                    border: 4px solid #34495e;
                    box-shadow: 0 8px 25px rgba(44, 62, 80, 0.4);">
            <h3 style="margin: 0; 
                      font-weight: bold; 
                      font-size: 1.1em; 
                      color: #ffffff;
                      text-shadow: 1px 1px 2px rgba(0,0,0,0.5);">
                🚨 REMEMBER: Past performance ≠ Future results | Markets can crash | AI makes mistakes 🚨
            </h3>
        </div>
    </div>
    """)


    
    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            gr.HTML("<h3>🎯 Analysis Configuration</h3>")
            
            stock_input = gr.Dropdown(
                choices=["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN", "META", "NFLX", "NVDA", "ORCL", "CRM"],
                value="AAPL",
                label="📈 Select Stock Symbol",
                allow_custom_value=True,
                info="Choose popular stocks or enter any valid symbol"
            )
            
            model_input = gr.Radio(
                choices=["🚀 Chronos (Fast & Reliable)", "🎯 Moirai (Advanced)"],
                value="🚀 Chronos (Fast & Reliable)",
                label="🤖 AI Model Selection",
                info="Chronos: Faster, more stable | Moirai: More sophisticated (may fallback to Chronos)"
            )
            
            investment_input = gr.Slider(
                minimum=500,
                maximum=100000,
                value=5000,
                step=500,
                label="💰 Investment Amount ($)",
                info="Amount for profit/loss calculation"
            )
            
            analyze_btn = gr.Button(
                "🚀 Analyze Stock Now", 
                variant="primary", 
                size="lg",
                scale=1
            )
            
            gr.HTML("<br>")
            
            # Quick stats
            with gr.Group():
                gr.HTML("<h4>📊 Quick Metrics</h4>")
                current_price_display = gr.Textbox(
                    label="Current Price", 
                    interactive=False,
                    container=True
                )
                prediction_display = gr.Textbox(
                    label="7-Day Prediction", 
                    interactive=False,
                    container=True
                )
                decision_display = gr.Textbox(
                    label="AI Recommendation", 
                    interactive=False,
                    container=True
                )
            
        with gr.Column(scale=2, min_width=600):
            gr.HTML("<h3>📊 Analysis Results</h3>")
            
            analysis_output = gr.Markdown(
                value="""
# 👋 Welcome to AI Stock Predictor!
**Ready to analyze stocks with cutting-edge AI?**
🎯 **How to use:**
1. Select a stock symbol (or enter your own)
2. Choose AI model (Chronos recommended for beginners)
3. Set investment amount for scenario analysis
4. Click "Analyze Stock Now"
💡 **Tips:**
- Try popular stocks like AAPL, GOOGL, MSFT first
- Chronos model is faster and more reliable
- Analysis takes 30-60 seconds for first-time model loading
⚡ **Click the button to get started!**
""",
                container=True
            )
    
    with gr.Row():
        chart_output = gr.Plot(
            label="📈 Stock Price Chart & AI Predictions",
            container=True,
            show_label=True
        )
    
    # Event handlers
    analyze_btn.click(
        fn=analyze_stock,
        inputs=[stock_input, model_input, investment_input],
        outputs=[
            analysis_output, 
            chart_output, 
            decision_display,
            current_price_display,
            prediction_display
        ],
        show_progress=True
    )
    
    # Examples section
    gr.HTML("<h3>🎭 Try These Examples</h3>")
    gr.Examples(
        examples=[
            ["AAPL", "🚀 Chronos (Fast & Reliable)", 5000],
            ["TSLA", "🎯 Moirai (Advanced)", 10000],
            ["GOOGL", "🚀 Chronos (Fast & Reliable)", 2500],
            ["MSFT", "🎯 Moirai (Advanced)", 7500],
            ["NVDA", "🚀 Chronos (Fast & Reliable)", 3000],
        ],
        inputs=[stock_input, model_input, investment_input],
        label="Click any example to load it:",
        examples_per_page=5
    )
    
    # Footer
    gr.HTML("""
    <div style="text-align: center; padding: 20px; margin-top: 30px; border-top: 1px solid #eee;">
        <p><strong>🤖 AI Stock Predictor</strong> | Built with ❤️ using Gradio & Hugging Face</p>
        <p style="font-size: 12px; color: #666;">
            Powered by Amazon Chronos & Salesforce Moirai | 
            Educational Tool - Not Financial Advice
        </p>
    </div>
    """)

# Launch configuration - Fixed for Gradio 4.0+
if __name__ == "__main__":
    # Enable queue using new Gradio 4.0+ method
    demo.queue(max_size=20)
    
    demo.launch(
        share=True,  # Set to True for public sharing
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        max_threads=10
    )
