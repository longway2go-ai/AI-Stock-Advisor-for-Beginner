# ⚡ AI Stock Predictor

An advanced AI-powered stock price prediction application using state-of-the-art time series forecasting models from Amazon and Salesforce.

![AI Stock Predictor](https://img.shields.io/badge/AI-Stock%20Predictor-blue)
![Gradio](https://img.shields.io/badge/Built%20with-Gradio-orange)
![HuggingFace](https://img.shields.io/badge/Deployed%20on-HuggingFace%20Spaces-yellow)

## 🚀 Live Demo

**[Try the Live App Here →](https://your-username-ai-stock-predictor.hf.space)**

*Replace with your actual Hugging Face Spaces URL*

## 📊 Features

- **🤖 Dual AI Models**: Choose between Amazon Chronos (fast & reliable) or Salesforce Moirai (advanced accuracy)
- **📈 Real-time Stock Data**: Fetches live stock prices from Yahoo Finance
- **🔮 7-Day Predictions**: Generate AI-powered price forecasts with confidence intervals
- **💰 Investment Scenarios**: Calculate potential profit/loss for different investment amounts
- **📱 Interactive Interface**: User-friendly Gradio web interface with real-time progress tracking
- **📊 Visual Charts**: Interactive Plotly charts showing historical data and predictions
- **🛡️ Robust Error Handling**: Multiple fallback mechanisms ensure reliability
- **⚠️ Educational Focus**: Clear disclaimers and beginner-friendly guidance

## 🧠 AI Models Used

### Amazon Chronos Tiny
- **Fast loading** and prediction generation
- **Stable performance** across different stocks
- **Recommended for beginners**
- Built on T5 architecture optimized for time series

### Salesforce Moirai Small
- **Advanced accuracy** for complex patterns
- **Sophisticated forecasting** capabilities
- **Automatic fallback** to Chronos if unavailable
- Universal time series forecasting model

## 🛠️ Tech Stack

- **Frontend**: Gradio 4.0+
- **Backend**: Python 3.9+
- **AI Models**: Amazon Chronos, Salesforce Moirai
- **Data Source**: Yahoo Finance (yfinance)
- **Visualization**: Plotly
- **Deployment**: Hugging Face Spaces
- **ML Libraries**: PyTorch, Transformers, uni2ts, gluonts

## 📦 Installation

### Option 1: Run Locally

1. **Clone the repository**:
```
git clone https://github.com/longway2go-ai/ai-stock-predictor
cd ai-stock-predictor
```
2. **Install dependencies**:
```
pip install -r requirements.txt
```
3. **Run the application**:
```
python app.py
```
4. **Open your browser** to `http://localhost:7860`

### Option 2: Deploy on Hugging Face Spaces

1. **Create a new Space** on [Hugging Face](https://huggingface.co/spaces)
2. **Select Gradio SDK**
3. **Upload `app.py` and `requirements.txt`**
4. **Wait for automatic deployment**

## 📋 Requirements
```
gradio>=4.0.0
yfinance>=0.2.18
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.15.0
torch>=2.1.0,<2.5.0
uni2ts>=1.1.0
gluonts>=0.13.0
huggingface_hub>=0.23.0
transformers>=4.30.0
accelerate>=0.20.0
datasets
safetensors>=0.3.0
chronos-forecasting
```

## 🎯 How to Use

1. **Select a Stock**: Choose from popular options (AAPL, GOOGL, TSLA, etc.) or enter any valid symbol
2. **Choose AI Model**: 
   - **Chronos**: Faster, more reliable, good for beginners
   - **Moirai**: More sophisticated, higher accuracy potential
3. **Set Investment Amount**: Adjust slider from $500 to $100,000
4. **Click "Analyze Stock Now"**: Wait 30-60 seconds for AI analysis
5. **Review Results**: Get recommendations, predictions, and investment scenarios

## 🎨 Interface Highlights

- **📊 Real-time Progress Tracking**: See exactly what the AI is doing
- **🟢🟡🔴 Color-coded Recommendations**: Easy-to-understand buy/hold/sell signals
- **📈 Interactive Charts**: Historical data + AI predictions with confidence bands
- **💡 Beginner Tips**: Educational guidance for new investors
- **⚠️ Comprehensive Disclaimers**: Clear warnings about AI limitations
