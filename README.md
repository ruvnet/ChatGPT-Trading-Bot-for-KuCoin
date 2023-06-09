# CHATGPT Can Do The Crypto Trading | Script for KuCoin by Cody Krecicki

**Current prompt is not working with the API, in ChatGPT it says UP or DOWN. The API refuses to respond to our prompt**

## This is a trading script for KuCoin that continuously places buy and sell orders based on market data and a predictive model generated by OpenAI's GPT-3.5 language model.

### Requirements -- We use conda for enviroments and it is what the requirments.txt is made with.
ccxt
openai

### Setup
Install the required packages: pip install ccxt openai
Replace EXCHANGE_NAME with the name of the exchange you want to use.
Set sandbox mode to True or False by setting enabled to True or False.
Set your API keys by replacing the empty strings for apiKey, secret, and password.
Set the symbol you want to trade on KuCoin by replacing 'BTC/USDT' with your desired symbol.

### Setup your KuCoin API keys here https://www.kucoin.com/support/360015102174
Get your OpenAI keys https://platform.openai.com/account/api-keys

### How it works
The script continuously streams market data from KuCoin and fetches the current ticker information for the symbol. It then calculates the midpoint between the bid and ask prices and sets the premium for the sell order. The script uses a predictive model generated by OpenAI's GPT-3.5 language model to predict whether the market is going up or down. If the model predicts an upward trend, the script places a limit buy order at the midpoint price. If it predicts a downward trend, it places a limit sell order at the midpoint price. The script also checks for any open orders and updates them accordingly.

### Time frame
You can select the desired time frame by choosing from the following options:

'1m': '1min'
'3m': '3min'
'5m': '5min'
'15m': '15min'
'30m': '30min'
'1h': '1hour'
'2h': '2hour'
'4h': '4hour'
'6h': '6hour'
'8h': '8hour'
'12h': '12hour'
'1d': '1day'
'1w': '1week'

## Updates made by @ruv
Here are the main changes made to the original trading script and their benefits:

### Added RSI calculation:
Benefit: The RSI indicator helps to identify oversold and overbought market conditions, which can be used to make more informed trading decisions.

### Updated the trading logic:

Benefit: The new logic uses a combination of the RSI indicator and GPT-3.5-turbo predictions to make trading decisions. This helps to take into account both technical analysis and the understanding of market trends based on provided data, potentially leading to better results.

### Simplified the GPT-3.5-turbo prompt and reduced max_tokens:

Benefit: A simpler prompt makes it easier for GPT-3.5-turbo to understand the task and provide a relevant response. Reducing max_tokens to 5 ensures that we only get a single-word response ("up" or "down"), making the response processing more efficient.
Changed the GPT-3.5-turbo output processing:

Benefit: The updated gpt_up_down function directly returns the prediction as a string ("up" or "down"), making it easier to use the prediction in the trading logic.

### Using Replit secrets for API keys:

Benefit: Replit secrets allow you to securely store sensitive information such as API keys, preventing accidental exposure or unauthorized access. This ensures that your API keys are not hardcoded into the script, making it safer to share and collaborate on the code.

### To use Replit secrets, follow these steps:

In your Replit environment, click on the padlock icon in the left sidebar to open the "Secrets" tab.

Add a secret with the key OPENAI_API_KEY and the value set to your OpenAI API key.

Add another secret with the key CCXT_API_KEY and the value set to your CCXT API key.

Add a third secret with the key CCXT_SECRET and the value set to your CCXT secret.

These changes aim to simplify the script, make it more efficient, and potentially improve the trading results by combining technical analysis with GPT-3.5-turbo's understanding of market trends