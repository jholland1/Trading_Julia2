import alpaca_trade_api as tradeapi
import os

os.environ["APCA_API_KEY_ID"] = "PK314M0RYYHY0CFVA6QO"
os.environ["APCA_API_SECRET_KEY"] = "96CHq3BAaEPzz5hWZH2BDdp67RqRo5I0r6xDBQ8e"
os.environ["APCA_API_BASE_URL"] = "https://paper-api.alpaca.markets"
os.environ["APCA_API_DATA_URL"] = "https://data.alpaca.markets"
os.environ["APCA_API_VERSION"] = "v2"
os.environ["APCA_RETRY_MAX"] = "3"
os.environ["APCA_RETRY_CODES"]="429,504"

print(os.environ["APCA_API_KEY_ID"])

api = tradeapi.REST()

# Get our account information.
account = api.get_account()

api_tz = 'UTC'
save_tz = 'America/New_York'

# Check if our account is restricted from trading.
if account.trading_blocked:
    print('Account is currently restricted from trading.')

# Check how much money we can use to open new positions.
print('${} is available as buying power.'.format(account.buying_power))