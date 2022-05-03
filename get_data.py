#! python3
import pandas as pd
import os
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit

key="PK314M0RYYHY0CFVA6QO"
secret_key="96CHq3BAaEPzz5hWZH2BDdp67RqRo5I0r6xDBQ8e"
endpoint="https://paper-api.alpaca.markets"
data_url="https://data.alpaca.markets"

os.environ["APCA_API_KEY_ID"] = key
os.environ["APCA_API_SECRET_KEY"] = secret_key
os.environ["APCA_API_BASE_URL"] = endpoint
os.environ["APCA_API_DATA_URL"] = data_url
os.environ["APCA_API_VERSION"] = "v2"
os.environ["APCA_RETRY_MAX"] = "3"
os.environ["APCA_RETRY_CODES"]="429,504"

data_auth_string='curl -X GET \
    -H "APCA-API-KEY-ID: '+key+'" \
    -H "APCA-API-SECRET-KEY: '+secret_key+'" '\
    +data_url+'/v2/stocks/VOO/bars?timeframe=1Day&start=2021-01-01T00%3A00%3A00-05%3A00'

auth_string = 'curl -X GET \
    -H "APCA-API-KEY-ID: '+key+'" \
    -H "APCA-API-SECRET-KEY: '+secret_key+'" '\
    +endpoint+'/v2/account'

print(auth_string)
os.system(auth_string)
print('\n\n')
print(data_auth_string)
os.system(data_auth_string)
print('\n\n')

print(os.environ["APCA_API_KEY_ID"])

api = REST()

#Data for all variables seems to start at 12/01/2018
fd = pd.Timestamp('2013-3-5', tz='America/New_York').isoformat()
# fd = pd.Timestamp('2021-3-5', tz='America/New_York').isoformat()
td = pd.Timestamp('2022-4-23', tz='America/New_York').isoformat()

#API apparently reports in UTC now, need to add UTC timezone and convert to Eastern
#Not sure if I need to strip UTC or not anymore.
api_tz = 'UTC'
# save_tz = 'America/New_York'
save_tz = 'UTC' #USE THIS - DataFrames.jl not TZ aware, save in UTC and convert to Eastern in Julia later

tickers = ['AAPL','VIXY','VOO','SH','GLD']
# tickers=['AAPL']
tfunit=TimeFrame(1, TimeFrameUnit.Minute)
for ticker in tickers :
    print(ticker)
    gatekeeper = False
    mdata = api.get_bars(ticker, tfunit,start=fd, end=td, adjustment='split', limit=100000).df.tz_localize(None)
    mdata = mdata.tz_localize(api_tz)
    if pd.Timestamp(mdata.index[-1]) < pd.Timestamp(td) :
        gatekeeper = True
    while gatekeeper :
        last_end = mdata.index[-1]
        print(pd.Timestamp(last_end).isoformat())
        temp_data = api.get_bars(ticker, tfunit,start=pd.Timestamp(last_end).isoformat(),end=td,adjustment='split',limit=100000).df.tz_localize(None)
        temp_data = temp_data.tz_localize(api_tz)
        mdata=pd.concat([mdata, temp_data],axis=0)
        if pd.Timestamp(mdata.index[-1]) == last_end or pd.Timestamp(mdata.index[-1]) >= pd.Timestamp(td):
            gatekeeper = False
    mdata.drop_duplicates(inplace=True)
    mdata.dropna(inplace=True)
    ddata = api.get_bars(ticker, TimeFrame.Day, start=fd, end=td,adjustment='split').df.tz_localize(None)
    ddata = ddata.tz_localize(api_tz)
                
    mdata['volume'] = mdata['volume'].astype('int')
    ddata['volume'] = ddata['volume'].astype('int')
    mdata=mdata.tz_convert(save_tz)
    ddata=ddata.tz_convert(save_tz)
    mdata.to_csv('Data/'+ticker+'_min.csv')
    ddata.to_csv('Data/'+ticker+'_day.csv')

    print(mdata.index[-1])

    print(mdata)
    print(ddata)

    plt.figure()
    plt.plot(ddata['open'],label='Day')
    plt.plot(mdata['open'],label='Minute')
    plt.xlabel('Time')
    plt.ylabel('Open ($)')
    plt.title(ticker)
    plt.legend()
    plt.grid()
    plt.savefig('Data/'+ticker+'.png')
    # plt.show()
