import pandas as pd
import os
import time
from datetime import datetime

path = r"C:\Users\vveen\OneDrive\Documents\projects_personal\stocks\data\intraQuarter"


def Key_Stats(gather="Total Debt/Equity (mrq)"):
    statspath = path+'/_KeyStats'
    stock_list = [x[0] for x in os.walk(statspath)]
    df = pd.DataFrame(columns = ['Date','Unix','Ticker','DE Ratio'])
    
    # sp500_df = pd.read_csv(r"C:\Users\vveen\Documents\personal_projects\stocks\YAHOO-INDEX_GSPC.csv")
    sp500_df = pd.read_csv(r'C:\Users\vveen\Documents\personal_projects\stocks\YAHOO-INDEX_GSPC.csv')
    # print(stock_list)
    for each_dir in stock_list[1:]:
        each_file = os.listdir(each_dir)
        ticker = each_dir.split("\\")[1]
        if len(each_file) > 0:
            for file in each_file:
                date_stamp = datetime.strptime(file, '%Y%m%d%H%M%S.html')
                unix_time = time.mktime(date_stamp.timetuple())
                full_file_path = each_dir+'/'+file
                source = open(full_file_path,'r').read()
                try:
                    value = float(source.split(gather + ':</td><td class="yfnc_tabledata1">')[1].split('</td>')[0])
                    # time.sleep(15)
                    try: 
                        sp500_date = datetime.fromtimestamp(unix_time).strftime('%Y-%m-%d')
                        row = sp500_df[sp500_df["Date"] == sp500_date]
                        sp500_value = float(row["Adj Close"])
                    except:
                        sp500_date = datetime.fromtimestamp(unix_time-259200).strftime('%Y-%m-%d')
                        row = sp500_df[sp500_df["Date"] == sp500_date]
                        sp500_value = float(row["Adj Close"])


                    stock_price = float(source.split('</small><big><b>')[1].split('</b></big>')[0])
                    print("stock_price:",stock_price, "ticker:", ticker)

                    df = df.append({'Date': date_stamp, 'Unix':unix_time, 'Ticker':ticker, 'DE Ration':value}, ignore_index = True)
                except Exception as e:
                    pass
                    
    save = gather.replace(' ','').replace(')', '').replace('(', '').replace('/', '')+('.csv')
    print(save)
    df.to_csv(save)

Key_Stats()


