from datetime import datetime, date

import os, sys

def get_all_stock_data():
    os.system('rm -rf marcap')
    os.system('git clone "https://github.com/FinanceData/marcap.git" marcap')
    from marcap import marcap_data

    df = marcap_data('1995-05-02', datetime.today().strftime('%Y-%m-%d'))

    return df







