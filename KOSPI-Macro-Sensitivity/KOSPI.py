import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ======================================
# 1. DATA DOWNLOAD FUNCTION
# ======================================

def download_series(ticker, start="2000-01-01"):
    data = yf.download(ticker, start=start, interval="1mo")
    if "Adj Close" in data.columns:
        return data["Adj Close"]
    else:
        return data["Close"]

# Download data
kospi = download_series("^KS11")
sp500 = download_series("^GSPC")
usdkrw = download_series("KRW=X")

# ======================================
# 2. COMPUTE MONTHLY LOG RETURNS
# ======================================

kospi_ret = np.log(kospi / kospi.shift(1))
sp500_ret = np.log(sp500 / sp500.shift(1))
usdkrw_ret = np.log(usdkrw / usdkrw.shift(1))

data = pd.concat([kospi_ret, sp500_ret, usdkrw_ret], axis=1)
data.columns = ["KOSPI", "SP500", "USD_KRW"]
data.dropna(inplace=True)

print(data.head())

# ======================================
# 3. CORRELATION
# ======================================

print("\nCorrelation Matrix:")
print(data.corr())

# ======================================
# 4. FULL SAMPLE REGRESSION
# ======================================

X_full = sm.add_constant(data[["SP500", "USD_KRW"]])
y_full = data["KOSPI"]

full_model = sm.OLS(y_full, X_full).fit()
print(full_model.summary())

full_sample_beta = full_model.params["SP500"]

# ======================================
# 5. SCATTER PLOT
# ======================================

plt.figure()
plt.scatter(data["SP500"], data["KOSPI"])
plt.xlabel("S&P 500 Returns")
plt.ylabel("KOSPI Returns")
plt.title("KOSPI vs S&P 500 Monthly Returns")
plt.savefig("figures/scatter_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# ======================================
# 6. 36-MONTH ROLLING REGRESSION
# ======================================

window = 36
rolling_beta = []
rolling_r2 = []

data.index = pd.to_datetime(data.index)

for i in range(window, len(data) + 1):
    
    temp = data.iloc[i - window:i]
    
    Y = temp["KOSPI"]
    X = sm.add_constant(temp["SP500"])
    
    rolling_model = sm.OLS(Y, X).fit()
    
    rolling_beta.append(rolling_model.params["SP500"])
    rolling_r2.append(rolling_model.rsquared)

rolling_dates = data.index[window - 1:]

rolling_beta_series = pd.Series(rolling_beta, index=rolling_dates)
rolling_r2_series = pd.Series(rolling_r2, index=rolling_dates)

# ======================================
# 7. ROLLING BETA PLOT
# ======================================

plt.figure()
plt.plot(rolling_beta_series)
plt.axhline(y=full_sample_beta, linestyle="--")
plt.axvline(pd.to_datetime("2008-09-01"), linestyle="--")
plt.axvline(pd.to_datetime("2020-03-01"), linestyle="--")
plt.title("36-Month Rolling Beta of KOSPI to S&P 500")
plt.xlabel("Date")
plt.ylabel("Beta")
plt.savefig("figures/rolling_beta.png", dpi=300, bbox_inches="tight")
plt.show()

# ======================================
# 8. ROLLING R-SQUARED PLOT
# ======================================

plt.figure()
plt.plot(rolling_r2_series)
plt.axvline(pd.to_datetime("2008-09-01"), linestyle="--")
plt.axvline(pd.to_datetime("2020-03-01"), linestyle="--")
plt.title("36-Month Rolling R-Squared")
plt.xlabel("Date")
plt.ylabel("R-Squared")
plt.savefig("figures/rolling_R_squared.png", dpi=300, bbox_inches="tight")
plt.show()