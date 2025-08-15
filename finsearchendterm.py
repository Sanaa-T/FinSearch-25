import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

import yfinance as yf

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import statsmodels.api as sm

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

import gym
from gym import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv


SYMBOL = '^CNX100' 
LOOKBACK = 10  
INITIAL_CAPITAL = 100000.0
TRANSACTION_COST_PCT = 0.0005 
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

end_dt = pd.to_datetime('today').normalize()
start_dt = end_dt - pd.Timedelta(weeks=6)

print(f"Downloading {SYMBOL} from {start_dt.date()} to {end_dt.date()}")

df = yf.download(SYMBOL, start=start_dt.strftime('%Y-%m-%d'), end=(end_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d'), progress=False, auto_adjust=True)
if df.empty:
    raise RuntimeError('No data downloaded. Check symbol / internet connection in your environment.')


df = df.asfreq('B').ffill()


def add_features(df):
    df = df.copy()
    df['close'] = df['Close']
    df['return'] = df['close'].pct_change()
    df['logret'] = np.log(df['close'] / df['close'].shift(1))
    df['SMA5'] = df['close'].rolling(5).mean()
    df['SMA10'] = df['close'].rolling(10).mean()
    df['mom_3'] = df['close'].pct_change(3)
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / roll_down
    df['RSI14'] = 100.0 - (100.0 / (1.0 + rs))
    df = df.dropna()
    return df

feat = add_features(df)


split_date = feat.index[int(len(feat) * (4/6))]
train_df = feat[feat.index < split_date].copy()
test_df = feat[feat.index >= split_date].copy()

print('Train range:', train_df.index.min().date(), '->', train_df.index.max().date())
print('Test range :', test_df.index.min().date(), '->', test_df.index.max().date())

class TradingEnv(gym.Env):
    """Simple trading environment. Discrete actions: 0=hold, 1=buy, 2=sell.
    Agent always either fully in cash or fully invested in the index (fractional allowed).
    Reward: change in portfolio value.
    State: last N days of normalized features + current position flag + cash fraction.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, df, lookback=LOOKBACK, initial_capital=INITIAL_CAPITAL, transaction_cost_pct=TRANSACTION_COST_PCT):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.lookback = lookback
        self.initial_capital = initial_capital
        self.transaction_cost_pct = transaction_cost_pct

        self.features = ['close','SMA5','SMA10','mom_3','RSI14']
        self.n_features = len(self.features)

        self.action_space = spaces.Discrete(3)
     
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.lookback * self.n_features + 2,), dtype=np.float32)

        self._reset_internal()

    def _reset_internal(self):
        self.current_step = self.lookback
        self.position = 0 
        self.cash = self.initial_capital
        self.shares = 0.0
        self.portfolio_value = self.initial_capital
        self.history = []

        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaler.fit(self.df[self.features])
        self.scaler = scaler

    def reset(self):
        self._reset_internal()
        return self._get_obs()

    def _get_obs(self):
        window = self.df.loc[self.current_step-self.lookback:self.current_step-1, self.features].copy().values
        window = self.scaler.transform(window)
        obs = window.flatten()
        obs = np.concatenate([obs, [self.position, self.cash / self.initial_capital]])
        return obs.astype(np.float32)

    def step(self, action):
        done = False
        info = {}
        price = self.df.loc[self.current_step, 'close']

        prev_portfolio = self.portfolio_value

        if action == 1 and self.position == 0: 
            spend = self.cash * (1 - self.transaction_cost_pct)
            self.shares = spend / price
            self.cash = 0.0
            self.position = 1
        elif action == 2 and self.position == 1:
            proceeds = self.shares * price * (1 - self.transaction_cost_pct)
            self.cash = proceeds
            self.shares = 0.0
            self.position = 0


        self.portfolio_value = self.cash + self.shares * price

        reward = self.portfolio_value - prev_portfolio

        self.history.append({'step': int(self.current_step), 'price': price, 'action': int(action), 'portfolio': float(self.portfolio_value)})

        self.current_step += 1
        if self.current_step >= len(self.df):
            done = True

        return self._get_obs() if not done else np.zeros_like(self._get_obs()), reward, done, info

    def render(self, mode='human'):
        if len(self.history):
            last = self.history[-1]
            print(f"Step: {last['step']}, Price: {last['price']:.2f}, Action: {last['action']}, Portfolio: {last['portfolio']:.2f}")

def compute_metrics(portfolio_values, daily_index_returns):
    pv = np.array(portfolio_values)
    cum_return = pv[-1] / pv[0] - 1
    dr = np.diff(pv) / pv[:-1]
    vol = np.std(dr) * np.sqrt(252)
    sharpe = (np.mean(dr) / (np.std(dr) + 1e-9)) * np.sqrt(252)
    running_max = np.maximum.accumulate(pv)
    drawdowns = (pv - running_max) / running_max
    max_dd = drawdowns.min()
    return {'cumulative_return': cum_return, 'annualized_vol': vol, 'sharpe': sharpe, 'max_drawdown': max_dd}

buy_and_hold_values = []
initial_price = test_df['close'].iloc[0]
shares_bh = INITIAL_CAPITAL / initial_price
for p in test_df['close'].values:
    buy_and_hold_values.append(shares_bh * p)

def arima_strategy(train_df, test_df):
    history_prices = train_df['close'].copy().values
    portfolio = INITIAL_CAPITAL
    position = 0
    shares = 0.0
    values = []

    for t_idx in range(len(test_df)):
        model = sm.tsa.SARIMAX(history_prices, order=(5,1,0), enforce_stationarity=False, enforce_invertibility=False)
        res = model.fit(disp=False)
        fc = res.get_forecast(steps=1)
        pred = fc.predicted_mean[0]

        price_today = test_df['close'].iloc[t_idx]

        if pred > history_prices[-1] and position == 0:
            spend = portfolio * (1 - TRANSACTION_COST_PCT)
            shares = spend / price_today
            portfolio = 0.0
            position = 1
        elif pred <= history_prices[-1] and position == 1:
            proceeds = shares * price_today * (1 - TRANSACTION_COST_PCT)
            portfolio = proceeds
            shares = 0.0
            position = 0

        total = portfolio + shares * price_today
        values.append(total)

        history_prices = np.append(history_prices, price_today)

    return values

arima_values = arima_strategy(train_df, test_df)

def make_lstm_data(df, lookback=LOOKBACK):
    X, y = [], []
    features = ['close','SMA5','SMA10','mom_3','RSI14']
    scal = MinMaxScaler()
    scaled = scal.fit_transform(df[features])
    for i in range(lookback, len(df)):
        X.append(scaled[i-lookback:i])
        y.append((df['close'].iloc[i] / df['close'].iloc[i-1]) - 1)
    return np.array(X), np.array(y), scal

X_train, y_train, scaler_train = make_lstm_data(train_df)
X_test, y_test, scaler_test = make_lstm_data(pd.concat([train_df.tail(LOOKBACK), test_df]))

model_lstm = Sequential()
model_lstm.add(LSTM(64, input_shape=(LOOKBACK, X_train.shape[2]), return_sequences=False))
model_lstm.add(Dropout(0.2))
model_lstm.add(Dense(1, activation='linear'))
model_lstm.compile(optimizer=Adam(1e-3), loss='mse')

model_lstm.fit(X_train, y_train, epochs=50, batch_size=8, verbose=0)

lstm_preds = model_lstm.predict(X_test)

lstm_values = []
portfolio = INITIAL_CAPITAL
position = 0
shares = 0.0
test_prices = pd.concat([train_df.tail(LOOKBACK), test_df])['close'].iloc[LOOKBACK:].values
for i, pred in enumerate(lstm_preds.flatten()):
    price_today = test_prices[i]
    if pred > 0 and position == 0:
        spend = portfolio * (1 - TRANSACTION_COST_PCT)
        shares = spend / price_today
        portfolio = 0.0
        position = 1
    elif pred <= 0 and position == 1:
        proceeds = shares * price_today * (1 - TRANSACTION_COST_PCT)
        portfolio = proceeds
        shares = 0.0
        position = 0
    total = portfolio + shares * price_today
    lstm_values.append(total)

train_env = DummyVecEnv([lambda: TradingEnv(train_df, lookback=LOOKBACK)])

model = DQN('MlpPolicy', train_env, verbose=1, learning_rate=1e-4, buffer_size=50000, learning_starts=1000)
model.learn(total_timesteps=5000)

def run_rl_on_df(model, df):
    env = TradingEnv(df, lookback=LOOKBACK)
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(int(action))

    pv = [h['portfolio'] for h in env.history]
    return pv

try:
    rl_values = run_rl_on_df(model, pd.concat([train_df.tail(LOOKBACK), test_df]).reset_index(drop=True))
except Exception as e:
    print('RL evaluation error (likely due to env index mismatch):', e)
    rl_values = []

summary = {}
summary['buy_and_hold'] = compute_metrics(buy_and_hold_values, None)
summary['arima'] = compute_metrics(arima_values, None)
summary['lstm'] = compute_metrics(lstm_values, None)
if rl_values:
    summary['rl_dqn'] = compute_metrics(rl_values, None)

print('\nPerformance summary (on test period):')
for k,v in summary.items():
    print(f"\n{k}\n  cumulative_return: {v['cumulative_return']:.4f}  sharpe: {v['sharpe']:.4f}  vol: {v['annualized_vol']:.4f}  max_dd: {v['max_drawdown']:.4f}")

plt.figure(figsize=(10,6))
plt.plot(test_df.index, buy_and_hold_values, label='Buy & Hold')
if arima_values: plt.plot(test_df.index, arima_values, label='ARIMA')
if lstm_values: plt.plot(test_df.index, lstm_values, label='LSTM')
if rl_values:
    rl_idx = test_df.index[:len(rl_values)]
    plt.plot(rl_idx, rl_values, label='RL(DQN)')
plt.legend()
plt.title('Portfolio value during test period')
plt.show()

res_df = pd.DataFrame({'date': test_df.index})
res_df['buy_hold'] = buy_and_hold_values
res_df['arima'] = pd.Series(arima_values)
res_df['lstm'] = pd.Series(lstm_values)
if rl_values:
    res_df['rl'] = pd.Series(rl_values)

res_df.to_csv('results_nifty100_6weeks.csv', index=False)
print('Results saved to results_nifty100_6weeks.csv')
print('\nDone.\n')