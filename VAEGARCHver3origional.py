



import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['figure.figsize'] = [9, 6]
import sp500
# Set random seed for reproducible results 

#from IPython import display
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
#from tensorflow import keras
import keras
import time
from stable_baselines3 import PPO



print(tf.__version__)
print(keras.__version__)


num_history = 90
num_predict = 1
num_latent = 15

Predict = True
if Predict:
    num_input = num_history
    num_output = num_history + num_predict
else:
    num_input = num_history
    num_output = num_history

# Load data and set up dataset
data = sp500.sp500(num_output)
#data = garch.garch11(num_output)
r = tf.constant(data.r, dtype=tf.float32)
dataset = tf.data.Dataset.from_tensor_slices(r)
dataset = dataset.shuffle(dataset.cardinality().numpy())
train_num = round(0.75*dataset.cardinality().numpy())
test_num = dataset.cardinality().numpy() - train_num
train_dataset = dataset.take(train_num).batch(32, drop_remainder=True)
test_dataset = dataset.skip(train_num).batch(32, drop_remainder=True)

# Clear all previously registered custom objects
keras.saving.get_custom_objects().clear()

@tf.keras.utils.register_keras_serializable()

class VAE(keras.Model):
    def __init__(self, input_dim, latent_dim, output_dim, predict, **kwargs):
        super().__init__(**kwargs)

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.predict = predict


        def get_config(self):
            return {
        "input_dim": self.input_dim,
        "latent_dim": self.latent_dim,
        "output_dim": self.output_dim,
             }


        if predict:
            dec1_size = 67
            dec2_size = 267
        else:
            dec1_size = 50
            dec2_size = 200
        
        self.encoder = tf.keras.Sequential(
            [
            tf.keras.layers.InputLayer(shape=(input_dim,)),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(latent_dim + latent_dim)
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
            tf.keras.layers.InputLayer(shape=(latent_dim,)),
            tf.keras.layers.Dense(units=dec1_size, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=dec2_size, activation=tf.nn.relu),
            tf.keras.layers.Dense(units=output_dim)
            ]
        )


    def call(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        #print(">>> z shape:", z.shape)  # ← PLACE IT HERE
        x_hat = self.decoder(z)
        return x_hat, z, mean, logvar

       

    def encode(self, x):
        x = x[:, :self.input_dim]  # ensure correct input size
        x_encoded = self.encoder(x)
        mean, logvar = tf.split(x_encoded, num_or_size_splits=2, axis=1)
        return mean, logvar

    
    
    
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    
    def decode(self, z):
        x_hat = self.decoder(z)
        return x_hat    
    

    



    def get_config(self):
        config = super().get_config()
        config.update({ "input_dim" : self.input_dim,
                        "latent_dim" : self.latent_dim,
                        "output_dim" : self.output_dim,
                        "predict" : self.predict })
        return config

    @classmethod
    def from_config(cls, config):
        input_dim = config.pop("input_dim")
        latent_dim = config.pop("latent_dim")
        output_dim = config.pop("output_dim")
        predict = config.pop("predict")
        return cls(input_dim, latent_dim, output_dim, predict, **config)
  

# Convert the TF tensor r into a numpy array for the environment.
all_prices = data.r  # already numpy from your sp500 loader
# all_prices = r.numpy()  # alternative



@tf.function    
def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(32, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)









#def call(self, x):
#        mean, logvar = self.encode(x[:,:self.input_dim])
#        z = self.reparameterize(mean, logvar)
#        x_hat = self.decode(z)
#        return x_hat, z, mean, logvar




# Instantiate an optimizer
optimizer = tf.keras.optimizers.Adam(1e-4)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2.0 * np.pi)
    return tf.reduce_sum(-0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis)





def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_hat = model.decode(z)

    # Compute log-probabilities
    logpx_z = log_normal_pdf(x, x_hat, 0.0)
    logpz = log_normal_pdf(z, 0.0, 0.0)
    logqz_x = log_normal_pdf(z, mean, logvar)

    return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    

    logpx_z = log_normal_pdf(x, x_hat, 0.0)
    logpz = log_normal_pdf(z, 0.0, 0.0)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)






def compute_return(today_price, tomorrow_price):   #Build RL Envronment
    """
    Computes next-day percent return.
    Example: 1% return → 0.01
    """
    return (tomorrow_price - today_price) / today_price


def compute_reward(action, realized_return):
    """
    action: 1=Buy, -1=Sell, 0=Hold
    realized_return: actual next-day percent change
    """
    if action == 1:   # BUY
        return realized_return

    elif action == -1:  # SELL/SHORT
        return -realized_return

    else:  # HOLD
        return 0.0



class SP500VAEEnv(gym.Env):                              #PPO class setting up the environment
    """
    Simple 1-step-ahead trading environment:
    - Observations: [z from encoder, forecast from decoder]
    - Actions: 0=HOLD, 1=LONG, 2=SHORT
    - Reward: position * next_day_return
    """
    metadata = {"render_modes": []}

    def __init__(self, prices, model, window_size=90):
        super().__init__()
        self.prices = np.array(prices, dtype=np.float32)  # 1D array of prices or returns
        self.model = model
        self.window_size = window_size

        # Index where the current window ends (we predict t+1 from [t-window+1...t])
        self.current_idx = window_size

     
        obs_dim = model.latent_dim + 1

        # Observation space: continuous vector (z + forecast)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )



        # Action space: HOLD, LONG, SHORT
        self.action_space = spaces.Discrete(3)

        # Position: -1 (short), 0 (flat), 1 (long)
        self.position = 0
        self.prev_price = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # reset to somewhere safely away from the very end
        self.current_idx = self.window_size
        self.position = 0

        # initial window [0 : window_size]
        window = self.prices[self.current_idx - self.window_size : self.current_idx]
        z, forecast_val, state_vec = make_vae_state(self.model, window)

        # store last price for returns
        self.prev_price = float(self.prices[self.current_idx - 1])

        obs = state_vec.astype(np.float32)
        info = {
            "z": z,
            "forecast": forecast_val,
            "price": self.prev_price,
        }
        return obs, info

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        # Map action to position
        if action == 0:   # HOLD
            position = 0
        elif action == 1: # LONG
            position = 1
        else:             # SHORT
            position = -1

        # Move to next day
        # We will compute return from prev_price -> current_price
        if self.current_idx >= len(self.prices) - 1:
            # nothing more to step through
            done = True
            truncated = True
            reward = 0.0
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            info = {}
            return obs, reward, done, truncated, info

        price_today = float(self.prices[self.current_idx - 1])
        price_next = float(self.prices[self.current_idx])

        # daily return
        daily_return = (price_next - price_today) / price_today

        # reward is position * realized return
        reward = float(position * daily_return)

        # advance index
        self.current_idx += 1
        done = (self.current_idx >= len(self.prices) - 1)
        truncated = False

        # build next observation window ending at current_idx - 1
        window = self.prices[self.current_idx - self.window_size : self.current_idx]
        z, forecast_val, state_vec = make_vae_state(self.model, window)

        obs = state_vec.astype(np.float32)
        info = {
            "z": z,
            "forecast": forecast_val,
            "price_today": price_today,
            "price_next": price_next,
            "daily_return": daily_return,
            "position": position,
        }
        return obs, reward, done, truncated, info



class VAETradingEnv(gym.Env):
    def __init__(self, vae_model, price_data, history_length=90):
        super(VAETradingEnv, self).__init__()

        self.vae = vae_model
        self.data = price_data
        self.history_length = history_length
        self.current_step = 0

        # Sizes
        self.latent_dim = vae_model.latent_dim
        self.forecast_dim = 1                     # ONLY 1 DAY AHEAD FORECAST
        obs_dim = self.latent_dim + self.forecast_dim

        # Observation space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action space: Hold, Buy, Sell
        self.action_space = spaces.Discrete(3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.history_length
        obs = self._get_obs()
        return obs, {}  # ✅ return tuple (obs, info)



    def _get_obs(self):
        # Get 90‑day window
        past_window = self.data[self.current_step - self.history_length : self.current_step]
        past_window = past_window.astype(np.float32).reshape(1, -1)

        # Encode → z
        mean, logvar = self.vae.encode(past_window)
        z = self.vae.reparameterize(mean, logvar)            # shape (1, latent_dim)

        # Decode → forecast sequence
        x_hat = self.vae.decode(z)                           # shape (1, 91)

        # Extract next‑day forecast ONLY
        forecast = x_hat[:, -1:]                             # shape (1, 1)

        # Flatten and combine
        obs = np.concatenate([z.numpy().flatten(),
                              forecast.numpy().flatten()])

        return obs

    def step(self, action):
        today = self.data[self.current_step]
        tomorrow = self.data[self.current_step + 1]
        today_price = float(self.data[self.current_step][0])
        tomorrow_price = float(self.data[self.current_step + 1][0])

        price_change = float(tomorrow_price - today_price)




        # Reward logic
        if action == 1 and price_change > 0:        # Buy
            reward = 1
        elif action == 2 and price_change < 0:      # Sell
            reward = 1
        elif action == 0 and abs(price_change) < 0.001:


            reward = 0
        else:
            reward = -1
        obs = self._get_obs()

        # Move forward
        self.current_step += 1
        done = self.current_step >= len(self.data) - 2

        terminated = done
        truncated = False   # or add your own truncation logic

        return obs, reward, terminated, truncated, {}




# ===== Step 2: Build a function that maps a price window -> (z, forecast, state) =====

def make_vae_state(vae, window_1d):
    """
    window_1d: 1D numpy array of shape (window_size,)
               e.g., a sequence of daily returns or prices

    Returns:
        z: latent vector from encoder, shape (latent_dim,)
        forecast_value: scalar from decoder (we'll take the last element)
        state_vec: concatenation [z, forecast_value], shape (latent_dim + 1,)
    """
    # Ensure correct shape for the VAE: (batch_size, input_dim)
    x = np.asarray(window_1d, dtype=np.float32).reshape(1, -1)

    # Encoder → latent
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar).numpy()[0]       # shape: (latent_dim,)
    

    # Decoder → reconstruction / forecast
    decoded = vae.decoder(z.reshape(1, -1)).numpy()[0]  # shape: (output_dim,)

    # For now, treat the LAST element as the "1-day forecast"
    forecast_value = float(decoded[-1])

    # Build the RL state: [z, forecast]
    state_vec = np.concatenate([z, np.array([forecast_value], dtype=np.float32)])

    return z, forecast_value, state_vec

class PPOActor(tf.keras.Model):
    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.fc1 = tf.keras.layers.Dense(64, activation='relu')
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.out = tf.keras.layers.Dense(action_dim, activation='softmax')  # 3 actions: Buy, Sell, Hold

    def call(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.out(x)
    

    


def rl_env_step(model, prices, index, action):
    """
    model: trained VAE
    prices: numpy array of historical prices
    index: the current location in the price series
    action: Buy(1)/Sell(-1)/Hold(0)
    """

    # --------------------------------------
    # 1. Build current state (z + forecast)
    # --------------------------------------
    window = prices[index - 90:index]   # 90 = your VAE input size
    z, forecast_val, state_vec = make_vae_state(model, window)

    # --------------------------------------
    # 2. Compute the next-day true return
    # --------------------------------------
    today_price = prices[index]
    tomorrow_price = prices[index + 1]

    realized_return = compute_return(today_price, tomorrow_price)

    # --------------------------------------
    # 3. Compute reward from your action
    # --------------------------------------
    reward = compute_reward(action, realized_return)

    # --------------------------------------
    # 4. Build *next* state for RL
    # --------------------------------------
    next_window = prices[index - 90:index + 1]   # slide forward 1 day
    z2, forecast_val2, next_state_vec = make_vae_state(model, next_window)

    # --------------------------------------
    # 5. Return RL transition
    # --------------------------------------
    return state_vec, reward, next_state_vec













@tf.function
def train_step(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

print(">>> DEBUG CHECK")
print(">>> num_predict =", num_predict)
print(">>> num_output  =", num_output)
#print(">>> output_dim  =", output_dim)


Train =True

if Train:
    model = VAE(num_input, num_latent, num_output, Predict)
    epochs = 500
    for batch in train_dataset.take(1):
        print("Batch shape:", batch.shape)
    for epoch in range(epochs):
        start_time = time.time()
        for i, train_x in enumerate(train_dataset):
            train_step(model, train_x, optimizer)
        end_time = time.time()

        loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            loss(compute_loss(model, test_x))
        elbo = loss.result()
        print('Epoch: {}, Test set ELBO: {}, time elapse for current epoch: {}'
                .format(epoch, elbo, end_time - start_time))

    model.save('vae_keras_dense_predict.keras')
    model.summary()

    
    print(">>> test_x shape:", test_x.shape)

    for batch in test_dataset.take(1):
        sample_window = batch[0]

        z, forecast_val, state_vec = make_vae_state(model, sample_window)
        print("z:", z)
        print("forecast_val:", forecast_val)
        print("state_vec shape:", state_vec.shape)
        








# Load trained model and test it
new_model = keras.saving.load_model('vae_keras_dense_predict.keras')
new_model.trainable = False  # 🔒 Freeze the VAE

# Confirm freezing
for layer in new_model.layers:
    print(layer.name, "trainable:", layer.trainable)


new_model.summary()





for test_x in test_dataset:
    print(">>> test_x shape:", test_x.shape)
    break  # Only need one batch to check






for i, test_x in enumerate(test_dataset):
    #mean, logvar = new_model.encode(test_x)
    # NEW
# We take all rows (:), but only the first 90 columns (:90)
    mean, logvar = new_model.encode(test_x[:, :90])
    print(">>> test_x shape:", test_x.shape)  # ← add this
    z = new_model.reparameterize(mean, logvar)
    x_hat = new_model.decode(z)
    #x_hat, z, mean, logvar = new_model(test_x)

    if False:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
        ax1.plot(mean.numpy().T)
        ax2.plot(logvar.numpy().T)
        ax3.plot(test_x.numpy().T, color='blue')
        ax3.plot(x_hat.numpy().T, color='red')
        plt.show()
        for x, xh in zip(test_x, x_hat):
            fig, ax = plt.subplots()
            ax.plot(x, color='blue')
            ax.plot(xh, color='red')
            ymin, ymax = plt.ylim()
            ax.plot([89.5, 89.5],[ymin, ymax], color='black')
            plt.show()
    else:
        #fig, ax = plt.subplots()
        #ax.plot(test_x[0,:], color='blue', linewidth=10)
        #for j in range(20):
        #    x_hat, z, mean, logvar = new_model(test_x[:1,:])
        #    ax.plot(np.squeeze(x_hat), color='red')
        #ymin, ymax = plt.ylim()
        #ax.plot([89.5, 89.5],[ymin, ymax], color='black')
        #plt.show()

        # NEW (Fixing the shape)
        x_hat, z, mean, logvar = new_model(test_x[:1, :90])
        error = np.mean(np.abs(test_x[0, :] - np.squeeze(x_hat)))
        print(f"Avg reconstruction error: {error:.4f}")

        
        
env = VAETradingEnv(vae_model=new_model, price_data=data.r, history_length=90)




model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
       

