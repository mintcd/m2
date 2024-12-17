import numpy as np
import gym
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# example from A. Géron's Book
# CartPole with Q-Learning

# load the environnement cartPole
env = gym.make('CartPole-v0')

input_shape = env.observation_space.shape
n_outputs = env.action_space.n

# Q-value function replaced by a neural network
# 2 layers
model = keras.models.Sequential([
    keras.layers.Dense(32, activation="elu", input_shape=input_shape),
    keras.layers.Dense(32, activation="elu"),
    keras.layers.Dense(n_outputs)
    ])

# Epsilon-greedy policy to keep exploring the MDP
def epsilon_greedy_policy(state, epsilon = 0):
    if np.random.rand() < epsilon:
        # choose
        return np.random.randint(2)
    else:
        Q_values = model.predict(state[np.newaxis])
        # breakpoint()
        return np.argmax(Q_values[0])

# utilisation d'une liste "deque" pour la création de la mémoire de rejeu
from collections import deque

# Create a replay memory with deque
replay_memory = deque(maxlen = 2000)


def sample_experiences(batch_size):
    #  get a random samples 
    # generate a set of random inidces of size batch_size
    indices = np.random.randint(len(replay_memory), size=batch_size)
    # get the sample from replay memory
    batch = [replay_memory[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones

def play_one_step(env, state, epsilon):
    # get the action to play
    action = epsilon_greedy_policy(state, epsilon)
    # apply the action, get the next state and the reward
    next_state, reward, done, info = env.step(action)
    # add this step into replay memory
    replay_memory.append((state, action, reward, next_state, done))
    return next_state, reward, done, info



batch_size = # 32
discount_rate =  # 0.94
# learning rate
Alpha = # 0.02

# optimizer to use
optimizer = keras.optimizers.Adam(learning_rate= Alpha)
# loss function
loss_fn = keras.losses.mean_squared_error

def training_step(batch_size):
    # get a sample of experiences
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    # call the Neural network to predict the Q-values for the next states
    next_Q_values = model.predict(next_states)
    # compute the max value
    max_next_Q_values = np.max(next_Q_values, axis=1)
    # ... and the target Q-values
    target_Q_values = (rewards +
                       (1 - dones) * discount_rate * max_next_Q_values)
    target_Q_values = target_Q_values.reshape(-1, 1)

    # mask to remove Q-values for the not played action
    mask = tf.one_hot(actions, n_outputs)
    # breakpoint()
    # compute the gradient with tensorflow automatic differentiation
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        # loss is the mean square error between Q-values and target-Q-values
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    # optimize the loss : compute the gradients automatically
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


# train the model !
SEED = 33
env.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

rewards = [] 
best_score = 0

Nstep = 400

for episode in range(600):
    # reset the environnement
    obs = env.reset()  

    # decrease the value of epsilon through iterations
    epsilon = # 0.1 # max(1 - episode / 500, 0.01)

    # play the environnement for some steps
    for step in range(Nstep):
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        if done:
            break

    # Keep the best weights 
    rewards.append(step) 
    if step >= best_score: 
        print("best_score: ",best_score)
        best_weights = model.get_weights() 
        best_score = step 

    print("\rEpisode: {}, Steps: {}, eps: {:.3f}".format(episode, step + 1, epsilon), end="") # Not shown
    # after some episode, we begin to train the model
    if episode > 50:
        training_step(batch_size)

model.set_weights(best_weights)


# Plot the reward evolution during the learning process
plt.figure(figsize=(8, 4))
plt.plot(rewards)
plt.xlabel("Episode", fontsize=14)
plt.ylabel("Sum of rewards", fontsize=14)
plt.show()


env.seed(SEED)
state = env.reset()


# Environment simulation 
frames = []

for step in range(Nstep):
    action = epsilon_greedy_policy(state)
    state, reward, done, info = env.step(action)
    if done:
        break
    img = env.render(mode="rgb_array")
    frames.append(img)
    
#plot_animation(frames)
