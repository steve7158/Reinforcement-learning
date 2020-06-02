import gym
env=gym.make('MountainCar-v0')
env.reset()
done=False

print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)

while not done:
    action=2
    new_state, reward, done,_=env.step(action)
    print(env.observation_space.high)
    print(env.observation_space.low)
    #print(done)
    #done=False
    env.render()

env.close()
