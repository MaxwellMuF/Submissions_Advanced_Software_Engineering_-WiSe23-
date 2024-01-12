# Submission:
This is the submission for the task "Exercise Deep Reinforcement Learning: Flappy Bird" of the lecture "Advanced Software Engineering (WiSe23)".

name: Sebastian von Rohrscheidt

matrikel: 102090

# Setting:
Spider was used for training (but VS code for the repository).
1. The hyperparameters of the last training were as follows (Figure 1)
2. The printouts and saves of the model were as follows (Figure 2). A current model was saved every 10k iterations in case the process crashes. A model was saved for every 100k for later analysis.
3. The code was slightly modified for training and for optimizing the hyperparameters (Figure 3). The ideas implemented for this are explained in the next section.

# Training:
To get a better understanding of the flappy bird model, the agent was trained several times and with different hyperparameters:
1. flappy_bird_S-R_100_000: Default setting (given) of the hyperparameterr. This agent is hardly successful, which is probably due to the too small number of iterations (or game rounds). In order to train a more successful agent, the hyperparameters were changed slightly.
2. flappy_bird_S-R_100_000_lr_e-5: Here the learning rate has been made slightly higher. This should allow the agent to learn the required actions more quickly, i.e. it should perform better with the same number of iterations. Unfortunately, this is still not enough to achieve a recognizably better result. In addition, overfitting is encouraged.
3. flappy_bird_S-R_100000_high_lr_eps: After a few more attempts, this is the last setting of hyperparameters. Three changes have been made:
    1. the learning rate with dacay. The agent starts with a high learning rate, which decreases over time (latest: lr*0.1 every 50% of data).
    2. epsilon: The rate of random actions with decay. The exploration rate was strongly increased at the beginning and then a damping (decrease) was built in with an e-function until the final_epsilon was reached at the end of the training.
    3. random_action: The probability of making a jump as a random action has been reduced (again with decay). Initially only 10% of random actions are a jump. With a further decay, the rate will be 50/50 again at the end (or for the last 30% of the training).
Thus, an agent could be trained that could pass 1-2 or more tunnels (after 100k training iterations), at least in some scenarios. However, it is suspected that this agent is overtrained and it remains to be seen whether he can ever master the perfect game and whether he needs less than 1m iterations of training to do so.

# Link to Git Repo
https://github.com/MaxwellMuF/Submissions_Advanced_Software_Engineering_-WiSe23-/tree/main/Sub_2_flappy_bird