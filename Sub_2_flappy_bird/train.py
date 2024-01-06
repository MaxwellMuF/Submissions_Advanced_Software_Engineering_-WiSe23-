"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import os
import shutil
from random import random, randint, sample

import numpy as np
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

# Change the working directory to a new directory (replace with the path you want)
new_working_directory = r"C:\Users\sebas\Documents\Data_Science\WS_23_24\Fortgeschrittene Softwaretechnik\Submissions_Advanced_Software_Engineering_WiSe23\Sub_2_flappy_bird"
os.chdir(new_working_directory)
print("get here")
# Get and print the new current working directory
new_current_working_directory = os.getcwd()
print(f"The new current working directory is: {new_current_working_directory}")

from src.deep_q_network import DeepQNetwork
from src.flappy_bird import FlappyBird
from src.utils import pre_processing


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird""")
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--batch_size", type=int, default=32, help="The number of images per batch")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adam"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--initial_epsilon", type=float, default=0.9)
    parser.add_argument("--final_epsilon", type=float, default=1e-2)
    parser.add_argument("--num_iters", type=int, default=200_000)
    parser.add_argument("--replay_memory_size", type=int, default=50_000,
                        help="Number of epoches between testing phases")
    parser.add_argument("--log_path", type=str, default="tensorboard")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    model = DeepQNetwork()
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    writer = SummaryWriter(opt.log_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    criterion = nn.MSELoss()
    game_state = FlappyBird()
    image, reward, terminal = game_state.next_frame(0)
    image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)
    image = torch.from_numpy(image)
    if torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    replay_memory = []
    iter = 0
    interval_end = -np.log(opt.final_epsilon / opt.initial_epsilon) # project or transform interval 
    x_values = np.linspace(0, interval_end, opt.num_iters)          # of iteration on e-function
    # multiply lr by 1e4 and make decay ever 25% if iteration two lines below
    opt.lr = opt.lr * 1e4
    while iter < opt.num_iters:
        #  make lr decay ever 25%
        if iter % (opt.num_iters/4) == 0:
            opt.lr = opt.lr /10
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
        prediction = model(state)[0]
        # Exploration or exploitation
        epsilon = opt.initial_epsilon*np.exp(-x_values[iter]) # Exponential decay of the exploration-rate
        u = random()
        random_action = u <= epsilon
        if random_action:
            print("Perform a random action")
            # The following decay reduces the initially high probability of not jumping during a random action
            no_act_prob = 9 - int(iter*10 / opt.num_iters) # decay of no act prob [0] every 10% of iter
            no_act_prob = max(no_act_prob,1)
            # random action by sampling list of zeros and one one, e.g. [0,0,0,1]  
            action = sample([0]*no_act_prob + [1], 1)[0] # start with 90% [0] and 10% [1] end with 50% [0] and 50% [1]
                                                         # sequence of no_act_prob: [9,8,7,6,5,4,3,2,1,1]
        else:
            try:
                action = torch.argmax(prediction)[0]
            except:
                action = torch.argmax(prediction).item()

        next_image, reward, terminal = game_state.next_frame(action)
        next_image = pre_processing(next_image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size,
                                    opt.image_size)
        next_image = torch.from_numpy(next_image)
        if torch.cuda.is_available():
            next_image = next_image.cuda()
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]
        replay_memory.append([state, action, reward, next_state, terminal])
        if len(replay_memory) > opt.replay_memory_size:
            del replay_memory[0]
        batch = sample(replay_memory, min(len(replay_memory), opt.batch_size))
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = zip(*batch)

        state_batch = torch.cat(tuple(state for state in state_batch))
        action_batch = torch.from_numpy(
            np.array([[1, 0] if action == 0 else [0, 1] for action in action_batch], dtype=np.float32))
        reward_batch = torch.from_numpy(np.array(reward_batch, dtype=np.float32)[:, None])
        next_state_batch = torch.cat(tuple(state for state in next_state_batch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            next_state_batch = next_state_batch.cuda()
        current_prediction_batch = model(state_batch)
        next_prediction_batch = model(next_state_batch)

        y_batch = torch.cat(
            tuple(reward if terminal else reward + opt.gamma * torch.max(prediction) for reward, terminal, prediction in
                  zip(reward_batch, terminal_batch, next_prediction_batch)))

        q_value = torch.sum(current_prediction_batch * action_batch, dim=1)
        optimizer.zero_grad()
        # y_batch = y_batch.detach()
        loss = criterion(q_value, y_batch)
        loss.backward()
        optimizer.step()

        state = next_state
        iter += 1
        print("Iteration: {}/{}, Action: {}, Loss: {}, Epsilon {}, Reward: {}, Q-value: {}".format(
            iter + 1,
            opt.num_iters,
            action,
            loss,
            epsilon, reward, torch.max(prediction)))
        writer.add_scalar('Train/Loss', loss, iter)
        writer.add_scalar('Train/Epsilon', epsilon, iter)
        writer.add_scalar('Train/Reward', reward, iter)
        writer.add_scalar('Train/Q-value', torch.max(prediction), iter)
        if (iter+1) % 100_000 == 0:
            torch.save(model, "{}/flappy_bird_S-R_{}_high_lr_eps".format(opt.saved_path, iter+1))
        if (iter+1) % 10_000 == 0:
            torch.save(model, "{}/flappy_bird_S-R_current_high_lr_eps".format(opt.saved_path))
    torch.save(model, "{}/flappy_bird".format(opt.saved_path))


if __name__ == "__main__":
    opt = get_args()
    train(opt)
