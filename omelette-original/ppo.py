from collections import deque, namedtuple
from LambdaLang import LambdaLang
from PropLang import PropLang
from rejoice.lib import Language

import torch_geometric as pyg
import argparse
import os
import random
import time
from distutils.util import strtobool
from typing import Union
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from MathLang import MathLang
from LambdaLang import LambdaLang
from rejoice import envs, EGraph
import time

from rejoice.networks import SAGENetwork, GATNetwork, GCNNetwork, GraphTransformerNetwork, GINNetwork


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, cuda will be enabled by default")

    # omelette-specific configuration
    parser.add_argument("--mode", type=str, choices=["single_task_sat", "single_task_explodes", "bc", "multitask"], default="single_task_sat")
    parser.add_argument("--expr-str", type=str, default=None, help="Expression to run")
    parser.add_argument('--expr-list', nargs='+', default=[], help="list of expressions to multitask learn")

    parser.add_argument("--agent-weights-path", type=str, default=None,
                        help="Whether or not to pretrain the value and policy networks") 
    parser.add_argument("--multitask-count", type=int, default=16,
                        help="the number of tasks to generate for multitask eval")
    parser.add_argument("--print-actions", type=bool, default=False,
                        help="print the (action, reward) tuples that make up each episode")
    parser.add_argument("--pretrained-weights-path", type=str, default=None,
                        help="Whether or not to pretrain the value and policy networks")
    parser.add_argument("--lang", type=str, default="PROP",
                        help="The language to use. One of PROP, MATH, TENSOR.")
    parser.add_argument("--use-action-mask", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, action masking is enabled")
    parser.add_argument("--use-edge-attr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, use edge attributes denoting difference between e-class member and e-node child edges")
    parser.add_argument("--use-shrink-action", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="if toggled, include the shrink-and-reexpand action in the action-space")
    parser.add_argument("--node-limit", type=int, default=10_000,
                        help="egraph node limit")
    parser.add_argument("--num-egg-iter", type=int, default=7,
                        help="number of iterations to run egg for")
    parser.add_argument("--max-episode-steps", type=int, default=10,
                        help="the maximum number of steps in any episode")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="egraph-v0",
                        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=1_000_000,
                        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=32,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128, 
                        help="the number of steps to run in each environment per policy rollout")

    # Below - these shouldn't be changed much.
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
                        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
                        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
                        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
                        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
                        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
                        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
                        help="the target KL divergence threshold")

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args



def make_env(env_id, seed, idx: int, input_expr, run_name: str, max_episode_steps: int, lang_name: str, use_shrink_action: bool, node_limit: int, num_egg_iter: int, mode: str, multitask_count = 0):

    def run_egg(lang: Language, expr):
        print(f"running egg for env {idx}", "expr", expr)
        first_stamp = int(round(time.time() * 1000))
        egraph = EGraph()
        egraph.add(expr)
        stop_reason, num_applications, num_enodes, num_eclasses = egraph.run(lang.rewrite_rules(), iter_limit=num_egg_iter, node_limit=node_limit)
        print(stop_reason, "num_applications", num_applications, "num_enodes", num_enodes, "num_eclasses", num_eclasses)
        best_cost, best_expr = egraph.extract(expr)
        second_stamp = int(round(time.time() * 1000))
        # Calculate the time taken in milliseconds
        time_taken = second_stamp - first_stamp
        # egraph.graphviz("egg_best.png")
        print(f"env {idx} egg best cost:", best_cost, "in",
              f"{time_taken}ms", "best expr: ", best_expr)

    def thunk():
        # TODO: refactor so the lang and expr can be passed in
        lang = get_lang_from_str(lang_name)

        if input_expr is None:
            if mode == "single_task_sat":
                single_task_exprs = lang.get_single_task_exprs()
                expr = single_task_exprs.saturatable
                if idx == 1:
                    run_egg(lang, expr)
            elif mode == "single_task_explodes":
                single_task_exprs = lang.get_single_task_exprs()
                expr = single_task_exprs.explodes
                if idx == 1:
                    run_egg(lang, expr)
            elif mode == "multitask":
                tasks = lang.get_multi_task_exprs()
                expr = tasks[idx % len(tasks)]
        else:
            expr = lang.eval_expr(input_expr)

        env = gym.make(env_id, disable_env_checker=True, lang=lang, expr=expr, use_shrink_action=use_shrink_action, node_limit=node_limit)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed)
        env.action_space.seed(seed)
        return env

    return thunk


def get_lang_from_str(name: str) -> Language:
    if name in ["PROP", "PropLang"]:
        return PropLang()
    elif name in ["MATH", "MathLang"]:
        return MathLang()
    elif name in ["LAMBDA", "LambdaLang"]:
        return LambdaLang()


class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, mask=None, device=torch.device("cuda")):
        self.mask = mask
        self.device = device
        if self.mask is None:
            super(CategoricalMasked, self).__init__(
                probs, logits, validate_args)
        else:
            self.mask = mask.type(torch.BoolTensor).to(device)
            # make probabilities of invalid actions impossible
            logits = torch.where(self.mask, logits,
                                 torch.tensor(-1e+8).to(device))
            super(CategoricalMasked, self).__init__(
                probs, logits, validate_args)

    def entropy(self):
        if self.mask is None:
            return super(CategoricalMasked, self).entropy()

        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.mask, p_log_p,
                              torch.tensor(0.).to(self.device))
        return -p_log_p.sum(-1)


class PPOAgent(nn.Module):
    def __init__(self, n_actions: int, n_node_features: int, weights_path=None, use_dropout=False, use_edge_attr=True, device=torch.device("cuda")):
        super().__init__()
        print("use_edge_attr", use_edge_attr, "use_dropout", use_dropout, "weights_path", weights_path)
        self.device = device
        self.critic = GATNetwork(num_node_features=n_node_features,
                                 n_actions=1,
                                 n_layers=3,
                                 hidden_size=64,
                                 dropout=(0.3 if use_dropout else 0.0),
                                 use_edge_attr=use_edge_attr,
                                 out_std=1.)
        self.actor = GATNetwork(num_node_features=n_node_features,
                                n_actions=n_actions,
                                n_layers=3,
                                hidden_size=64,
                                dropout=(0.3 if use_dropout else 0.0),
                                use_edge_attr=use_edge_attr,
                                out_std=0.001)  # make probability of each action similar to start with

        if weights_path is not None:
            self.actor.load_state_dict(torch.load(weights_path))
            keys_vin = torch.load(weights_path)
            current_model = self.critic.state_dict()
            new_state_dict = {k: v if v.size() == current_model[k].size() else current_model[k] for k, v in zip(
                current_model.keys(), keys_vin.values())}
            self.critic.load_state_dict(new_state_dict, strict=False)

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None, invalid_action_mask=None):
        logits = self.actor(x)

        if invalid_action_mask is not None:
            probs = CategoricalMasked(
                logits=logits, mask=invalid_action_mask, device=self.device)
        else:
            probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        return action, probs.log_prob(action), probs.entropy(), self.critic(x)

class DictObj:
    def __init__(self, in_dict:dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(self, key, [DictObj(x) if isinstance(x, dict) else x for x in val])
            else:
                setattr(self, key, DictObj(val) if isinstance(val, dict) else val)

def run_ppo(**kwargs):
    all_args = vars(parse_args()) | kwargs
    args = DictObj(all_args)
    torch.cuda.empty_cache()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"ppo_logs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % (
            "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    writer.add_text(
        "expression",
        str(args.expr_str)
    )

    writer.add_text(
        "costs/max",
        str(args.max_cost)
    )
    writer.add_text(
        "costs/egg",
        str(args.egg_cost)
    )

    writer.add_text(
        "egg_expr",
        str(args.egg_expr)
    )

    # Seed random number generators for reproducible results
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    print("lang", args.lang, "device:", device, "use_shrink", args.use_shrink_action)
    lang = get_lang_from_str(args.lang)

    weights_output_path = "ppo_agent_weights"
    if not os.path.exists(weights_output_path):
        os.makedirs(weights_output_path)

    # env setup
    print("spawning envs")
    if args.num_envs == 1:
        envs = gym.vector.SyncVectorEnv([make_env(env_id=args.env_id, input_expr=args.expr_str, seed=args.seed + i, idx=i, run_name=run_name,
                    max_episode_steps=args.max_episode_steps, lang_name=args.lang, use_shrink_action=args.use_shrink_action, node_limit=args.node_limit, num_egg_iter=args.num_egg_iter, mode=args.mode, multitask_count=args.multitask_count) for i in range(args.num_envs)])
    else:
        envs = gym.vector.AsyncVectorEnv(
            [make_env(env_id=args.env_id, input_expr=args.expr_str, seed=args.seed + i, idx=i, run_name=run_name,
                    max_episode_steps=args.max_episode_steps, lang_name=args.lang, use_shrink_action=args.use_shrink_action, node_limit=args.node_limit, num_egg_iter=args.num_egg_iter, mode=args.mode, multitask_count=args.multitask_count) for i in range(args.num_envs)],
            shared_memory=False,
            copy=False
        )

    action_names = [r[0] for r in lang.all_rules()] + ["end"]

    if args.use_shrink_action:
        action_names.append("shrink")

    agent = PPOAgent(
        n_node_features=envs.single_observation_space.num_node_features, n_actions=envs.single_action_space.n, weights_path=args.pretrained_weights_path, use_dropout=False, use_edge_attr=args.use_edge_attr, device=device).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    if args.agent_weights_path is not None:
        print("Loading learned weights from RL agent")
        agent.load_state_dict(torch.load(args.agent_weights_path))

    # ALGO Logic: Storage setup
    obs = np.empty(args.num_steps, dtype=object)
    actions = torch.zeros((args.num_steps, args.num_envs) +
                          envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    invalid_action_masks = torch.zeros((args.num_steps, args.num_envs) +
                                       (envs.single_action_space.n,)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    # Intial observation
    next_obs = pyg.data.Batch.from_data_list(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    for update in range(1, num_updates + 1):
        # Annealing the learning rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Policy rollout across envs
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs

            if update == 1:
                envs.call("set_global_step", step)
            # nvs.set_attr("global_step", global_step)
            # add the batch of 4 env observations to the obs list at index step
            obs[step] = next_obs
            dones[step] = next_done

            if args.use_action_mask:
                invalid_action_masks[step] = next_obs.action_mask.reshape(
                    (args.num_envs, envs.single_action_space.n))

            # log the action, logprob, and value for this step into our data storage
            with torch.no_grad():  # no grad b/c we're just rolling out, not training
                if args.use_action_mask:
                    action, logprob, _, value = agent.get_action_and_value(
                        next_obs, invalid_action_mask=invalid_action_masks[step])
                else:
                    action, logprob, _, value = agent.get_action_and_value(
                        next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            # execute the chosen action
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            # log the reward into data storage at this step
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_done = torch.Tensor(done).to(device)
            # convert next obs to a pytorch geometric batch
            next_obs = pyg.data.Batch.from_data_list(next_obs).to(device)


            if "episode" in info.keys():
                for env_ind, env_ep_info in enumerate(info["episode"]):
                    if env_ep_info is not None:
                        writer.add_scalar("charts/episodic_return",
                                            env_ep_info["r"], global_step)
                        writer.add_scalar("charts/episodic_length",
                                            env_ep_info["l"], global_step)
                        writer.add_scalar("charts/episodic_cost",
                                            info["actual_cost"][env_ind], global_step) 

                        print(f"global_step={global_step:7}, episode_length={env_ep_info['l']:3}, episodic_return={env_ep_info['r']:5.2f}, episodic_cost={info['actual_cost'][env_ind]:5.2f}")

            # for key, values in info.items():
            #     for value in values:
            #         writer.add_scalar(f"charts/{key}", value, global_step)

            # for env_ind, item in enumerate(info):
            #     if "actions_available" in item.keys():
            #         writer.add_scalar("charts/actions_available",
            #                           item["actions_available"], global_step)
            #     if "episode" in item.keys():
            #         writer.add_scalar("charts/episodic_return",
            #                           item["episode"]["r"], global_step)
            #         writer.add_scalar("charts/episodic_length",
            #                           item["episode"]["l"], global_step)
            #         writer.add_scalar("charts/episodic_cost",
            #                           item["actual_cost"], global_step)
                    
            #         acc_rw = item.get("acc_rewrites")
            #         if acc_rw is not None:
            #             writer.add_scalar("charts/acc_rewrites",
            #                             acc_rw, global_step)

            #         print(
            #             f"global_step={global_step:7}, episode_length={item['episode']['l']:3}, episodic_return={item['episode']['r']:5.2f}, episodic_cost={item['actual_cost']:5.2f}, acc_rw={acc_rw}")

            #         if args.print_actions:
            #             # TODO: clean this up
            #             start = (step + 1) - item["episode"]["l"]
            #             if start < 0:
            #                 # start of episode is at end of buffer
            #                 start_before = args.num_steps + start
            #                 ep_actions = [action_names[int(i)] for i in torch.cat(
            #                     [actions[start_before:][:, env_ind], actions[0:step + 1][:, env_ind]])]
            #                 ep_rewards = [x.item() for x in list(
            #                     torch.cat([rewards[start_before:][:, env_ind], rewards[0:step + 1][:, env_ind]]))]
            #             else:
            #                 ep_actions = [
            #                     action_names[int(i)] for i in actions[start:step + 1][:, env_ind]]
            #                 ep_rewards = [round(x.item(), 10) for x in list(
            #                     rewards[start:step + 1][:, env_ind])]
            #             ep_a_r = list(zip(ep_actions, ep_rewards))
            #             print(ep_a_r)
            #             print()
            #         break

        # Remember that actions is (n_steps, n_envs, n_actions)
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            if args.gae:
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * \
                        nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * \
                        args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + args.gamma * \
                        nextnonterminal * next_return
                advantages = returns - values

        # flatten the batch
        all_obs_raw = []
        for step_batch in obs:
            # TODO: this is a performance hog. Find a diff way?
            all_obs_raw += step_batch.to_data_list()
        b_obs = pyg.data.Batch.from_data_list(all_obs_raw)

        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        # Optimizing the policy and value network (learn!)
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        # update_epochs num of gradient updates
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                mb_batch_obs = pyg.data.Batch.from_data_list(b_obs[mb_inds])

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    mb_batch_obs, b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() >
                                   args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (
                        mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * \
                    torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * \
                        ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - \
            np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate",
                          optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl",
                          old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance",
                          explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step /
                          (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

    # Save weights of agent now that it's been trained
    torch.save(agent.state_dict(), f"{weights_output_path}/{args.exp_name}")
    return agent


if __name__ == "__main__":
    args = parse_args()
    run_ppo(**vars(args))
