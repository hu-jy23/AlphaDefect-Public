"""Synchronous, critic-free full-run experiments for GRPO, DAPO and DSPO.

Complete same-seed groups use one frozen behavior policy. All sampled decisions,
including rejected groups, count towards the environment budget. See
docs/experiments/group_policy/Plan.md for the game adaptation.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import signal
import time

import numpy as np
import torch

from sts2env import VecEnv
from sts2env.protocol import format_for
from .content import CATALOG_VERSION
from .catalog import Catalog
from .features import Ragged, collate_ragged
from .group_objectives import (
    action_weights, clipped_terms, overlong_penalty, reference_kl,
    relative_advantages, sequence_coefficients,
)
from .model import ModelConfig, PolicyValueNet, use_tf32
from .muon import Muon, select as muon_select
from .train import guarded_step, params_finite

STOP_REQUESTED = False
CONTRACT = "alphadefect-game-groups-v1"


@dataclass
class Episode:
    seed: str
    observations: list = field(default_factory=list)
    actions: list[int] = field(default_factory=list)
    logp: list[float] = field(default_factory=list)
    reward: float = 0.0
    victory: bool = False
    truncated: bool = False
    terminal: bool = False
    final_floor: int = 0


def write_json(path, payload):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append_json(path, payload):
    with path.open("a") as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n")


def model_batch(observations, device):
    modes = np.ones(len(observations), dtype=np.int64)
    return collate_ragged(Ragged.from_obs(observations, mode=modes)).to(device)


def selected_logp(net, observations, actions, device):
    logits, _ = net.forward_infer(model_batch(observations, device))
    distribution = torch.distributions.Categorical(logits=logits.float())
    return distribution.log_prob(torch.as_tensor(actions, device=device)), distribution.entropy()


def milestone_reward(before, after, rewards):
    """Public full-run milestones, matching the upstream Ironclad coefficients."""
    value = max(0, after.floor - before.floor) * rewards["floor"]
    for act in range(before.act + 1, after.act + 1):
        value += rewards.get("enter_act_" + str(act), 0.0)
    if before.act == after.act == 2 and before.header[24] < 1 <= after.header[24]:
        value += rewards["act3_first_boss"]
    if after.done and after.victory:
        value += rewards["victory"] + rewards["act3_second_boss"]
    return value


@torch.no_grad()
def collect_wave(net, env, seeds, cfg, device, remaining, deadline):
    group_size = cfg["group_size"]
    episodes = [Episode(seed) for seed in seeds for _ in range(group_size)]
    requests = [
        dict(mode="run", character=cfg["character"], ascension=10, seed=e.seed,
             route_summary=True, obs_format=29, loop_cut=True, stall_turns=20,
             privileged=False, future_content=False)
        for e in episodes
    ]
    observations = env.reset(requests)
    active = list(range(len(episodes)))
    spent = 0
    net.eval()
    while active and spent < remaining and time.monotonic() < deadline and not STOP_REQUESTED:
        # A budget boundary can interrupt a wave. Such unfinished groups are censored.
        taking = active[:remaining - spent]
        current = [observations[i] for i in taking]
        if any(o.num_actions < 1 or o.done or o.stuck for o in current):
            raise RuntimeError("An active game has no valid decision")
        logits, _ = net.forward_infer(model_batch(current, device))
        distribution = torch.distributions.Categorical(logits=logits.float())
        actions = distribution.sample()
        logp = distribution.log_prob(actions)
        if not torch.isfinite(logp).all():
            raise FloatingPointError("Non-finite behavior log probabilities")
        choices = actions.cpu().tolist()
        probabilities = logp.cpu().tolist()
        commands = [None] * len(episodes)
        for i, action, probability in zip(taking, choices, probabilities):
            episode = episodes[i]
            observation = observations[i]
            if not 0 <= action < observation.num_actions:
                raise RuntimeError("Policy selected a padded action")
            episode.observations.append(observation)
            episode.actions.append(action)
            episode.logp.append(probability)
            commands[i] = action
        following = env.step(commands)
        spent += len(taking)
        for i in taking:
            after = following[i]
            before = observations[i]
            episode = episodes[i]
            if after.stuck:
                raise RuntimeError(f"Engine reported a stuck run for {episode.seed}")
            episode.reward += milestone_reward(before, after, cfg["rewards"])
            episode.final_floor = after.floor
            observations[i] = after
            if after.done:
                episode.victory = after.victory
                episode.terminal = True
                active.remove(i)
            elif len(episode.actions) >= cfg["max_episode_steps"]:
                episode.truncated = True
                active.remove(i)
    groups = [episodes[i:i + group_size] for i in range(0, len(episodes), group_size)]
    return groups, spent


def prepare_group(group, cfg):
    complete = all(e.terminal and not e.truncated for e in group)
    # DAPO uses its explicit overlong scoring for episodes reaching the horizon.
    if cfg["algorithm"] == "dapo":
        complete = all(e.terminal or e.truncated for e in group)
    rewards = np.array([e.reward for e in group], dtype=np.float64)
    if cfg["algorithm"] == "dapo":
        rewards += overlong_penalty(
            [len(e.actions) for e in group], cfg["max_episode_steps"], cfg["overlong_buffer"])
    advantages = relative_advantages(rewards)
    informative = bool(np.any(advantages))
    retained = complete and (cfg["algorithm"] == "grpo" or informative)
    record = dict(
        seed=group[0].seed, rewards=rewards.tolist(),
        lengths=[len(e.actions) for e in group], victories=[e.victory for e in group],
        floors=[e.final_floor for e in group], truncated=[e.truncated for e in group],
        complete=complete, informative=informative, retained=retained,
        reward_std=float(rewards.std(ddof=0)),
    )
    return retained, advantages, record


def update_policy(net, reference, optimizer, episodes, advantages, cfg, device):
    observations = [o for e in episodes for o in e.observations]
    actions = np.array([a for e in episodes for a in e.actions], dtype=np.int64)
    old_logp = np.array([p for e in episodes for p in e.logp], dtype=np.float32)
    lengths = np.array([len(e.actions) for e in episodes], dtype=np.int64)
    owners = np.repeat(np.arange(len(episodes)), lengths)
    advantages = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    old = torch.as_tensor(old_logp, device=device)
    owner = torch.as_tensor(owners, device=device)
    sizes = torch.as_tensor(lengths, device=device)
    algorithm = cfg["algorithm"]
    weights = None
    if algorithm != "dspo":
        weights = torch.as_tensor(action_weights(lengths, algorithm), device=device)
    beta = cfg["reference_kl_beta"] if algorithm == "grpo" else 0.0
    reference_values = None
    if beta:
        reference_values = torch.empty_like(old)
        with torch.no_grad():
            for begin in range(0, len(actions), cfg["minibatch"]):
                end = min(begin + cfg["minibatch"], len(actions))
                reference_values[begin:end], _ = selected_logp(
                    reference, observations[begin:end], actions[begin:end], device)
    # Dropout is zero. Both DSPO passes use exactly the same policy and fp32 path.
    net.eval()
    metrics = []
    for _ in range(cfg["epochs"]):
        optimizer.zero_grad(set_to_none=True)
        policy_loss = 0.0
        kl_value = 0.0
        entropy_value = 0.0
        sequence_coefficient = None
        if algorithm == "dspo":
            sums = torch.zeros(len(episodes), device=device)
            with torch.no_grad():
                for begin in range(0, len(actions), cfg["minibatch"]):
                    end = min(begin + cfg["minibatch"], len(actions))
                    logp, _ = selected_logp(net, observations[begin:end], actions[begin:end], device)
                    sums.index_add_(0, owner[begin:end], logp - old[begin:end])
            exact_loss, sequence_coefficient = sequence_coefficients(
                sums / sizes, advantages, cfg["clip_low"], cfg["clip_high"])
            policy_loss = float(exact_loss)
        for begin in range(0, len(actions), cfg["minibatch"]):
            end = min(begin + cfg["minibatch"], len(actions))
            logp, entropy = selected_logp(net, observations[begin:end], actions[begin:end], device)
            ids = owner[begin:end]
            if algorithm == "dspo":
                loss = (logp * sequence_coefficient[ids] / sizes[ids]).sum()
            else:
                step_weight = weights[begin:end]
                terms = clipped_terms(logp - old[begin:end], advantages[ids],
                                      cfg["clip_low"], cfg["clip_high"])
                loss = -(terms * step_weight).sum()
                policy_loss += float(loss.detach())
                if beta:
                    divergence = (reference_kl(logp, reference_values[begin:end]) * step_weight).sum()
                    loss = loss + beta * divergence
                    kl_value += float(divergence.detach())
            entropy_value += float(entropy.detach().sum()) / len(actions)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite policy loss")
            loss.backward()
        if guarded_step(net, optimizer, cfg["max_grad_norm"]).item():
            raise FloatingPointError("Non-finite gradients; optimizer update was skipped")
        metrics.append(dict(policy_loss=policy_loss, reference_kl=kl_value,
                            entropy=entropy_value, gradient_norm=float(guarded_step.last_gn)))
    return {k: float(np.mean([m[k] for m in metrics])) for k in metrics[0]}


def checkpoint(path, net, optimizer, cfg, counters):
    if not params_finite(net):
        raise FloatingPointError("Refusing to save non-finite weights")
    payload = dict(
        contract=CONTRACT, algorithm=cfg["algorithm"], group_config=cfg,
        state=net.state_dict(), opt=optimizer.state_dict(),
        muon=optimizer.muon.state_dict(), config=dict(model=cfg["model"]),
        version=counters["updates"], total_steps=counters["sampled_decisions"],
        counters=counters, obs_format=format_for(cfg["character"]),
        catalog_version=CATALOG_VERSION, vocab_n=Catalog().vocab_size,
        content_names=list(getattr(net, "content_names", ())),
        torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
    )
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def validate_config(cfg):
    if cfg["algorithm"] not in ("grpo", "dapo", "dspo"):
        raise ValueError("Unknown algorithm")
    for name in ("group_size", "groups_per_wave", "groups_per_update", "max_sampling_waves",
                 "max_episode_steps", "minibatch", "epochs", "total_decisions",
                 "max_updates", "max_runtime_seconds"):
        if not isinstance(cfg[name], int) or cfg[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if cfg["group_size"] < 2 or cfg["groups_per_wave"] > cfg["groups_per_update"]:
        raise ValueError("Invalid group sizes")
    if not 0 < cfg["clip_low"] < 1 or cfg["clip_high"] <= 0:
        raise ValueError("Invalid clipping thresholds")
    if not 0 < cfg["overlong_buffer"] <= cfg["max_episode_steps"]:
        raise ValueError("Invalid overlong buffer")
    if cfg["lr"] <= 0 or cfg["muon_lr"] <= 0 or cfg["reference_kl_beta"] < 0:
        raise ValueError("Invalid optimizer configuration")


def main():
    global STOP_REQUESTED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    validate_config(cfg)
    run = Path(cfg["run_dir"])
    run.mkdir(parents=True, exist_ok=True)
    write_json(run / "config.json", cfg)
    torch.set_num_threads(cfg["cpu_threads"])
    torch.manual_seed(cfg["seed"])
    use_tf32()
    device = torch.device(cfg["device"])
    initial = Path(cfg["initial_checkpoint"])
    initial_sha = hashlib.sha256(initial.read_bytes()).hexdigest()
    if initial_sha != cfg["initial_sha256"]:
        raise ValueError("Shared initial checkpoint hash differs")
    baseline = torch.load(initial, map_location="cpu", weights_only=False)
    if baseline["config"]["model"] != cfg["model"]:
        raise ValueError("Initial architecture differs from the experiment")
    net = PolicyValueNet(ModelConfig(**cfg["model"])).to(device).eval()
    net.load_state_dict(baseline["state"], strict=True)
    net.infer_ready()
    reference = copy.deepcopy(net).eval().requires_grad_(False) if cfg["algorithm"] == "grpo" else None
    optimizer = torch.optim.Adam(net.parameters(), lr=cfg["lr"], eps=1e-5, fused=device.type == "cuda")
    optimizer.muon = Muon(muon_select(net), lr=cfg["muon_lr"])
    counters = dict(updates=0, sampled_decisions=0, trained_decisions=0, sampled_groups=0,
                    retained_groups=0, rejected_groups=0, next_seed=0, elapsed_seconds=0.0)
    path = run / "latest.pt"
    if path.exists():
        saved = torch.load(path, map_location=device, weights_only=False)
        old = saved["group_config"]
        mutable = {"total_decisions", "max_updates", "max_runtime_seconds"}
        if saved["contract"] != CONTRACT or any(old[k] != cfg[k] for k in cfg if k not in mutable):
            raise ValueError("Frozen experiment contract differs")
        net.load_state_dict(saved["state"], strict=True)
        optimizer.load_state_dict(saved["opt"])
        optimizer.muon.load_state_dict(saved["muon"])
        counters = saved["counters"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        torch.cuda.set_rng_state_all([x.cpu() for x in saved["cuda_rng"]])
        print(f"RESUMED {cfg['algorithm']} update={counters['updates']} sampled={counters['sampled_decisions']}", flush=True)

    def request_stop(*_):
        global STOP_REQUESTED
        STOP_REQUESTED = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    started = time.monotonic()
    prior_elapsed = counters["elapsed_seconds"]
    deadline = started + max(0, cfg["max_runtime_seconds"] - prior_elapsed)
    status = "RUNNING"
    environment = None
    reason = None

    def publish_status():
        counters["elapsed_seconds"] = prior_elapsed + time.monotonic() - started
        write_json(run / "status.json", dict(
            status=status, stop_reason=reason, algorithm=cfg["algorithm"],
            time=time.time(), pid=os.getpid(), counters=counters,
            initial_sha256=initial_sha, code_commit=cfg["code_commit"],
            torch=torch.__version__, catalog_version=CATALOG_VERSION))

    try:
        count = cfg["groups_per_wave"] * cfg["group_size"]
        with (run / "engine.log").open("ab") as engine_log:
            environment = VecEnv(count, log=engine_log, engine="csharp")
            publish_status()
            while (counters["sampled_decisions"] < cfg["total_decisions"]
                   and counters["updates"] < cfg["max_updates"]
                   and time.monotonic() < deadline and not STOP_REQUESTED):
                tick = time.monotonic()
                accepted, advantages = [], []
                attempts = 0
                for _ in range(cfg["max_sampling_waves"]):
                    seeds = [f"ALPHA_RL_TRAIN_{cfg['seed']}_{counters['next_seed'] + i:08d}"
                             for i in range(cfg["groups_per_wave"])]
                    counters["next_seed"] += len(seeds)
                    remaining = cfg["total_decisions"] - counters["sampled_decisions"]
                    groups, spent = collect_wave(net, environment, seeds, cfg, device, remaining, deadline)
                    counters["sampled_decisions"] += spent
                    for group in groups:
                        keep, adv, record = prepare_group(group, cfg)
                        record.update(version=counters["updates"], time=time.time())
                        counters["sampled_groups"] += 1
                        counters["retained_groups" if keep else "rejected_groups"] += 1
                        append_json(run / "groups.jsonl", record)
                        attempts += 1
                        if keep:
                            accepted.extend(group)
                            advantages.extend(adv.tolist())
                    publish_status()
                    enough = len(accepted) >= cfg["groups_per_update"] * cfg["group_size"]
                    exhausted = counters["sampled_decisions"] >= cfg["total_decisions"]
                    if enough or exhausted or STOP_REQUESTED or time.monotonic() >= deadline:
                        break
                if STOP_REQUESTED or time.monotonic() >= deadline:
                    break
                if not accepted:
                    append_json(run / "log.jsonl", dict(time=time.time(), event="no_informative_groups",
                                sampled_decisions=counters["sampled_decisions"], attempted_groups=attempts))
                    checkpoint(path, net, optimizer, cfg, counters)
                    continue
                # A bounded sampling window may yield a smaller, explicitly logged batch.
                metrics = update_policy(net, reference, optimizer, accepted, advantages, cfg, device)
                counters["updates"] += 1
                used = sum(len(e.actions) for e in accepted)
                counters["trained_decisions"] += used
                elapsed = time.monotonic() - tick
                record = dict(time=time.time(), algorithm=cfg["algorithm"], version=counters["updates"],
                              **counters, **metrics, trained_batch_decisions=used,
                              retained_batch_groups=len(accepted) // cfg["group_size"],
                              attempted_batch_groups=attempts, batch_seconds=elapsed,
                              win_rate=float(np.mean([e.victory for e in accepted])),
                              mean_floor=float(np.mean([e.final_floor for e in accepted])),
                              cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                              cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
                append_json(run / "log.jsonl", record)
                publish_status()
                checkpoint(path, net, optimizer, cfg, counters)
                print(json.dumps(record, allow_nan=False), flush=True)
            status = "STOPPED" if STOP_REQUESTED else "COMPLETE"
            reason = ("signal" if STOP_REQUESTED else "wall_time_budget" if time.monotonic() >= deadline
                      else "sample_budget" if counters["sampled_decisions"] >= cfg["total_decisions"]
                      else "update_budget")
    except BaseException as error:
        status, reason = "FAILED", str(error)
        raise
    finally:
        publish_status()
        try:
            checkpoint(path, net, optimizer, cfg, counters)
        finally:
            if environment is not None:
                environment.close()
