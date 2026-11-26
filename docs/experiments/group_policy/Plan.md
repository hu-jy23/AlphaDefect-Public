# Grouped policy experiments on Slay the Spire 2

Each branch selects one published algorithm adapted to the game and one
private server. The base contains shared collection and objective code.

| Branch | Algorithm | Assigned server |
| --- | --- | --- |
| rl/grpo | DeepSeekMath outcome-supervised GRPO | AutoDL RTX 3090 |
| rl/dapo | DAPO | Primary RTX 3090 |
| rl/dspo | Dynamic-filter Sequence-level Policy Optimization | AutoDL RTX 3080 Ti |

## Mapping to the game

A task is a complete Ironclad A10 run from a fixed seed. Four independent
policy samples replay that seed. A game decision is one action in the
trajectory likelihood. State transitions and enemy actions contribute no
policy log probability. The behavior network stays frozen through collection.

GRPO normalizes episode scores within each group, clips action ratios
symmetrically and gives each episode equal total weight. It includes a
sampled k3 KL penalty against the frozen initial policy.

DAPO uses group normalization, asymmetric clipping (0.2 below, 0.28 above),
decision-weighted loss and dynamic filtering of zero-variance groups.
Its overlong window penalizes the last 2,000 decisions of the 20,000-decision
episode limit.

DSPO clips the geometric mean of action importance ratios per complete
episode. A no-gradient first pass computes complete episode ratios; a second,
memory-bounded pass applies their exact chain-rule coefficients. Chunk
boundaries never redefine the sequence. A regression compares this gradient
to full autograd with unequal trajectory lengths and both clipping directions.
Dynamic filtering rejects constant-score groups.

## Shared contract

All three load the exact same 19,615,981-parameter initial checkpoint,
identified by SHA256. They use observation format 29, game v0.114.0, the
original encoder and both action heads. Value and auxiliary outputs have no
losses. Collection requests no privileged information.

The score is the public milestone component of the upstream Ironclad
configuration: 0.02 per newly reached floor, 0.1 and 0.5 for entering Acts 2
and 3, 1.5 for defeating the first Act-3 boss, and 2 + 2 on victory.
All three use that same episode score. Potion penalties and combat-practice
episodes are excluded from this pilot. DAPO adds its explicit overlong penalty.

The papers use binary task correctness. This adaptation filters zero-variance
milestone scores, extending their binary mixed-outcome condition to a
continuous score. Filtering solely on A10 wins from random initialization
would provide almost no updates. Binary-only rewards are a separate protocol.

The earlier AlphaDefect PPO pilot includes combat practice, critics and
auxiliary losses. Comparing it to these runs combines those training changes
with the algorithm change. Within the three grouped runs, initialization,
score, seed stream, group sizes, optimizer and epochs are shared.

Optimization reuses upstream Adam + Muon selection and the finite-gradient
guard. Each epoch accumulates the complete batch objective before stepping;
minibatches bound memory. GAE, value regression, asynchronous policy lag and
auxiliary losses are absent.

## Bounds and recovery

The pilot allows 1,048,576 sampled decisions or 7,200 cumulative seconds,
whichever occurs first. Every decision counts, including filtered, censored
and unused samples. At most 16 sampling waves are attempted per update.
A smaller informative batch is logged explicitly. An empty batch is logged
without an optimizer update.

A wave has two seeds and four samples per seed, using eight game hosts.
Each update targets four retained groups. Sampled and trained decisions
have separate counters. Training seeds use the ALPHA_RL_TRAIN namespace;
fixed evaluation uses ALPHA_RL_EVAL.

Atomic checkpoints save model, Adam, Muon, sampled/trained counters, next
seed index, CPU/CUDA RNG and elapsed budget. Resume checks the frozen
contract, allowing only bound changes. Games reset after recovery.
Partial groups on a graceful stop are censored and their cost is retained.
SIGTERM requests a saved shutdown and closes game hosts.

## Sources

- DeepSeekMath, GRPO outcome supervision and k3 KL, section 4.1.
  https://arxiv.org/html/2402.03300v3
- DAPO objectives and four components.
  https://arxiv.org/html/2503.14476v2
- DSPO geometric sequence ratio, filtering and loss, equations 6, 9–11.
  https://arxiv.org/html/2510.09255v4
- DSPO's public repository currently has the paper and a code-release notice.
  This fork implements the published objective for the game.
  https://github.com/AgnesAI-Labs/DSPO
