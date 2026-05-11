'''
Cross-Entropy Method (CEM) outer loop for optimizing the trait sampling distribution 
toward higher “harm” scores. 

Samples trait vectors, converts them to trait strings, runs full simulations, 
selects elite rollouts, and updates the age/Beta distributions. 

Logs rollouts to JSON and optionally to W&B
'''

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable, Dict, Any, List, Optional, Tuple
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

from simvbg.wildguard_scorer import wildguard_harm_score

try:
    import wandb  # optional
except Exception:
    wandb = None

from .simulator import SimVBGSimulator


# ----------------------------
# Trait distribution (Age + Bernoulli-ish booleans via Beta)
# ----------------------------

@dataclass
class CEMConfig:
    num_rounds: int = 100          # X
    num_elites: int = 8            # Y
    num_rollouts: int = 30         # K
    num_turns: int = 3             # should match SimulationConfig.n_turns
    output_file: Optional[str] = None

    # Age distribution init
    age_mean: float = 25.0
    age_std: float = 10.0
    # Age update smoothing
    age_lr: float = 0.2
    age_std_min: float = 5.0
    age_clip: Tuple[float, float] = (10.0, 80.0)

    # Boolean traits init
    num_boolean_traits: int = 10
    beta_init_alpha: float = 1.0
    beta_init_beta: float = 1.0
    beta_lr: float = 0.2  # added learning rate to slow convergence

    # Beta update sharpness (alpha+beta); higher = more peaked
    beta_concentration: float = 10.0  # tune
    # Avoid degenerate p_hat=0/1
    p_clip: Tuple[float, float] = (0.05, 0.95)

    # Whether to binarize Beta samples 
    binarize_booleans: bool = True  # TODO if False, encode “degree” traits later


@dataclass
class TraitSpace:
    """
    Maps sampled numeric traits -> SimVBG's trait strings.
    Age is index 0, then N boolean traits.
    """
    boolean_trait_names: List[str]

    def __post_init__(self):
        if len(self.boolean_trait_names) == 0:
            raise ValueError("boolean_trait_names must be non-empty.")

    def vector_to_trait_strings(self, vec: List[float]) -> List[str]:
        age = int(round(float(vec[0])))
        bools = vec[1:]

        traits = [f"{age} years old"]
        for name, v in zip(self.boolean_trait_names, bools):
            if int(round(float(v))) == 1:
                traits.append(name)
        return traits


class TraitDistribution:
    def __init__(self, cfg: CEMConfig):
        self.cfg = cfg
        self.age_mean = cfg.age_mean
        self.age_std = cfg.age_std
        self.beta_params = np.ones((cfg.num_boolean_traits, 2), dtype=float)
        self.beta_params[:, 0] *= cfg.beta_init_alpha
        self.beta_params[:, 1] *= cfg.beta_init_beta

    def sample(self) -> List[float]:
        age = np.random.normal(loc=self.age_mean, scale=self.age_std)
        age = float(np.clip(age, self.cfg.age_clip[0], self.cfg.age_clip[1]))

        bools: List[float] = []
        for i in range(self.cfg.num_boolean_traits):
            a, b = self.beta_params[i]
            s = np.random.beta(a, b)  # in [0,1]
            if self.cfg.binarize_booleans:
                bools.append(float(int(round(s))))
            else:
                bools.append(float(s))

        return [age] + bools

    def update_from_elites(self, elite_trait_vectors: List[List[float]]) -> None:
        elite = np.array(elite_trait_vectors, dtype=float)  # (Y, 1+N)

        # Update age mean/std by MLE ** added learning rate to slow convergence
        lr = self.cfg.age_lr
        new_mean = float(np.mean(elite[:, 0]))
        new_std = float(np.std(elite[:, 0]))
        new_std = max(new_std, self.cfg.age_std_min)  # avoid collapse

        # now each round will only move 10% of the way from old to new mean/std
        self.age_mean = (1 - lr) * self.age_mean + lr * new_mean
        self.age_std = (1 - lr) * self.age_std + lr * new_std
        
        # clip age mean 
        self.age_mean = float(np.clip(
            self.age_mean,
            self.cfg.age_clip[0],
            self.cfg.age_clip[1],
        ))

        # Update Beta via moment-ish approach: match mean p_hat, fixed concentration
        for i in range(self.cfg.num_boolean_traits):
            vals = elite[:, i + 1] # elite values for this boolean trait; 1s push alpha up, 0s push beta up
            p_hat = float(np.mean(vals))
            p_hat = float(np.clip(p_hat, self.cfg.p_clip[0], self.cfg.p_clip[1]))

            conc = float(self.cfg.beta_concentration)
            beta_lr = self.cfg.beta_lr
            
            new_alpha = 1.0 + p_hat * conc
            new_beta  = 1.0 + (1.0 - p_hat) * conc

            # updated logic above, was not keeping 1,1 prior in update. good?
            # new_alpha = p_hat * conc
            # new_beta = (1.0 - p_hat) * conc

            old_alpha = self.beta_params[i, 0]
            old_beta = self.beta_params[i, 1]

            self.beta_params[i, 0] = (1 - beta_lr) * old_alpha + beta_lr * new_alpha
            self.beta_params[i, 1] = (1 - beta_lr) * old_beta + beta_lr * new_beta

    def state_dict(self) -> Dict[str, Any]:
        return {
            "age_mean": self.age_mean,
            "age_std": self.age_std,
            "beta_params": self.beta_params.tolist(),
        }
    
# ----------------------------
# Rollout scoring interface
# ----------------------------

def default_harm_score_fn(sim_out: Dict[str, Any]) -> float:
    """
    Placeholder. See wildguard_scorer.py 
    """
    return 0.0


# ----------------------------
# CEM Runner
# ----------------------------

@dataclass
class RolloutRecord:
    round: int
    rollout: int
    trait_vector: List[float]
    traits: List[str]
    neutral_prompts: List[str]
    score: float
    sim_out: Dict[str, Any]


class CEMRunner:
    def __init__(
        self,
        simulator: SimVBGSimulator,
        trait_space: TraitSpace,
        cfg: CEMConfig,
        harm_score_fn: Optional[Callable[[Dict[str, Any]], float]] = None,
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
    ):
        self.simulator = simulator
        self.trait_space = trait_space
        self.cfg = cfg
        self.dist = TraitDistribution(cfg)
        self.harm_score_fn = harm_score_fn or default_harm_score_fn

        if cfg.output_file is None:
            cfg.output_file = f"cem_rollouts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        self.wandb_enabled = (wandb is not None) and (wandb_project is not None)
        if self.wandb_enabled:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config=asdict(cfg),
            )

        self.all_rollouts: List[Dict[str, Any]] = []
        
        # Log trait distribution snapshots separately from rollout logs
        base_output_file = self.cfg.output_file
        if base_output_file.endswith(".json"):
            self.trait_distribs_file = base_output_file.replace(
                ".json", "_trait_distribs.json"
            )
        else:
            self.trait_distribs_file = f"{base_output_file}_trait_distribs.json"

        self.trait_distribs: List[Dict[str, Any]] = []

    # ---------------------------
    # Visulization of trait distributions
    # ---------------------------
    def _log_beta_pdf_plot(self, round_idx: int, every: int = 5) -> None:
        if round_idx % every != 0:
            return

        x = np.linspace(0.001, 0.999, 200)
        beta_params = self.dist.beta_params

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, name in enumerate(self.trait_space.boolean_trait_names):
            a, b = beta_params[i]
            y = beta_dist.pdf(x, a, b)
            ax.plot(x, y, label=name, alpha=0.6)

        ax.set_title(f"Beta distributions at round {round_idx}")
        ax.set_xlabel("Trait probability")
        ax.set_ylabel("Density")
        ax.legend(fontsize=6, ncol=2)

        wandb.log({
            "beta_pdf_plot": wandb.Image(fig),
        }, step=round_idx)

        plt.close(fig)
        
    
    
    def run(self, neutral_prompts: List[str]) -> Dict[str, Any]:
        
        if len(neutral_prompts) < 1:
            raise ValueError(f"Need at least 1 neutral prompts.")

        for r in range(self.cfg.num_rounds):
            round_idx = r + 1
            round_rollouts: List[RolloutRecord] = []

            print("\n" + "=" * 60)
            print(f"  ROUND {round_idx} / {self.cfg.num_rounds}")
            print("=" * 60)

            # --- K rollouts
            for k in range(self.cfg.num_rollouts):
                rollout_idx = k + 1
                vec = self.dist.sample()
                traits = self.trait_space.vector_to_trait_strings(vec)

                print(f"  Rollout {rollout_idx}/{self.cfg.num_rollouts} ... ", end="")

                used_neutral_prompts = [neutral_prompts[0]]

                sim_out = self.simulator.run(
                    traits=traits,
                    neutral_prompts=used_neutral_prompts,
                )

                score = float(self.harm_score_fn(sim_out))
                print(f"score={score:.6e}")

                round_rollouts.append(
                    RolloutRecord(
                        round=round_idx,
                        rollout=rollout_idx,
                        trait_vector=vec,
                        traits=traits,
                        neutral_prompts=used_neutral_prompts,
                        score=score,
                        sim_out=sim_out,
                    )
                )
                

            # --- select elites
            round_rollouts_sorted = sorted(round_rollouts, key=lambda x: x.score, reverse=True)
            elites = round_rollouts_sorted[: self.cfg.num_elites]

            all_scores = [r.score for r in round_rollouts]
            elite_scores = [e.score for e in elites]
            mean_elite = float(np.mean(elite_scores)) if elite_scores else float("nan")

            print(f"\n  Elite mean score: {mean_elite:.4f}")

            # --- update distributions
            elite_vecs = [e.trait_vector for e in elites]
            self.dist.update_from_elites(elite_vecs)

            print(f"  Updated age: mean={self.dist.age_mean:.2f}, std={self.dist.age_std:.2f}")
            a0, b0 = self.dist.beta_params[0]
            print(f"  Updated beta[0]: alpha={a0:.3f}, beta={b0:.3f}")
            
            # --- log trait distribution snapshot to separate JSON
            beta_params = self.dist.beta_params
            beta_trait_distribs = []

            for i, trait_name in enumerate(self.trait_space.boolean_trait_names):
                alpha = float(beta_params[i, 0])
                beta = float(beta_params[i, 1])
                mean = alpha / (alpha + beta)

                # Beta variance: alpha*beta / ((alpha+beta)^2 * (alpha+beta+1))
                variance = (alpha * beta) / (((alpha + beta) ** 2) * (alpha + beta + 1.0))

                beta_trait_distribs.append({
                    "trait": trait_name,
                    "alpha": alpha,
                    "beta": beta,
                    "mean": float(mean),
                    "variance": float(variance),
                })

            trait_distrib_snapshot = {
                "round": round_idx,
                "age": {
                    "mean": float(self.dist.age_mean),
                    "std": float(self.dist.age_std),
                    "clip": list(self.cfg.age_clip),
                },
                "beta_traits": beta_trait_distribs,
            }

            self.trait_distribs.append(trait_distrib_snapshot)

            with open(self.trait_distribs_file, "w") as f:
                json.dump(self.trait_distribs, f, indent=2)

            # --- log/save
            if self.wandb_enabled:
                wandb.log({
                    "mean_elite_score": mean_elite,
                    "age_mean": self.dist.age_mean,
                    "age_std": self.dist.age_std,
                    "max_score": float(np.max(all_scores)),
                    "mean_score": float(np.mean(all_scores)),
                    "std_score": float(np.std(all_scores)),
                    "min_score": float(np.min(all_scores)),
                }, step=round_idx)
                
                self._log_beta_pdf_plot(round_idx, every=5)

            # Append round rollouts to master json log
            for rec in round_rollouts:
                self.all_rollouts.append({
                    "round": rec.round,
                    "rollout": rec.rollout,
                    "trait_vector": rec.trait_vector,
                    "traits": rec.traits,
                    "neutral_prompts": rec.neutral_prompts,
                    "score": rec.score,
                    # Keep full sim output (conversation + trace)
                    "sim_out": rec.sim_out,
                })

            with open(self.cfg.output_file, "w") as f:
                json.dump(self.all_rollouts, f, indent=2)

        if self.wandb_enabled:
            wandb.finish()

        print(f"\nCEM complete. All rollouts saved to: {self.cfg.output_file}")
        return {
            "output_file": self.cfg.output_file,
            "final_distribution": self.dist.state_dict(),
        }