#!/usr/bin/env python3
"""masked_lstm_policy.py - RecurrentPPO policy with the bench's legal-action mask.

sb3-contrib's RecurrentPPO has no action masking. The proposal (Section 3.2) applies the same
mask as MaskablePPO by setting the logits of illegal schedules to -inf before sampling. Because
the hardware limit is static, the mask is the same at every step, so it can live inside the
policy as a fixed buffer: RecurrentPPO's rollout buffer, GAE and update loop stay untouched.

All of RecurrentActorCriticPolicy's paths (forward during rollouts, evaluate_actions during the
update, get_distribution/predict at evaluation) build their distribution through
_get_action_dist_from_latent, so overriding that one method masks all of them.

-1e8 is used instead of a literal -inf, as sb3-contrib's MaskableCategorical does: the
probability is exactly 0 in float32 (so illegal schedules are never sampled and receive no
gradient through the log-probability), while entropy and KL stay finite.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch as th
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from stable_baselines3.common.distributions import MultiCategoricalDistribution

MASK_LOGIT = -1e8


class MaskedRecurrentActorCriticPolicy(RecurrentActorCriticPolicy):
    """RecurrentActorCriticPolicy whose MultiDiscrete logits carry a fixed legal-action mask.

    Args (in addition to the parent's):
        action_mask: flat boolean mask over the concatenated MultiDiscrete logits, the same
            layout MaskablePPO uses ([20 schedule | 19 assignment] for the ICCCN'26 tables).
            None means no masking.
    """

    def __init__(self, *args, action_mask: Optional[Sequence[bool]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.action_dist, MultiCategoricalDistribution):
            raise TypeError("MaskedRecurrentActorCriticPolicy expects a MultiDiscrete action space")
        n_logits = int(sum(self.action_dist.action_dims))
        mask = np.ones(n_logits, dtype=bool) if action_mask is None else np.asarray(action_mask, dtype=bool)
        if mask.shape != (n_logits,):
            raise ValueError(f"action_mask has {mask.shape[0]} entries, policy has {n_logits} logits")
        if any(not seg.any() for seg in np.split(mask, np.cumsum(self.action_dist.action_dims)[:-1])):
            raise ValueError("action_mask leaves a head with no legal action")
        bias = np.where(mask, 0.0, MASK_LOGIT).astype(np.float32)
        # A buffer moves with .to(device) and is saved in the state dict.
        self.register_buffer("action_logit_bias", th.as_tensor(bias), persistent=True)
        self._action_mask_list = mask.tolist()

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor):
        logits = self.action_net(latent_pi) + self.action_logit_bias
        return self.action_dist.proba_distribution(action_logits=logits)

    def _get_constructor_parameters(self):
        data = super()._get_constructor_parameters()
        data["action_mask"] = self._action_mask_list
        return data
