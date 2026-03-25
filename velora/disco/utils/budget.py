# Copyright 2025 Achronus
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import random


def sample_budget(rng: random.Random) -> int:
    """
    Sample a step budget from a lifetime distribution.

    Budgets are drawn from `{5M, 10M, 20M, 50M}` environment steps,
    weighted inversely proportional to their size so shorter lifetimes are
    sampled more frequently.

    Parameters
    ----------
    rng : random.Random
        Python RNG instance

    Returns
    -------
    budget : int
        Sampled step budget
    """
    LIFETIME_BUDGETS = [5_000_000, 10_000_000, 20_000_000, 50_000_000]
    LIFETIME_WEIGHTS = [1 / b for b in LIFETIME_BUDGETS]

    return rng.choices(LIFETIME_BUDGETS, weights=LIFETIME_WEIGHTS, k=1)[0]
