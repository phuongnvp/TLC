"""UCT search over solvent identities, then integer percentage allocations."""
from dataclasses import dataclass, field
import math
import random

import pandas as pd
import torch
from graph_encoding import encode_attentive, encode_mpnn
from ensemble import Batch, WEIGHT, canonicalize_smiles, smiles_to_graph
from solvent_rules import natural_solvent_pool, natural_choices, solvent_label, display_solvent, modifier_limits_ok, miscibility_flag


@dataclass
class Node:
    visits: int = 0
    reward: float = 0.
    children: dict = field(default_factory=dict)
    remaining: list | None = None


@torch.inference_mode()
def search(ensemble, smiles, target, tolerance=0., count=2, trials=10000, seed=42, natural_products=False):
    if not (0 <= target <= 1 and 0 <= tolerance <= 1):
        raise ValueError("Target and tolerance must be between zero and one.")
    solvents = natural_solvent_pool(ensemble.solvents) if natural_products else ensemble.solvents
    if not 1 <= count <= min(4, len(solvents)):
        raise ValueError("Choose one to four solvents.")
    allowed_choices = natural_choices(solvents, count) if natural_products else None
    if natural_products and not allowed_choices:
        raise ValueError("No compatible solvent systems are available for this component count.")
    graph = Batch.from_data_list([smiles_to_graph(canonicalize_smiles(smiles))])
    models = (ensemble.attentive, ensemble.mpnn)
    embeddings = [encode_attentive(models[0].model, graph), encode_mpnn(models[1].model, graph)]
    columns = [[p.solvents.index(s) for s in solvents] for p in models]

    def score(keys):
        predictions = []
        for p, embedding, mapping in zip(models, embeddings, columns):
            inputs = torch.zeros(len(keys), len(p.inputs))
            for row, (ids, percentages) in enumerate(keys):
                inputs[row, [mapping[i] for i in ids]] = torch.tensor(percentages) / 100
            encoded = p.model.solvent_encoder(inputs)
            predictions.append(p.model.regressor(torch.cat(
                (embedding.expand(len(keys), -1), encoded), dim=1)).flatten().tolist())
        return [WEIGHT*a + (1-WEIGHT)*b for a,b in zip(*predictions)]

    def options(prefix):
        depth = len(prefix)
        if depth < count:
            if natural_products:
                return allowed_choices[prefix]
            first = prefix[-1]+1 if depth else 0
            return range(first, len(solvents)-(count-depth)+1)
        # Reserve at least one percent for each remaining component.
        return range(1, 101-sum(prefix[count:])-(2*count-depth-1))

    rng = random.Random(seed)
    root = Node()
    cache = {}
    best = {}
    depth_limit = 2*count-1
    for offset in range(0, min(trials,len(solvents)) if count == 1 else trials, 32):
        pending = []
        budget = min(trials,len(solvents)) if count == 1 else trials
        for _ in range(min(32, budget-offset)):
            node, prefix, path = root, (), [root]
            while len(prefix) < depth_limit:
                if node.remaining is None:
                    node.remaining = list(options(prefix))
                    rng.shuffle(node.remaining)
                if node.remaining and (not node.children or node.visits >= 10 or node is root):
                    action = node.remaining.pop()
                    node.children[action] = Node()
                    path.append(node.children[action])
                    prefix += (action,)
                    break
                action = max(node.children, key=lambda a: (
                    node.children[a].reward / max(1,node.children[a].visits)
                    + .3*math.sqrt(math.log(1+node.visits)/max(1,node.children[a].visits))))
                node = node.children[action]
                path.append(node)
                prefix += (action,)
            while len(prefix) < depth_limit:
                prefix += (rng.choice(options(prefix)),)
            parts = prefix[count:]
            key = (prefix[:count], parts+(100-sum(parts),))
            for ancestor in path:
                ancestor.visits += 1
            names = [solvents[i] for i in key[0]]
            if natural_products and not modifier_limits_ok(names, key[1]):
                continue
            pending.append((key,path))
        fresh = list(dict.fromkeys(key for key,_ in pending if key not in cache))
        if fresh:
            cache.update(zip(fresh,score(fresh)))
        for key,path in pending:
            error = abs(cache[key]-target)
            reward = 1-max(0.,error-tolerance)-0.1*error
            for ancestor in path:
                ancestor.reward += reward
            ids = key[0]
            if ids not in best or error < abs(cache[best[ids]]-target):
                best[ids] = key

    rows = []
    ranked = sorted(cache, key=lambda k:(abs(cache[k]-target),k))
    best_keys = set(best.values())
    top_indices = []
    for i, key in enumerate(ranked):
        if key in best_keys and len(top_indices) < 10:
            top_indices.append(i)
        ids, percentages = key
        err = abs(cache[key]-target)
        rows.append({"Solvent system":" + ".join(display_solvent(solvents[i]) for i in ids),
                     "Ratio (%)":":".join(map(str,percentages)), "Predicted Rf":cache[key],
                     "|ΔRf|":err})
        names = [solvents[j] for j in ids]
        rows[-1]["Miscibility"] = miscibility_flag(names)
        if tolerance:
            rows[-1]["Within range"] = err <= tolerance
    all_mixtures = pd.DataFrame(rows)
    return all_mixtures.iloc[top_indices].copy(), all_mixtures

