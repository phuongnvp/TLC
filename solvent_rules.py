"""Natural-product search names and solvent compatibility rules."""
from itertools import combinations

NATURAL_SOLVENTS = (
    "n-Butanol", "Acetic acid", "Acetone", "Chloroform", "Diethyl ether",
    "Ethyl acetate", "Formic acid", "n-Hexane", "NH3", "Toluene",
    "Triethylamine", "Water",
)
_ALIASES = {name.casefold(): name for name in NATURAL_SOLVENTS}
_ALIASES.update({
    "1-butanol": "n-Butanol", "butan-1-ol": "n-Butanol",
    "glacial acetic acid": "Acetic acid", "acetic acid glacial": "Acetic acid",
    "ammonia solution": "NH3", "ammonia": "NH3",
})


def solvent_label(name):
    return _ALIASES.get(name.strip().casefold(), name)

MODIFIER_MAX_PCT = {
    "Acetic acid": 2,
    "Formic acid": 2,
    "NH3": 2,
    "Triethylamine": 2,
}

def modifier_limits_ok(names, percentages):
    return all(
        pct <= MODIFIER_MAX_PCT.get(solvent_label(name), 100)
        for name, pct in zip(names, percentages)
    )

# Extend using a validated miscibility reference.
WATER_GROUP = {
    "Acetic acid",
    "Formic acid",
    "NH3",
    "Triethylamine",
    "Water",
}

NONPOLAR_GROUP = {
    "Diethyl ether",
    "n-Hexane",
    "Chloroform",
}

def miscibility_flag(names):
    labels = {solvent_label(name) for name in names}
    water_members = labels & WATER_GROUP
    nonpolar_members = labels & NONPOLAR_GROUP
    if water_members and nonpolar_members:
        return "Check miscibility"
    return "Not assessed"

def display_solvent(name):
    label = solvent_label(name)
    return {"n-Butanol": "n-Butanol", "NH3": "Ammonia"}.get(label, name)


def natural_solvent_pool(available):
    # Keep the checkpoint's column names; aliases are for selection/display only.
    by_label = {}
    for name in available:
        label = solvent_label(name)
        if label in NATURAL_SOLVENTS:
            if label not in by_label or name.casefold() == label.casefold():
                by_label[label] = name
    return [by_label[label] for label in NATURAL_SOLVENTS if label in by_label]


def compatible(names):
    labels = {solvent_label(name) for name in names}
    acids = labels & {"Acetic acid", "Formic acid"}
    bases = labels & {"NH3", "Triethylamine"}
    return len(acids) <= 1 and len(bases) <= 1 and not (acids and bases)


def natural_choices(solvents, count):
    """Only allow prefixes that can reach a complete compatible system."""
    choices = {}
    for ids in combinations(range(len(solvents)), count):
        if compatible(solvents[i] for i in ids):
            for depth, action in enumerate(ids):
                choices.setdefault(ids[:depth], set()).add(action)
    return {prefix: sorted(actions) for prefix, actions in choices.items()}
