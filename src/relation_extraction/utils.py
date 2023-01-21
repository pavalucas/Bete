BETE_RELATIONS = ['O', 'causes', 'prevents', 'treats', 'has', 'diagnoses', 'complicates']
# construct map label to id and id to label
bete_relation2id = {tag: i for i, tag in enumerate(BETE_RELATIONS)}
bete_id2relation = {i: tag for i, tag in enumerate(BETE_RELATIONS)}

EHEALTH_RELATIONS = [
    "O",
    "is-a",
    "part-of",
    "has-property",
    "causes",
    "entails",
    "in-context",
    "in-place",
    "in-time",
    "subject",
    "target",
    "domain",
    "arg",
    "same-as"
]
# construct map label to id and id to label
ehealth_relation2id = {tag: i for i, tag in enumerate(EHEALTH_RELATIONS)}
ehealth_id2relation = {i: tag for i, tag in enumerate(EHEALTH_RELATIONS)}