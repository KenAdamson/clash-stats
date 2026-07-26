import os, resource, sys
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from tracker.simulation.battles_repo import compute_simulation_data
from tracker.simulation.interaction_matrix import detect_sub_archetypes

def rss_mb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024  # KB->MB on Linux
e = create_engine(os.environ["DATABASE_URL"]); s = Session(e)
print(f"baseline peak RSS: {rss_mb():.0f} MB")
cd = compute_simulation_data(s)
print(f"after corpus_data load: peak RSS {rss_mb():.0f} MB")
# structure sizes
tot_decks = sum(len(v) for v in cd.archetype_decks.values())
print(f"archetype_decks: {len(cd.archetype_decks)} archetypes, {tot_decks} total decks")
print(f"pair_counts: {len(cd.pair_counts)}  card_counts: {len(cd.card_counts)}")
# distinct str identity check on card names
import itertools
sample = list(itertools.islice((c for v in cd.archetype_decks.values() for d in v for c in d['card_names']), 200000))
print(f"card-name refs sampled: {len(sample)}  distinct id(): {len(set(map(id,sample)))}  distinct value: {len(set(sample))}")
# per-WC deck counts (how big is one worker's real working set)
for wc in ["Hog Rider","Mega Knight","Golem"]:
    n = sum(1 for v in cd.archetype_decks.values() for d in v if wc in d['card_names'])
    print(f"  decks containing {wc}: {n}")
r0 = rss_mb()
subs = detect_sub_archetypes("Hog Rider", sim_data=cd, min_cluster_size=10, similarity_threshold=0.55)
print(f"after Hog Rider clustering: peak RSS {rss_mb():.0f} MB (delta {rss_mb()-r0:.0f})  -> {len(subs)} sub-archetypes")
