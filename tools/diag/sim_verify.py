import os, resource, time
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from tracker.simulation.battles_repo import compute_simulation_data
from tracker.simulation.interaction_matrix import detect_sub_archetypes
def rss(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
e=create_engine(os.environ["DATABASE_URL"]); s=Session(e)
t0=time.time(); cd=compute_simulation_data(s); t1=time.time()
tot=sum(len(v) for v in cd.archetype_decks.values())
print(f"corpus_data load: {t1-t0:.0f}s  peak RSS {rss():.0f} MB  unique-deck records={tot} (was 10.4M per-battle)")
for wc in ["Hog Rider","Golem"]:
    t=time.time(); subs=detect_sub_archetypes(wc, sim_data=cd, min_cluster_size=10, similarity_threshold=0.55); dt=time.time()-t
    top=subs[0] if subs else {}
    print(f"  {wc}: {len(subs)} sub-archetypes in {dt:.1f}s  peak RSS {rss():.0f} MB"
          + (f" | top: n={top.get('count')} wr={top.get('win_rate',0):.2f} elix={top.get('avg_elixir')} sig={top.get('signature_cards',[])[:4]}" if top else ""))
