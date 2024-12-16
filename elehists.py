import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.processor import Runner, IterativeExecutor, FuturesExecutor, DaskExecutor
import hist
import re

from scaleout import setup_cluster_on_submit
from eleprocessor import EleProcessor
from files import get_rootfiles

hostid = 'cmseos.fnal.gov'
path = '/store/user/eertorer/OptimizedCAE_logOriginalCALQSum_normalization_on/'
files = get_rootfiles(hostid, path)[:] #Get all the files

f0 = uproot.open(files[0])
trees = []
for key in f0.keys():
    if '/' not in key:
        trees.append(key[:-2]+'/HGCalTriggerNtuple')
        print(trees[-1])

#if len(files) > 10:
#    cluster, client = setup_cluster_on_submit(1, 500)
#
#    runner = Runner(
#        executor=DaskExecutor(client=client, status=True),
#        schema=NanoAODSchema,
#    )
#else:
runner = Runner(
    executor=IterativeExecutor(),
    schema=NanoAODSchema,
)

for tree in trees:
    splitted = [s for s in re.split("([A-Z][^A-Z]*)", tree) if s]
    name = tree.split('/')[0] #splitted[1]
    if name == 'Dummy':
        name = name + splitted[0]
    print(name)

    out = runner(
        {'Events': files},
        treename=tree,
        processor_instance=EleProcessor(),
    )

    with open("eles_%s.pkl"%name, 'wb') as f:
        import pickle
        pickle.dump(out, f)

#if len(files) > 10:
#    client.close()
#    cluster.close()
