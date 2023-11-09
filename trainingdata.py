from files import get_rootfiles
from coffea.nanoevents import NanoEventsFactory
import awkward as ak
import numpy as np

hostid = 'cmseos.fnal.gov'
basepath = '/store/group/lpcpfnano/srothman/Nov08_2023_ECON_trainingdata'
tree = 'FloatingpointThreshold0DummyHistomaxDummynTuple/HGCalTriggerNtuple'

files = get_rootfiles(hostid, basepath)

#loop over all the files
for file in files:
    #open the file
    x = NanoEventsFactory.from_root(file, treepath=tree).events()

    #wafer attributes
    #we'll flatten along the event axis
    #if we care about event level information, we would have to add something here
    waferid = ak.to_numpy(ak.flatten(x.wafer.id)) #unique id for each wafer
    zside = ak.to_numpy(ak.flatten(x.wafer.zside)) #which endcap the wafer is in
    layer = ak.to_numpy(ak.flatten(x.wafer.layer)) #which layer the wafer is in
    subdet = ak.to_numpy(ak.flatten(x.wafer.subdet)) #separates the various subdetectors. tbh not sure what the values are
    waferu = ak.to_numpy(ak.flatten(x.wafer.waferu)) #u coordinate of the wafer
    waferv = ak.to_numpy(ak.flatten(x.wafer.waferv)) #v coordinate of the wafer
    wafertype = ak.to_numpy(ak.flatten(x.wafer.wafertype)) #wafertype
    mipPt = ak.to_numpy(ak.flatten(x.wafer.mipPt)) #total mipPt of wafer
    energy = ak.to_numpy(ak.flatten(x.wafer.energy)) #total energy of wafer
    simenergy = ak.to_numpy(ak.flatten(x.wafer.simenergy)) #total simenergy of wafer
    eta = ak.to_numpy(ak.flatten(x.wafer.eta)) #eta of wafer
    phi = ak.to_numpy(ak.flatten(x.wafer.phi)) #phi of wafer
    waferx = ak.to_numpy(ak.flatten(x.wafer.x)) #x of wafer
    wafery = ak.to_numpy(ak.flatten(x.wafer.y)) #y of wafer
    waferz = ak.to_numpy(ak.flatten(x.wafer.z)) #z of wafer

    #training data
    inputs = []
    for i in range(64):
        inputs.append(ak.to_numpy(ak.flatten(x.wafer['AEin%d'%i]))) 
    
    inputs = np.stack(inputs, axis=-1) #stack all 64 inputs
    inputs = np.reshape(inputs, (-1, 8, 8)) #reshape to 8x8
    #I really hope the reshape logic in numpy is the same as in tensorflow
    #NB these do NOT need to be remapped at ALL
    #should be identical to the 8x8 in CMSSW, and as described by Danny's pdf

    #I leave it to you to do something with the data
    #Could save it onto disk in csvs or something
    #Or load it all into memory or a dataloader or something
    #Whatever is most convenient
    raise NotImplementedError("Nate: do something with the data")
    break;
