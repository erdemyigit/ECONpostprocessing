from coffea import processor
import awkward as ak
import numpy as np
import hist

def find_closest(genEta, genPhi, recoEta, recoPhi):
    deta = genEta - recoEta
    dphi = genPhi - recoPhi
    dphi = ak.where(dphi > np.pi, dphi - 2*np.pi, dphi)
    dphi = ak.where(dphi < -np.pi, dphi + 2*np.pi, dphi)

    dR = np.sqrt(np.square(deta) + np.square(dphi))
    minIdx = ak.argmin(dR, axis=-1, keepdims=True)

    return minIdx, ak.firsts(dR[minIdx])

class EleProcessor(processor.ProcessorABC):
    def __init__(self):
        pass

    def process(self, events):
        # Convert filename to an integer
        filename = events.metadata['filename'].split('ntuple_')[1].split('.root')[0]
        
        filename_int = int(filename)

        
        
        genPt = ak.firsts(events.gen.pt)
        genEEplus = events.gen.eta > 0
        

        genPlusEta = ak.firsts(events.gen.eta[genEEplus])
        genPlusPhi = ak.firsts(events.gen.phi[genEEplus])

        genMinusEta = ak.firsts(events.gen.eta[~genEEplus])
        genMinusPhi = ak.firsts(events.gen.phi[~genEEplus])

        recoPt = events.cl3d.pt
        recoEta = events.cl3d.eta
        recoPhi = events.cl3d.phi
        EEplus = recoEta > 0
        EEminus = recoEta < 0

        plusIdx, plusDR = find_closest(genPlusEta, genPlusPhi, recoEta[EEplus], recoPhi[EEplus])
        passPlus = plusDR < 0.1
        minusIdx, minusDR = find_closest(genMinusEta, genMinusPhi, recoEta[EEminus], recoPhi[EEminus])
        passMinus = minusDR < 0.1

        passPlus = ak.fill_none(passPlus, False)
        passMinus = ak.fill_none(passMinus, False)

        recoPlusPt = ak.firsts(recoPt[EEplus][plusIdx])
        recoPlusEta = ak.firsts(recoEta[EEplus][plusIdx])
        recoPlusPhi = ak.firsts(recoPhi[EEplus][plusIdx])

        recoMinusPt = ak.firsts(recoPt[EEminus][minusIdx])
        recoMinusEta = ak.firsts(recoEta[EEminus][minusIdx])
        recoMinusPhi = ak.firsts(recoPhi[EEminus][minusIdx])

        plusPtErr = (recoPlusPt - genPt) / genPt
        minusPtErr = (recoMinusPt - genPt) / genPt

        H = hist.Hist(
            hist.axis.Regular(200, -1, 3, name='ptErr', label='(reco pt - gen pt)/gen pt'),
            hist.axis.Regular(50, 0, 100, name='genPt', label='gen pt'),
            hist.axis.Regular(5, 1.4, 3.0, name='eta', label='gen eta'),
        )

        H.fill(
            ptErr=plusPtErr[passPlus],
            genPt=genPt[passPlus],
            eta=np.abs(genPlusEta[passPlus]),
        )

        H.fill(
            ptErr=minusPtErr[passMinus],
            genPt=genPt[passMinus],
            eta=np.abs(genMinusEta[passMinus]),
        )
        
        combined_genPt = ak.concatenate([genPt[passPlus], genPt[passMinus]], axis=0)
        combined_ptErr = ak.concatenate([plusPtErr[passPlus], minusPtErr[passMinus]], axis=0)
        combined_eta = ak.concatenate([genPlusEta[passPlus], genMinusEta[passMinus]], axis=0)
        
        # Generate event index
        event_index = np.arange(len(combined_eta))

        # Create a unique event ID by combining filename and event index
        unique_event_id = filename_int * (10 ** (len(str(len(event_index)))+3)) + event_index
        
        # Convert unique_event_id to Awkward Array
        combined_event_id = ak.Array(unique_event_id)
        
        # Combine plus and minus data
        
        # Collect the point-level data with identifiers
        data = {
            'event_id': combined_event_id.to_list(),
            'genPt': combined_genPt.to_list(),
            'ptErr': combined_ptErr.to_list(),
            'eta': combined_eta.to_list(),
        }

        return {'hist': H, 'data': data}


    def postprocess(self, accumulator):
        pass
