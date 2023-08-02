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
        passPlus = plusDR < 0.05

        minusIdx, minusDR = find_closest(genMinusEta, genMinusPhi, recoEta[EEminus], recoPhi[EEminus])
        passMinus = minusDR < 0.05


        recoPlusPt =  ak.firsts(recoPt[EEplus][plusIdx])
        recoPlusEta = ak.firsts(recoEta[EEplus][plusIdx])
        recoPlusPhi = ak.firsts(recoPhi[EEplus][plusIdx])

        recoMinusPt =  ak.firsts(recoPt[EEminus][minusIdx])
        recoMinusEta = ak.firsts(recoEta[EEminus][minusIdx])
        recoMinusPhi = ak.firsts(recoPhi[EEminus][minusIdx])

        plusPtErr = (recoPlusPt - genPt)/genPt
        minusPtErr = (recoMinusPt - genPt)/genPt

        H = hist.Hist(
            hist.axis.Regular(100, -1, 1, name='ptErr', label='(reco pt - gen pt)/gen pt'),
            hist.axis.Regular(50, 0, 100, name='genPt', label='gen pt'),
            hist.axis.Regular(50, 1.4, 3.0, name='eta', label='gen eta'),
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

        return {'hist': H}

    def postprocess(self, accumulator):
        pass
