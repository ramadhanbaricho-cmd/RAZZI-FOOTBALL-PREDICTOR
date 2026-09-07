import math
from collections import defaultdict
import numpy as np
import pandas as pd
class PoissonModel:
    def __init__(self, shrinkage=.35,max_goals=8): self.shrinkage=shrinkage; self.max_goals=max_goals
    def fit(self,df):
        self.home_avg=max(df.home_goals.mean(),.05); self.away_avg=max(df.away_goals.mean(),.05); self.teams=sorted(set(df.home_team)|set(df.away_team))
        hs=defaultdict(list); aws=defaultdict(list); hc=defaultdict(list); ac=defaultdict(list)
        for r in df.itertuples(): hs[r.home_team].append(r.home_goals); hc[r.home_team].append(r.away_goals); aws[r.away_team].append(r.away_goals); ac[r.away_team].append(r.home_goals)
        self.aH={};self.aA={};self.dH={};self.dA={}
        for t in self.teams:
            vals=[np.mean(hs[t])/self.home_avg if hs[t] else 1,np.mean(aws[t])/self.away_avg if aws[t] else 1,np.mean(hc[t])/self.away_avg if hc[t] else 1,np.mean(ac[t])/self.home_avg if ac[t] else 1]
            vals=[1+(x-1)*(1-self.shrinkage) for x in vals];self.aH[t],self.aA[t],self.dH[t],self.dA[t]=vals
        return self
    def predict(self,h,a):
        if h not in self.teams or a not in self.teams: raise ValueError('Unknown team')
        lh=float(np.clip(self.home_avg*self.aH[h]*self.dA[a],.05,5)); la=float(np.clip(self.away_avg*self.aA[a]*self.dH[h],.05,5))
        p=lambda l: np.array([math.exp(-l)*l**k/math.factorial(k) for k in range(self.max_goals+1)])
        m=np.outer(p(lh),p(la)); hw=np.tril(m,-1).sum(); dr=np.trace(m); aw=np.triu(m,1).sum(); ov=sum(m[i,j] for i in range(9) for j in range(9) if i+j>=3); bt=sum(m[i,j] for i in range(1,9) for j in range(1,9)); i,j=np.unravel_index(np.argmax(m),m.shape)
        return {'home':hw,'draw':dr,'away':aw,'over25':ov,'under25':1-ov,'btts':bt,'score':f'{i}-{j}','confidence':max(hw,dr,aw),'xg':(lh,la)}
