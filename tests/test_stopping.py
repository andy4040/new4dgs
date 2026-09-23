import pytest
import torch
from stopping import Plateau
from geometry import Gaussians,split_gaussians


def test_plateau_and_best_are_distinct():
    s=Plateau(patience=2,min_delta=.1,min_steps=2)
    assert s.update(1.,0)==(True,False)
    assert s.update(.96,2)==(True,False)
    assert s.update(.95,3)==(True,True)
    assert s.best_step==3
    s.reset_patience()
    assert s.update(.94,4)==(True,False)


def test_accumulated_improvement_and_nonfinite():
    s=Plateau(patience=3,min_delta=.1,min_steps=0)
    s.update(1.,0);s.update(.96,1);s.update(.92,2)
    assert s.update(.88,3)==(True,False)
    assert s.bad==0
    with pytest.raises(FloatingPointError):s.update(float('nan'),4)


def test_split_spd_and_ancestry():
    g=Gaussians(torch.randn(8,3),torch.full((8,3),.5),.1)
    with torch.no_grad():g.lower[:,0]=20.;g.log_scale[:,0]=-10
    new,event=split_gaussians(g,torch.arange(8.),.5)
    assert len(new.xyz)==12 and len(new.ids.unique())==12
    assert set(event['parent_ids'])==set(event['retired_ids'])
    assert set(event['retired_ids']).isdisjoint(new.ids.tolist())
    assert torch.isfinite(new.factor()).all()
    assert (new.factor().diagonal(dim1=-2,dim2=-1)>0).all()
