import numpy as np
import pytest
import torch

from flysurvivors import Connectome, LIFBrain
import flysurvivors.kernels as kernels

@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_grouped_propagation_matches_original():
    if not kernels.HAS_TRITON:
        pytest.skip("needs Triton")
    edges=[(0,8,10),(1,8,20),(2,9,-5),(3,9,15),(8,9,7)]
    n=10
    cn=Connectome(np.arange(1000,1000+n,dtype=np.int64),np.array([e[0] for e in edges],dtype=np.int32),np.array([e[1] for e in edges],dtype=np.int32),np.array([e[2] for e in edges],dtype=np.float32))
    brain=LIFBrain(cn,device="cuda",backend="triton")
    grouped=getattr(kernels,"propagate_grouped_kernel",None)
    assert grouped is not None, "propagate_grouped_kernel todavía no existe"
    brain.hist.zero_()
    brain.hist_pos.zero_()
    read_row=(int(brain.hist_pos.item())-brain.delay_steps)%brain.hist_len
    brain.hist[read_row,:4]=1.0
    brain.hist[read_row,8]=1.0
    original=torch.zeros_like(brain.g_inc)
    candidate=torch.zeros_like(brain.g_inc)
    kernels.propagate_kernel[(brain.n,)](brain.hist,brain.hist_pos,brain.rowptr,brain.col,brain.val,original,brain.n,brain.delay_steps,brain.hist_len,BLOCK=128)
    grouped[((brain.n+7)//8,)](brain.hist,brain.hist_pos,brain.rowptr,brain.col,brain.val,candidate,brain.n,brain.delay_steps,brain.hist_len,BLOCK=128,GROUP=8)
    torch.cuda.synchronize()
    torch.testing.assert_close(candidate,original,rtol=1e-5,atol=1e-6)
