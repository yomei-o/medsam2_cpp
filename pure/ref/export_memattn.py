# Extract the MedSAM2 memory_attention (4 layers, RoPE self+cross attn) weights + a synthetic-input
# parity ref. curr [N=4096,1,256] self-attends (RoPE); cross-attends to memory [M,1,64] (RoPE, k=mem+pos).
import os, numpy as np, torch
from sam2.build_sam import build_sam2
HERE=os.path.dirname(__file__)
m=build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", os.path.join(HERE,"MedSAM2_latest.pt"), device="cpu"); m.eval()
ma=m.memory_attention
def P(t): return t.detach().cpu().numpy()
blob=bytearray()
def put(a): blob.extend(np.ascontiguousarray(np.asarray(a,np.float32)).ravel().tobytes())
def lin(x): put(P(x.weight).T); put(P(x.bias))
def ln(x): put(P(x.weight)); put(P(x.bias))
for L in ma.layers:
    lin(L.self_attn.q_proj); lin(L.self_attn.k_proj); lin(L.self_attn.v_proj); lin(L.self_attn.out_proj); ln(L.norm1)
    lin(L.cross_attn_image.q_proj); lin(L.cross_attn_image.k_proj); lin(L.cross_attn_image.v_proj); lin(L.cross_attn_image.out_proj); ln(L.norm2)
    lin(L.linear1); lin(L.linear2); ln(L.norm3)
ln(ma.norm)
open(os.path.join(HERE,"mem_weights.bin"),"wb").write(blob)

# synthetic parity ref: 1 memory frame (64x64), 0 obj ptrs
N=64*64
curr=torch.sin(torch.arange(N*256,dtype=torch.float32).reshape(N,1,256)*0.001)
curr_pos=torch.cos(torch.arange(N*256,dtype=torch.float32).reshape(N,1,256)*0.0007)
memory=torch.sin(torch.arange(N*64,dtype=torch.float32).reshape(N,1,64)*0.0013+1.0)
memory_pos=torch.cos(torch.arange(N*64,dtype=torch.float32).reshape(N,1,64)*0.0009)
with torch.no_grad():
    out=ma(curr=curr, memory=memory, curr_pos=curr_pos, memory_pos=memory_pos, num_obj_ptr_tokens=0)
P(curr).tofile(os.path.join(HERE,"mem_curr.bin")); P(curr_pos).tofile(os.path.join(HERE,"mem_currpos.bin"))
P(memory).tofile(os.path.join(HERE,"mem_mem.bin")); P(memory_pos).tofile(os.path.join(HERE,"mem_mempos.bin"))
P(out).tofile(os.path.join(HERE,"mem_out.bin"))
print(f"mem_weights.bin {len(blob)/1e6:.2f} MB")
print("pos_enc_at_input:", ma.pos_enc_at_input, "| out", tuple(out.shape))
