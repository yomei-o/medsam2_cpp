# Build a 2-frame propagation parity reference using SAM2's own methods, + export the propagation
# constants/weights. frame0 (prompt) -> memory; frame1 conditioned on frame0's memory. Saves the memory
# bank + the conditioned feature (memory_attention output) so C++ can parity-check the wiring.
import os, numpy as np, torch, torch.nn.functional as F
from sam2.build_sam import build_sam2
from sam2.modeling.sam2_utils import get_1d_sine_pe
HERE=os.path.dirname(__file__)
m=build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", os.path.join(HERE,"MedSAM2_latest.pt"), device="cpu"); m.eval()
def P(t): return t.detach().cpu().numpy()

# ---- constants + weights ----
blob=bytearray()
def put(a): blob.extend(np.ascontiguousarray(np.asarray(a,np.float32)).ravel().tobytes())
put(P(m.no_mem_embed)); put(P(m.no_mem_pos_enc)); put(P(m.maskmem_tpos_enc)); put(P(m.no_obj_embed_spatial)); put(P(m.no_obj_ptr))
for l in m.obj_ptr_proj.layers: put(P(l.weight).T); put(P(l.bias))
put(P(m.obj_ptr_tpos_proj.weight).T); put(P(m.obj_ptr_tpos_proj.bias))
# precompute maskmem_pos_enc (sine) at 64x64 -> [1,64,64,64]
pe = m.memory_encoder.position_encoding
mm_pos = pe(torch.zeros(1,64,64,64))          # [1,64,64,64]
put(P(mm_pos))
open(os.path.join(HERE,"prop_weights.bin"),"wb").write(blob)

# ---- run 2 frames ----
torch.manual_seed(0)
img0=torch.sin(torch.arange(3*1024*1024,dtype=torch.float32).reshape(1,3,1024,1024)*0.0005)
img1=torch.cos(torch.arange(3*1024*1024,dtype=torch.float32).reshape(1,3,1024,1024)*0.0006)
def enc(x):
    o=m.image_encoder(x); fpn=o["backbone_fpn"]; pos=o["vision_pos_enc"]
    # current_vision_feats = flattened top level [(HW),B,C]; feat_sizes
    feats=[f.flatten(2).permute(2,0,1) for f in fpn]; posf=[p.flatten(2).permute(2,0,1) for p in pos]
    sizes=[(f.shape[-2],f.shape[-1]) for f in fpn]
    return fpn, feats, posf, sizes
with torch.no_grad():
    fpn0, feats0, pos0, sizes0 = enc(img0)
    # frame0 conditioning: directly add no_mem_embed
    B=1; C=256; H,W=sizes0[-1]
    pix0 = (feats0[-1] + m.no_mem_embed).permute(1,2,0).view(B,C,H,W)
    # decode frame0 with a point prompt
    hr=[m.sam_mask_decoder.conv_s0(fpn0[0]), m.sam_mask_decoder.conv_s1(fpn0[1])]
    pt=torch.tensor([[[500.,460.]]]); lb=torch.tensor([[1]])
    sp,de=m.sam_prompt_encoder(points=(pt,lb),boxes=None,masks=None)
    lowres,iou,mask_tok,objsc=m.sam_mask_decoder(image_embeddings=pix0, image_pe=m.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sp, dense_prompt_embeddings=de, multimask_output=True, repeat_image=False, high_res_features=hr)
    best=int(iou[0].argmax()); sam_tok=mask_tok[:,best]
    obj_ptr0=m.obj_ptr_proj(sam_tok)
    hires=F.interpolate(lowres[:,best:best+1], size=(1024,1024), mode="bilinear", align_corners=False)
    mm0, mmpos0 = m._encode_new_memory(current_vision_feats=[feats0[-1]], feat_sizes=sizes0,
        pred_masks_high_res=hires, object_score_logits=objsc, is_mask_from_pts=True)
    # frame1: prepare memory-conditioned features
    fpn1, feats1, pos1, sizes1 = enc(img1)
    outdict={"cond_frame_outputs":{0:{"maskmem_features":mm0,"maskmem_pos_enc":mmpos0,"obj_ptr":obj_ptr0}}, "non_cond_frame_outputs":{}}
    pix1=m._prepare_memory_conditioned_features(frame_idx=1, is_init_cond_frame=False,
        current_vision_feats=[feats1[-1]], current_vision_pos_embeds=[pos1[-1]], feat_sizes=sizes1,
        output_dict=outdict, num_frames=2)
    # replicate the memory-bank assembly to save it for parity isolation
    to_mem=[]; to_pos=[]
    to_mem.append(mm0.flatten(2).permute(2,0,1))                     # spatial [4096,1,64]
    me=mmpos0[-1].flatten(2).permute(2,0,1) + m.maskmem_tpos_enc[m.num_maskmem-0-1]
    to_pos.append(me)
    op=obj_ptr0.reshape(-1,1,256//64,64).permute(0,2,1,3).flatten(0,1)   # [4,1,64]
    ops=get_1d_sine_pe(torch.tensor([1.0])/(16-1), dim=256); ops=m.obj_ptr_tpos_proj(ops).unsqueeze(1).expand(-1,1,64).repeat_interleave(4,dim=0)
    to_mem.append(op); to_pos.append(ops)
    memory=torch.cat(to_mem,0); memory_pos=torch.cat(to_pos,0)
P(feats1[-1]).tofile(os.path.join(HERE,"prop_feat1.bin"))     # [4096,1,256]
P(pos1[-1]).tofile(os.path.join(HERE,"prop_pos1.bin"))
P(mm0).tofile(os.path.join(HERE,"prop_mm0.bin"))              # [1,64,64,64]
P(obj_ptr0).tofile(os.path.join(HERE,"prop_objptr0.bin"))     # [1,256]
P(objsc).tofile(os.path.join(HERE,"prop_objsc0.bin"))
P(pix1).tofile(os.path.join(HERE,"prop_pix1.bin"))
P(memory[:,0]).tofile(os.path.join(HERE,"prop_memory.bin"))
P(memory_pos[:,0]).tofile(os.path.join(HERE,"prop_mempos.bin"))
print("memory",tuple(memory.shape))
# frame1 decode (no prompt -> dummy point (0,0) label -1) using the conditioned feature pix1
with torch.no_grad():
    hr1=[m.sam_mask_decoder.conv_s0(fpn1[0]), m.sam_mask_decoder.conv_s1(fpn1[1])]
    dpc=torch.zeros(1,1,2); dpl=-torch.ones(1,1,dtype=torch.int32)
    sp1,de1=m.sam_prompt_encoder(points=(dpc,dpl),boxes=None,masks=None)
    lr1,iou1,mt1,os1=m.sam_mask_decoder(image_embeddings=pix1, image_pe=m.sam_prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sp1, dense_prompt_embeddings=de1, multimask_output=False, repeat_image=False, high_res_features=hr1)
P(fpn1[0]).tofile(os.path.join(HERE,"prop_hr256_1.bin")); P(fpn1[1]).tofile(os.path.join(HERE,"prop_hr128_1.bin"))
P(lr1).tofile(os.path.join(HERE,"prop_mask1.bin")); P(iou1).tofile(os.path.join(HERE,"prop_iou1.bin"))
print("frame1 mask", tuple(lr1.shape), "iou", float(iou1.ravel()[0]), "sparse1", tuple(sp1.shape))
print("prop_weights.bin", len(blob)/1e6, "MB")
print("obj_ptr0", tuple(obj_ptr0.shape), "mm0", tuple(mm0.shape), "pix1", tuple(pix1.shape))
