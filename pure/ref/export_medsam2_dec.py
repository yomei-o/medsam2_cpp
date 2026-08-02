# Extract the MedSAM2 SAM2 decoder (prompt encoder + mask decoder) into a forward-order weight blob +
# parity refs. SAM2 decoder = SAM's + obj_score_token + conv_s0/conv_s1 high-res fusion + obj-score head.
# Prompt encoder is identical to SAM. B=1. Uses the real encoder features from a fixed image.
import os, numpy as np, torch
from sam2.build_sam import build_sam2
HERE=os.path.dirname(__file__)
m=build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", os.path.join(HERE,"MedSAM2_latest.pt"), device="cpu"); m.eval()
pe, md = m.sam_prompt_encoder, m.sam_mask_decoder
def P(t): return t.detach().cpu().numpy()
blob=bytearray()
def put(a): blob.extend(np.ascontiguousarray(np.asarray(a,np.float32)).ravel().tobytes())
def lin(x): put(P(x.weight).T); put(P(x.bias))
def conv(x): put(P(x.weight)); put(P(x.bias))
def ln(x): put(P(x.weight)); put(P(x.bias))
def emb(x): put(P(x.weight))
# ---- prompt encoder (SAM order) ----
put(P(pe.pe_layer.positional_encoding_gaussian_matrix))
for i in range(4): emb(pe.point_embeddings[i])
emb(pe.not_a_point_embed); emb(pe.no_mask_embed)
mds=pe.mask_downscaling; conv(mds[0]); ln(mds[1]); conv(mds[3]); ln(mds[4]); conv(mds[6])
# ---- mask decoder (SAM2): obj_score + iou + mask tokens ----
emb(md.obj_score_token); emb(md.iou_token); emb(md.mask_tokens)
def attn(a): lin(a.q_proj); lin(a.k_proj); lin(a.v_proj); lin(a.out_proj)
for L in md.transformer.layers:
    attn(L.self_attn); ln(L.norm1); attn(L.cross_attn_token_to_image); ln(L.norm2)
    lin(L.mlp.layers[0]); lin(L.mlp.layers[1]); ln(L.norm3); ln(L.norm4); attn(L.cross_attn_image_to_token)
attn(md.transformer.final_attn_token_to_image); ln(md.transformer.norm_final_attn)
up=md.output_upscaling
put(P(up[0].weight)); put(P(up[0].bias)); ln(up[1]); put(P(up[3].weight)); put(P(up[3].bias))
conv(md.conv_s0); conv(md.conv_s1)                                    # high-res fusion convs
for mm in md.output_hypernetworks_mlps:
    for l in mm.layers: lin(l)
for l in md.iou_prediction_head.layers: lin(l)
for l in md.pred_obj_score_head.layers: lin(l)
open(os.path.join(HERE,"dec_weights.bin"),"wb").write(blob)
np.frombuffer(bytes(blob),np.float32).astype(np.float16).tofile(os.path.join(HERE,"dec_weights_fp16.bin"))
with open(os.path.join(HERE,"dec_config.txt"),"w") as f:
    f.write("embed_dim 256\nimage_embed 64\ninput_image 1024\nnum_mask_tokens 4\nheads 8\ntf_depth 2\n")

# ---- parity ref: real encoder features + a point -> masks/iou/obj ----
x=torch.sin(torch.arange(1*3*1024*1024,dtype=torch.float32).reshape(1,3,1024,1024)*0.0005)
with torch.no_grad():
    out=m.image_encoder(x); fpn=out["backbone_fpn"]                  # [256²,128²,64²]
    image_embed=fpn[-1]
    high_res=[md.conv_s0(fpn[0]), md.conv_s1(fpn[1])]              # conv_s0/s1 applied: 256²/32, 128²/64
    pt=torch.tensor([[[500.0,460.0]]]); lb=torch.tensor([[1]])
    sparse,dense=pe(points=(pt,lb),boxes=None,masks=None)
    image_pe=pe.get_dense_pe()
    masks,iou,_,obj=md(image_embeddings=image_embed, image_pe=image_pe,
        sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense,
        multimask_output=True, repeat_image=False, high_res_features=high_res)
P(image_embed).tofile(os.path.join(HERE,"dec_emb.bin"))              # [1,256,64,64]
P(fpn[0]).tofile(os.path.join(HERE,"dec_hr256.bin")); P(fpn[1]).tofile(os.path.join(HERE,"dec_hr128.bin"))
P(masks).tofile(os.path.join(HERE,"dec_masks.bin")); P(iou).tofile(os.path.join(HERE,"dec_iou.bin")); P(obj).tofile(os.path.join(HERE,"dec_obj.bin"))
print(f"dec_weights.bin {len(blob)/1e6:.2f} MB")
print("masks",tuple(masks.shape),"iou",P(iou).ravel(),"obj",float(P(obj).ravel()[0]))
