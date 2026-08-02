# 3-frame propagation reference to validate the MULTI-frame memory bank (cond frame t_pos=0 + recent
# frame t_pos=6 + 2 object pointers). Uses SAM2's own methods. Captures frame2's conditioned feature.
import os, numpy as np, torch, torch.nn.functional as F
from sam2.build_sam import build_sam2
HERE=os.path.dirname(__file__)
m=build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", os.path.join(HERE,"MedSAM2_latest.pt"), device="cpu"); m.eval()
def P(t): return t.detach().cpu().numpy()
def enc(x):
    o=m.image_encoder(x); fpn=o["backbone_fpn"]
    return fpn,[f.flatten(2).permute(2,0,1) for f in fpn],[p.flatten(2).permute(2,0,1) for p in o["vision_pos_enc"]],[(f.shape[-2],f.shape[-1]) for f in fpn]
imgs=[torch.sin(torch.arange(3*1024*1024,dtype=torch.float32).reshape(1,3,1024,1024)*(0.0005+0.0001*k)) for k in range(3)]
with torch.no_grad():
    # frame0 (prompt)
    fpn0,feats0,pos0,sz=enc(imgs[0]); pix0=(feats0[-1]+m.no_mem_embed).permute(1,2,0).view(1,256,64,64)
    hr=[m.sam_mask_decoder.conv_s0(fpn0[0]),m.sam_mask_decoder.conv_s1(fpn0[1])]
    sp,de=m.sam_prompt_encoder(points=(torch.tensor([[[500.,460.]]]),torch.tensor([[1]])),boxes=None,masks=None)
    lr,iou,mt,osc=m.sam_mask_decoder(image_embeddings=pix0,image_pe=m.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sp,dense_prompt_embeddings=de,multimask_output=True,repeat_image=False,high_res_features=hr)
    b=int(iou[0].argmax()); optr0=m.obj_ptr_proj(mt[:,b]); hi=F.interpolate(lr[:,b:b+1],size=(1024,1024),mode="bilinear",align_corners=False)
    mm0,mp0=m._encode_new_memory(current_vision_feats=[feats0[-1]],feat_sizes=sz,pred_masks_high_res=hi,object_score_logits=osc,is_mask_from_pts=True)
    od={"cond_frame_outputs":{0:{"maskmem_features":mm0,"maskmem_pos_enc":mp0,"obj_ptr":optr0}},"non_cond_frame_outputs":{}}
    # frame1 (propagate)
    fpn1,feats1,pos1,sz=enc(imgs[1])
    pix1=m._prepare_memory_conditioned_features(frame_idx=1,is_init_cond_frame=False,current_vision_feats=[feats1[-1]],current_vision_pos_embeds=[pos1[-1]],feat_sizes=sz,output_dict=od,num_frames=3)
    hr1=[m.sam_mask_decoder.conv_s0(fpn1[0]),m.sam_mask_decoder.conv_s1(fpn1[1])]
    sp1,de1=m.sam_prompt_encoder(points=(torch.zeros(1,1,2),-torch.ones(1,1,dtype=torch.int32)),boxes=None,masks=None)
    lr1,iou1,mt1,osc1=m.sam_mask_decoder(image_embeddings=pix1,image_pe=m.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sp1,dense_prompt_embeddings=de1,multimask_output=False,repeat_image=False,high_res_features=hr1)
    optr1=m.obj_ptr_proj(mt1[:,0]); hi1=F.interpolate(lr1,size=(1024,1024),mode="bilinear",align_corners=False)
    mm1,mp1=m._encode_new_memory(current_vision_feats=[feats1[-1]],feat_sizes=sz,pred_masks_high_res=hi1,object_score_logits=osc1,is_mask_from_pts=False)
    od["non_cond_frame_outputs"][1]={"maskmem_features":mm1,"maskmem_pos_enc":mp1,"obj_ptr":optr1}
    # frame2 (propagate, MULTI-frame memory)
    fpn2,feats2,pos2,sz=enc(imgs[2])
    pix2=m._prepare_memory_conditioned_features(frame_idx=2,is_init_cond_frame=False,current_vision_feats=[feats2[-1]],current_vision_pos_embeds=[pos2[-1]],feat_sizes=sz,output_dict=od,num_frames=3)
# save frame2 inputs + the two memories + two obj ptrs + conditioned feat
P(feats2[-1]).tofile(os.path.join(HERE,"p3_feat2.bin")); P(pos2[-1]).tofile(os.path.join(HERE,"p3_pos2.bin"))
P(mm0).tofile(os.path.join(HERE,"p3_mm0.bin")); P(mm1).tofile(os.path.join(HERE,"p3_mm1.bin"))
P(optr0).tofile(os.path.join(HERE,"p3_optr0.bin")); P(optr1).tofile(os.path.join(HERE,"p3_optr1.bin"))
P(pix2).tofile(os.path.join(HERE,"p3_pix2.bin"))
print("frame2 conditioned", tuple(pix2.shape))
