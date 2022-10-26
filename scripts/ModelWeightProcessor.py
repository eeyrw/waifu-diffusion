import os
from pathlib import Path
import torch
import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--input', '-I', type=str, help='Input file to prune', required = True)
# args = parser.parse_args()
# file = args.input


# def prune_it(p):
#     print(f"prunin' in path: {p}")
#     size_initial = os.path.getsize(p)
#     nsd = dict()
#     sd = torch.load(p, map_location="cpu")
#     print(sd.keys())
#     for k in sd.keys():
#         if k != "optimizer_states":
#             nsd[k] = sd[k]
#     else:
#         print(f"removing optimizer states for path {p}")
#     if "global_step" in sd:
#         print(f"This is global step {sd['global_step']}.")
#     if keep_only_ema:
#         sd = nsd["state_dict"].copy()
#         # infer ema keys
#         ema_keys = {k: "model_ema." + k[6:].replace(".", "") for k in sd.keys() if k.startswith('model.')}
#         new_sd = dict()

#         for k in sd:
#             if k in ema_keys:
#                 print(k, ema_keys[k])
#                 new_sd[k] = sd[ema_keys[k]]
#             elif not k.startswith("model_ema.") or k in ["model_ema.num_updates", "model_ema.decay"]:
#                 new_sd[k] = sd[k]

#         assert len(new_sd) == len(sd) - len(ema_keys)
#         nsd["state_dict"] = new_sd
#     else:
#         sd = nsd['state_dict'].copy()
#         new_sd = dict()
#         for k in sd:
#             new_sd[k] = sd[k]
#         nsd['state_dict'] = new_sd

#     fn = f"{os.path.splitext(p)[0]}-pruned.ckpt" if not keep_only_ema else f"{os.path.splitext(p)[0]}-ema-pruned.ckpt"
#     print(f"saving pruned checkpoint at: {fn}")
#     torch.save(nsd, fn)
#     newsize = os.path.getsize(fn)
#     MSG = f"New ckpt size: {newsize*1e-9:.2f} GB. " + \
#           f"Saved {(size_initial - newsize)*1e-9:.2f} GB by removing optimizer states"
#     if keep_only_ema:
#         MSG += " and non-EMA weights"
#     print(MSG)


if __name__ == "__main__":
    weightFilePath = '/root/autodl-tmp/2022-10-25T13-20-49_fp16-no-ema-finetune/checkpoints/last.ckpt'
    size_initial = os.path.getsize(weightFilePath)
    weightDict = torch.load(weightFilePath, map_location="cpu")
    print(weightDict.keys())
    if "global_step" in weightDict:
        print(f"This is global step {weightDict['global_step']}.")
    if "epoch" in weightDict:
        print(f"This is epoch {weightDict['epoch']}.")
    with open('weight_keys.txt','w') as f:
        f.writelines([k+'\n' for k in weightDict['state_dict'].keys()])
    newStateDict = dict()
    keepKeys = ['model.diffusion_model','first_stage_model']
    for k in weightDict['state_dict'].keys():
        needKeep = False
        for keepKey in keepKeys:
            if k.startswith(keepKey):
                needKeep = True
                break
        if not needKeep:
            continue
        newStateDict[k] = weightDict['state_dict'][k]
    newWeightDict = dict({'state_dict':newStateDict})
    fn = 'minimal.ckpt'
    torch.save(newWeightDict, fn)
    newsize = os.path.getsize(fn)
    MSG = f"New ckpt size: {newsize*1e-9:.2f} GB. " + \
          f"Saved {(size_initial - newsize)*1e-9:.2f} GB"
    print(MSG)

    
