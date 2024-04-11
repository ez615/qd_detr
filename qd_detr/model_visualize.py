from torchviz import make_dot
import torch
from qd_detr.model import build_model
from qd_detr.config import BaseOptions
from sys import argv

device = torch.device("cuda")

src_txt = torch.randn(1, 23, 512).to(device, non_blocking=False)
src_txt_mask = torch.randn(1, 23).to(device, non_blocking=False)
src_vid = torch.randn(1, 75, 2818).to(device, non_blocking=False)
src_vid_mask = torch.randn(1, 75).to(device, non_blocking=False)


opt = BaseOptions().parse()
model, criterion = build_model(opt)
model.to(device)

output = model(src_txt, src_txt_mask, src_vid, src_vid_mask)
make_dot((output["pred_logits"], output["pred_spans"]), params=dict(model.named_parameters()), show_attrs=True, show_saved=True).render('model_viz', format='png')