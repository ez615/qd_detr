import os
import json
from glob import glob
import math
import numpy as np
from collections import OrderedDict, defaultdict
import multiprocessing as mp
from standalone_eval.utils import compute_average_precision_detection, \
    compute_temporal_iou_batch_cross, compute_temporal_iou_batch_paired, load_jsonl, get_ap

from run_on_video.data_utils import VideoLoader
import matplotlib.pyplot as plt

from qd_detr.config import BaseOptions
from qd_detr.start_end_dataset import \
    StartEndDataset, start_end_collate, prepare_batch_inputs
from tqdm import tqdm, trange
import random
import torch
from torch.utils.data import DataLoader
from qd_detr.span_utils import generalized_temporal_iou, span_cxw_to_xx

from qd_detr.config import TestOptions
import torch.backends.cudnn as cudnn
from qd_detr.model import build_model
from torch.utils.data import DataLoader
from qd_detr.inference import setup_model


def compute_average_precision_detection_wrapper(
        input_triple, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    qid, ground_truth, prediction = input_triple
    scores = compute_average_precision_detection(
        ground_truth, prediction, tiou_thresholds=tiou_thresholds)
    return qid, scores

def compute_mr_ap(submission, ground_truth, iou_thds=np.linspace(0.5, 0.95, 10),
                  max_gt_windows=None, max_pred_windows=10, num_workers=8, chunksize=50):
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    pred_qid2data = defaultdict(list)
    for d in submission:
        pred_windows = d["pred_relevant_windows"][:max_pred_windows] \
            if max_pred_windows is not None else d["pred_relevant_windows"]
        qid = d["qid"]
        for w in pred_windows:
            pred_qid2data[qid].append({
                "video-id": d["qid"],  # in order to use the API
                "t-start": w[0],
                "t-end": w[1],
                "score": w[2]
            })

    gt_qid2data = defaultdict(list)
    for d in ground_truth:
        gt_windows = d["relevant_windows"][:max_gt_windows] \
            if max_gt_windows is not None else d["relevant_windows"]
        qid = d["qid"]
        for w in gt_windows:
            gt_qid2data[qid].append({
                "video-id": d["qid"],
                "t-start": w[0],
                "t-end": w[1]
            })
    qid2ap_list = {}
    # start_time = time.time()
    data_triples = [[qid, gt_qid2data[qid], pred_qid2data[qid]] for qid in pred_qid2data]
    from functools import partial
    compute_ap_from_triple = partial(
        compute_average_precision_detection_wrapper, tiou_thresholds=iou_thds)

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            for qid, scores in pool.imap_unordered(compute_ap_from_triple, data_triples, chunksize=chunksize):
                qid2ap_list[qid] = scores
    else:
        for data_triple in data_triples:
            qid, scores = compute_ap_from_triple(data_triple)
            qid2ap_list[qid] = scores

    # print(f"compute_average_precision_detection {time.time() - start_time:.2f} seconds.")
    ap_array = np.array(list(qid2ap_list.values()))  # (#queries, #thd)
    # ap_thds = ap_array.mean(0)  # mAP at different IoU thresholds.
    # iou_thd2ap = dict(zip([str(e) for e in iou_thds], ap_thds))
    # iou_thd2ap["average"] = np.mean(ap_thds)
    # # formatting
    # iou_thd2ap = {k: float(f"{100 * v:.2f}") for k, v in iou_thd2ap.items()}
    return ap_array


def compute_mr_r1(submission, ground_truth, iou_thds=np.linspace(0.5, 0.95, 10), no_sorted=False):
    """If a predicted segment has IoU >= iou_thd with one of the 1st GT segment, we define it positive"""
    iou_thds = [float(f"{e:.2f}") for e in iou_thds]
    pred_qid2window = {d["qid"]: d["pred_relevant_windows"][0][:2] for d in submission}  # :2 rm scores
    # gt_qid2window = {d["qid"]: d["relevant_windows"][0] for d in ground_truth}
    gt_qid2window = {}
    for d in ground_truth:
        cur_gt_windows = d["relevant_windows"]
        cur_qid = d["qid"]
        cur_max_iou_idx = 0
        if len(cur_gt_windows) > 0:  # select the GT window that has the highest IoU
            cur_ious = compute_temporal_iou_batch_cross(
                np.array([pred_qid2window[cur_qid]]), np.array(d["relevant_windows"])
            )[0]
            cur_max_iou_idx = np.argmax(cur_ious)
        gt_qid2window[cur_qid] = cur_gt_windows[cur_max_iou_idx]

    qids = list(pred_qid2window.keys())
    vids = [d["vid"] for d in submission]
    queries = [d["query"] for d in submission]
    pred_windows = np.array([pred_qid2window[k] for k in qids]).astype(float)
    gt_windows = np.array([gt_qid2window[k] for k in qids]).astype(float)
    pred_gt_iou = compute_temporal_iou_batch_paired(pred_windows, gt_windows)
    
    # plot iou histogram
    # plt.hist(pred_gt_iou)
    # plt.savefig('new_loss_iou_hist.png')
    # plt.close()

    # plot query len & iou scatter
    # query_lens = [len(q.split(' ')) for q in queries]
    # plt.scatter(query_lens, pred_gt_iou)
    # plt.savefig('query_len_scatter.png')
    # plt.close()

    iou = [{"qid": qids[i], "iou": pred_gt_iou[i], "vid": vids[i], "pred_wds": pred_windows[i].tolist(), "gt_wds":gt_windows[i].tolist(), "query": queries[i]} for i in range(len(qids))]
    # iou_thd2recall_at_one = {}
    # for thd in iou_thds:
    #     iou_thd2recall_at_one[str(thd)] = float(f"{np.mean(pred_gt_iou >= thd) * 100:.2f}")

    # plot query len histogram
    # query_len = [len(d['query'].split(' ')) for d in iou if d['iou'] == 0]
    # plt.hist(query_len)
    # plt.savefig('query_len_hist.png')
    # plt.close()
    if no_sorted:
        return iou
    else:
        return sorted(iou, key=lambda x:x['iou'])


def result_evaluation(submission, gt, save_dir, exp_name):
    # with open('result_AP.txt', 'w') as f:
    #     ap_array = compute_mr_ap(submission, gt)
    #     for ap in ap_array:
    #         f.write(f'{ap}\n')

    pred_gt_iou = compute_mr_r1(submission, gt)    

    ious = [d['iou'] for d in pred_gt_iou]

    # plt.ylim(0, 300)
    n, bins, _ = plt.hist(ious)

    if not os.path.exists(f'{save_dir}'):
        os.makedirs(f'{save_dir}')

    for i in range(10):
        # print(n[i])
        # print(bins)
        if n[i] > 0:
            plt.text(bins[i] + 0.05, n[i] + 1, str(int(n[i])), fontsize=10, ha='center')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{exp_name}_iou_hist.png')
    plt.close()

    with open(f'{save_dir}/{exp_name}_result_IOU.jsonl', 'w') as f:    
        for iou in pred_gt_iou:
            f.write(f"{json.dumps(iou)}\n")

def window2clips(window, clip_len=2):
    return [int(window[0] / clip_len), int(window[1] / clip_len)]

def video_visualization(submission, gt, save_dir):
    print(f"Start submission visualization")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f'>> save path: {save_dir}')

    pred_gt_iou = compute_mr_r1(submission, gt)

    video_loader = VideoLoader(framerate=1/2, size=224, centercrop=True)

    ct = 0
    for d in tqdm(pred_gt_iou):
        pred_st, pred_end = window2clips(d['pred_wds'])
        gt_st, gt_end = window2clips(d['gt_wds'])

        video_frames = video_loader.read_video_from_file(video_path=f'../val/{d["vid"]}.mp4')
        video_frames = video_frames.permute(0, 2, 3, 1) / 255.0

        pred_frames = video_frames[pred_st:pred_end]
        gt_frames = video_frames[gt_st:gt_end]

        pred_len, gt_len = len(pred_frames), len(gt_frames)
        max_len = max(pred_len, gt_len)

        n_rows, n_cols = gt_len // 10 + pred_len // 10 + 2, min(10, max_len)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows), constrained_layout=True)
        fig.suptitle(f'query: {d["query"]}')
        for axis in axes.flatten():
            axis.axis('off')

        for i in range(gt_len):
            r = i // n_cols
            c = i % n_cols
            if i == 0:
                axes[r][c].set_title(f'Ground Truth {gt_st + i}', fontsize=6)
            else:
                axes[r][c].set_title(f'{gt_st + i}', fontsize=6)
            axes[r][c].imshow(gt_frames[i])
            # axes[r][c].axis('off')

        for i in range(pred_len):
            r = i // n_cols + gt_len // 10 + 1
            c = i % n_cols
            if i == 0:
                axes[r][c].set_title(f'Prediction {pred_st + i}', fontsize=6)
            else:
                axes[r][c].set_title(f'{pred_st + i}', fontsize=6)
            axes[r][c].imshow(pred_frames[i])
            # axes[r][c].axis('off')

        fig.savefig(f'./{save_dir}/{d["iou"]}_{d["qid"]}_{d["vid"]}.png', bbox_inches='tight', pad_inches=0)
        plt.close()
        ct += 1

        # if ct >= 20:
        #     break

def video_visualization_full(submission, gt, save_dir):
    print(f"Start submission visualization")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    print(f'>> save path: {save_dir}')

    pred_gt_iou = compute_mr_r1(submission, gt)

    video_loader = VideoLoader(framerate=1/2, size=224, centercrop=True)

    ct = 0
    for d in tqdm(pred_gt_iou):
        # if d['iou'] == 0 or d['iou'] > 0.8:
        vid_path = os.path.join("/workspace/val", d['vid'] + ".mp4")
        video_frames = video_loader.read_video_from_file(video_path=vid_path)
        video_frames = video_frames.permute(0, 2, 3, 1) / 255.0

        vid_len = len(video_frames)
        pred_st, pred_end = window2clips(d['pred_wds'])
        gt_st, gt_end = window2clips(d['gt_wds'])

        n_rows, n_cols = 5, 15
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows), constrained_layout=True)
        
        fig.suptitle(f'Query: {d["query"]}')

        for axis in axes.flatten():
            axis.axis('off')
        
        for i in range(vid_len):
            r = i // n_cols
            c = i % n_cols

            axes[r][c].set_title(str(i), fontsize=6)
            if gt_st <= i < gt_end:
                axes[r][c].text(0, 0, 'GT', fontsize=6, color='green')

            axes[r][c].imshow(video_frames[i])
            
            # if pred_st <= i <= pred_end:
            #     axes[r][c].text(170, 0, 'Pred', fontsize=6, color='blue')
            #     axes[r][c].imshow(video_frames[i])
            
            # else:
            #     axes[r][c].imshow(video_frames[i], alpha=0.6)
        
        fig.savefig(f'{save_dir}/{d["vid"]}_{d["qid"]}.png', bbox_inches='tight', pad_inches=0)
        plt.close()


def similar_clip_eval(threshold=0.9):
    opt = BaseOptions().parse()

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    dataset_config = dict(
        dset_name=opt.dset_name,
        data_path=opt.train_path,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        data_ratio=opt.data_ratio,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=opt.txt_drop_ratio
    )

    dataset_config["data_path"] = opt.train_path
    train_dataset = StartEndDataset(**dataset_config)

    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
        pin_memory=opt.pin_memory
    )

    num_training_examples = len(train_loader)
    
    sim_count = []
    zero_ct = 0
    for batch_idx, batch in tqdm(enumerate(train_loader),
                                 desc="Training Iteration",
                                 total=num_training_examples):
        # vids = batch[0]['qid']
        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)

        # src_txt = model_inputs['src_txt']
        src_vid = model_inputs['src_vid'][:,:,2305:-1]   #(bsz, #clips, vid_D=512)

        targets = targets['span_labels']
        tgt_spans = [(span_cxw_to_xx(t['spans']) * 75).detach().cpu() for t in targets]  # (#spans, 2)
        # print(f'tgt_spans: {tgt_spans}')

        for vid, span in zip(src_vid, tgt_spans):  # vid shape: (#clips, 512)
            ct = 0
            # sim_clips_idx = []
            for s in span:
                st, end = int(s[0]), int(s[1])
                mid = (st + end) // 2

                tgt_clip = vid[mid].detach()
                # print(f'tgt_vid_shape: {tgt_clip.shape}')
                
                other_clips = torch.cat([vid[:st], vid[end + 1:]], dim=0).detach()
                # indices = torch.cat([range(0, st), range(end + 1, 75)], dim=0)
                # print(f'other vids shape: {other_clips.shape}')

                tgt_clip /= tgt_clip.norm(dim=-1, keepdim=True) + 1e-6
                other_clips /= other_clips.norm(dim=-1, keepdim=True) +1e-6
                
                for other in other_clips:
                    sim = tgt_clip @ other.T
                    # print(f'sim: {sim}')
                    
                    if sim >= threshold:
                        ct += 1
                        # sim_clips_idx.append(idx)
            if ct == 0:
                zero_ct += 1
            sim_count.append(ct / len(span))

        
        # print(f'sim_count: {sim_count}')
    print(f'sim count len: {len(sim_count)} max: {max(sim_count)}')
    plt.hist(sim_count, bins=30)
    plt.savefig(f'sim_hist_{threshold}_zero{zero_ct}.png')
    plt.close()

def compare_result(baseline, compare, gt, save_dir):
    print(f"Start compare submission to baseline and visualize")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        # os.mkdir(save_dir + '/same')
        os.mkdir(save_dir + '/dec')
        os.mkdir(save_dir + '/inc')

        print(f'>> save path: {save_dir}')

    baseline = compute_mr_r1(baseline, gt)
    compare = compute_mr_r1(compare, gt)

    baseline = sorted(baseline, key=lambda x: x['qid'])
    compare = sorted(compare, key=lambda x: x['qid'])

    video_loader = VideoLoader(framerate=1/2, size=224, centercrop=True)

    def compare_visualize(mode, delta):
        b_st, b_end = window2clips(b['pred_wds'])
        c_st, c_end = window2clips(c['pred_wds'])
        gt_st, gt_end = window2clips(b['gt_wds'])

        b_frames, gt_frames = video_loader.extract_clips(video_path=f'../val/{b["vid"]}.mp4', 
                                pred_st=b_st, pred_end=b_end, gt_st=gt_st, gt_end=gt_end)
        c_frames, _ = video_loader.extract_clips(video_path=f'../val/{b["vid"]}.mp4', 
                                pred_st=c_st, pred_end=c_end, gt_st=gt_st, gt_end=gt_end)

        b_len, c_len = len(b_frames), len(c_frames)
        pred_len, gt_len = b_len + c_len, len(gt_frames)
        max_len = max(pred_len, gt_len)

        n_rows, n_cols = gt_len // 10 + b_len // 10 + c_len // 10 + 3, min(10, max_len)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows), constrained_layout=True)
        fig.suptitle(f'query: {b["query"]}\nbaseline iou: {b["iou"]:.2f}    compare iou: {c["iou"]:.2f}')
        for axis in axes.flatten():
            axis.axis('off')

        for i in range(gt_len):
            row = i // n_cols
            col = i % n_cols
            if i == 0:
                axes[row][col].set_title(f'Ground Truth {gt_st + i}', fontsize=6)
            else:
                axes[row][col].set_title(f'{gt_st + i}', fontsize=6)
            axes[row][col].imshow(gt_frames[i])

        for i in range(b_len):
            row = i // n_cols + gt_len // 10 + 1
            col = i % n_cols
            if i == 0:
                axes[row][col].set_title(f'Baseline {b_st + i}', fontsize=6)
            else:
                axes[row][col].set_title(f'{b_st + i}', fontsize=6)
            axes[row][col].imshow(b_frames[i])
        
        for i in range(c_len):
            row = i // n_cols + gt_len // 10 + b_len // 10 + 2
            col = i % n_cols
            if i == 0:
                axes[row][col].set_title(f'Compare {c_st + i}', fontsize=6)
            else:
                axes[row][col].set_title(f'{c_st + i}', fontsize=6)
            axes[row][col].imshow(c_frames[i])

        fig.savefig(f'{save_dir}/{mode}/{delta:.2f}_{b["qid"]}_{b["vid"]}.png', bbox_inches='tight', pad_inches=0)
        plt.close()

    # print([b['qid'] != c['qid'] for b, c in zip(baseline, compare)])

    for b, c in zip(tqdm(baseline), compare):
        delta = c['iou'] - b['iou']

        if delta == 0:
            # compare_visualize('same', delta)
            continue
        elif delta > 0:
            compare_visualize('inc', delta)
        else:
            compare_visualize('dec', -delta)

def chk_training_process(submissions_path, save_dir, n_chk=20, save_jsonl=True):
    if not os.path.exists(save_dir):
        print(f'\n>>> saved at {save_dir}\n')
        os.makedirs(save_dir)

    subm_files_lst = sorted(glob(submissions_path + '/*'), key=os.path.getctime)
    sumb_lst = [load_jsonl(s) for s in subm_files_lst]

    best_ckpt_subm = sorted(sumb_lst[-1], key=lambda x: max(x['iou']))
    chk_lst = best_ckpt_subm[:n_chk] + best_ckpt_subm[-n_chk:]
    chk_qid_lst = [c['qid'] for c in chk_lst]

    ## for debug
    # for c in chk_lst:
    #     print(c['vid'], end='.mp4 ')
    # print()
    # return

    chk_sumb_lst = []
    for sumb in sumb_lst:
        chk_sumb_lst.append(sorted([s for s in sumb if s['qid'] in chk_qid_lst], key=lambda x: x['qid']))

    ### for debug
    # for i, cs in enumerate(chk_sumb_lst[-1]):
    #     print(f'i: {i}\tsumb qid: {cs["qid"]}\tiou: {max(cs["iou"])}')
    # return

    n_ckpts = len(chk_sumb_lst)
    total_n_chk = 2 * n_chk

    if save_jsonl:
        os.makedirs(f'{save_dir}/jsonl', exist_ok=True)
        for n in range(n_ckpts):
            with open(f'{save_dir}/jsonl/ckpt{n}_chk_sumb.jsonl', 'w') as f:    
                for cs in chk_sumb_lst[n]:
                    f.write(f"{json.dumps(cs)}\n")

    video_loader = VideoLoader(framerate=1/2, size=224, centercrop=True)

    for n in tqdm(range(total_n_chk), desc='for each videos'):
        vid = chk_sumb_lst[0][n]['vid']
        qid = chk_sumb_lst[0][n]['qid']
        final_iou = max(chk_sumb_lst[-1][n]["iou"])

        vid_path = f'/workspace/qv_train/{vid}.mp4'
        video_frames = video_loader.read_video_from_file(vid_path)
        video_frames = video_frames.permute(0, 2, 3, 1) / 255.0
        vid_len = len(video_frames)

        for ckpt in tqdm(range(n_ckpts), desc='for each ckpts'):
            s = chk_sumb_lst[ckpt][n]

            for i, (sim_loss, iou, pred_span, gt_span) in enumerate(zip(s['sim_loss'], s['iou'], s['pred_span'], s['gt_span'])):
                img_save_path = os.path.join(save_dir, f'{final_iou:.2f}_{qid}_{vid}_{i}')
                os.makedirs(img_save_path, exist_ok=True)

                if iou == 1:
                    sim = 1
                else:
                    sim = 2 - sim_loss / (1 - iou)

                pred_st, pred_end = pred_span
                gt_st, gt_end = gt_span

                n_rows, n_cols = 5, math.ceil(vid_len / 5)
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows), constrained_layout=True)
                
                fig.suptitle(f'Query: {s["query"]}\niou:  {iou:.4f}\nsim_loss:  {sim_loss:.4f}    sim:  {sim:.4f}')

                for axis in axes.flatten():
                    axis.axis('off')
                
                for i in range(vid_len):
                    r = i // n_cols
                    c = i % n_cols

                    axes[r][c].set_title(str(i), fontsize=6)
                    if gt_st <= i <= gt_end:
                        axes[r][c].text(0, 0, 'GT', fontsize=6, color='green')
                        axes[r][c].imshow(video_frames[i])
                    
                    if pred_st <= i <= pred_end:
                        axes[r][c].text(170, 0, 'Pred', fontsize=6, color='blue')
                        axes[r][c].imshow(video_frames[i])
                    
                    else:
                        axes[r][c].imshow(video_frames[i], alpha=0.6)
                
                fig.savefig(f'{img_save_path}/ckpt_{ckpt}_{iou:.4f}.png', bbox_inches='tight', pad_inches=0)
                plt.close()


@torch.no_grad()
def chk_similarity(exp, gt, loss_type, img_save_path, top_k=10):
    opt = TestOptions().parse()

    opt.eval_split_name = 'val'
    opt.eval_path = '/workspace/qd_detr/data/highlight_val_release.jsonl'
    opt.resume = os.path.join(exp, 'model_best.ckpt')
    opt.loss_type = loss_type

    cudnn.benchmark = True
    cudnn.deterministic = False

    eval_dataset = StartEndDataset(
        dset_name=opt.dset_name,
        data_path=opt.eval_path,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        data_ratio=opt.data_ratio,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        load_labels=True,  # opt.eval_split_name == "val",
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=0,
        dset_domain=opt.dset_domain,
    )

    model = setup_model(opt)[0]

    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=opt.pin_memory
    )

    video_loader = VideoLoader(framerate=1/2, size=224, centercrop=True)

    print(f'\n>>> saved at {img_save_path}\n')
    os.makedirs(img_save_path, exist_ok=True)

    submission = load_jsonl(os.path.join(exp, 'best_hl_val_preds.jsonl'))
    mr_result = compute_mr_r1(submission, gt, no_sorted=True)  # dict list

    model.eval()

    for i, (query_meta, batch) in enumerate(tqdm(eval_loader, desc="chk similarity")):
        model_inputs, _ = prepare_batch_inputs(batch, opt.device, non_blocking=opt.pin_memory)

        result_idx = i * opt.eval_bsz
        result = mr_result[result_idx:result_idx + opt.eval_bsz]

        sims = model.get_similarity_map(**model_inputs)[1]

        # print(f'subm: {[d["qid"] for d in subm]}')
        # print(f'meta: {[d["qid"] for d in query_meta]}')
        for idx, sim in enumerate(tqdm(sims, desc='in batch')):
            result_i = result[idx]

            gt_st, gt_ed = window2clips(result_i['gt_wds'])
            gt_ed -= 1

            if gt_ed < 1:
                continue
            
            pred_st, pred_ed = window2clips(result_i['pred_wds'])
            pred_ed -= 1

            gt_sim = sim[gt_st:gt_ed, :]
            indices = gt_sim.sort(descending=True)[1][:, :top_k]
            max_sims = gt_sim.max(dim=1)[0]
            # print(f'max_sims: {max_sims}\nmax: {max_sims.max().item()}\tmean:{max_sims.mean()}\tstd: {max_sims.std()}')

            vid_path = f'/workspace/val/{result_i["vid"]}.mp4'
            video_frames = video_loader.read_video_from_file(vid_path)
            video_frames = video_frames.permute(0, 2, 3, 1) / 255.0
            
            n_rows, n_cols = max(indices.shape[0], 2), top_k + 1
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows), constrained_layout=True)
            
            fig.suptitle(f'Query: {result_i["query"]}\niou:  {result_i["iou"]:.4f}\ngt span: [{gt_st}, {gt_ed}]    pred_span: [{pred_st}, {pred_ed}]\nmax: {max_sims.max().item():.4f}    mean:{max_sims.mean():.4f}    std: {max_sims.std():.4f}\nCV: {max_sims.std() / max_sims.mean()}')

            for axis in axes.flatten():
                axis.axis('off')
            
            for r in range(n_rows):
                gt_i = gt_st + r
                axes[r][0].set_title(f'GT {gt_i}', fontsize=6)
                for c in range(n_cols):
                    if c == 0:
                        axes[r][c].imshow(video_frames[gt_i])
                    else:
                        indice = indices[r][c - 1]
                        axes[r][c].set_title(f'clip {indice}: {sim[gt_i, indice].item():.4f}', fontsize=6)
                        try:
                            if pred_st <= indice < pred_ed:
                                axes[r][c].imshow(video_frames[indice])
                            else:
                                axes[r][c].imshow(video_frames[indice], alpha=0.6)
                        except:
                            print(f'video_frames len:{len(video_frames)}\tindice: {indice}\ngt span: [{gt_st}, {gt_ed}]')
            
            fig.savefig(f'{img_save_path}/{result_i["iou"]:.4f}_{result_i["qid"]}.png', bbox_inches='tight', pad_inches=0)
            plt.close()
            
        break

@torch.no_grad()
def chk_self_sim(exp, gt, loss_type, img_save_path):
    opt = TestOptions().parse()

    opt.eval_split_name = 'val'
    opt.eval_path = '/workspace/qd_detr/data/highlight_val_release.jsonl'
    opt.resume = os.path.join(exp, 'model_best.ckpt')
    opt.loss_type = loss_type

    cudnn.benchmark = True
    cudnn.deterministic = False

    eval_dataset = StartEndDataset(
        dset_name=opt.dset_name,
        data_path=opt.eval_path,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        data_ratio=opt.data_ratio,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        load_labels=True,  # opt.eval_split_name == "val",
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=0,
        dset_domain=opt.dset_domain,
    )

    model = setup_model(opt)[0]

    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=False,
        pin_memory=opt.pin_memory
    )

    video_loader = VideoLoader(framerate=1/2, size=224, centercrop=True)

    print(f'\n>>> saved at {img_save_path}\n')
    os.makedirs(img_save_path, exist_ok=True)

    submission = load_jsonl(os.path.join(exp, 'best_hl_val_preds.jsonl'))
    mr_result = compute_mr_r1(submission, gt, no_sorted=True)  # dict list

    model.eval()

    for i, (query_meta, batch) in enumerate(tqdm(eval_loader, desc="chk similarity")):
        model_inputs, _ = prepare_batch_inputs(batch, opt.device, non_blocking=opt.pin_memory)

        result_idx = i * opt.eval_bsz
        result = mr_result[result_idx:result_idx + opt.eval_bsz]

        sims = model.get_similarity_map(**model_inputs)[1]

        # print(f'subm: {[d["qid"] for d in subm]}')
        # print(f'meta: {[d["qid"] for d in query_meta]}')
        for idx, sim in enumerate(tqdm(sims, desc='in batch')):
            result_i = result[idx]

            gt_st, gt_ed = window2clips(result_i['gt_wds'])
            
            pred_st, pred_ed = window2clips(result_i['pred_wds'])

            self_sims = sim[gt_st:gt_ed, gt_st:gt_ed].diag()
            cv = self_sims.std() / self_sims.mean()
            # print(f'max_sims: {max_sims}\nmax: {max_sims.max().item()}\tmean:{max_sims.mean()}\tstd: {max_sims.std()}')

            vid_path = f'/workspace/val/{result_i["vid"]}.mp4'
            video_frames = video_loader.read_video_from_file(vid_path)
            video_frames = video_frames.permute(0, 2, 3, 1) / 255.0
            vid_len = len(video_frames)
            
            n_rows, n_cols = 5, math.ceil(vid_len / 5)
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows), constrained_layout=True)
            
            fig.suptitle(f'Query: {result_i["query"]}\niou:  {result_i["iou"]:.4f}\nmean:{self_sims.mean():.4f}    std: {self_sims.std():.4f}\nCV: {cv}')

            for axis in axes.flatten():
                axis.axis('off')
            
            idx = 0
            for j in range(vid_len):
                r = j // n_cols
                c = j % n_cols

                axes[r][c].set_title(str(j), fontsize=6)
                if gt_st <= j < gt_ed:
                    axes[r][c].text(0, 0, f'GT    {self_sims[idx]:.4f}', fontsize=6, color='green')
                    axes[r][c].imshow(video_frames[j])
                    idx += 1
                
                if pred_st <= j <= pred_ed:
                    axes[r][c].text(170, 0, 'Pred', fontsize=6, color='blue')
                    axes[r][c].imshow(video_frames[j])
                
                else:
                    axes[r][c].imshow(video_frames[j], alpha=0.6)
            
            fig.savefig(f'{img_save_path}/CV_{cv:.4f}_{result_i["iou"]:.4f}_{result_i["qid"]}.png', bbox_inches='tight', pad_inches=0)
            plt.close()
        
        break
        
if __name__ == "__main__":
    gt_path = '/workspace/qd_detr/data/highlight_val_release.jsonl'
    submission_path = '/workspace/qd_detr/results/loss0/no_pt_re-2024_01_11_08_44_41/best_hl_val_preds.jsonl'

    submission = load_jsonl(submission_path)
    gt = load_jsonl(gt_path)

    exp_name = ('-').join(submission_path.split('/')[-2].split('-')[:-1])
    # save_dir = os.path.join('evaluation', submission_path.split('/')[4])
    
    # result_evaluation(submission, gt, save_dir, exp_name)

    save_dir = os.path.join('visualize', submission_path.split('/')[4], exp_name)

    video_visualization(submission, gt, save_dir + '/video_vis')
    # similar_clip_eval(0.6)
    # similar_clip_eval(0.8)
    # similar_clip_eval(0.9)

    # baseline_path = '/workspace/QD-DETR/results/loss0/no_pt-2023_12_26_08_49_07/best_hl_val_preds.jsonl'
    # baseline = load_jsonl(baseline_path)

    # compare_result(baseline, submission, gt, save_dir + '/compare')

    # exp_path = '/workspace/qd_detr/results/loss2/temp_re-2024_03_27_07_39_41'
    # exp_name = ('-').join(exp_path.split('/')[-1].split('-')[:-1])
    # save_dir = os.path.join('visualize', exp_path.split('/')[4], exp_name)

    # chk_training_process(exp_path + '/submissions', save_dir + '/chk_training')

    # chk_similarity(exp_path, gt, '2', save_dir + '/chk_similarity_CV')
    # chk_self_sim(exp_path, gt, '2', save_dir + '/chk_self_sim2')