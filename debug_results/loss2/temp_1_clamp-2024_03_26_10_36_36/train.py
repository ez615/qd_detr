import os
import time
import json
import pprint
import random
import numpy as np
from tqdm import tqdm, trange
from collections import defaultdict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from qd_detr.config import BaseOptions
from qd_detr.start_end_dataset import \
    StartEndDataset, start_end_collate, prepare_batch_inputs
from qd_detr.span_utils import span_cxw_to_xx
from qd_detr.start_end_dataset_audio import \
    StartEndDataset_audio, start_end_collate_audio, prepare_batch_inputs_audio
from qd_detr.postprocessing_qd_detr import PostProcessorDETR
from qd_detr.inference import eval_epoch, start_inference, setup_model
from utils.basic_utils import AverageMeter, dict_to_markdown, save_jsonl
from utils.model_utils import count_parameters


import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def set_seed(seed, use_cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer):
    logger.info(f"[Epoch {epoch_i+1}]")
    model.train()
    criterion.train()

    ### ADDED
    giou_coeffs = np.array([0., 0., 0., 0., 0., 1., 1., 1., 1., 1.,])  # linear: [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]
    sim_coeffs = np.flip(giou_coeffs)

    # init meters
    time_meters = defaultdict(AverageMeter)
    loss_meters = defaultdict(AverageMeter)

    num_training_examples = len(train_loader)
    timer_dataloading = time.time()

    ### ADDED
    mr_res = []
    temp_scale = 0

    for batch_idx, batch in tqdm(enumerate(train_loader),
                                 desc="Training Iteration",
                                 total=num_training_examples):
        time_meters["dataloading_time"].update(time.time() - timer_dataloading)

        timer_start = time.time()
        if opt.a_feat_dir is None:
            model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, non_blocking=opt.pin_memory)
        else:
            model_inputs, targets = prepare_batch_inputs_audio(batch[1], opt.device, non_blocking=opt.pin_memory)
        time_meters["prepare_inputs_time"].update(time.time() - timer_start)
        timer_start = time.time()

        ### ADDED
        targets['durations'] = [b["duration"] // opt.clip_length for b in batch[0]] 

        outputs = model(**model_inputs)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        if opt.save_pred:
            query_meta = batch[0]
            # # compose predictions
            # for idx, (meta, spans, score) in enumerate(zip(query_meta, pred_spans.cpu(), scores.cpu())):
            #     # print(f'idx: {idx}\tspans: {spans}')
            #     spans = span_cxw_to_xx(spans) * meta["duration"]
            #     # # (#queries, 3), [st(float), ed(float), score(float)]
            #     cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1).tolist()
            #     if not opt.no_sort_results:
            #         cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True)
            #     cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds]
            #     cur_query_pred = dict(
            #         qid=meta["qid"],
            #         query=meta["query"],
            #         vid=meta["vid"],
            #         sim_loss=criterion.sim[idx],
            #         ious=criterion.ious[idx],
            #         pred_relevant_windows=cur_ranked_preds,
            #         pred_saliency_scores=saliency_scores[idx],
            #     )
            for idx, meta in enumerate(query_meta):
                # meta = query_meta[idx]
                sim_loss = criterion.sim_losses[idx]
                iou = criterion.ious[idx]

                cur_query_pred = dict(
                    idx=idx,
                    qid=meta["qid"],
                    query=meta["query"],
                    vid=meta["vid"],
                    sim_loss=sim_loss,
                    iou=iou,
                    sim=[2. - s / (1. - i) for s, i in zip(sim_loss, iou)],
                    pred_span=criterion.pred_spans[idx],
                    gt_span=criterion.gt_spans[idx],
                )
                mr_res.append(cur_query_pred)
            
            # post_processor = PostProcessorDETR(
            #     clip_length=2, min_ts_val=0, max_ts_val=150,
            #     min_w_l=2, max_w_l=150, move_window_method="left",
            #     process_func_names=("clip_ts", "round_multiple")
            # )
            # mr_res = post_processor(mr_res)

        ### ADDED
        if opt.scheduling:
            coef_idx = epoch_i // 20

            ### sim + giou sched (linear)
            if opt.scheduling == 1:  
                # loss_dict['loss_sim'] = sim_coeffs[coef_idx] * loss_dict['loss_sim']
                # loss_dict['loss_giou'] = giou_coeffs[coef_idx] * loss_dict['loss_giou']

                ### linear LDF cross
                weight_dict['loss_sim'] = (1 - epoch_i / opt.n_epoch)  
                weight_dict['loss_giou'] = (epoch_i / opt.n_epoch)

                for i in range(opt.dec_layers - 1):
                    # loss_dict[f'loss_sim_{i}'] = sim_coeffs[coef_idx] * loss_dict[f'loss_sim_{i}']
                    # loss_dict[f'loss_giou_{i}'] = giou_coeffs[coef_idx] * loss_dict[f'loss_giou_{i}']

                    ### linear LDF cross
                    weight_dict[f'loss_sim_{i}'] = (1 - epoch_i / opt.n_epoch)  # linear LDF
                    weight_dict[f'loss_giou_{i}'] = (epoch_i / opt.n_epoch)

            ### sim + giou non-linear
            elif opt.scheduling == 2:  
                ### cos DF cross
                weight_dict['loss_sim'] = (1 + math.cos(epoch_i * math.pi / opt.n_epoch)) / 2  
                weight_dict['loss_giou'] = (1 - math.cos(epoch_i * math.pi / opt.n_epoch)) / 2

                ### inverse sqrt DF cross
                # weight_dict['loss_sim'] = 1 / math.sqrt(epoch_i + 1)
                # weight_dict['loss_giou'] = 1 - (1 / math.sqrt(epoch_i + 1))

                ### sigmoid DF cross
                # weight_dict['loss_sim'] = 1 / (1 + np.exp(epoch_i - opt.n_epoch / 2))
                # weight_dict['loss_giou'] = 1 / (1 + np.exp(-epoch_i + opt.n_epoch / 2))

                for i in range(opt.dec_layers - 1):
                    ### cos DF cross
                    weight_dict[f'loss_sim_{i}'] = (1 + math.cos(epoch_i * math.pi / opt.n_epoch)) / 2  
                    weight_dict[f'loss_giou_{i}'] = (1 - math.cos(epoch_i * math.pi / opt.n_epoch)) / 2

                    ### inverse sqrt DF cross
                    # weight_dict[f'loss_sim_{i}'] = 1 / math.sqrt(epoch_i + 1)  
                    # weight_dict[f'loss_giou_{i}'] = 1 - (1 / math.sqrt(epoch_i + 1))

                    ### sigmoid DF cross
                    # weight_dict[f'loss_sim_{i}'] = 1 / (1 + np.exp(epoch_i - opt.n_epoch / 2))
                    # weight_dict[f'loss_giou_{i}'] = 1 / (1 + np.exp(-epoch_i + opt.n_epoch / 2))

            ### sim + sim sched
            elif opt.scheduling == 3:  
                loss_dict['loss_sim'] = sim_coeffs[coef_idx] * loss_dict['loss_sim']
                loss_dict['loss_sim2'] = giou_coeffs[coef_idx] * loss_dict['loss_sim2']

                for i in range(opt.dec_layers - 1):
                    loss_dict[f'loss_sim_{i}'] = sim_coeffs[coef_idx] * loss_dict[f'loss_sim_{i}']
                    loss_dict[f'loss_sim2_{i}'] = giou_coeffs[coef_idx] * loss_dict[f'loss_sim2_{i}']
                
            ### for logging
            if epoch_i % 10 == 0 and batch_idx == 0:
                with open(opt.train_log_filepath, "a") as f:
                    # f.write(f"weight_dict['loss_sim']: {sim_coeffs[coef_idx]}\tweight_dict['loss_giou']: {giou_coeffs[coef_idx]}\n")
                    f.write(f"weight_dict['loss_sim']: {weight_dict['loss_sim']}\tweight_dict['loss_giou']: {weight_dict['loss_giou']}\n")
            
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        time_meters["model_forward_time"].update(time.time() - timer_start)

        timer_start = time.time()
        optimizer.zero_grad()
        losses.backward()
        if opt.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), opt.grad_clip)
        optimizer.step()

        ### clipping temperature scaling factor
        # temp_scale += model.temp_scale.data
        temp_scale += outputs['temp_scale']
        # model.temp_scale.data = torch.clamp(model.temp_scale.data, 0, 4.6052)

        time_meters["model_backward_time"].update(time.time() - timer_start)

        loss_dict["loss_overall"] = float(losses)  # for logging only
        for k, v in loss_dict.items():
            loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

        timer_dataloading = time.time()
        if opt.debug and batch_idx == 1:
            break

    # print/add logs
    tb_writer.add_scalar("Train/lr", float(optimizer.param_groups[0]["lr"]), epoch_i+1)
    for k, v in loss_meters.items():
        tb_writer.add_scalar("Train/{}".format(k), v.avg, epoch_i+1)

    to_write = opt.train_log_txt_formatter.format(
        time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
        epoch=epoch_i+1,
        loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in loss_meters.items()]))
    with open(opt.train_log_filepath, "a") as f:
        f.write(to_write)

    ### ADDED
    tb_writer.add_scalar("Train/temp_scale", temp_scale / len(train_loader), epoch_i+1)

    for i, s in enumerate(outputs['sims']):
            if len(s) > 0:
                tb_writer.add_histogram(f"Train/sim_{i}", s[0], epoch_i+1)

    logger.info("Epoch time stats:")
    for name, meter in time_meters.items():
        d = {k: f"{getattr(meter, k):.4f}" for k in ["max", "min", "avg"]}
        logger.info(f"{name} ==> {d}")

    return mr_res


def train(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt):
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)

    tb_writer = SummaryWriter(opt.tensorboard_log_dir)
    tb_writer.add_text("hyperparameters", dict_to_markdown(vars(opt), max_str_len=None))
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    if opt.a_feat_dir is None:
        train_loader = DataLoader(
            train_dataset,
            collate_fn=start_end_collate,
            batch_size=opt.bsz,
            num_workers=opt.num_workers,
            shuffle=True,
            pin_memory=opt.pin_memory
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            collate_fn=start_end_collate_audio,
            batch_size=opt.bsz,
            num_workers=opt.num_workers,
            shuffle=True,
            pin_memory=opt.pin_memory
        )

    prev_best_score = 0.
    es_cnt = 0
    # start_epoch = 0
    if opt.start_epoch is None:
        start_epoch = -1 if opt.eval_untrained else 0
    else:
        start_epoch = opt.start_epoch
    save_submission_filename = "latest_{}_{}_preds.jsonl".format(opt.dset_name, opt.eval_split_name)
    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        if epoch_i > -1:
            mr_res = train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer)
            lr_scheduler.step()
        eval_epoch_interval = 5
        if opt.eval_path is not None and (epoch_i + 1) % eval_epoch_interval == 0:
            with torch.no_grad():
                metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = \
                    eval_epoch(model, val_dataset, opt, save_submission_filename, epoch_i, criterion, tb_writer)

            # log
            to_write = opt.eval_log_txt_formatter.format(
                time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i,
                loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in eval_loss_meters.items()]),
                eval_metrics_str=json.dumps(metrics_no_nms))

            with open(opt.eval_log_filepath, "a") as f:
                f.write(to_write)
            logger.info("metrics_no_nms {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))
            if metrics_nms is not None:
                logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))

            metrics = metrics_no_nms
            for k, v in metrics["brief"].items():
                tb_writer.add_scalar(f"Eval/{k}", float(v), epoch_i+1)

            stop_score = metrics["brief"]["MR-full-mAP"]
                
            if stop_score > prev_best_score:
                es_cnt = 0
                prev_best_score = stop_score

                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt
                }
                torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"))

                best_file_paths = [e.replace("latest", "best") for e in latest_file_paths]
                for src, tgt in zip(latest_file_paths, best_file_paths):
                    os.renames(src, tgt)
                logger.info("The checkpoint file has been updated.")

                ### ADDED
                with open(opt.train_log_filepath, "a") as f:
                    f.write(f"The checkpoint file has been updated at epoch {epoch_i}\n")

                if opt.save_pred:
                    train_chk_path = os.path.join(opt.results_dir, 'submissions')
                    os.makedirs(train_chk_path, exist_ok=True)
                    save_jsonl(mr_res, os.path.join(train_chk_path, f'{epoch_i}_submissions.jsonl'))
            else:
                es_cnt += 1
                if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:  # early stop
                    with open(opt.train_log_filepath, "a") as f:
                        f.write(f"Early Stop at epoch {epoch_i}")
                    logger.info(f"\n>>>>> Early stop at epoch {epoch_i}  {prev_best_score}\n")
                    break

            # save ckpt
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt"))

        # save_interval = 10 if "subs_train" in opt.train_path else 50  # smaller for pretrain
        # if (epoch_i + 1) % save_interval == 0 or (epoch_i + 1) % opt.lr_drop == 0:  # additional copies
        #     checkpoint = {
        #         "model": model.state_dict(),
        #         "optimizer": optimizer.state_dict(),
        #         "epoch": epoch_i,
        #         "opt": opt
        #     }
        #     torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", f"_e{epoch_i:04d}.ckpt"))

        if opt.debug:
            break

    tb_writer.close()



def train_hl(model, criterion, optimizer, lr_scheduler, train_dataset, val_dataset, opt):
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)

    tb_writer = SummaryWriter(opt.tensorboard_log_dir)
    tb_writer.add_text("hyperparameters", dict_to_markdown(vars(opt), max_str_len=None))
    opt.train_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str}\n"
    opt.eval_log_txt_formatter = "{time_str} [Epoch] {epoch:03d} [Loss] {loss_str} [Metrics] {eval_metrics_str}\n"

    train_loader = DataLoader(
        train_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.bsz,
        num_workers=opt.num_workers,
        shuffle=True,
        pin_memory=opt.pin_memory
    )

    prev_best_score = 0.
    es_cnt = 0
    # start_epoch = 0
    if opt.start_epoch is None:
        start_epoch = -1 if opt.eval_untrained else 0
    else:
        start_epoch = opt.start_epoch
    save_submission_filename = "latest_{}_{}_preds.jsonl".format(opt.dset_name, opt.eval_split_name)
    for epoch_i in trange(start_epoch, opt.n_epoch, desc="Epoch"):
        if epoch_i > -1:
            train_epoch(model, criterion, train_loader, optimizer, opt, epoch_i, tb_writer)
            lr_scheduler.step()
        eval_epoch_interval = 5
        if opt.eval_path is not None and (epoch_i + 1) % eval_epoch_interval == 0:
            with torch.no_grad():
                metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = \
                    eval_epoch(model, val_dataset, opt, save_submission_filename, epoch_i, criterion, tb_writer)

            # log
            to_write = opt.eval_log_txt_formatter.format(
                time_str=time.strftime("%Y_%m_%d_%H_%M_%S"),
                epoch=epoch_i,
                loss_str=" ".join(["{} {:.4f}".format(k, v.avg) for k, v in eval_loss_meters.items()]),
                eval_metrics_str=json.dumps(metrics_no_nms))

            with open(opt.eval_log_filepath, "a") as f:
                f.write(to_write)
            logger.info("metrics_no_nms {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))
            if metrics_nms is not None:
                logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))

            metrics = metrics_no_nms
            for k, v in metrics["brief"].items():
                tb_writer.add_scalar(f"Eval/{k}", float(v), epoch_i+1)

            # stop_score = metrics["brief"]["MR-full-mAP"]
            stop_score = metrics["brief"]["mAP"]
            if stop_score > prev_best_score:
                es_cnt = 0
                prev_best_score = stop_score

                checkpoint = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch_i,
                    "opt": opt
                }
                torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"))

                best_file_paths = [e.replace("latest", "best") for e in latest_file_paths]
                for src, tgt in zip(latest_file_paths, best_file_paths):
                    os.renames(src, tgt)
                logger.info("The checkpoint file has been updated.")

                ### ADDED
                with open(opt.train_log_filepath, "a") as f:
                    f.write(f"The checkpoint file has been updated at epoch {epoch_i}\n")
            else:
                es_cnt += 1
                if opt.max_es_cnt != -1 and es_cnt > opt.max_es_cnt:  # early stop
                    with open(opt.train_log_filepath, "a") as f:
                        f.write(f"Early Stop at epoch {epoch_i}")
                    logger.info(f"\n>>>>> Early stop at epoch {epoch_i}  {prev_best_score}\n")
                    break

            # save ckpt
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            # torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", "_latest.ckpt"))

        save_interval = 10 if "subs_train" in opt.train_path else 50  # smaller for pretrain
        if (epoch_i + 1) % save_interval == 0 or (epoch_i + 1) % opt.lr_drop == 0:  # additional copies
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch_i,
                "opt": opt
            }
            # torch.save(checkpoint, opt.ckpt_filepath.replace(".ckpt", f"_e{epoch_i:04d}.ckpt"))

        if opt.debug:
            break

    tb_writer.close()




def start_training():
    logger.info("Setup config, data and model...")
    opt = BaseOptions().parse()
    set_seed(opt.seed)
    if opt.debug:  # keep the model run deterministically
        # 'cudnn.benchmark = True' enabled auto finding the best algorithm for a specific input/net config.
        # Enable this only when input size is fixed.
        cudnn.benchmark = False
        cudnn.deterministic = True
    print('##################')
    print(opt.a_feat_dir is None)
    print(opt.a_feat_dir)
    print('##################')
    if opt.a_feat_dir is None:
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
            txt_drop_ratio=opt.txt_drop_ratio,
            dset_domain=opt.dset_domain,
        )
        dataset_config["data_path"] = opt.train_path
        train_dataset = StartEndDataset(**dataset_config)
    else:
        dataset_config = dict(
            dset_name=opt.dset_name,
            data_path=opt.train_path,
            v_feat_dirs=opt.v_feat_dirs,
            q_feat_dir=opt.t_feat_dir,
            a_feat_dir=opt.a_feat_dir,
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
            txt_drop_ratio=opt.txt_drop_ratio,
            dset_domain=opt.dset_domain,
        )
        dataset_config["data_path"] = opt.train_path
        train_dataset = StartEndDataset_audio(**dataset_config)



    if opt.eval_path is not None:
        dataset_config["data_path"] = opt.eval_path
        dataset_config["txt_drop_ratio"] = 0
        dataset_config["q_feat_dir"] = opt.t_feat_dir.replace("sub_features", "text_features")  # for pretraining
        # dataset_config["load_labels"] = False  # uncomment to calculate eval loss
        if opt.a_feat_dir is None:
            eval_dataset = StartEndDataset(**dataset_config)
        else:
            eval_dataset = StartEndDataset_audio(**dataset_config)
    else:
        eval_dataset = None

    model, criterion, optimizer, lr_scheduler = setup_model(opt)
    logger.info(f"Model {model}")
    count_parameters(model)
    logger.info("Start Training...")
    
    # For tvsum dataset, use train_hl function
    if opt.dset_name in ['tvsum']:
        train_hl(model, criterion, optimizer, lr_scheduler, train_dataset, eval_dataset, opt)
    else:
        train(model, criterion, optimizer, lr_scheduler, train_dataset, eval_dataset, opt)
    
    return opt.ckpt_filepath.replace(".ckpt", "_best.ckpt"), opt.eval_split_name, opt.eval_path, opt.debug, opt


if __name__ == '__main__':
    best_ckpt_path, eval_split_name, eval_path, debug, opt = start_training()
    if not debug:
        input_args = ["--resume", best_ckpt_path,
                      "--eval_split_name", eval_split_name,
                      "--eval_path", eval_path]

        import sys
        sys.argv[1:] = input_args
        logger.info("\n\n\nFINISHED TRAINING!!!")
        logger.info("Evaluating model at {}".format(best_ckpt_path))
        logger.info("Input args {}".format(sys.argv[1:]))
        start_inference(opt)
