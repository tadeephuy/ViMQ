import logging
import os
from tqdm.auto import tqdm, trange


from data_loader import ViMQ
from utils import load_tokenizer, MODEL_CLASSES, compute_metrics, convert_spacy_to_iob, get_iou_score, get_entity_label, get_IoU
from early_stopping import EarlyStopping


import torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AdamW, get_linear_schedule_with_warmup

logger = logging.getLogger(__name__)

class Trainer(object):
    def __init__(self, args):
        self.args = args
        # INIT DATA
        self.tokenizer = load_tokenizer(args)
        _, self.index2label = get_entity_label(args)
        self.dev_dataset = ViMQ(args,
                                self.tokenizer,
                                mode='dev',
                                predictions=None,
                                iteration=-1)
        self.test_dataset = ViMQ(args,
                                self.tokenizer,
                                mode='test',
                                predictions=None,
                                iteration=-1)
        # INIT MODEL
        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]
        if args.pretrained:
            self.model = self.model_class(
                config=self.config,
                args=args
            )
            # load model
        else:
            self.config = self.config_class.from_pretrained(args.model_name_or_path)
            self.model = self.model_class(
                config=self.config,
                args=args
            )
        # GPU or CPU
        torch.cuda.set_device(self.args.gpu_id)
        # self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.args.device)
        

    def train(self):
        # INIT TRAINING SET
        self.train_dataset = ViMQ(self.args,
                                self.tokenizer,
                                mode='train',
                                predictions=None,
                                iteration=-1)

        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)
        writer = SummaryWriter(log_dir=self.args.model_dir)
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = (
                self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
            )
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        
#         results = self.evaluate("dev")
        # Prepare optimizer and schedule (linear warmup and decay)
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total
        )

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        predictions_train = {}
        self.model.zero_grad()

        
        early_stopping = EarlyStopping(patience=self.args.early_stopping, verbose=True)
        for iter in range(1, self.args.num_iteration):
            train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
            for epoch in train_iterator:
                print("\nEpoch", epoch)
                epoch_iterator = tqdm(train_dataloader, desc="Iteration", position=0, leave=True)
                for step, batch in enumerate(epoch_iterator):
                    self.model.train()
                    # input
                    input_ids = batch[0].to(self.args.device)
                    attention_mask = batch[1].to(self.args.device)
                    first_subword = batch[2].to(self.args.device)
                    seq_len = batch[3].to(self.args.device)
                    char_ids = batch[4].to(self.args.device)
                    # output
                    label = batch[5].to(self.args.device)
                    guid = batch[-1]

                    inputs = {
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "first_subword": first_subword,
                        "seq_len": seq_len,
                        "char_ids": char_ids,
                        "label": label
                    }
                    outputs, loss = self.model(**inputs)
                    

                    if self.args.gradient_accumulation_steps > 1:
                        loss = loss / self.args.gradient_accumulation_steps
                    
                    loss.backward()
                    tr_loss += loss.item()
                    if iter > self.args.omega:
                        predictions_train = self.store_pred(outputs, label, guid, predictions_train)
                    if (step + 1) % self.args.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                        optimizer.step()
                        scheduler.step()  # Update learning rate schedule
                        self.model.zero_grad()
                        global_step += 1

#                     if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
#                         print("\nTuning metrics:", self.args.tuning_metric)
#                         results = self.evaluate("dev")
#                         writer.add_scalar("Loss/validation", results["loss"], epoch)
#                         writer.add_scalar("Precision/validation", results["f1_score"], epoch)
#                         writer.add_scalar("Recall/validation", results["f1_score"], epoch)
#                         writer.add_scalar("F1/validation", results["f1_score"], epoch)
#                         early_stopping(results[self.args.tuning_metric], self.model, self.args)
#                         if early_stopping.early_stop:
#                             print("Early stopping")
#                             break
                    if 0 < self.args.max_steps < global_step:
                        epoch_iterator.close()
                        break

                if 0 < self.args.max_steps < global_step or early_stopping.early_stop:
                    train_iterator.close()
                    break
                print("Loss/train", tr_loss / global_step, epoch)
            if early_stopping.early_stop:
                break
            self.train_dataset = ViMQ(self.args,
                                self.tokenizer,
                                mode='train',
                                predictions=predictions_train,
                                iteration=iter)
            train_sampler = RandomSampler(self.train_dataset)
            train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)
        return global_step, tr_loss / global_step   

    def write_evaluation_result(self, out_file, results):
        out_file = self.args.model_dir + "/" + out_file
        w = open(out_file, "w", encoding="utf-8")
        w.write("***** Eval results *****\n")
        for key in sorted(results.keys()):
            to_write = " {key} = {value}".format(key=key, value=str(results[key]))
            w.write(to_write)
            w.write("\n")
        w.close()

    def evaluate(self, mode):
        if mode == "test":
            dataset = self.test_dataset
        elif mode == "dev":
            dataset = self.dev_dataset
        else:
            raise Exception("Only dev and test dataset available")
        
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)
        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        labels = None
        seq_lens = None
        
        preds_decode = []
        labels_decode = []
        # guids = []


        self.model.eval()
        for step, batch in enumerate(eval_dataloader):
            # input
            input_ids = batch[0].to(self.args.device)
            attention_mask = batch[1].to(self.args.device)
            first_subword = batch[2].to(self.args.device)
            seq_len = batch[3].to(self.args.device)
            char_ids = batch[4].to(self.args.device)
            # output
            label = batch[5].to(self.args.device)
            guid = batch[-1]

            with torch.no_grad():
                inputs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "first_subword": first_subword,
                    "seq_len": seq_len,
                    "char_ids": char_ids,
                    "label": label
                }
                outputs, loss = self.model(**inputs)
                eval_loss += loss.mean().item()
            nb_eval_steps += 1
            if labels is None:
                preds = outputs.detach().cpu().numpy()
                labels = inputs["label"].detach().cpu().numpy()
            else:
                preds = np.append(preds, outputs.detach().cpu().numpy(), axis=0)
                labels = np.append(
                    labels, inputs["label"].detach().cpu().numpy(), axis=0
                )
            if seq_lens is None:
                seq_lens = seq_len.detach().cpu().numpy().reshape(-1)
            else:
                seq_lens = np.append(seq_lens, seq_len.detach().cpu().numpy().reshape(-1), axis=0)
            # guids.extend(guid)
        eval_loss = eval_loss / nb_eval_steps
        results = {"loss": eval_loss}
        preds = np.argmax(preds, axis=-1)
        
        for p, l in zip(preds, labels):
            preds_decode.append(self.span_decode(p))
            labels_decode.append(self.span_decode(l))
        # Convert spacy format to iob format
        preds_decode, labels_decode = convert_spacy_to_iob(preds_decode, labels_decode, seq_lens)
        total_result = compute_metrics(preds_decode, labels_decode)
        results.update(total_result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        return results
    
    def store_pred(self, outputs, labels, guids, prediction):
        """
        outputs: Batch x Seq_len X Seq_len X C
        labels: Batch x Seq_len X Seq_len

        return prediction - dict {
            guid: label
        }
        """

        outputs_ = outputs.detach().cpu().numpy()
        labels_ = labels.detach().cpu().numpy()
        outputs_ = np.argmax(outputs_, axis=-1) # B X Seq_len X Seq_len
        for output, label, guid in zip(outputs_, labels_, guids): # Sample iter
            output = self.span_decode(output)
            label = self.span_decode(label)
            new_label = []
            for gt in label:
                for o in output:
                    iou = get_IoU(gt, o)
                    if iou > self.args.threshold_iou:
                        new_label.append(o)
            if guid not in prediction:
                prediction[guid] = new_label
            else:
                prediction[guid].append(new_label)
        return prediction


    def span_decode(self, logits):
        arg_index = []
        for i in range(len(logits)):
            for j in range(i, len(logits[i])):
                if logits[i][j] > 0:
                    arg_index.append([i, j, self.index2label.get(int(logits[i][j]), 'UNK')])
        return arg_index
    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = self.model_class.from_pretrained(
                self.args.model_dir,
                args=self.args
            )
            self.model.to(self.args.device)
            logger.info("***** Model Loaded *****")
        except Exception:
            raise Exception("Some model files might be missing...")