#!/usr/bin/env/python3
"""
Recipe for "direct" (speech -> semantics) SLU with ASR-based transfer learning.

We encode input waveforms into features using a model trained on LibriSpeech,
then feed the features into a seq2seq model to map them to semantics.

(Adapted from the LibriSpeech seq2seq ASR recipe written by Ju-Chieh Chou, Mirco Ravanelli, Abdel Heba, and Peter Plantinga.)

Run using:
> python train.py hparams/train.yaml

Authors
 * Loren Lugosch 2020
 * Mirco Ravanelli 2020
"""

import sys
import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
import jsonlines
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# Define training procedure
class Captioning(sb.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches to the output probabilities."""
        batch = batch.to(self.device)
        current_epoch = self.hparams.epoch_counter.current
        src_list = list()
        src_list1 = list()
        if ('audio' in self.hparams.modality):
            wavs_audio, wav_lens = batch.audio
            wavs_audio, wav_lens = wavs_audio.to(self.device), wav_lens.to(self.device)
            feats_audio = self.hparams.audio_compute_features(wavs_audio)
            # feats_audio = self.hparams.audio_normalize(feats_audio, wav_lens, epoch=current_epoch)
            # with open('feature.npy', 'wb') as f:
            #     np.save(f, feats_audio.cpu().detach().numpy())
            # print(feats_audio.size())
            src_audio = self.hparams.audio_CNN(feats_audio)
            src_list.append(src_audio)
            src_audio1 = self.hparams.audio_CNN1(feats_audio)
            src_list1.append(src_audio1)
            # print(src_audio.size())
        if ('acc' in self.hparams.modality):
            wavs_acc, wav_lens = batch.acc
            wavs_acc, wav_lens = wavs_acc.to(self.device), wav_lens.to(self.device)
            feats_acc = self.hparams.acc_compute_features(wavs_acc)
            feats_acc = self.hparams.acc_normalize(feats_acc, wav_lens, epoch=current_epoch)
            # print(feats_acc.size())
            src_acc = self.hparams.acc_CNN(feats_acc)
            src_list.append(src_acc)
            src_acc1 = self.hparams.acc_CNN1(feats_acc)
            src_list1.append(src_acc1)
            # print(src_acc.size())
        if ('csi' in self.hparams.modality):
            wavs_csi, wav_lens = batch.csi
            wavs_csi, wav_lens = wavs_csi.to(self.device), wav_lens.to(self.device)
            feats_csi = self.hparams.csi_compute_features(wavs_csi)
            feats_csi = self.hparams.csi_normalize(feats_csi, wav_lens, epoch=current_epoch)
            # print(feats_csi.size())
            src_csi = self.hparams.csi_CNN(feats_csi)
            src_list.append(src_csi)
            src_csi1 = self.hparams.csi_CNN1(feats_csi)
            src_list1.append(src_csi1)
            # print(src_csi.size())

        tokens_bos, tokens_bos_lens = batch.tokens_bos

        # print(src_audio.size(), src_acc.size(), src_csi.size())
        src_modalitywise = src_list[0]
        for i in range(1, len(src_list)):
            src_modalitywise = torch.cat((src_modalitywise,src_list[i]),dim = 1)

        src_modalitywise = src_modalitywise.view(src_modalitywise.size(0),src_modalitywise.size(1), -1)

        src_framewise = src_list1[0]
        for i in range(1, len(src_list1)):
            src_framewise = torch.cat((src_framewise,src_list1[i]),dim = 2)

        src_framewise = src_framewise.view(src_framewise.size(0),src_framewise.size(1), -1)

        src = torch.cat((src_modalitywise, src_framewise), dim = 1)
        # print(src.size())
        # print(src.size())

        # print(src.size())
        
        # src = self.hparams.CNN(src)
        # data argumentation
        # if stage == sb.Stage.TRAIN:
        #     if hasattr(self.hparams, "augmentation"):
        #         feats = self.hparams.augmentation(feats)

        # forward modules
        # print(self.hparams.modality)
        # print(src.size())
        # print(src_acc.size(), src_gyro.size(), src_audio.size(), src_csi.size())

        enc_out, pred = self.hparams.Transformer(
            src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index
        )
        # print(src.size(), tokens_bos.size())
        # print(enc_out.size(), pred.size())
        # return

        # output layer for ctc log-probabilities
        logits = self.hparams.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        # output layer for seq2seq log-probabilities
        pred = self.hparams.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        # Compute outputs
        hyps = None
        if stage == sb.Stage.TRAIN:
            hyps = None
        elif stage == sb.Stage.VALID:
            hyps = None
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch % self.hparams.valid_search_interval == 0:
                # for the sake of efficiency, we only perform beamsearch with limited capacity
                # and no LM to give user some idea of how the AM is doing
                hyps, _ = self.hparams.valid_search(enc_out.detach(), wav_lens)
        elif stage == sb.Stage.TEST:
            hyps, _ = self.hparams.test_search(enc_out.detach(), wav_lens)
        
        return p_ctc, p_seq, wav_lens, hyps

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (NLL) given predictions and targets."""
        (p_ctc, p_seq, wav_lens, hyps,) = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        if hasattr(self.modules, "env_corrupt") and stage == sb.Stage.TRAIN:
            tokens_eos = torch.cat([tokens_eos, tokens_eos], dim=0)
            tokens_eos_lens = torch.cat(
                [tokens_eos_lens, tokens_eos_lens], dim=0
            )
            tokens = torch.cat([tokens, tokens], dim=0)
            tokens_lens = torch.cat([tokens_lens, tokens_lens], dim=0)

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )
        loss_ctc = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)
        loss = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_seq
        )

        if stage != sb.Stage.TRAIN:
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if current_epoch % valid_search_interval == 0 or (
                stage == sb.Stage.TEST
            ):
                # Decode token terms to words
                predicted_words = [
                    tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in hyps
                ]
                target_words = [semantics.split(" ") for semantics in batch.semantics]
                self.wer_metric.append(ids, predicted_words, target_words)

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)
        return loss

    def fit_batch(self, batch):
        """Train the parameters given a single batch in input"""
        # check if we need to switch optimizer
        # if so change the optimizer from Adam to SGD
        self.check_and_reset_optimizer()

        predictions = self.compute_forward(batch, sb.Stage.TRAIN)
        loss = self.compute_objectives(predictions, batch, sb.Stage.TRAIN)

        # normalize the loss by gradient_accumulation step
        (loss / self.hparams.gradient_accumulation).backward()

        if self.step % self.hparams.gradient_accumulation == 0:
            # gradient clipping & early stop if loss is not fini
            self.check_gradients(loss)

            self.optimizer.step()
            self.optimizer.zero_grad()

            # anneal lr every update
            self.hparams.noam_annealing(self.optimizer)

        return loss.detach()

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage != sb.Stage.TRAIN:
            self.acc_metric = self.hparams.acc_computer()
            self.wer_metric = self.hparams.error_rate_computer()

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ACC"] = self.acc_metric.summarize()
            current_epoch = self.hparams.epoch_counter.current
            valid_search_interval = self.hparams.valid_search_interval
            if (
                current_epoch % valid_search_interval == 0
                or stage == sb.Stage.TEST
            ):
                stage_stats["WER"] = self.wer_metric.summarize("error_rate")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():

            # report different epoch stages according current stage
            current_epoch = self.hparams.epoch_counter.current
            if current_epoch <= self.hparams.stage_one_epochs:
                lr = self.hparams.noam_annealing.current_lr
                steps = self.hparams.noam_annealing.n_steps
                optimizer = self.optimizer.__class__.__name__
            else:
                lr = self.hparams.lr_sgd
                steps = -1
                optimizer = self.optimizer.__class__.__name__

            epoch_stats = {
                "epoch": epoch,
                "lr": lr,
                "steps": steps,
                "optimizer": optimizer,
            }
            self.hparams.train_logger.log_stats(
                stats_meta=epoch_stats,
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ACC": stage_stats["ACC"], "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=5,
            )

        elif stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                stats_meta={"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stage_stats,
            )
            with open(self.hparams.wer_file, "w") as w:
                self.wer_metric.write_stats(w)

            # save the averaged checkpoint at the end of the evaluation stage
            # delete the rest of the intermediate checkpoints
            # ACC is set to 1.1 so checkpointer only keeps the averaged checkpoint
            self.checkpointer.save_and_keep_only(
                meta={"ACC": 1.1, "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=1,
            )
    
    def check_and_reset_optimizer(self):
        """reset the optimizer if training enters stage 2"""
        current_epoch = self.hparams.epoch_counter.current
        if not hasattr(self, "switched"):
            self.switched = False
            if isinstance(self.optimizer, torch.optim.SGD):
                self.switched = True

        if self.switched is True:
            return

        if current_epoch > self.hparams.stage_one_epochs:
            self.optimizer = self.hparams.SGD(self.modules.parameters())

            if self.checkpointer is not None:
                self.checkpointer.add_recoverable("optimizer", self.optimizer)

            self.switched = True

    def on_fit_start(self):
        """Initialize the right optimizer on the training start"""
        super().on_fit_start()

        # if the model is resumed from stage two, reinitialize the optimizer
        current_epoch = self.hparams.epoch_counter.current
        current_optimizer = self.optimizer
        if current_epoch > self.hparams.stage_one_epochs:
            del self.optimizer
            self.optimizer = self.hparams.SGD(self.modules.parameters())

            # Load latest checkpoint to resume training if interrupted
            if self.checkpointer is not None:

                # do not reload the weights if training is interrupted right before stage 2
                group = current_optimizer.param_groups[0]
                if "momentum" not in group:
                    return

                self.checkpointer.recover_if_possible(
                    device=torch.device(self.device)
                )

    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint averge if needed"""
        super().on_evaluate_start()

        ckpts = self.checkpointer.find_checkpoints(
            max_key=max_key, min_key=min_key
        )
        ckpt = sb.utils.checkpoints.average_checkpoints(
            ckpts, recoverable_name="model", device=self.device
        )

        self.hparams.model.load_state_dict(ckpt, strict=True)
        self.hparams.model.eval()

def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""

    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["csv_train"], replacements={"data_root": data_folder},
    )

    if hparams["sorting"] == "ascending":
        # we sort training data to speed up training and get better results.
        train_data = train_data.filtered_sorted(sort_key="duration")
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "descending":
        train_data = train_data.filtered_sorted(
            sort_key="duration", reverse=True
        )
        # when sorting do not shuffle in dataloader ! otherwise is pointless
        hparams["dataloader_opts"]["shuffle"] = False

    elif hparams["sorting"] == "random":
        pass

    else:
        raise NotImplementedError(
            "sorting must be random, ascending or descending"
        )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["csv_valid"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["csv_test"], replacements={"data_root": data_folder},
    )
    test_data = test_data.filtered_sorted(sort_key="duration")

    datasets = [train_data, valid_data, test_data]

    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides(
        "audio", "acc", "gyro", "csi")
    def audio_pipeline(wav):
        # sig = sb.dataio.dataio.read_audio(wav)
        ''' Read npz '''
        signal = np.load(wav)
        audio = signal['audio_signal']
        audio = audio[::3]
        audio = torch.from_numpy(audio).float().to('cpu')
        yield audio
        acc = signal['acc_matrix']
        acc = np.transpose(acc)
        acc = torch.from_numpy(acc).float().to('cpu')
        yield acc
        gyro = signal['gyro_matrix']
        gyro = np.transpose(gyro)
        gyro = torch.from_numpy(gyro).float().to('cpu')
        yield gyro
        csi = signal ['csi_matrix']
        csi = np.transpose(csi)
        csi = torch.from_numpy(csi).float().to('cpu')
        # filepath = wav.rsplit("/", 1)[0]
        # filename = wav.rsplit("/", 1)[1]
        # csi_file = filepath + '_pca/' + filename[:-4] + '.npz'
        # signal = np.load(csi_file)
        # csi = signal ['csi_matrix']
        # csi = np.transpose(csi)
        # csi = torch.from_numpy(csi).float().to('cpu')
        yield csi

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("semantics")
    @sb.utils.data_pipeline.provides(
        "semantics", "token_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(semantics):
        yield semantics
        tokens_list = tokenizer.encode_as_ids(semantics)
        # print(tokenizer.encode_as_pieces(semantics))
        # print(tokens_list)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets,
        ["id", "audio", "acc", "gyro", "csi", "semantics", "tokens_bos", "tokens_eos", "tokens"],
    )
    return train_data, valid_data, test_data, tokenizer


if __name__ == "__main__":

    # Load hyperparameters file with command-line overrides
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    # hparams_file --> hparams file; run_opts --> command line
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    show_results_every = 200  # plots results every N iterations

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Dataset prep (parsing SLURP)
    from prepare import prepare_Egolife  # noqa

    # multi-gpu (ddp) save data preparation
    run_on_main(
        prepare_Egolife,
        kwargs={
            "dataset_csv": hparams["dataset_csv"],
            "save_folder": hparams["output_folder"],
            "caption_type": "uniact",
            "skip_prep": hparams["skip_prep"],
            "seed": hparams["seed"],
        },
    )

    # here we create the datasets objects as well as tokenization and encoding
    (train_set, valid_set, test_set, tokenizer,) = dataio_prepare(hparams)

    # # We download and pretrain the tokenizer
    run_on_main(hparams["pretrainer"].collect_files)
    hparams["pretrainer"].load_collected(device=run_opts["device"])

    # Brain class initialization
    cap_brain = Captioning(
        modules=hparams["modules"],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # adding objects to trainer:
    cap_brain.tokenizer = tokenizer

    # Training
    cap_brain.fit(
        cap_brain.hparams.epoch_counter,
        train_set,
        valid_set,
        train_loader_kwargs=hparams["train_dataloader_opts"],
        valid_loader_kwargs=hparams["valid_dataloader_opts"],
    )

    # Testing

    cap_brain.hparams.wer_file = os.path.join(
        hparams["output_folder"], "wer_{}.txt".format('100_test')
    )
    cap_brain.evaluate(
        test_set,
        max_key="ACC",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
