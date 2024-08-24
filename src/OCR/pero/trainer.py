import json
import os
from typing import Union, Dict, Optional

import numpy as np
import torch
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from tqdm import tqdm
import Levenshtein

from src.OCR.pero.ocr_engine import transformer
from src.OCR.pero.dataset import Dataset
from src.OCR.utils import set_seed


ALPHABET = ['<PAD>', '<START>', '<NAN>', '<END>',
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q',
            'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q',
            'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
            'ä', 'ö', 'ü', 'ſ', 'ẞ', 'à',
            '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
            ' ', ',', '.', '?', '!', '-', '_', ':', ';', '/', '(', ')',
            '\"', '\'', '&', '+' '~']

LR = 1e-4
CROP_HEIGHT = 64
VALID_EVERY = 12800
BATCH_SIZE = 128


class Trainer:
    """Class to train models."""

    def __init__(
            self,
            model: transformer.TransformerOCR,
            traindataset: Dataset,
            testdataset: Dataset,
            optimizer: Optimizer,
            name: str,
            cuda: int = 0,
    ) -> None:
        """
        Trainer class to train models.

        Args:
            model: model to train
            traindataset: dataset to train on
            testdataset: dataset to validate model while trainings process
            optimizer: optimizer to use
            name: name of the model in save-files and tensorboard
            cuda: number of used cuda device
        """
        print(f"{torch.cuda.is_available()=}")
        self.device = (
            torch.device(f"cuda:{cuda}")
            if torch.cuda.is_available() and cuda >= 0
            else torch.device("cpu")
        )
        print(f"using {self.device}")

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.tokenizer = traindataset.tokenizer
        self.loss_fn = torch.nn.CrossEntropyLoss()

        self.trainloader = DataLoader(
            traindataset, batch_size=1, shuffle=True, num_workers=24
        )
        self.testloader = DataLoader(
            testdataset, batch_size=1, shuffle=False, num_workers=24
        )

        self.bestavgloss: Union[float, None] = None
        self.epoch = 0
        self.step = 0
        self.name = name

        # setup tensorboard
        train_log_dir = f"logs/runs/{self.name}"
        print(f"{train_log_dir=}")
        self.writer = SummaryWriter(train_log_dir)  # type: ignore

        self.valid_example_image, self.valid_example_label, self.valid_example_text = testdataset[2]
        self.train_example_image, self.train_example_label, self.train_example_text = traindataset[
            0]

        self.log_text(dataset='Valid', step=self.step, ground_truth=self.valid_example_text)
        self.log_text(dataset='Training', step=self.step, ground_truth=self.train_example_text)

    def save(self, name: str = "") -> None:
        """
        Save the model in models folder.

        Args:
            name: name of the model
        """
        os.makedirs("models/", exist_ok=True)
        torch.save(
            self.model.state_dict(),
            f"models/{name}",
        )

    def load(self, name: str = "") -> None:
        """
        Load the given model.

        Args:
            name: name of the model
        """
        self.model.load_state_dict(
            torch.load(f"models/{name}.pt")
        )

    def train(self, epoch: int) -> None:
        """
        Train model for given number of epochs.

        Args:
            epoch: number of epochs
        """
        for self.epoch in range(1, epoch + 1):
            print(f"start epoch {self.epoch}:")
            self.train_epoch()

        # save model after training
        self.save(f"{self.name}_end.pt")

    def train_epoch(self) -> None:
        """Trains one epoch."""
        loss_lst = []

        process_bar = tqdm(self.trainloader, desc="training")
        for crops, labels, _ in process_bar:
            self.step += 1
            crops = crops.to(self.device)
            labels = labels.to(self.device)

            if self.step % BATCH_SIZE == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            output = self.model.forward(crops, labels[:, :-1])
            loss = self.loss_fn(output.squeeze(), labels[:, 1:].squeeze())
            loss.backward()

            loss_lst.append(loss.cpu().detach())

            process_bar.set_description(f"training loss: {loss_lst[-1]}")

            if self.step % VALID_EVERY == 0:
                self.log_loss('Training', step=self.step,
                              step_loss=np.mean(loss_lst[-VALID_EVERY:]))
                avgloss = self.valid()

                # early stopping
                if self.bestavgloss is None or self.bestavgloss > avgloss:
                    self.bestavgloss = avgloss
                    self.save(f"{self.name}_es.pt")

            del crops, labels, output, loss

        self.log_loss('Training', loss=np.mean(loss_lst))

        del loss_lst

    def valid(self) -> float:
        """
        Validates current model on validation set.

        Returns:
            current loss
        """
        loss_lst = []
        levenshtein_lst = []

        counter = 0
        for crops, labels, text in tqdm(self.testloader, desc="validation"):
            counter += 1
            crops = crops.to(self.device)
            labels = labels.to(self.device)

            output = self.model.forward(crops, labels[:, :-1]).view(-1, len(ALPHABET))
            loss = self.loss_fn(output, labels[:, 1:].view(-1))
            loss_lst.append(loss.cpu().detach())

            pred_text = self.predict(crops)
            ratio = Levenshtein.distance(pred_text, text) / max(len(pred_text), len(text))
            levenshtein_lst.append(ratio)

            del crops, labels, output, loss, pred_text, ratio

            if counter >= int(VALID_EVERY * 0.1):
                break

        self.log_loss('Valid',
                      step=self.step,
                      loss=np.mean(loss_lst))

        self.log_loss('Valid',
                      step=self.step,
                      levenshtein=np.mean(levenshtein_lst))

        self.log_examples('Training')
        self.log_examples('Valid')

        return np.mean(loss_lst)  # type: ignore

    def predict(self, images: torch.Tensor, max_length: int = 100,
                start_token_idx: int = 1,
                eos_token_idx: int = 3):
        """
            Perform autoregressive prediction using the transformer decoder.

            Args:
                images: Input tensor for the encoder, expected shape [batch_size, seq_length].
                max_length: Maximum len
                gth of the sequence to be generated.
                start_token_idx: Index of the start token in the vocabulary.
                eos_token_idx: Index of the end token in the vocabulary.

            Returns:
                generated_sequences: The predicted sequences, shape [batch_size, max_length].
            """

        # Step 1: Encode the input sequence
        encoder_output = self.model.encode(images)

        # Step 2: Initialize the generated sequences with the start token
        batch_size = images.size(0)
        generated_sequences = torch.full((1, batch_size), start_token_idx, dtype=torch.long).to(
            images.device)

        # Step 3: Iteratively generate the sequence
        for i in range(max_length - 1):  # Already have <START> as the first token
            # Get the current length of the generated sequence
            tgt_len = generated_sequences.size(0)

            # Step 3a: Get the mask for the current length of the generated sequence
            dec_mask = self.model.get_mask(tgt_len).to(images.device)

            # Step 3b: Get the embeddings and apply positional encoding
            tgt_embs = self.model.dec_embeder(generated_sequences)
            tgt_embs = self.model.pos_encoder(tgt_embs)

            # print(f"{i}: {tgt_embs.shape=}")

            # Step 3c: Pass through the decoder
            decoder_output = self.model.trans_decoder(tgt_embs, encoder_output, tgt_mask=dec_mask)
            # print(f"{i}: {decoder_output.shape=}")

            # Step 3d: Project the decoder output to vocabulary size and get the next token
            # Use the last step's output for next token
            output_logits = self.model.dec_out_proj(decoder_output[-1, :, :])
            next_token = output_logits.argmax(dim=-1,
                                              keepdim=True)  # Get the most likely next token

            # Append the next token to the generated sequence
            generated_sequences = torch.cat([generated_sequences, next_token], dim=0)

            # Stop if all sequences have generated the <eos> token
            if (next_token == eos_token_idx).all():
                break

        # Return batch-first output shape [batch_size, seq_length]
        pred = generated_sequences.permute(1, 0)
        return self.tokenizer.to_text(pred[0])

    def log_examples(self, dataset: str) -> None:
        """
        Predicts and logs a example image form the training- and from the validation set.

        Args:
            dataset: dataset to log
        """
        self.model.eval()

        example = self.train_example_image if dataset == 'Training' else self.valid_example_image
        text = self.train_example_text if dataset == 'Training' else self.valid_example_text

        pred_text = self.predict(example[None].to(self.device))
        ratio = Levenshtein.distance(pred_text, text) / max(len(pred_text), len(text))
        print(f"{dataset} prediction (L: {ratio}): \"{pred_text}\"")

        self.log_text(dataset=dataset,
                      step=self.step,
                      prediction=pred_text)

        self.model.train()

    def log_text(self, dataset: str, step: Optional[int] = None,
                 **kwargs: Dict[str, torch.Tensor]) -> None:
        """
        Logs given images under the given dataset label.

        Args:
            dataset: dataset to log the images under ('Training' or 'Validation')
            step: Optional value for step (default is current epoch)
            kwargs: Dict with names (keys) and texts (string) to log
        """
        for key, text in kwargs.items():
            # log in tensorboard
            self.writer.add_text(
                f"{dataset}/{key}",
                text,  # type: ignore
                global_step=self.epoch if step is None else step
            )  # type: ignore

        self.writer.flush()  # type: ignore

    def log_loss(self, dataset: str, step: Optional[int] = None,
                 **kwargs: Dict[str, float]) -> None:
        """
        Logs the loss values to tensorboard.

        Args:
            dataset: Name of the dataset the loss comes from ('Training' or 'Valid')
            step: Optional value for step (default is current epoch)
            kwargs: dict with loss names (keys) and loss values (values)

        """
        # logging
        for key, value in kwargs.items():
            self.writer.add_scalar(
                f"{dataset}/{key}",
                value,
                global_step=self.epoch if step is None else step
            )  # type: ignore

        self.writer.flush()  # type: ignore


def main():
    set_seed(42)

    trainset = Dataset(image_path='data/preprocessedOCR/train',
                       target_path='data/preprocessedOCR/train',
                       alphabet=ALPHABET,
                       pad=False,
                       cache_images=True)

    validset = Dataset(image_path='data/preprocessedOCR/valid',
                       target_path='data/preprocessedOCR/valid',
                       alphabet=ALPHABET,
                       pad=False,
                       cache_images=True)

    print(f"training with {len(trainset)} samples for training and {len(validset)} "
          f"samples for validation.")

    with open("train/config.json", "r") as file:
        json_data = json.load(file)

    net: transformer.TransformerOCR = transformer.build_net(net=json_data,
                                                input_height=CROP_HEIGHT,
                                                input_channels=3,
                                                nb_output_symbols=len(ALPHABET) - 2)
    optimizer = AdamW(net.parameters(), lr=LR)

    trainer = Trainer(model=net,
                      traindataset=trainset,
                      testdataset=validset,
                      optimizer=optimizer,
                      name='test29')

    trainer.train(2)


if __name__ == '__main__':
    main()
