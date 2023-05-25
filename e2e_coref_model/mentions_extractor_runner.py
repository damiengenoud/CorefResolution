import os
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader
import mentions_extractor
import utils
import custom_dataset


class MentionsExtractorRunner:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.encoder = AutoModel.from_pretrained("SpanBERT/spanbert-base-cased").to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        self.model = mentions_extractor.MentionsExtractor(768, 0.25, 15).to(self.device)
        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=5e-5
        )
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(  #not used for now
            self.optim, step_size=2, gamma=0.95
        )

    @staticmethod
    def compute_loss(
            # [k], [k]
            scores, labels
    ):
        loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
        return loss(scores, labels)

    def get_validation_loss(self, validation_dataloader):
        total_loss = 0

        # Iterate over all validation documents
        for data in validation_dataloader:
            # Get encoded text and true entities
            encoded_text = data["encoding"].squeeze()
            true_entities = data["doc_entities"].squeeze()

            # Run the mentions extractor
            top_spans_start_indices, top_spans_end_indices, top_spans_scores = self.model(encoded_text)

            # Get the labels of the extracted spans
            labels = utils.get_mentions_labels(true_entities, top_spans_start_indices, top_spans_end_indices).to(self.device)

            # Compute the loss
            loss = self.compute_loss(top_spans_scores, labels)

            # Print information
            # print('  validation document {} loss: {}'.format(docId, loss.item()))

            # Add the loss of this document
            total_loss += loss.item()

        # Return average loss of the validation
        return total_loss / len(validation_dataloader)

    def train_one_epoch(self, training_dataloader):
        total_loss = 0

        # Iterate over all training documents
        for data in training_dataloader:
            # Get encoded text and true entities
            encoded_text = data["encoding"].squeeze()
            true_entities = data["doc_entities"].squeeze()

            self.optim.zero_grad()

            # Run the mentions extractor
            top_spans_start_indices, top_spans_end_indices, top_spans_scores = self.model(encoded_text)

            # Get the labels of the extracted spans
            labels = utils.get_mentions_labels(true_entities, top_spans_start_indices, top_spans_end_indices).to(self.device)

            # Compute the loss
            loss = self.compute_loss(top_spans_scores, labels)
            loss.backward()

            # Update the weights
            self.optim.step()

            # Print information
            # print('  training document {} loss: {}'.format(docId, loss.item()))

            # Add the loss of this document
            total_loss += loss.item()

        # Return average loss of the training
        return total_loss / len(training_dataloader)

    def train(self):
        # Prepare for training
        num_epochs = 50
        best_validation_loss = float('inf')

        print('STARTING MENTIONS EXTRACTOR TRAINING')
        print('------------------------------------')

        # Get the data ready
        with torch.no_grad():
            train_data_list = utils.prepare_data(self.tokenizer, self.encoder, "train")
            train_dataset = custom_dataset.CustomDataset(train_data_list)
            train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
            dev_data_list = utils.prepare_data(self.tokenizer, self.encoder, "dev")
            dev_dataset = custom_dataset.CustomDataset(dev_data_list)
            dev_dataloader = DataLoader(dev_dataset, batch_size=1, shuffle=True)

        for epoch in range(num_epochs):
            print('EPOCH {}:'.format(epoch + 1))

            # Train the model and get the training loss of the current epoch
            training_loss = self.train_one_epoch(train_dataloader)

            # Compute the loss on the validation set
            with torch.no_grad():
                validation_loss = self.get_validation_loss(dev_dataloader)

            print('LOSS: train {}, valid {}'.format(training_loss, validation_loss))

            # Save the best model (in terms of validation loss)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                model_path = 'saved_models/mentions_extractor'
                os.makedirs(model_path, exist_ok=True)
                torch.save(self.model.state_dict(), model_path + '/mentions_extractor.pt')

        print('------------------------------------')
        print('MENTIONS EXTRACTOR TRAINING FINISHED')

