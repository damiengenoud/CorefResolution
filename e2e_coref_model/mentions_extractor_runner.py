import os
from datetime import datetime
import torch
import bert
import mentions_extractor


class MentionsExtractorRunner:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = mentions_extractor.MentionsExtractor(1024, 0.25, 15).to(self.device)
        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=5e-5
        )
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optim, step_size=100, gamma=0.999
        )

    @staticmethod
    def compute_loss(
            # [k], [k]
            scores, labels
    ):
        loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
        return loss(scores, labels)

    def get_validation_loss(self, validation_set_data):
        total_loss = 0

        # Iterate over all validation documents
        for i, data in enumerate(validation_set_data):
            # Get encoded text and true mentions
            encoded_text = data

            # Run the mentions extractor
            top_spans_start_indices, _, top_spans_scores, _ = self.model(encoded_text)

            # TODO parse conll and get labels
            labels = torch.where(top_spans_start_indices % 2 == 0, 1.0, 0.0).to(self.device)

            # Compute the loss
            loss = self.compute_loss(top_spans_scores, labels)

            # Print information
            print('  validation document {} loss: {}'.format(i + 1, loss.item()))

            # Add the loss of this document
            total_loss += loss.item()

        # Return average loss of the validation
        return total_loss / len(validation_set_data)

    def train_one_epoch(self, encoded_training_texts_data):
        total_loss = 0

        # Iterate over all training documents
        for i, data in enumerate(encoded_training_texts_data):
            # Get encoded text and true mentions
            encoded_text = data

            self.optim.zero_grad()

            # Run the mentions extractor
            top_spans_start_indices, _, top_spans_scores, _ = self.model(encoded_text)

            # TODO parse conll and get labels
            labels = torch.where(top_spans_start_indices % 2 == 0, 1.0, 0.0).to(self.device)

            # Compute the loss
            loss = self.compute_loss(top_spans_scores, labels)
            loss.backward()

            # Update the weights
            self.optim.step()

            # Print information
            print('  training document {} loss: {}'.format(i + 1, loss.item()))

            # Add the loss of this document
            total_loss += loss.item()

        # Return average loss of the training
        return total_loss / len(encoded_training_texts_data)

    def train(self):
        # Prepare for training
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        num_epochs = 100
        best_validation_loss = float('inf')

        # Prepare sample dataset for training
        text = "I'm a penguin who likes icecream, I find it very good. My mother always said that it is bad for me to eat so much of it. I hope she is not angry about it."
        encoded_text = bert.spanbert_encode(text).detach()
        encoded_training_texts_data = [encoded_text]
        encoded_validation_texts_data = [encoded_text]

        for epoch in range(num_epochs):
            print('EPOCH {}:'.format(epoch + 1))

            # Train the model and get the training loss of the current epoch
            training_loss = self.train_one_epoch(encoded_training_texts_data)

            # Compute the loss on the validation set
            with torch.no_grad():
                validation_loss = self.get_validation_loss(encoded_validation_texts_data)

            print('LOSS: train {}, valid {}'.format(training_loss, validation_loss))

            # Save the best model (in terms of validation loss)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                model_path = 'saved_models/mentions_extractor'
                os.makedirs(model_path, exist_ok=True)
                torch.save(self.model.state_dict(), model_path + '/model.pt'.format(timestamp, epoch + 1))

