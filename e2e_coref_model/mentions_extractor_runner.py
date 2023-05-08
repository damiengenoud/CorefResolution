from datetime import datetime
import torch
import mentions_extractor


class MentionsExtractorRunner:
    def __init__(self):
        self.model = mentions_extractor.MentionsExtractor()
        self.optim = torch.optim.Adam(
            self.model.parameters(), lr=5e-5
        )
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optim, step_size=100, gamma=0.999
        )

    @staticmethod
    def compute_loss(
            # [num_spans], [num_spans], [num_spans]
            scores, top_scoring_spans_mask, true_mentions_labels
    ):
        loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
        return loss(
            torch.masked_select(scores, top_scoring_spans_mask),  # [num_top_spans]
            torch.masked_select(true_mentions_labels, top_scoring_spans_mask)  # [num_top_spans]
        )

    def train_one_epoch(self, encoded_training_texts_data):
        total_loss = 0

        # Iterate over all training documents
        for i, data in enumerate(encoded_training_texts_data):
            # Get encoded text and true mentions
            encoded_text, true_mentions_labels_list = data  # true_mentions_label_list: [num_spans]
            true_mentions_labels = torch.Tensor([true_mentions_labels_list])  # [num_spans]

            self.optim.zero_grad()

            # Run the mentions extractor
            mentions_scores, top_scoring_spans_mask, _ = self.model(encoded_text)

            # Compute the loss
            loss = self.compute_loss(mentions_scores, top_scoring_spans_mask, true_mentions_labels)
            loss.backward()

            # Update the weights
            self.optim.step()

            # Print information
            print('  training document {} loss: {}'.format(i + 1, loss.item()))

            # Add the loss of this document
            total_loss += loss.item()

        # Return average loss of the training
        return total_loss / len(encoded_training_texts_data)

    def get_validation_loss(self, encoded_validation_texts_data):
        total_loss = 0

        # Iterate over all validation documents
        for i, data in enumerate(encoded_validation_texts_data):
            # Get encoded text and true mentions
            encoded_text, true_mentions_labels_list = data  # true_mentions_label_list: [num_spans]
            true_mentions_labels = torch.Tensor([true_mentions_labels_list])  # [num_spans]

            # Run the mentions extractor
            mentions_scores, top_scoring_spans_mask, _ = self.model(encoded_text)

            # Compute the loss
            loss = self.compute_loss(mentions_scores, top_scoring_spans_mask, true_mentions_labels)

            # Print information
            print('  validation document {} loss: {}'.format(i + 1, loss.item()))

            # Add the loss of this document
            total_loss += loss.item()

        # Return average loss of the validation
        return total_loss / len(encoded_validation_texts_data)

    def train(self):
        # Prepare for training
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        num_epochs = 100
        best_validation_loss = float('inf')

        # Prepare dataset for training
        encoded_training_texts_data = ["", "a"]
        encoded_validation_texts_data = ["", "a"]

        for epoch in range(num_epochs):
            print('EPOCH {}:'.format(epoch + 1))

            # Train the model and get the training loss of the current epoch
            self.model.train(True)
            training_loss = self.train_one_epoch(encoded_training_texts_data)
            self.model.train(False)

            # Compute the loss on the validation set
            validation_loss = self.get_validation_loss(encoded_validation_texts_data)

            print(print('LOSS: train {}, valid {}'.format(training_loss, validation_loss)))

            # Save the best model (in terms of validation loss)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                model_path = 'saved_models/model_{}_{}'.format(timestamp, epoch)
                torch.save(self.model.state_dict(), model_path)

