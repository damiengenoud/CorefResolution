import os
import torch
import mentions_extractor
import utils
import config


class MentionsExtractorTrainer:
    def __init__(self, encoder, tokenizer, bert_embedding_size):
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.extractor = mentions_extractor.MentionsExtractor(bert_embedding_size, config.TOP_SPAN_RATIO, config.MAX_SPAN_LENGTH).to(utils.device)
        self.optim = torch.optim.Adam(
            self.extractor.parameters(), lr=config.MENTIONS_EXTRACTOR_LR
        )
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(  # not used for now
            self.optim, step_size=config.MENTIONS_EXTRACTOR_LR_DECAY_STEP, gamma=config.MENTIONS_EXTRACTOR_LR_DECAY
        )

    @staticmethod
    def compute_loss(
            # [k], [k]
            scores, labels
    ):
        loss = torch.nn.BCEWithLogitsLoss(reduction='sum')
        return loss(scores, labels)

    @staticmethod
    def get_mentions_labels(
            true_entities: torch.Tensor,  # [num_true_mentions, 3]
            predicted_start_indices: torch.Tensor,  # [num_predicted_spans]
            predicted_end_indices: torch.Tensor  # [num_predicted_spans]
    ) -> torch.Tensor:
        with torch.no_grad():
            predicted_indices = torch.stack([predicted_start_indices, predicted_end_indices], dim=1)
            true_indices = true_entities[:, :2]
            # Expands the tensors to be able to use a sort of "contains" function, see :
            # https://stackoverflow.com/questions/73524338/pytorch-test-each-row-of-the-first-2d-tensor-also-exist-in-the-second-tensor
            predicted_expand = predicted_indices.unsqueeze(1).expand(-1, true_indices.shape[0],
                                                                     predicted_indices.shape[1])
            true_expand = true_indices.unsqueeze(0).expand(predicted_indices.shape[0], -1,
                                                           predicted_indices.shape[1]).to(
                utils.device
            )

            # Similar to a "contains" function
            labels = (predicted_expand == true_expand).all(-1).any(-1).float()  # [num_predicted_spans]
            return labels

    def one_pass(self, dataloader, training: bool):
        total_loss = 0

        # Iterate over all training documents
        for data in dataloader:
            # Get encoded text and true entities
            encoded_text = data["encoding"].squeeze()
            true_entities = data["doc_entities"].squeeze()

            if training:
                self.optim.zero_grad()

            # Run the mentions extractor
            top_spans_start_indices, top_spans_end_indices, top_spans_scores = self.extractor(encoded_text)

            # Get the labels of the extracted spans
            labels = self.get_mentions_labels(true_entities, top_spans_start_indices, top_spans_end_indices).to(
                utils.device)

            # Compute the loss
            loss = self.compute_loss(top_spans_scores, labels)

            if training:
                loss.backward()
                # Update the weights
                self.optim.step()

            # Add the loss of this document
            total_loss += loss.item()

        # Return average loss of the training
        return total_loss / len(dataloader)

    def train(self, train_dataloader, val_dataloader):
        # Prepare for training
        num_epochs = 50
        best_validation_loss = float('inf')

        print('STARTING MENTIONS EXTRACTOR TRAINING')
        print('------------------------------------')

        for epoch in range(num_epochs):
            print('EPOCH {}:'.format(epoch + 1))

            # Train the model and get the training loss of the current epoch
            training_loss = self.one_pass(train_dataloader, training=True)

            # Compute the loss on the validation set
            with torch.no_grad():
                validation_loss = self.one_pass(val_dataloader, training=False)

            print('LOSS: train {}, valid {}'.format(training_loss, validation_loss))

            # Save the best model (in terms of validation loss)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                model_path = 'saved_models/my_models'
                os.makedirs(model_path, exist_ok=True)
                torch.save(self.extractor.state_dict(), model_path + '/mentions_extractor.pt')

        print('------------------------------------')
        print('MENTIONS EXTRACTOR TRAINING FINISHED')
