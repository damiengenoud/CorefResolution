from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import DataLoader
import mentions_extractor_trainer
import mentions_linker
import conll
import config
import utils


class CorefModel:
    def __init__(self):
        self.encoder = AutoModel.from_pretrained(config.SPANBERT_MODEL)
        self.tokenizer = AutoTokenizer.from_pretrained(config.SPANBERT_TOKENIZER)

        self.linker = mentions_linker.MentionsLinker().to(utils.device)

        self.optim = torch.optim.Adam(
            self.linker.parameters(), lr=config.MENTIONS_LINKER_LR
        )
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(  # not used for now
            self.optim, step_size=config.MENTIONS_LINKER_LR_DECAY_STEP, gamma=config.MENTIONS_LINKER_LR_DECAY
        )

    @staticmethod
    def compute_loss(
            all_pairs_scores,  # [k, num_antecedents + 1]
            antecedents_labels  # [k, num_antecedents + 1]
    ):
        true_pairs_score = all_pairs_scores + torch.log(antecedents_labels)  # [k, num_antecedents + 1]

        log_sum_true_antecedents_score = torch.logsumexp(true_pairs_score, 1)  # [k]
        log_sum_all_antecedents_score = torch.logsumexp(all_pairs_scores, 1)  # [k]

        loss = torch.sum(log_sum_true_antecedents_score - log_sum_all_antecedents_score)  # []
        return -loss

    @staticmethod
    def get_labels(true_entities,  # [num_true_spans, 3]
                   predicted_start_indices,  # [k]
                   predicted_end_indices,  # [k]
                   antecedents,  # [k, num_antecedents]
                   antecedents_mask  # [k, num_antecedents]
                   ) -> torch.Tensor:

        true_entities = true_entities.to(utils.device)
        predicted_indices = torch.stack([predicted_start_indices, predicted_end_indices], dim=1)  # [k]
        true_indices = true_entities[:, :2]  # [num_true_spans, 2]
        cluster_ids = true_entities[:, 2]  # [num_true_spans]

        predicted_expand = predicted_indices.unsqueeze(1).expand(-1, true_indices.shape[0],
                                                                 predicted_indices.shape[1])  # [k, num_true_spans, 2]
        true_expand = true_indices.unsqueeze(0).expand(predicted_indices.shape[0], -1,
                                                       predicted_indices.shape[1])  # [k, num_true_spans, 2]

        same_span = (predicted_expand == true_expand).all(-1)  # [k, num_true_spans]
        candidate_labels = torch.matmul(same_span.float(),
                                        torch.unsqueeze(cluster_ids, 1).float()).squeeze().long()  # [k]

        antecedents_cluster = candidate_labels[antecedents] + torch.log(antecedents_mask.float())  # [k, num_antecedents]

        antecedent_same_cluster = torch.logical_and(torch.eq(antecedents_cluster, candidate_labels.unsqueeze(1)),
                                                    antecedents_cluster > 0)  # [k, k]
        has_dummy_antecedent = torch.logical_not(antecedent_same_cluster.any(dim=1, keepdim=True))  # [k, 1]
        antecedents_labels = torch.cat([antecedent_same_cluster, has_dummy_antecedent], 1).float()  # [k, num_antecedents + 1]

        return antecedents_labels  # [k, num_antecedents + 1]

    def one_pass(self, dataloader, training: bool):
        total_loss = 0

        # Iterate over all training documents
        for data in dataloader:
            # Get encoded text and true entities
            encoded_text = data["encoding"].squeeze().to(utils.device)
            true_entities = data["doc_entities"].squeeze().to(utils.device)

            if training:
                self.optim.zero_grad()

            # Run the mentions linker
            spans_start_indices, spans_end_indices, pairs_scores, antecedents, antecedents_mask = self.linker(
                encoded_text)

            # Get the labels of the extracted spans
            labels = self.get_labels(true_entities, spans_start_indices, spans_end_indices, antecedents,
                                     antecedents_mask).to(utils.device)

            # Compute the loss
            loss = self.compute_loss(pairs_scores, labels)

            if training:
                loss.backward()
                self.optim.step()
                # self.lr_scheduler.step()

            # Add the loss of this document
            total_loss += loss.item()

        # Return average loss of the training and
        return total_loss / len(dataloader)

    def train(self, model_save_folder_path: str = config.MY_MODELS_FOLDER):
        # Prepare the dataset
        train_dataloader = DataLoader(utils.get_dataset(self.tokenizer, self.encoder, 'train'), batch_size=1,
                                      shuffle=True)
        val_dataloader = DataLoader(utils.get_dataset(self.tokenizer, self.encoder, 'val'), batch_size=1,
                                    shuffle=True)

        # Train the mentions extractor
        extractor_trainer = mentions_extractor_trainer.MentionsExtractorTrainer()
        extractor_trainer.train(train_dataloader, val_dataloader, model_save_folder_path)

        # Load trained mentions extractor in linker
        self.linker.load_extractor(model_save_folder_path)

        print('STARTING FULL MODEL TRAINING')
        best_validation_loss = float('inf')

        for epoch in range(config.MENTIONS_LINKER_EPOCHS):
            print('EPOCH {}:'.format(epoch + 1))

            # Train the model and get the training loss of the current epoch
            training_loss = self.one_pass(train_dataloader, True)

            # Compute the loss on the validation set
            with torch.no_grad():
                validation_loss = self.one_pass(val_dataloader, False)

            print('LOSS: train {}, valid {}'.format(training_loss, validation_loss))

            # Save the best model (in terms of validation loss)
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                torch.save(self.linker.state_dict(), model_save_folder_path + config.COREF_MODEL_NAME)

        print('FULL MODEL TRAINING FINISHED')
        print('------------------------------------')

    @torch.no_grad()
    def evaluate(self, model_folder_path: str = config.PRETRAINED_MODELS_FOLDER):
        # Prepare eval data
        test_dataloader = DataLoader(utils.get_dataset(self.tokenizer, self.encoder, 'test'), batch_size=1,
                                     shuffle=True)

        print('STARTING MODEL EVALUATION')

        # Load linker
        linker = mentions_linker.MentionsLinker().to(utils.device)
        linker.load_state_dict(torch.load(model_folder_path + config.COREF_MODEL_NAME))

        # Iterate over all documents
        for data in test_dataloader:
            # Get encoded text and true entities
            encoded_text = data["encoding"].squeeze().to(utils.device)

            # Run the mentions linker
            spans_start_indices, spans_end_indices, pairs_scores, _, _ = linker(encoded_text)

            # Clusterize the result
            clusters = utils.clusterize_antecedents(pairs_scores)
            
            print(clusters)

            # Write the CONLL files for scorer
            conll.write_conll(data["doc_id"][0], spans_start_indices, spans_end_indices, clusters)

            # TODO add scorer call, does not work on windows...

        print('------------------------------------')
        print('MODEL EVALUATION FINISHED')
