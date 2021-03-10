from parser_utils import get_args
from preprocess_docs import CorpusPreProcessor
from dataloader import DocumentGraphDataset
from learner import Learner

if __name__ == "__main__":

    args, device = get_args()

    ######################################
    # Initiate learner
    ######################################
    learner = Learner(
        experiment_name=args.experiment_name,
        device=device,
        multi_label=False
    )

    if args.do_train:

        ######################################
        # Load dataset
        ######################################
        corpus_prepper = CorpusPreProcessor(
            min_freq_word=args.min_freq_word,
            multi_label=False
        )
        # Read data
        docs, labels, n_labels, word2idx = corpus_prepper.load_clean_corpus(args.path_to_dataset)
        # Split into train/val
        train_docs, dev_docs, train_labels, dev_labels = corpus_prepper.split_corpus(docs, labels, args.percentage_dev)
        # Load embeddings
        embeddings = corpus_prepper.load_embeddings(args.path_to_embeddings, word2idx, embedding_type='word2vec')

        # Instantiate dataloader
        dataset_train = DocumentGraphDataset(
            docs=train_docs,
            labels=train_labels,
            word2idx=word2idx,
            window_size=args.window_size,
            use_master_node=args.use_master_node,
            normalize_edges=args.normalize,
            use_directed_edges=args.directed
        )

        dataloader_train = dataset_train.to_dataloader(
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True
        )

        dataset_dev = DocumentGraphDataset(
            docs=dev_docs,
            labels=dev_labels,
            word2idx=word2idx,
            window_size=args.window_size,
            use_master_node=args.use_master_node,
            normalize_edges=args.normalize,
            use_directed_edges=args.directed
        )

        dataloader_dev = dataset_dev.to_dataloader(
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False
        )

        ######################################
        # Initiate model
        ######################################

        if args.pretrained_model is None:
            # Initialize a new model
            learner.init_model(
                args.model_type,
                lr=args.lr,
                n_feat=embeddings.shape[1],
                n_message_passing=args.message_passing_layers,
                n_hid=args.hidden,
                n_penultimate=args.penultimate,
                n_class=n_labels,
                dropout=args.dropout,
                embeddings=embeddings,
                use_master_node=args.use_master_node

            )
        else:
            # Load pretrained
            learner.load_model(
                path=args.pretrained_model,
                lr=args.lr
            )

        ######################################
        # Start training
        ######################################

        eval_every = len(dataloader_train) if args.eval_every == 'epoch' else int(args.eval_every)

        for epoch in range(args.epochs):

            learner.train_epoch(
                dataloader_train,
                eval_every=eval_every,
                batch_size=args.batch_size
            )

            learner.evaluate(
                dataloader_dev
            )

    ######################################
    # Infer Test Set
    ######################################
    if args.do_evaluate:

        if args.do_train:
            print("Loading best model to infer test set...")
            learner.load_best_model()
        else: # Load other pretrained model
            assert args.pretrained_model is not None, "--pretrained-model must be given when --do-train is False and --do-evaluate is True"
            print("Loading pretrained model for evaluation...")
            learner.load_model(args.pretrained_model)

        ######################################
        # Load dataset
        ######################################






