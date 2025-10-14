# Usage:
#   make -n learn-all -- Will print which LeaPR feature files are still missing
#   make learn-all -- Runs all LeaPR training jobs:
#       make results/features/[...].json -- Runs a single LeaPR training job.
#   make train-all -- Train and evaluate random forest from existing LeaPR-learned features:
#       make results/evals/[...].json -- Train and evaluate from a single LeaPR feature file
#   make stockfish-eval -- Evaluate a learned chess value function on Stockfish accuracy
#       make results/stockfish-evals/method__chess__model.json
#

MODELS := gpt-5-mini gpt-4o-mini # claude-4-sonnet
METHODS := did3 funsearch 
METHODS_WITH_COMBO := did3 funsearch combo
DOMAINS := chess image_classification_mnist image_classification_fashion_mnist text_classification_ghostbuster

TRANSFER_PAIRS := image_classification_mnist__image_classification_fashion_mnist image_classification_fashion_mnist__image_classification_mnist

# Baseline variables
NN_MODELS := resnet50 efficientnet
NN_DATASETS := mnist fashion_mnist
NN_INITS := random imagenet

FEATURE_TARGETS := $(foreach model,$(MODELS), \
			$(foreach method,$(METHODS), \
				$(foreach domain,$(DOMAINS), \
					results/features/$(method)__$(domain)__$(model).json)))

TRAIN_TARGETS := $(foreach model,$(MODELS), \
			$(foreach method,$(METHODS), \
				$(foreach domain,$(DOMAINS), \
					results/evals/$(method)__$(domain)__$(model).json)))

NN_TARGETS := $(foreach model,$(NN_MODELS), \
		$(foreach dataset,$(NN_DATASETS), \
			$(foreach init,$(NN_INITS), \
				results/nn/$(model)__$(dataset)__$(init).pth)))

COMBO_TARGETS := $(foreach model,$(MODELS), \
			$(foreach domain,$(DOMAINS), \
				results/features/combo__$(domain)__$(model).json))

TRANSFER_TARGETS := $(foreach model,$(MODELS), \
			$(foreach method,$(METHODS), \
				$(foreach domain,$(TRANSFER_PAIRS), \
					results/evals/transfer/$(method)__$(domain)__$(model).json)))

CHESS_EVAL_TARGETS := $(foreach model,$(MODELS), \
						$(foreach method,$(METHODS), \
							results/evals/chess/$(method)__$(model).json))

CHESS_ACCURACY_TARGETS := $(foreach model,$(MODELS), \
							$(foreach method,$(METHODS), \
								results/evals/chess/accuracy/$(method)__$(model).json))

results/features/%.json:
	$(eval parts := $(subst __, ,$*))
	$(eval method := $(word 1,$(parts)))
	$(eval domain_dataset := $(word 2,$(parts)))
	$(eval model := $(word 3,$(parts)))
	python launch.py --leapr --learner $(method) --domain $(domain_dataset) --model $(model)

results/features/combo__%.json:
	$(eval parts := $(subst __, ,$*))
	$(eval domain_dataset := $(word 1,$(parts)))
	$(eval model := $(word 2,$(parts)))
	python launch.py --combine --output $@ --model $(model) --domain $(domain_dataset)

results/features/raw__%.json:
	python launch.py --raw --output $@ --domain $*

results/evals/%.json: results/features/%.json
	$(eval parts := $(subst __, ,$*))
	$(eval method := $(word 1,$(parts)))
	$(eval domain_dataset := $(word 2,$(parts)))
	$(eval model := $(word 3,$(parts)))
	python launch.py --train --learner $(method) --domain $(domain_dataset) --model $(model)

results/evals/transfer/%.json:
	$(eval parts := $(subst __, ,$*))
	$(eval method := $(word 1,$(parts)))
	$(eval domain_dataset := $(word 2,$(parts)))
	$(eval transfer_dataset := $(word 3,$(parts)))
	$(eval model := $(word 4,$(parts)))
	python launch.py --train --learner $(method) --domain $(domain_dataset) --model $(model) --transfer-from $(transfer_dataset)

results/nn/resnet50__%.pth:
	$(eval parts := $(subst __, ,$*))
	$(eval dataset := $(word 1,$(parts)))
	$(eval init := $(word 2,$(parts)))
	python train.py --config-path=config/trainer --config-name=resnet50 dataset=$(dataset) trainer.initialization=$(init) trainer.lr=0.03 trainer.batch_size=1024 trainer.n_steps=4000 output=$@

results/nn/efficientnet__%.pth:
	$(eval parts := $(subst __, ,$*))
	$(eval dataset := $(word 1,$(parts)))
	$(eval init := $(word 2,$(parts)))
	python train.py --config-path=config/trainer --config-name=efficientnet dataset=$(dataset) trainer.initialization=$(init) trainer.lr=0.001 trainer.batch_size=1024 trainer.n_steps=4000 output=$@

results/chess_transformer.pt:
	DONT_PARSE_BOARD=1 python train.py trainer=transformer max_size=100000000 trainer.n_steps=250000 trainer.batch_size=400

results/evals/chess_transformer/%.json:
	python evaluation.py evaluator=state_value policy=value_softmax policy.model_path=results/transformer_ckpt_$*.pt evaluator.output=results/evals/chess_transformer_$*.json

results/evals/chess_transformer/accuracy/%.json:
	python evaluation.py evaluator=accuracy policy=value_softmax policy.model_path=results/transformer_ckpt_$*.pt evaluator.output=results/evals/chess_transformer/accuracy/$*.json evaluator.n_jobs=1

results/evals/chess/%.json:
	$(eval parts := $(subst __, ,$*))
	$(eval method := $(word 1,$(parts)))
	$(eval model := $(word 2,$(parts)))
	python evaluation.py evaluator=state_value policy=value_softmax policy.model_path=results/models/$(method)__chess__$(model).pkl evaluator.output=results/evals/chess/$*.json

results/evals/chess/accuracy/%.json:
	$(eval parts := $(subst __, ,$*))
	$(eval method := $(word 1,$(parts)))
	$(eval model := $(word 2,$(parts)))
	python evaluation.py evaluator=accuracy policy=value_softmax policy.model_path=results/models/$(method)__chess__$(model).pkl evaluator.output=results/evals/chess/accuracy/$*.json

eval-chess-random-policy:
	python evaluation.py evaluator=accuracy policy=uniform evaluator.output=results/evals/chess/accuracy/random.json

train-all: $(TRAIN_TARGETS)

learn-all: $(FEATURE_TARGETS)

raw-all: results/features/raw__chess.json # results/features/raw__image_classification_mnist

image-baselines-all: $(NN_TARGETS)

eval-transformers: \
	 $(foreach n,1000 10000 25000 50000 100000 last, results/evals/chess_transformer/$(n).json)

eval-transformers-accuracy: \
	 $(foreach n,25000 50000 100000 150000 200000 last, results/evals/chess_transformer/accuracy/$(n).json)

chess-transformer: results/chess_transformer.pt

funsearch-waterbird: results/features/funsearch__image_classification_waterbird__gpt-5-mini.json

combine: $(COMBO_TARGETS)

transfer: $(TRANSFER_TARGETS)

eval-chess-models: $(CHESS_EVAL_TARGETS) $(CHESS_ACCURACY_TARGETS)

all: train-all learn-all image-baselines-all chess-transformer

.PHONY: all check-features train-all image-baselines-all chess-transformer funsearch-waterbird eval-transformers eval-chess-models
