# Usage:
#   make learn-v2 -- Runs all V2 LeaPR training jobs (official test splits - DEFAULT)
#   make train-v2 -- Train and evaluate from V2 features (official test splits - DEFAULT)
#   make learn-all -- Runs all V1 LeaPR training jobs (original splits)
#   make train-all -- Train and evaluate from V1 features (original splits)
#

MODELS := gpt-5-mini gpt-4o-mini # claude-4-sonnet
METHODS := did3 funsearch 
METHODS_WITH_COMBO := did3 funsearch combo

# V1 domains (original splits)
DOMAINS_V1 := chess image_classification_mnist image_classification_fashion_mnist text_classification_ghostbuster

# V2 domains (official splits - excludes chess)
DOMAINS_V2 := image_classification_mnist image_classification_fashion_mnist text_classification_ghostbuster

TRANSFER_PAIRS := image_classification_mnist__image_classification_fashion_mnist image_classification_fashion_mnist__image_classification_mnist

# Baseline variables
NN_MODELS := resnet50 efficientnet
NN_DATASETS := mnist fashion_mnist
NN_INITS := random imagenet

# V1 targets (original splits)
FEATURE_TARGETS_V1 := $(foreach model,$(MODELS), \
			$(foreach method,$(METHODS), \
				$(foreach domain,$(DOMAINS_V1), \
					results/features/v1/$(method)__$(domain)__$(model).json)))

TRAIN_TARGETS_V1 := $(foreach model,$(MODELS), \
			$(foreach method,$(METHODS), \
				$(foreach domain,$(DOMAINS_V1), \
					results/evals/v1/$(method)__$(domain)__$(model).json)))

# V2 targets (official splits)
FEATURE_TARGETS_V2 := $(foreach model,$(MODELS), \
			$(foreach method,$(METHODS), \
				$(foreach domain,$(DOMAINS_V2), \
					results/features/v2/$(method)__$(domain)__$(model).json)))

TRAIN_TARGETS_V2 := $(foreach model,$(MODELS), \
			$(foreach method,$(METHODS), \
				$(foreach domain,$(DOMAINS_V2), \
					results/evals/v2/$(method)__$(domain)__$(model).json)))

NN_TARGETS := $(foreach model,$(NN_MODELS), \
		$(foreach dataset,$(NN_DATASETS), \
			$(foreach init,$(NN_INITS), \
				results/nn/$(model)__$(dataset)__$(init).pth)))

COMBO_TARGETS_V1 := $(foreach model,$(MODELS), \
			$(foreach domain,$(DOMAINS_V1), \
				results/features/v1/combo__$(domain)__$(model).json))

COMBO_TARGETS_V2 := $(foreach model,$(MODELS), \
			$(foreach domain,$(DOMAINS_V2), \
				results/features/v2/combo__$(domain)__$(model).json))

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

# V1 feature learning
results/features/v1/%.json:
	$(eval parts := $(subst __, ,$*))
	$(eval method := $(word 1,$(parts)))
	$(eval domain_dataset := $(word 2,$(parts)))
	$(eval model := $(word 3,$(parts)))
	mkdir -p results/features/v1
	python launch.py --leapr --learner $(method) --domain $(domain_dataset) --model $(model) --output v1/$*

# V2 feature learning
results/features/v2/%.json:
	$(eval parts := $(subst __, ,$*))
	$(eval method := $(word 1,$(parts)))
	$(eval domain_dataset := $(word 2,$(parts)))
	$(eval model := $(word 3,$(parts)))
	mkdir -p results/features/v2
	python launch.py --leapr --learner $(method) --domain $(domain_dataset) --model $(model) --output v2/$*

# V1 combo features
results/features/v1/combo__%.json:
	$(eval parts := $(subst __, ,$*))
	$(eval domain_dataset := $(word 1,$(parts)))
	$(eval model := $(word 2,$(parts)))
	mkdir -p results/features/v1
	python launch.py --combine --output $@ --model $(model) --domain $(domain_dataset)

# V2 combo features
results/features/v2/combo__%.json:
	$(eval parts := $(subst __, ,$*))
	$(eval domain_dataset := $(word 1,$(parts)))
	$(eval model := $(word 2,$(parts)))
	mkdir -p results/features/v2
	python launch.py --combine --output $@ --model $(model) --domain $(domain_dataset)

results/features/raw__%.json:
	python launch.py --raw --output $@ --domain $*

# V1 training
results/evals/v1/%.json: results/features/v1/%.json
	$(eval parts := $(subst __, ,$*))
	$(eval method := $(word 1,$(parts)))
	$(eval domain_dataset := $(word 2,$(parts)))
	$(eval model := $(word 3,$(parts)))
	mkdir -p results/evals/v1
	mkdir -p results/models/v1
	python launch.py --train --learner $(method) --domain $(domain_dataset) --model $(model) --version v1

# V2 training (default)
results/evals/v2/%.json: results/features/v2/%.json
	$(eval parts := $(subst __, ,$*))
	$(eval method := $(word 1,$(parts)))
	$(eval domain_dataset := $(word 2,$(parts)))
	$(eval model := $(word 3,$(parts)))
	mkdir -p results/evals/v2
	mkdir -p results/models/v2
	python launch.py --train --learner $(method) --domain $(domain_dataset) --model $(model) --version v2

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

# V2 targets (official splits - DEFAULT)
train-v2: $(TRAIN_TARGETS_V2)

learn-v2: $(FEATURE_TARGETS_V2)

combine-v2: $(COMBO_TARGETS_V2)

all-v2: learn-v2 train-v2

# V1 targets (original splits)
train-all: $(TRAIN_TARGETS_V1)

learn-all: $(FEATURE_TARGETS_V1)

combine-v1: $(COMBO_TARGETS_V1)

all-v1: learn-all train-all

raw-all: results/features/raw__chess.json

image-baselines-all: $(NN_TARGETS)

eval-transformers: \
	 $(foreach n,1000 10000 25000 50000 100000 last, results/evals/chess_transformer/$(n).json)

eval-transformers-accuracy: \
	 $(foreach n,25000 50000 100000 150000 200000 last, results/evals/chess_transformer/accuracy/$(n).json)

chess-transformer: results/chess_transformer.pt

funsearch-waterbird: results/features/funsearch__image_classification_waterbird__gpt-5-mini.json

transfer: $(TRANSFER_TARGETS)

eval-chess-models: $(CHESS_EVAL_TARGETS) $(CHESS_ACCURACY_TARGETS)

all: all-v2 all-v1 image-baselines-all chess-transformer

.PHONY: all train-all learn-all train-v2 learn-v2 all-v1 all-v2 combine-v1 combine-v2 image-baselines-all chess-transformer funsearch-waterbird eval-transformers eval-chess-models
