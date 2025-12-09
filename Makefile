ECR := <your-account>.dkr.ecr.eu-central-1.amazonaws.com
SERVICES := ingestion preprocess ml_infer nlp_infer fusion alerting feedback orchestrator
TAG := latest

login-ecr:
	aws ecr get-login-password | docker login --username AWS --password-stdin $(ECR)

build:
	@for s in $(SERVICES); do \
	  docker build -t $(ECR)/$$s:$(TAG) services/$$s ; \
	done

push: login-ecr
	@for s in $(SERVICES); do \
	  docker push $(ECR)/$$s:$(TAG) ; \
	done

deploy:
	kubectl apply -f k8s/