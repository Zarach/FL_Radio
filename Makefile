TAG ?= $(shell git describe --tags --always --dirty)
REGISTRY ?= harbor.foresight-next.plaiful.org/federated-learning

docker-build-server:
	docker build -t ${REGISTRY}/radio-server:${TAG} -f Dockerfile.server .

docker-build: docker-build-server

docker-push: docker-build
	docker push ${REGISTRY}/radio-server:${TAG}