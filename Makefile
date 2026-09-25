.PHONY: install test build verify test-whatsapp test-inference check-quantization

install:
	npm --prefix apps/whatsapp-adapter install

test: test-whatsapp test-inference check-quantization

build:
	npm --prefix apps/whatsapp-adapter run build

verify: test build

test-whatsapp:
	npm --prefix apps/whatsapp-adapter test

test-inference:
	cd services/inference-runtime && python -m unittest discover -s tests

check-quantization:
	python -m py_compile pipelines/qwen-quantization/quantize.py pipelines/qwen-quantization/benchmark.py pipelines/qwen-quantization/performance.py
