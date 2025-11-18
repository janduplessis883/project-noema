install:
	@pip install -e .

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -f .DS_Store
	@rm -rf */.ipynb_checkpoints
	@rm -Rf build
	@rm -Rf */__pycache__
	@rm -Rf */*.pyc
	@echo "🧽 Cleaned up successfully!"

all: install clean

data:
	@python noema/data.py

llm:
	@python noema/llm.py

test:
	@pytest -v tests

# Specify package name
lint:
	@black package/
