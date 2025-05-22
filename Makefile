VENV_DIR := .venv
PYTHON := python3
PIP := $(VENV_DIR)/bin/pip
ACTIVATE := source $(VENV_DIR)/bin/activate

.PHONY: all install clean activate

all: install

install:
	@echo "Creating virtual environment in $(VENV_DIR)..."
	@$(PYTHON) -m venv $(VENV_DIR)
	@echo "Installing requirements..."
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt
	@echo "Environment setup complete."

activate:
	@echo "To activate the environment, run:"
	@echo "$(ACTIVATE)"

clean:
	@echo "Removing virtual environment..."
	@rm -rf $(VENV_DIR)
	@echo "Clean complete."
