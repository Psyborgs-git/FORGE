.PHONY: help dev backend frontend docker-up docker-down install-backend install-frontend test clean

help:
	@echo "FORGE Development Commands"
	@echo "=========================="
	@echo "make dev              - Start all services (Docker + Frontend dev server)"
	@echo "make backend          - Start backend dev server"
	@echo "make frontend         - Start frontend dev server"
	@echo "make docker-up        - Start Docker services"
	@echo "make docker-down      - Stop Docker services"
	@echo "make install-backend  - Install backend dependencies"
	@echo "make install-frontend - Install frontend dependencies"
	@echo "make test             - Run all tests"
	@echo "make clean            - Clean all build artifacts"

dev: docker-up
	@echo "Starting frontend dev server..."
	cd forge-frontend && npm run dev

backend:
	@echo "Starting backend dev server..."
	cd forge-backend && source .venv/bin/activate && uvicorn main:app --reload --port 8000

frontend:
	@echo "Starting frontend dev server..."
	cd forge-frontend && npm run dev

docker-up:
	@echo "Starting Docker services..."
	docker compose up -d

docker-down:
	@echo "Stopping Docker services..."
	docker compose down

install-backend:
	@echo "Installing backend dependencies..."
	cd forge-backend && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

install-frontend:
	@echo "Installing frontend dependencies..."
	npm install

test:
	@echo "Running backend tests..."
	cd forge-backend && source .venv/bin/activate && pytest
	@echo "Running frontend tests..."
	cd forge-frontend && npm test

clean:
	@echo "Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "node_modules" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
