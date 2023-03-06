activate=. venv/bin/activate

build: venv
	$(activate) && maturin build --release

dev: venv
	$(activate) && maturin develop && python main.py

clear_logs:
	rm -rf ./lightning_logs

clean:
	cargo clean