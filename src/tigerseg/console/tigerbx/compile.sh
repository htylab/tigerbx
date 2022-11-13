pipenv install --site-packages https://github.com/htylab/tigerseg/archive/main.zip
pipenv install pyinstaller
pipenv run pyinstaller -c --icon=ico.ico -F --add-data libonnxruntime_providers_shared.so:onnxruntime/capi --add-data mprage_v0004_bet_full.onnx:tigerseg/models --add-data mprage_v0002_bet_kuor128.onnx:tigerseg/models --add-data mprage_v0005_aseg43_full.onnx:tigerseg/models --add-data mprage_v0001_aseg43_MXRWr128.onnx:tigerseg/models tigerbet.py
