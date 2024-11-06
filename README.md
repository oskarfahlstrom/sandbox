# Setup guide
Create a virtual environment for the project. 

```shell
python -m venv .venv  # aligned with .gitignore file
```

After doing that, add it as the python interpreter for your workspace/environment. 

For VSCode I added a file `.env` and add the following to it. Adjust to fit your folder structure.\
`PYTHONPATH=<path to your venv>\Scripts\python.exe`

Next install the required dependencies.

```shell
python.exe -m pip install --upgrade pip
pip install mediapipe>=0.10.18
pip install opencv-python>=4.10
```

Now you should be able to launch the `hand.py` program and have hand recognition using your webcam.
