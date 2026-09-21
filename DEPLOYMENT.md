# Deployment

The active application is the root Flask service in `app.py`. It loads the frozen checkpoint from `models/best_model.pth` and serves the existing templates and static files.

## Install

```powershell
python -m pip install -r requirements.txt
```

## Start

Use the Waitress WSGI server for a production-style process:

```powershell
python wsgi.py
```

The default address is `http://127.0.0.1:8000/`. Set `PORT` to the port supplied by the hosting environment. The process must run from the repository root so the model and source paths resolve correctly.

The application is intended for educational and research use. It is not a clinical diagnostic service.