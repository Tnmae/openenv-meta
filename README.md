install uv using pipx/pip and then activate a virtual env. git clone OpenEnv meta-pytorch repo, and then cd into that repo. inside it run "uv pip install -e .", then openenv cli would be available. then run openenv init my_env to initialize a openenv environment. Then run that initialized openenv environment using "uv run server --host 0.0.0.0 --port 8000".
This repo is temporary, would migrate from this to a different repo, which will be used for submission. For Testing using curl: 
->curl http://localhost:8000/state for /state endpoint.
->curl http://localhost:8000/health for /health endpoint. 
->curl http://localhost:8000/schema for /schema endpoint.
->curl -X POST http://localhost:8000/step -H "Content-Type: application/json" -d "{\"action\": {\"message\": \"This is the message to echo back\"}}" for /step endpoint.
->curl http://localhost:8000/reset for /reset endpoint.

With this inference, further development towards bringing the final idea to completion can be done.

/baseline, /tasks and /grader endpoints are yet to be defined, which are a part of qualification criteria.