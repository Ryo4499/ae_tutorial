# Python Docker Template

This is Python Docker Template for Rootless Docker.  
The container is bind on specified a local machine user by `.env` file.  
Also, this directory bind mount to volumes.

## Usage

```sh
git clone $REPO_URL
cd PythonDockerTemplate
cp .env.sample .env
# Specify your environments
vi .env
docker compose build
docker compose up -d
docker compose exec app sh
uv run hello.py
uvx --with pytest-cov pytest-mock pytest
docker compose down
```
