cd my-langserve/

docker build . -t langserve:latest

docker run -p 8001:8001 -e PORT=8001 langserve:latest


Go to: http://localhost:8001/my-app/playground/
