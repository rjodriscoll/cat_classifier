# cat classifier 

Installation

Clone the repository and navigate to the project directory.
Create a virtual environment and activate it:

```bash
python -m venv venv
source venv/bin/activate
```
Install the dependencies:

```bash
pip install -r requirements.txt
```
Usage




# Gradio
This is primarily usable via a local ui built in gradio. To run the web app, use 

```python
python3 grad.py
```



# api
Start the API server:

```bash
uvicorn main:app --reload
```


# docker instructions
Install Docker on your machine. You can download Docker from https://www.docker.com/get-started.
Build the Docker image with the following command:

```bash
docker build -t cclf:latest .
```
Run the Docker container with the following command:

```bash
docker run -p 8000:8000 cclf:latest
```

Once the app/container is running, you can send a request to the API by making a POST request to predict.

