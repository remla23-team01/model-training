FROM python:3.9-slim
WORKDIR /root
COPY requirements.txt /root/
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]