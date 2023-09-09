FROM python:3.8-slim

# Create non-root user (Security First!)
RUN useradd --create-home appuser
WORKDIR /home/appuser
USER appuser

# copy whole files to docker
COPY . /home/appuser
RUN pip install -r requirements.txt

# compile py and delete py
RUN ./scripts/compile.sh

# open 50051/tcp port
EXPOSE 50051

# entrypoint
CMD ["./scripts/run.sh"]

