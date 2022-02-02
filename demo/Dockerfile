# https://pythonspeed.com/articles/base-image-python-docker-images/
FROM python:3.9-slim-bullseye as base
MAINTAINER Andreas van Cranenburgh <a.w.van.cranenburgh@rug.nl>
# https://snyk.io/blog/best-practices-containerizing-python-docker/
COPY requirements.txt /
RUN pip install -r /requirements.txt
ENV HOME=/usr
RUN python -c "import nltk; nltk.download('punkt')"

RUN groupadd -g 999 user && \
    useradd -r -u 999 -g user user

RUN mkdir --parents /usr/app/data /usr/app/templates && \
    chown --recursive user:user /usr/app
WORKDIR /usr/app

COPY --chown=user:user data data/
COPY --chown=user:user templates templates/
COPY --chown=user:user web.py .

USER 999
ENV MPLCONFIGDIR=/tmp
ENV GUNICORN_CMD_ARGS="--bind=0.0.0.0:5004 --workers=4 --preload"
CMD ["gunicorn", "web:APP"]
