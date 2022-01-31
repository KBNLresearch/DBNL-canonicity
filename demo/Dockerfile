# https://pythonspeed.com/articles/base-image-python-docker-images/
FROM python:3.9-slim-bullseye as base
MAINTAINER Andreas van Cranenburgh <a.w.van.cranenburgh@rug.nl>
# https://snyk.io/blog/best-practices-containerizing-python-docker/
COPY requirements.txt /
RUN pip install -r /requirements.txt

RUN groupadd -g 999 user && \
    useradd -r -u 999 -g user user

RUN mkdir /usr/app && chown user:user /usr/app
WORKDIR /usr/app

COPY --chown=user:user . .

USER 999
ENV MPLCONFIGDIR=/tmp
ENV GUNICORN_CMD_ARGS="--bind=0.0.0.0:5004 --workers=4 --preload"
CMD ["gunicorn", "web:APP"]
