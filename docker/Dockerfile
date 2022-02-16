FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-runtime

ARG TESTENV

ENV TESTENV=${TESTENV} \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

# Install and setup poetry
RUN pip install -U pip \
    && apt-get update \
    # opencv2 deps
    && apt-get install ffmpeg libsm6 libxext6  -y \
    && apt-get install -y curl netcat \
    && curl -sSL "https://install.python-poetry.org" | python -

ENV PATH="/root/.local/bin:${PATH}"

# Cleaning cache:
RUN apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false \
    && apt-get clean -y && rm -rf /var/lib/apt/lists/*

RUN poetry --version

WORKDIR /usr/src/code
COPY poetry.lock pyproject.toml /usr/src/code/
RUN poetry install --no-interaction --no-ansi

# copy in source code
COPY . /usr/src/code
