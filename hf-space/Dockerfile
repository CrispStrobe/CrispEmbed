FROM ubuntu:22.04 AS build
WORKDIR /src

RUN printf 'Acquire::Retries "5";\nAcquire::http::Timeout "30";\nAcquire::https::Timeout "30";\n' \
      > /etc/apt/apt.conf.d/80-retries && \
  apt-get update && \
  apt-get install -y --fix-missing build-essential cmake git curl ninja-build && \
  rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

ARG CRISPEMBED_REPO=https://github.com/CrispStrobe/CrispEmbed
ARG CRISPEMBED_REF=main

RUN git clone --depth 1 --recurse-submodules "${CRISPEMBED_REPO}" /src/CrispEmbed && \
  cd /src/CrispEmbed && \
  git fetch --depth 1 origin "${CRISPEMBED_REF}" && \
  git checkout --detach FETCH_HEAD && \
  git submodule update --init --recursive
WORKDIR /src/CrispEmbed
ARG CRISPEMBED_BUILD_JOBS
RUN jobs="${CRISPEMBED_BUILD_JOBS:-$(nproc)}" && \
  cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCRISPEMBED_BUILD_SHARED=ON && \
  cmake --build build -j"${jobs}" --target crispembed-shared crispembed-server crispembed-cli

FROM ubuntu:22.04 AS runtime
WORKDIR /space

RUN printf 'Acquire::Retries "5";\nAcquire::http::Timeout "30";\nAcquire::https::Timeout "30";\n' \
      > /etc/apt/apt.conf.d/80-retries && \
  apt-get update && \
  apt-get install -y --fix-missing bash curl python3 python3-pip && \
  rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

COPY --from=build /src/CrispEmbed/build /opt/crispembed/build
# In HF Space repo, files are at repo root (not hf-space/ subdirectory)
COPY requirements.txt /space/requirements.txt
RUN pip3 install --no-cache-dir -r /space/requirements.txt
RUN find /opt/crispembed/build -type f -name 'lib*.so*' -exec cp -a {} /usr/local/lib/ \; && ldconfig

COPY app.py start.sh /space/
RUN chmod +x /space/start.sh && \
  useradd -m -u 1000 app && \
  mkdir -p /models /cache && \
  chown -R app:app /space /opt/crispembed /models /cache

ENV PATH=/opt/crispembed/build:$PATH
ENV CRISPEMBED_CACHE_DIR=/cache
ENV CRISPEMBED_SERVER_URL=http://127.0.0.1:8090
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

USER app
EXPOSE 7860
ENTRYPOINT ["/space/start.sh"]
