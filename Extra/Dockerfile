ARG UBUNTU_VERSION=20.04
ARG CUDA_VERSION=11.2.1
ARG OPENMPI_VERSION="4.1.0"
ARG OPENMPI_VERSION_MM="4.1"

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}

# Install dependencies
RUN apt-get update -y
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
                                    build-essential \
                                    ca-certificates \
                                    gdb \
                                    gfortran \
                                    wget
RUN apt-get clean all
RUN rm -r /var/lib/apt/lists/*


# BUILD OPENMPI
ARG OPENMPI_CONFIGURE_OPTIONS="--enable-fast=all,O3 --prefix=/usr --with-cuda=/usr/local/cuda"
ARG OPENMPI_MAKE_OPTIONS="-j4"

ARG OPENMPI_VERSION="4.1.0"
ARG OPENMPI_VERSION_MM="4.1"
RUN mkdir -p /tmp/openmpi-build \
    && cd /tmp/openmpi-build \
    && wget "https://download.open-mpi.org/release/open-mpi/v${OPENMPI_VERSION_MM}/openmpi-${OPENMPI_VERSION}.tar.bz2" \
    && tar xjf openmpi-${OPENMPI_VERSION}.tar.bz2 \
    && cd openmpi-${OPENMPI_VERSION} \
    && ./configure ${OPENMPI_CONFIGURE_OPTIONS} \
    && make ${OPENMPI_MAKE_OPTIONS} \
    && make install \
    && ldconfig \
    && cd / \
RUN rm -rf /tmp/openmpi-build


# Final settings
WORKDIR /
CMD ["/bin/bash"]
