FROM ubuntu:24.04

ARG HADOOP_VERSION=3.4.1
ARG HADOOP_URL=https://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz

ENV DEBIAN_FRONTEND=noninteractive \
    JAVA_HOME=/usr/lib/jvm/java \
    HADOOP_HOME=/opt/hadoop \
    HADOOP_CONF_DIR=/opt/hadoop/etc/hadoop
ENV PATH=${HADOOP_HOME}/bin:${HADOOP_HOME}/sbin:${PATH}

# Java 11 (supporté par Hadoop 3.4) + outils requis par les scripts Hadoop
RUN apt-get update \
 && apt-get install -y --no-install-recommends openjdk-11-jdk-headless curl ca-certificates procps \
 && ln -s /usr/lib/jvm/java-11-openjdk-$(dpkg --print-architecture) ${JAVA_HOME} \
 && rm -rf /var/lib/apt/lists/*

# Hadoop
RUN curl -fSL "${HADOOP_URL}" -o /tmp/hadoop.tgz \
 && tar -xzf /tmp/hadoop.tgz -C /opt \
 && mv /opt/hadoop-${HADOOP_VERSION} ${HADOOP_HOME} \
 && rm -rf /tmp/hadoop.tgz ${HADOOP_HOME}/share/doc

# Utilisateur non-root + répertoires de données (hérités par les volumes nommés)
RUN useradd -m -s /bin/bash hadoop \
 && mkdir -p /hadoop/dfs/name /hadoop/dfs/data ${HADOOP_HOME}/logs \
 && chown -R hadoop:hadoop /hadoop ${HADOOP_HOME}

COPY --chmod=755 entrypoint.sh /entrypoint.sh

USER hadoop
WORKDIR /home/hadoop
ENTRYPOINT ["/entrypoint.sh"]
