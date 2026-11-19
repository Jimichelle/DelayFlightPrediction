#!/usr/bin/env bash
set -euo pipefail

case "${1:-}" in
  namenode)
    # Formatage uniquement au premier démarrage (volume vide)
    if [ ! -f /hadoop/dfs/name/current/VERSION ]; then
      echo "[entrypoint] Formatage du NameNode (clusterId=${CLUSTER_ID})"
      hdfs namenode -format -nonInteractive -clusterid "${CLUSTER_ID}"
    fi
    exec hdfs namenode
    ;;
  datanode)
    # Les arguments suivants (-D clé=valeur) surchargent hdfs-site.xml pour ce DataNode
    shift
    exec hdfs datanode "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
