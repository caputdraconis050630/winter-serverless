#!/usr/bin/env bash
# Rebuild the Knative testbed used by scripts/testbed_measure.py,
# scripts/testbed_e2e.py and scripts/testbed_cohort.py.
#
# The original cluster was created interactively and never scripted, so the
# testbed was not reproducible. This script is that missing artifact: it
# recreates the same topology (single-node kind + Knative Serving + Kourier
# exposed on NodePort 31118, domain 127.0.0.1.sslip.io, namespace
# serverless-test, three real runtime images) from scratch.
#
# Version pin rationale: kind v0.27 ships Kubernetes v1.32 node images and
# Knative v1.18.0 declares a v1.31 minimum, so 1.18.0 is the newest release
# that is certain to run on this kind version.
#
# Usage:  bash testbed/setup_cluster.sh          # create
#         bash testbed/setup_cluster.sh --delete # tear down
set -euo pipefail

CLUSTER=serverless-test
NS=serverless-test
KNATIVE_VERSION=knative-v1.18.0
NODEPORT=31118
KIND_NODE_IMAGE=""     # empty = kind's default for this kind version
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${1:-}" == "--delete" ]]; then
  kind delete cluster --name "$CLUSTER"
  exit 0
fi

# ---------------------------------------------------------------- preflight
echo "== preflight"
root_dir=$(docker info 2>/dev/null | sed -n 's/ *Docker Root Dir: //p')
echo "   docker root: ${root_dir:-unknown}"
case "$root_dir" in
  /data/*) findmnt -no OPTIONS /data | grep -q '\brw\b' \
             || { echo "FATAL: /data is not read-write"; exit 1; }
           avail=$(df -BG --output=avail /data | tail -1 | tr -dc '0-9') ;;
  *)       avail=$(df -BG --output=avail / | tail -1 | tr -dc '0-9') ;;
esac
echo "   available: ${avail}G"
[[ "$avail" -ge 10 ]] || { echo "FATAL: need >=10G free for images"; exit 1; }

# ------------------------------------------------------------------ cluster
echo "== kind cluster"
if kind get clusters 2>/dev/null | grep -qx "$CLUSTER"; then
  echo "   already exists, reusing"
else
  cat >/tmp/kind-${CLUSTER}.yaml <<EOF
kind: Cluster
apiVersion: kind.x-k8s.io/v1alpha4
nodes:
  - role: control-plane
    extraPortMappings:
      - containerPort: ${NODEPORT}
        hostPort: ${NODEPORT}
        protocol: TCP
EOF
  if [[ -n "$KIND_NODE_IMAGE" ]]; then
    kind create cluster --name "$CLUSTER" --config /tmp/kind-${CLUSTER}.yaml \
                        --image "$KIND_NODE_IMAGE"
  else
    kind create cluster --name "$CLUSTER" --config /tmp/kind-${CLUSTER}.yaml
  fi
fi
kubectl config use-context "kind-${CLUSTER}"

# ------------------------------------------------------------------ knative
echo "== knative serving ${KNATIVE_VERSION}"
base=https://github.com/knative/serving/releases/download/${KNATIVE_VERSION}
kubectl apply -f ${base}/serving-crds.yaml
kubectl wait --for=condition=Established --all crd --timeout=120s
kubectl apply -f ${base}/serving-core.yaml
echo "   waiting for serving core..."
kubectl wait --for=condition=Available deployment --all \
             -n knative-serving --timeout=600s

echo "== kourier ingress"
kubectl apply -f \
  https://github.com/knative-extensions/net-kourier/releases/download/${KNATIVE_VERSION}/kourier.yaml
kubectl wait --for=condition=Available deployment --all \
             -n kourier-system --timeout=600s

kubectl patch configmap/config-network -n knative-serving --type merge \
  -p '{"data":{"ingress-class":"kourier.ingress.networking.knative.dev"}}'
# sslip.io domain: the measurement scripts address services as
# <svc>.<ns>.127.0.0.1.sslip.io via a Host header.
kubectl patch configmap/config-domain -n knative-serving --type merge \
  -p '{"data":{"127.0.0.1.sslip.io":""}}'
# Expose kourier on a fixed NodePort so the load generators can hit the node
# IP directly (the convention baked into testbed_measure.py: NODE_IP:31118).
kubectl patch service/kourier -n kourier-system --type merge \
  -p "{\"spec\":{\"type\":\"NodePort\",\"ports\":[{\"name\":\"http2\",\"port\":80,\"targetPort\":8080,\"nodePort\":${NODEPORT},\"protocol\":\"TCP\"}]}}"

# ------------------------------------------------------------------- images
echo "== runtime images"
for d in python-ml node-api java-svc; do
  case "$d" in
    python-ml) tag=dev.local/testbed-python-ml:v1 ;;
    node-api)  tag=dev.local/testbed-node-api:v1 ;;
    java-svc)  tag=dev.local/testbed-java-svc:v1 ;;
  esac
  echo "   building $tag"
  docker build -q -t "$tag" "${HERE}/${d}"
  kind load docker-image "$tag" --name "$CLUSTER"
done

# ----------------------------------------------------------------- services
echo "== namespace + services"
kubectl create namespace "$NS" --dry-run=client -o yaml | kubectl apply -f -
kubectl label namespace "$NS" knative.dev/serving=true --overwrite
kubectl apply -f "${HERE}/ksvc-all.yaml"
for s in python-ml node-api-real java-svc; do
  kubectl wait --for=condition=Ready ksvc/$s -n "$NS" --timeout=300s || true
done

# -------------------------------------------------------------------- ready
NODE_IP=$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' "${CLUSTER}-control-plane")
echo
echo "== cluster ready"
echo "   NODE_IP=${NODE_IP}  PORT=${NODEPORT}"
echo "   autoscaler config (record this with every measurement):"
kubectl get configmap config-autoscaler -n knative-serving \
        -o jsonpath='{.data.stable-window}{" / "}{.data.scale-to-zero-grace-period}{"\n"}' \
  || true
echo "   smoke test:"
echo "     curl -s -o /dev/null -w '%{time_total}\\n' -H 'Host: python-ml.${NS}.127.0.0.1.sslip.io' http://${NODE_IP}:${NODEPORT}"
