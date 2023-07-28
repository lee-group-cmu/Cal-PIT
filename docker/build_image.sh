NVC_TAG=22.04
MY_TAG="ngc-${NVC_TAG}-v0"

docker build --platform linux/amd64 --build-arg nvc_tag=$NVC_TAG-py3 -t biprateep/pytorch:$MY_TAG -f Dockerfile .