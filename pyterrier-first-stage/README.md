# PyTerrier Baseline for LongEval'25

This directory contains a PyTerrier baseline for the retrieval task of [LongEval 2025](https://clef-longeval.github.io/). This baseline uses the [LongEval ir_datasets extension](https://github.com/clef-longeval/ir-datasets-longeval) without modification and tracks resource consumption for indexing and retrieval in the ir_metadata format.

## Submitted runs

```
./fusion-with-core.py --dataset longeval-sci/clef-2025-test --output output-fusion-with-core --index indexes --core-documents top-core-documents.jsonl.gz
```


```
./qrel-boost-core.py --dataset longeval-sci/clef-2025-test --prior-stage output-fusion-with-core --output output-qrel-boost-core
```

```
./baseline.py --dataset longeval-sci/clef-2025-test --output output --index indexes
```



## Development

This directory is [configured as DevContainer](https://code.visualstudio.com/docs/devcontainers/containers), i.e., you can open this directory with VS Code or some other DevContainer compatible IDE to work directly in the Docker container with all dependencies installed.

If you want to run it locally, please install the dependencies via `pip3 install -r requirements.txt`.
